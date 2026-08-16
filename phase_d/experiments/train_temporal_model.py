"""
train_temporal_model.py
-------------------------
The training entry point docs/02_TRAINING_GUIDE.txt Step 3 describes:

    python experiments/train_temporal_model.py \
      --model gru --dataset daisee --window-size 90 --epochs 30 \
      --output experiments/runs/gru_daisee_w90/

Run three times (--model gru/lstm/transformer) with everything else
held fixed for the Section 15 ablation; then a --window-size sweep on
the winner; then repeat with --dataset self_collected. This single
script drives every one of those runs -- the only things that vary are
CLI flags, never code, which is what makes the resulting comparisons
fair (see models/common.py's docstring).

Every run writes config.json, metrics.json, confusion_matrix.png, and
best.pt into --output, per the continuation brief's requirement.

Reusable as a library: main(args) is separated from the CLI parsing so
Phase D's own tests can call it directly on --dataset synthetic without
shelling out to a subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.daisee_loader import load_daisee_dataset
from data.self_collected_loader import load_self_collected_dataset
from data.synthetic import generate_synthetic_dataset
from data.types import BEHAVIOR_CLASSES, NUM_CLASSES, NUM_FEATURES, RawSequence, window_sequence, windows_to_arrays
from models.common import build_model, count_parameters

DATASET_NAMES = ("synthetic", "daisee", "self_collected")


@dataclass
class RunConfig:
    model: str
    dataset: str
    window_size: float
    stride_size: float
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    num_layers: int
    dropout: float
    seed: int
    val_fraction: float
    daisee_labels_csv: str | None = None
    daisee_features_dir: str | None = None
    self_collected_csvs: list[str] | None = None


def _load_sequences(cfg: RunConfig) -> list[RawSequence]:
    if cfg.dataset == "synthetic":
        return generate_synthetic_dataset(sequences_per_class=20, duration_seconds=300, seed=cfg.seed)

    if cfg.dataset == "daisee":
        if not cfg.daisee_labels_csv or not cfg.daisee_features_dir:
            raise ValueError("--dataset daisee requires --daisee-labels-csv and --daisee-features-dir")
        sequences, stats = load_daisee_dataset(cfg.daisee_labels_csv, cfg.daisee_features_dir)
        print(f"[daisee] loaded {stats['loaded']} clips "
              f"(unmappable={stats['unmappable']}, missing_features={stats['missing_features']})")
        return sequences

    if cfg.dataset == "self_collected":
        if not cfg.self_collected_csvs:
            raise ValueError("--dataset self_collected requires one or more --self-collected-csv paths")
        sequences, stats = load_self_collected_dataset(cfg.self_collected_csvs)
        print(f"[self_collected] loaded {stats['sessions']} sessions from {stats['rows_read']} rows "
              f"(diagnostic rows skipped={stats['rows_diagnostic_skipped']})")
        return sequences

    raise ValueError(f"unknown dataset {cfg.dataset!r}; expected one of {DATASET_NAMES}")


def _split_sequences(
    sequences: list[RawSequence], val_fraction: float, seed: int
) -> tuple[list[RawSequence], list[RawSequence]]:
    """Split at the SEQUENCE level, not the window level: windows from
    the same source clip/session are highly correlated (overlapping or
    adjacent), so splitting after windowing would leak near-duplicate
    examples between train and val and inflate val accuracy. Splitting
    sequences first, then windowing each split independently, is the
    only leak-free option here."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sequences))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(sequences) * val_fraction))) if len(sequences) > 1 else 0
    val_idx = set(indices[:n_val].tolist())
    train = [s for i, s in enumerate(sequences) if i not in val_idx]
    val = [s for i, s in enumerate(sequences) if i in val_idx]
    return train, val


def _windows_to_dataset(sequences: list[RawSequence], window_size: float, stride_size: float) -> TensorDataset:
    all_windows = []
    for seq in sequences:
        all_windows.extend(window_sequence(seq, window_size, stride_size))
    if not all_windows:
        raise ValueError(
            "windowing produced zero windows -- check that --window-size is not longer "
            "than your shortest sequence, and that sequences have labeled frames"
        )
    x, y = windows_to_arrays(all_windows)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _macro_prf1(cm: np.ndarray) -> tuple[float, float, float]:
    precisions, recalls, f1s = [], [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def _save_confusion_matrix_png(cm: np.ndarray, class_names: list[str], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless -- this script runs on servers/CI with no display
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(cfg: RunConfig, output_dir: str) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)

    sequences = _load_sequences(cfg)
    if len(sequences) < 2:
        raise ValueError("need at least 2 sequences to make a train/val split")
    train_seqs, val_seqs = _split_sequences(sequences, cfg.val_fraction, cfg.seed)

    train_ds = _windows_to_dataset(train_seqs, cfg.window_size, cfg.stride_size)
    val_ds = _windows_to_dataset(val_seqs, cfg.window_size, cfg.window_size)  # non-overlapping for a clean val count

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg.model, NUM_FEATURES, NUM_CLASSES, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        y_true_all, y_pred_all = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                y_pred_all.append(logits.argmax(dim=1).numpy())
                y_true_all.append(yb.numpy())
        val_loss /= len(val_ds)
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        cm = _confusion_matrix(y_true, y_pred, NUM_CLASSES)
        precision, recall, f1 = _macro_prf1(cm)
        accuracy = float((y_true == y_pred).mean())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "val_macro_precision": precision,
                "val_macro_recall": recall,
                "val_macro_f1": f1,
            }
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={accuracy:.4f}  val_f1={f1:.4f}"
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {
                "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "confusion_matrix": cm,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    assert best_state is not None

    # Inference latency + model size + param count, measured on the
    # BEST checkpoint, on CPU (the deployment target -- see the training
    # guide's Step 5 latency requirement).
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    model.to("cpu")
    window_frames = max(1, round(cfg.window_size))  # 1 Hz synthetic/typical assumption for a latency smoke check
    dummy = torch.zeros(1, window_frames, NUM_FEATURES)
    n_latency_runs = 20
    with torch.no_grad():
        model(dummy)  # warm-up, excluded from timing
        start = time.perf_counter()
        for _ in range(n_latency_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    inference_latency_ms = (elapsed / n_latency_runs) * 1000.0

    checkpoint_path = os.path.join(output_dir, "best.pt")
    torch.save(best_state["model_state_dict"], checkpoint_path)
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    metrics = {
        "best_epoch": best_state["epoch"],
        "accuracy": best_state["accuracy"],
        "macro_precision": best_state["precision"],
        "macro_recall": best_state["recall"],
        "macro_f1": best_state["f1"],
        "confusion_matrix": best_state["confusion_matrix"].tolist(),
        "class_names": BEHAVIOR_CLASSES,
        "inference_latency_ms_per_window_cpu": inference_latency_ms,
        "model_size_mb": model_size_mb,
        "parameter_count": count_parameters(model),
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "history": history,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _save_confusion_matrix_png(
        best_state["confusion_matrix"], BEHAVIOR_CLASSES, os.path.join(output_dir, "confusion_matrix.png")
    )

    print(f"\nbest val macro-F1={best_val_f1:.4f} (epoch {best_state['epoch'] + 1}) -> {output_dir}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Phase D temporal engagement classifier.")
    p.add_argument("--model", choices=["gru", "lstm", "transformer"], required=True)
    p.add_argument("--dataset", choices=list(DATASET_NAMES), required=True)
    p.add_argument("--window-size", type=float, default=90.0, help="window length in SECONDS")
    p.add_argument("--stride-size", type=float, default=None, help="window stride in seconds; defaults to --window-size")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--output", required=True)
    p.add_argument("--daisee-labels-csv", default=None)
    p.add_argument("--daisee-features-dir", default=None)
    p.add_argument("--self-collected-csv", action="append", default=None, dest="self_collected_csvs")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    cfg = RunConfig(
        model=args.model,
        dataset=args.dataset,
        window_size=args.window_size,
        stride_size=args.stride_size if args.stride_size is not None else args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        val_fraction=args.val_fraction,
        daisee_labels_csv=args.daisee_labels_csv,
        daisee_features_dir=args.daisee_features_dir,
        self_collected_csvs=args.self_collected_csvs,
    )
    return run(cfg, args.output)


if __name__ == "__main__":
    main()
