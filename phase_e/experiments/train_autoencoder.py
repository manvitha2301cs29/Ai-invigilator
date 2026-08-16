"""
train_autoencoder.py
----------------------
docs/02_TRAINING_GUIDE.txt Step 4:

    python experiments/train_autoencoder.py \
      --dataset self_collected --mode per_participant \
      --output experiments/runs/autoencoder_v1/

Trains SequenceAutoencoder (models/autoencoder.py) on unlabeled,
unsupervised reconstruction of feature windows -- reusing Phase D's
RawSequence + this phase's window_sequence_unlabeled (see
data/windowing.py for why an unlabeled variant is needed).

THREE MODES (Section 5's "per-student ... or pooled + fine-tuned"):
  pooled            One autoencoder trained on every participant's
                     windows together. Fastest to train, weakest fit to
                     any individual student's baseline -- mainly useful
                     as a cold-start model for a brand-new student with
                     no session history yet.
  per_participant   One independently-initialized-and-trained
                     autoencoder per participant_id, using ONLY that
                     participant's own sessions. Matches Section 5's
                     language most directly ("comparing a student's
                     current session pattern to their own established
                     baseline") but needs enough per-student history to
                     train a usable model.
  pooled_finetuned  Train one pooled model first (as in `pooled`), then
                     make a copy per participant and continue training
                     each copy on only that participant's windows for a
                     few more epochs. A practical middle ground: new
                     students start from a sensible pooled prior instead
                     of random initialization, per the training guide's
                     "pretrained on a pooled set and fine-tuned per
                     student" phrasing.

Every mode writes, per participant (or a single top-level model for
`pooled`): model.pt, metrics.json (train reconstruction-error
mean/std -- this IS "the student's own session baseline" as a concrete
number) into --output/<participant_id>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.participants import group_by_participant, read_session_to_participant_map
from data.reuse import NUM_FEATURES, RawSequence, load_self_collected_dataset
from data.synthetic_fatigue import generate_normal_session
from data.windowing import window_sequence_unlabeled, windows_to_array
from models.autoencoder import SequenceAutoencoder

MODES = ("pooled", "per_participant", "pooled_finetuned")


def _sequences_to_dataset(sequences: list[RawSequence], window_seconds: float, stride_seconds: float) -> TensorDataset:
    windows = []
    for seq in sequences:
        windows.extend(window_sequence_unlabeled(seq, window_seconds, stride_seconds))
    if not windows:
        raise ValueError("windowing produced zero windows for this participant/pool")
    x = windows_to_array(windows)
    return TensorDataset(torch.from_numpy(x))


def _load_grouped_sequences(dataset: str, self_collected_csvs: list[str] | None, seed: int) -> dict[str, list[RawSequence]]:
    if dataset == "synthetic":
        # A handful of synthetic "students," each with several normal
        # sessions -- enough to exercise per_participant training without
        # needing real diary-protocol data.
        rng = np.random.default_rng(seed)
        grouped: dict[str, list[RawSequence]] = {}
        for student_i in range(3):
            student_id = f"synthetic_student_{student_i}"
            grouped[student_id] = [
                generate_normal_session(
                    f"{student_id}_session_{s}", seconds=200, seed=int(rng.integers(0, 2**31 - 1))
                )
                for s in range(4)
            ]
        return grouped

    if dataset == "self_collected":
        if not self_collected_csvs:
            raise ValueError("--dataset self_collected requires one or more --self-collected-csv paths")
        sequences, stats = load_self_collected_dataset(self_collected_csvs, keep_diagnostic_labels=True)
        print(f"[self_collected] loaded {stats['sessions']} sessions from {stats['rows_read']} rows")
        session_to_participant = read_session_to_participant_map(self_collected_csvs)
        return group_by_participant(sequences, session_to_participant)

    raise ValueError(f"unknown dataset {dataset!r}; expected 'synthetic' or 'self_collected'")


def _train_one_model(
    dataset: TensorDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_size: int,
    num_layers: int,
    init_state_dict: dict | None = None,
) -> tuple[SequenceAutoencoder, dict]:
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = SequenceAutoencoder(NUM_FEATURES, latent_size=latent_size, num_layers=num_layers)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(dataset)
        history.append({"epoch": epoch, "train_mse": total_loss})
        print(f"  epoch {epoch + 1}/{epochs}  train_mse={total_loss:.5f}")

    # The final-epoch, per-window reconstruction errors on this
    # participant's OWN training data are recorded as their baseline
    # (mean/std) -- this is the concrete number Section 5's "relative to
    # the student's own session baseline" cashes out to at inference
    # time (Phase F reads baseline_mean/baseline_std, not raw MSE, when
    # deciding what counts as anomalous for THIS student).
    model.eval()
    errors = []
    with torch.no_grad():
        for (xb,) in DataLoader(dataset, batch_size=64, shuffle=False):
            recon = model(xb)
            per_window = ((recon - xb) ** 2).mean(dim=(1, 2))
            errors.extend(per_window.tolist())
    errors_arr = np.array(errors)

    metrics = {
        "history": history,
        "num_windows": len(dataset),
        "baseline_reconstruction_error_mean": float(errors_arr.mean()),
        "baseline_reconstruction_error_std": float(errors_arr.std()),
    }
    return model, metrics


def run(
    dataset: str,
    mode: str,
    output_dir: str,
    window_seconds: float = 60.0,
    stride_seconds: float | None = None,
    epochs: int = 20,
    finetune_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_size: int = 16,
    num_layers: int = 1,
    seed: int = 0,
    self_collected_csvs: list[str] | None = None,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    grouped = _load_grouped_sequences(dataset, self_collected_csvs, seed)
    per_participant_metrics: dict[str, dict] = {}

    if mode == "per_participant":
        for participant_id, sequences in grouped.items():
            print(f"training per-participant autoencoder for {participant_id!r} ({len(sequences)} sessions)")
            ds = _sequences_to_dataset(sequences, window_seconds, stride_seconds)
            model, metrics = _train_one_model(ds, epochs, batch_size, lr, latent_size, num_layers)
            _save_participant(output_dir, participant_id, model, metrics)
            per_participant_metrics[participant_id] = metrics

    else:  # pooled or pooled_finetuned
        all_sequences = [seq for seqs in grouped.values() for seq in seqs]
        pooled_ds = _sequences_to_dataset(all_sequences, window_seconds, stride_seconds)
        print(f"training pooled autoencoder ({len(grouped)} participants, {len(pooled_ds)} windows)")
        pooled_model, pooled_metrics = _train_one_model(pooled_ds, epochs, batch_size, lr, latent_size, num_layers)
        _save_participant(output_dir, "__pooled__", pooled_model, pooled_metrics)
        per_participant_metrics["__pooled__"] = pooled_metrics

        if mode == "pooled_finetuned":
            for participant_id, sequences in grouped.items():
                print(f"fine-tuning pooled model for {participant_id!r} ({len(sequences)} sessions)")
                ds = _sequences_to_dataset(sequences, window_seconds, stride_seconds)
                model, metrics = _train_one_model(
                    ds,
                    finetune_epochs,
                    batch_size,
                    lr * 0.1,  # smaller LR for fine-tuning, standard practice
                    latent_size,
                    num_layers,
                    init_state_dict=copy.deepcopy(pooled_model.state_dict()),
                )
                _save_participant(output_dir, participant_id, model, metrics)
                per_participant_metrics[participant_id] = metrics

    summary = {"dataset": dataset, "mode": mode, "window_seconds": window_seconds, "participants": per_participant_metrics}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_participant(output_dir: str, participant_id: str, model: SequenceAutoencoder, metrics: dict) -> None:
    participant_dir = os.path.join(output_dir, participant_id)
    os.makedirs(participant_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(participant_dir, "model.pt"))
    with open(os.path.join(participant_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "num_features": model.num_features,
                "latent_size": model.latent_size,
                **metrics,
            },
            f,
            indent=2,
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the Phase E fatigue-pattern autoencoder.")
    p.add_argument("--dataset", choices=["synthetic", "self_collected"], required=True)
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window-size", type=float, default=60.0, dest="window_seconds")
    p.add_argument("--stride-size", type=float, default=None, dest="stride_seconds")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-collected-csv", action="append", default=None, dest="self_collected_csvs")
    args = p.parse_args(argv)

    run(
        dataset=args.dataset,
        mode=args.mode,
        output_dir=args.output,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        seed=args.seed,
        self_collected_csvs=args.self_collected_csvs,
    )


if __name__ == "__main__":
    main()
