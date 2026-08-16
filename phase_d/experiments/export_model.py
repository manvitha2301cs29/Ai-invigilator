"""
export_model.py
----------------
docs/02_TRAINING_GUIDE.txt Step 5:

    python experiments/export_model.py \
      --checkpoint experiments/runs/gru_daisee_w90/best.pt \
      --format torchscript \
      --output watcher/models/temporal_gru.pt

Wraps a trained checkpoint's state_dict (as saved by
train_temporal_model.py) into a TorchScript module for fast CPU
inference inside the live watcher. Reads the sibling config.json from
the checkpoint's run directory to know which architecture/hyperparameters
to rebuild before loading weights -- a state_dict alone doesn't carry
that information, and re-deriving it from CLI flags at export time would
risk mismatching the actual trained architecture.

ONNX export is stubbed but not wired to a default `--format` choice:
TorchScript is sufficient for this project's single-process, PyTorch-only
watcher, and keeps the deployment dependency surface smaller (no onnx /
onnxruntime package needed in the watcher's requirements.txt). The
--format onnx path is left in place because Section 9 leaves the choice
open and a future watcher rewrite (e.g. Phase H's PyInstaller packaging)
may prefer ONNX's smaller runtime -- exercise this path deliberately, not
by default, if that need arises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.types import NUM_CLASSES, NUM_FEATURES
from models.common import build_model


def _load_config(checkpoint_path: str) -> dict:
    run_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"expected a config.json next to {checkpoint_path!r} (written by "
            "train_temporal_model.py) -- export needs it to know which "
            "architecture/hyperparameters produced this checkpoint"
        )
    with open(config_path) as f:
        return json.load(f)


def export(checkpoint_path: str, output_path: str, fmt: str, example_window_frames: int | None = None) -> None:
    cfg = _load_config(checkpoint_path)
    model = build_model(
        cfg["model"], NUM_FEATURES, NUM_CLASSES, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    window_frames = example_window_frames or max(1, round(cfg["window_size"]))
    example_input = torch.zeros(1, window_frames, NUM_FEATURES)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if fmt == "torchscript":
        traced = torch.jit.trace(model, example_input)
        traced.save(output_path)
    elif fmt == "onnx":
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    else:
        raise ValueError(f"unknown export format {fmt!r}; expected 'torchscript' or 'onnx'")

    latency_note = (
        f"exported {cfg['model']} (window={window_frames} frames, "
        f"hidden_size={cfg['hidden_size']}, num_layers={cfg['num_layers']}) -> {output_path}"
    )
    print(latency_note)
    print(
        "Before wiring this into the watcher, confirm inference latency is "
        "well under your feature-sampling interval (see the training guide's "
        "Step 5 latency requirement) -- metrics.json in the run directory "
        "already reports a CPU latency estimate from training time."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export a trained Phase D checkpoint for the watcher.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    p.add_argument("--output", required=True)
    p.add_argument("--example-window-frames", type=int, default=None)
    args = p.parse_args(argv)
    export(args.checkpoint, args.output, args.format, args.example_window_frames)


if __name__ == "__main__":
    main()
