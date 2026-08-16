"""
common.py
---------
Shared interface every temporal model (GRU/LSTM/Transformer) implements,
plus the build_model() factory train.py uses to select one by name.

WHY A SHARED INTERFACE: Section 15 of AI_Study_Invigilator_Project.txt
asks for a "fair ablation" between the three architectures -- that's
only meaningful if train.py, the loss function, the optimizer, the
metrics computation, and the export step are IDENTICAL for all three,
with architecture as the only varying factor. Concretely: every model
below takes (batch, seq_len, NUM_FEATURES) float32 and returns
(batch, NUM_CLASSES) raw logits, nothing else -- no model-specific extra
outputs, no model-specific preprocessing.

TemporalClassifier is a typing.Protocol (structural typing), not a
literal shared base class, since GRU/LSTM/Transformer's internals differ
enough (recurrent hidden state vs. attention) that forcing a common
nn.Module base class buys nothing beyond what the shared forward()
signature already guarantees.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

MODEL_NAMES = ("gru", "lstm", "transformer")


class TemporalClassifier(Protocol):
    """Structural interface: anything with this forward signature can be
    trained by experiments/train_temporal_model.py."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) float32.
        returns: (batch, num_classes) raw logits (no softmax -- train.py
        applies CrossEntropyLoss, which expects raw logits)."""
        ...


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    name: str,
    num_features: int,
    num_classes: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory used by train.py's --model flag. Keeping this in
    common.py (rather than each model file registering itself) keeps the
    set of valid --model values explicit and greppable in one place."""
    name = name.lower()
    if name == "gru":
        from .gru import GRUClassifier

        return GRUClassifier(num_features, num_classes, hidden_size, num_layers, dropout)
    if name == "lstm":
        from .lstm import LSTMClassifier

        return LSTMClassifier(num_features, num_classes, hidden_size, num_layers, dropout)
    if name == "transformer":
        from .transformer import TransformerClassifier

        return TransformerClassifier(num_features, num_classes, hidden_size, num_layers, dropout)
    raise ValueError(f"unknown model name {name!r}; expected one of {MODEL_NAMES}")
