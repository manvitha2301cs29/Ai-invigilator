"""
lstm.py
-------
COMPARISON baseline #1 for the Section 15 ablation. Architecturally
identical in spirit to gru.py (multi-layer recurrent encoder, classify
from the final hidden state) so the comparison isolates GRU-vs-LSTM,
not some other unrelated difference like hidden size or head design --
every hyperparameter here matches GRUClassifier's defaults exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _c_n) = self.lstm(x)  # c_n (cell state) unused by the
        # classification head -- kept only because nn.LSTM always
        # returns it; discarding it here (rather than the hidden state)
        # matches the same "final layer's final-timestep hidden state"
        # convention as GRUClassifier.
        last_layer_hidden = h_n[-1]
        return self.head(self.dropout(last_layer_hidden))
