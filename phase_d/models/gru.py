"""
gru.py
------
The PRIMARY temporal model (per the continuation brief: "GRU primary
temporal model"). A standard multi-layer GRU over the window's feature
sequence, classifying from the final timestep's hidden state -- GRU is
chosen as primary over LSTM for the deployment reason documented in
Section 9 of AI_Study_Invigilator_Project.txt: fewer parameters than an
equivalent LSTM (no separate cell state, no output gate) for comparable
sequence-modeling capacity, which matters directly for the watcher's
CPU-only, continuous-inference deployment target (see
docs/02_TRAINING_GUIDE.txt Step 5's latency requirement).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        _, h_n = self.gru(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's
        # final-timestep hidden state, the standard "sequence summary" for
        # a many-to-one classification head.
        return self.head(self.dropout(last_layer_hidden))
