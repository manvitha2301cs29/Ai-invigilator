"""
transformer.py
--------------
COMPARISON baseline #2 for the Section 15 ablation. A small
Transformer encoder (nn.TransformerEncoder) over the window's feature
sequence, with a learned positional encoding (windows are short and
fixed-length, so a learned embedding table is simpler and just as
effective here as sinusoidal encoding, and keeps this file dependency-
free). Classifies from a mean-pooled representation across timesteps
(NOT a CLS token -- no natural "start of sequence" token exists in a
continuous sensor stream the way it does in text, so mean pooling is
the more defensible choice here and is called out explicitly for the
report per the training guide's instruction to document non-obvious
choices).

`hidden_size` and `num_layers` are reused with the exact same meaning
as gru.py/lstm.py's constructor arguments (embedding dim, encoder-layer
count) purely so the three models can share one --hidden-size /
--num-layers CLI flag pair in train.py -- see models/common.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_WINDOW_FRAMES = 512  # generous upper bound -- covers the training
# guide's largest swept window size (120s at up to ~4 Hz) with headroom;
# raise if a future window-size sweep needs longer sequences.


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            # nn.MultiheadAttention requires this; fail loudly at
            # construction time rather than inside the first forward().
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.input_proj = nn.Linear(num_features, hidden_size)
        self.positional_embedding = nn.Embedding(MAX_WINDOW_FRAMES, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if seq_len > MAX_WINDOW_FRAMES:
            raise ValueError(
                f"window has {seq_len} frames, exceeds MAX_WINDOW_FRAMES={MAX_WINDOW_FRAMES}"
            )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        h = self.input_proj(x) + self.positional_embedding(positions)
        h = self.encoder(h)  # (batch, seq_len, hidden_size)
        pooled = h.mean(dim=1)  # mean pooling -- see module docstring
        return self.head(self.dropout(pooled))
