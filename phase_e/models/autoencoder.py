"""
autoencoder.py
---------------
A GRU encoder-decoder that reconstructs a window's feature sequence.
Implements Section 5's FATIGUED-PATTERN: "Produced by an anomaly /
reconstruction model (autoencoder), not a hard classifier -- explicitly
comparative, never an absolute claim about tiredness." The output of
this model is a per-window reconstruction ERROR (a number), never a
"fatigued: yes/no" label -- turning that number into any
student-facing wording happens in Phase F (thresholds) and Phase G
(LLM phrasing), never here.

ARCHITECTURE
  Encoder: GRU over the input window -> final hidden state = a fixed-
  size latent vector summarizing the window.
  Decoder: the latent vector is broadcast (repeated) across every
  output timestep as the decoder GRU's input at each step ("repeat
  vector" pattern) -- simpler than teacher-forcing the decoder with a
  shifted copy of the true sequence, and appropriate here since, unlike
  language modeling, there's no notion of "the previous token" for a
  continuous sensor stream that the decoder should condition on beyond
  what the latent already captures.
  A final Linear layer projects each decoder timestep back to
  NUM_FEATURES, and MSE against the original input is the reconstruction
  loss AND the anomaly score at inference time.

WHY NOT A SINGLE SHARED-WEIGHTS FIT ACROSS ALL STUDENTS: Section 5 is
explicit that this must be comparative to "THE STUDENT'S OWN SESSION
BASELINE," not a population norm -- see experiments/train_autoencoder.py
for how --mode per_participant enforces that at the training-run level;
this class itself is student-agnostic and simply gets trained once per
student (or pooled-then-fine-tuned, per the training guide).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceAutoencoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        latent_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.latent_size = latent_size

        self.encoder = nn.GRU(
            input_size=num_features,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=latent_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(latent_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) -> reconstruction of the
        same shape."""
        batch, seq_len, _ = x.shape
        _, h_n = self.encoder(x)
        latent = h_n[-1]  # (batch, latent_size) -- final layer's summary

        # "repeat vector": feed the same latent as decoder input at
        # every timestep (see module docstring for why).
        decoder_input = latent.unsqueeze(1).expand(batch, seq_len, self.latent_size)
        decoded, _ = self.decoder(decoder_input)
        return self.output_proj(decoded)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent vector only -- exposed for callers (e.g. a future
        clustering/visualization tool) that want the summary without a
        reconstruction."""
        _, h_n = self.encoder(x)
        return h_n[-1]


def reconstruction_error(model: SequenceAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-window mean-squared reconstruction error: (batch,) tensor,
    one scalar anomaly score per window. This -- and only this number --
    is what Phase F's recommendation engine and Phase G's LLM layer are
    allowed to see; neither ever sees the reconstructed sequence or the
    latent vector directly (Section 5 / the project's determinism
    principle: numbers come from plain code, comparative statements
    about a number are what get phrased downstream)."""
    with torch.no_grad():
        recon = model(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))
