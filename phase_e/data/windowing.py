"""
windowing.py
------------
Phase D's window_sequence() (reused via reuse.py) REQUIRES every window
to resolve to a majority-vote LABEL, and drops unlabeled frames by
default -- correct for Phase D's supervised classifier, but wrong for
this phase: the autoencoder is unsupervised (Section 5: "an anomaly /
reconstruction model," trained on reconstruction error, not on labels),
so a window with no label at all should still be usable here.

This is the one piece of genuinely new windowing code in Phase E (see
data/__init__.py's reuse-decision docstring for why everything else is
imported, not copied) -- it reuses FeatureFrame.to_array() and
RawSequence from Phase D unchanged, and mirrors window_sequence's
"whole seconds via sample_rate_hz, drop a short tail rather than pad it"
semantics exactly, so the two windowing functions behave identically
except for the label requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reuse import RawSequence


@dataclass
class UnlabeledWindow:
    source_id: str
    start_time: float
    features: np.ndarray  # shape (T, NUM_FEATURES)


def window_sequence_unlabeled(
    seq: RawSequence,
    window_seconds: float,
    stride_seconds: float | None = None,
) -> list[UnlabeledWindow]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    window_frames = max(1, round(window_seconds * seq.sample_rate_hz))
    stride_frames = max(1, round(stride_seconds * seq.sample_rate_hz))

    windows: list[UnlabeledWindow] = []
    n = len(seq.frames)
    start = 0
    while start + window_frames <= n:
        chunk = seq.frames[start : start + window_frames]
        arr = np.stack([f.to_array() for f in chunk], axis=0)
        windows.append(UnlabeledWindow(source_id=seq.source_id, start_time=chunk[0].timestamp, features=arr))
        start += stride_frames

    return windows


def windows_to_array(windows: list[UnlabeledWindow]) -> np.ndarray:
    """Stack same-length UnlabeledWindows into (N, T, NUM_FEATURES)."""
    if not windows:
        raise ValueError("windows_to_array called with an empty list")
    lengths = {w.features.shape[0] for w in windows}
    if len(lengths) != 1:
        raise ValueError(f"windows have inconsistent lengths: {lengths}")
    return np.stack([w.features for w in windows], axis=0)
