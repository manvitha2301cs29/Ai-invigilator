"""
types.py
--------
Dataset-agnostic core of Phase D: a RawSequence/windowing abstraction
that every loader (synthetic, DAiSEE, self-collected) converts its
source data INTO, so the training loop (train.py) and the three model
architectures (models/) never know or care where a sequence came from.

Implements the "shared interface for a fair ablation" requirement from
Section 15 of AI_Study_Invigilator_Project.txt: GRU, LSTM, and
Transformer all consume the exact same Window objects, produced by the
exact same windowing function, so any accuracy difference between them
is attributable to architecture, not to accidentally different
preprocessing per model.

FEATURE SCHEMA -- matches phase_c/backend/models.py's StateEvent table
and docs/02_TRAINING_GUIDE.txt's self-collected CSV columns exactly
(yaw, pitch, roll, gaze_x, gaze_y, eyes_closed, posture_angle,
movement_delta, face_conf, phone_detected, second_person_detected):
this is deliberate, not a coincidence -- it's what lets a model trained
here be fed features straight from Phase B's live FeatureVector /
Phase C's StateEvent rows with no schema translation step at inference
time in the watcher.

BEHAVIOR TAXONOMY -- Engaged / Idle-present / Distracted-present only.
"Away" is intentionally excluded from the set of trained classes: per
Section 4 Layer 3 of the project doc, Away is handled as a deterministic
rule (no face detected for a sustained period) in Phase C's monitor.py,
not learned, since it's trivially and reliably detectable without a
model and training on it would just waste model capacity on an already-
solved problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Order matters: this list IS the label-index mapping used everywhere
# in Phase D (loss functions, confusion matrices, argmax decoding).
# "Away" is deliberately not in this list -- see module docstring.
BEHAVIOR_CLASSES: list[str] = [
    "engaged",
    "idle_present",
    "distracted_present",
]
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(BEHAVIOR_CLASSES)}
NUM_CLASSES = len(BEHAVIOR_CLASSES)

# Column order for the numeric feature vector at each timestep. Booleans
# are represented as 0.0/1.0 so the whole sequence can live in one float
# array -- this matches how StateEvent's typed boolean columns
# (eyes_closed, phone_detected, second_person_detected) get exported by
# any SQL query into a training CSV.
FEATURE_COLUMNS: list[str] = [
    "yaw",
    "pitch",
    "roll",
    "gaze_x",
    "gaze_y",
    "eyes_closed",
    "posture_angle",
    "movement_delta",
    "face_conf",
    "phone_detected",
    "second_person_detected",
]
NUM_FEATURES = len(FEATURE_COLUMNS)


@dataclass
class FeatureFrame:
    """One timestep's numeric feature vector plus its ground-truth label
    (if labeled -- unlabeled frames are only used by Phase E's
    autoencoder, never by this phase's supervised classifier)."""

    timestamp: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    eyes_closed: bool = False
    posture_angle: float = 0.0
    movement_delta: float = 0.0
    face_conf: float = 1.0
    phone_detected: bool = False
    second_person_detected: bool = False
    label: str | None = None

    def to_array(self) -> np.ndarray:
        """Numeric vector in FEATURE_COLUMNS order, float32, ready to be
        stacked into a sequence array."""
        return np.array(
            [
                self.yaw,
                self.pitch,
                self.roll,
                self.gaze_x,
                self.gaze_y,
                1.0 if self.eyes_closed else 0.0,
                self.posture_angle,
                self.movement_delta,
                self.face_conf,
                1.0 if self.phone_detected else 0.0,
                1.0 if self.second_person_detected else 0.0,
            ],
            dtype=np.float32,
        )


@dataclass
class RawSequence:
    """One continuous, chronologically-ordered run of FeatureFrames from
    a single source clip/session (one DAiSEE video, one self-collected
    diary segment, or one synthetic session). sample_rate_hz documents
    the assumed spacing between frames, since DAiSEE clips and
    self-collected CSVs may be sampled at different rates -- windowing
    always operates in SECONDS (window_seconds), not raw frame counts,
    so a --window-size 90 run means "90 seconds" regardless of source
    sample rate.
    """

    source_id: str
    frames: list[FeatureFrame] = field(default_factory=list)
    sample_rate_hz: float = 1.0

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class Window:
    """A fixed-length slice of a RawSequence, ready for a model: a dense
    (T, NUM_FEATURES) float32 array and a single label for the whole
    window. Windows -- not raw sequences -- are what train.py's
    DataLoader actually batches, since sequences vary in length but
    windows are fixed-size by construction.

    label is majority-vote over the window's frame labels (see
    window_sequence's WINDOW_LABEL_STRATEGY note) rather than e.g. the
    label of the last frame, since a single window can straddle a brief
    labeling transition in DAiSEE/self-collected data and majority vote
    is the more robust choice for a short window.
    """

    source_id: str
    start_time: float
    features: np.ndarray  # shape (T, NUM_FEATURES)
    label: str
    label_index: int


def window_sequence(
    seq: RawSequence,
    window_seconds: float,
    stride_seconds: float | None = None,
    drop_unlabeled: bool = True,
) -> list[Window]:
    """Slice a RawSequence into fixed-length Windows.

    window_seconds / stride_seconds are in SECONDS, converted to frame
    counts via seq.sample_rate_hz -- this is what makes `--window-size
    90` mean the same real-world duration whether the underlying source
    was sampled at 1 Hz or 5 Hz. stride defaults to window_seconds
    (non-overlapping windows); pass a smaller stride for overlapping
    windows (more training examples, some correlation between them --
    a legitimate trade-off to sweep during hyperparameter tuning, not
    something this function decides for the caller).

    Windows with fewer than window_frames frames (i.e. the tail end of
    a sequence shorter than one window) are dropped, since padding a
    short window changes what the model actually experiences relative to
    real-time deployment, where the watcher always has a full window's
    worth of history before it ever calls the model.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    window_frames = max(1, round(window_seconds * seq.sample_rate_hz))
    stride_frames = max(1, round(stride_seconds * seq.sample_rate_hz))

    windows: list[Window] = []
    n = len(seq.frames)
    start = 0
    while start + window_frames <= n:
        chunk = seq.frames[start : start + window_frames]

        if drop_unlabeled and any(f.label is None for f in chunk):
            start += stride_frames
            continue

        # WINDOW_LABEL_STRATEGY: majority vote across the window's
        # per-frame labels. Ties broken by earliest-appearing label in
        # BEHAVIOR_CLASSES order (deterministic, documented, not random)
        # so results are reproducible run to run.
        counts: dict[str, int] = {}
        for f in chunk:
            if f.label is not None:
                counts[f.label] = counts.get(f.label, 0) + 1
        if not counts:
            start += stride_frames
            continue
        best_label = max(
            counts.items(),
            key=lambda kv: (kv[1], -BEHAVIOR_CLASSES.index(kv[0]) if kv[0] in BEHAVIOR_CLASSES else -999),
        )[0]

        if best_label not in LABEL_TO_INDEX:
            # e.g. a DAiSEE-mapped "unmappable" bucket -- skip rather
            # than silently mis-training on it.
            start += stride_frames
            continue

        arr = np.stack([f.to_array() for f in chunk], axis=0)
        windows.append(
            Window(
                source_id=seq.source_id,
                start_time=chunk[0].timestamp,
                features=arr,
                label=best_label,
                label_index=LABEL_TO_INDEX[best_label],
            )
        )
        start += stride_frames

    return windows


def windows_to_arrays(windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of same-length Windows into (N, T, NUM_FEATURES) and
    (N,) arrays for training. Raises if window lengths differ -- callers
    should window a whole dataset with the same window_seconds so this
    never happens; a clear error here is better than a silent shape
    mismatch inside PyTorch three function calls later."""
    if not windows:
        raise ValueError("windows_to_arrays called with an empty list")
    lengths = {w.features.shape[0] for w in windows}
    if len(lengths) != 1:
        raise ValueError(f"windows have inconsistent lengths: {lengths}")
    x = np.stack([w.features for w in windows], axis=0)
    y = np.array([w.label_index for w in windows], dtype=np.int64)
    return x, y
