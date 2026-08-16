"""
daisee_loader.py
------------------
Loads the DAiSEE benchmark into RawSequence objects, applying the
mapping documented in daisee_label_mapping.py.

TWO-STEP PIPELINE (matches docs/02_TRAINING_GUIDE.txt Step 1's
instruction to "extract per-clip features ... rather than training on
raw pixels"):
  1. experiments/extract_daisee_features.py runs once, offline, over the
     raw DAiSEE video files (which you download separately -- see the
     training guide) and writes one CSV per clip into a features
     directory, using the exact FEATURE_COLUMNS schema from
     data/types.py. This step needs OpenCV + the same MediaPipe models
     as the live watcher (phase_a/phase_c's models/ folder) and is NOT
     run by this loader or by any test in this phase.
  2. THIS module (daisee_loader.py) reads DAiSEE's official Labels CSV
     (columns: ClipID, Boredom, Engagement, Confusion, Frustration --
     the format DAiSEE ships) plus the per-clip feature CSVs from step
     1, and produces RawSequence objects with every frame carrying the
     SAME mapped label (one DAiSEE clip = one label for its whole
     duration; DAiSEE does not provide finer-grained per-frame labels).

This loader never touches a video file or a MediaPipe model directly --
that separation is what keeps it unit-testable with a couple of fake
CSVs (see tests/test_dataset.py) instead of requiring the actual
9,068-clip download to run `pytest`.
"""

from __future__ import annotations

import csv
import os

from .daisee_label_mapping import DaiseeLabels, map_daisee_labels
from .types import FEATURE_COLUMNS, FeatureFrame, RawSequence

# DAiSEE's official label CSV uses these column names (case as shipped
# by the dataset maintainers at time of writing -- if a future DAiSEE
# release renames columns, update this tuple, not the parsing logic
# below).
_LABEL_CSV_COLUMNS = ("ClipID", "Boredom", "Engagement", "Confusion", "Frustration")


def _clip_id_to_source_id(clip_id: str) -> str:
    # DAiSEE ClipIDs typically include a file extension (e.g.
    # "1100011002.avi"); strip it so it matches the feature CSV filename
    # stem produced by extract_daisee_features.py.
    return os.path.splitext(clip_id)[0]


def _read_labels(labels_csv_path: str) -> dict[str, str | None]:
    """clip_id (no extension) -> mapped label, or None if unmappable."""
    mapped: dict[str, str | None] = {}
    with open(labels_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in _LABEL_CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(
                f"DAiSEE labels CSV {labels_csv_path!r} is missing expected "
                f"columns {missing_cols}; found {reader.fieldnames}"
            )
        for row in reader:
            clip_id = _clip_id_to_source_id(row["ClipID"])
            labels = DaiseeLabels(
                engagement=int(row["Engagement"]),
                boredom=int(row["Boredom"]),
                confusion=int(row["Confusion"]),
                frustration=int(row["Frustration"]),
            )
            mapped[clip_id] = map_daisee_labels(labels)
    return mapped


def _read_feature_csv(path: str) -> list[FeatureFrame]:
    frames: list[FeatureFrame] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in ("timestamp", *FEATURE_COLUMNS) if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"feature CSV {path!r} missing columns {missing_cols}")
        for row in reader:
            frames.append(
                FeatureFrame(
                    timestamp=float(row["timestamp"]),
                    yaw=float(row["yaw"]),
                    pitch=float(row["pitch"]),
                    roll=float(row["roll"]),
                    gaze_x=float(row["gaze_x"]),
                    gaze_y=float(row["gaze_y"]),
                    eyes_closed=row["eyes_closed"].strip().lower() in ("1", "true", "yes"),
                    posture_angle=float(row["posture_angle"]),
                    movement_delta=float(row["movement_delta"]),
                    face_conf=float(row["face_conf"]),
                    phone_detected=row["phone_detected"].strip().lower() in ("1", "true", "yes"),
                    second_person_detected=row["second_person_detected"].strip().lower() in ("1", "true", "yes"),
                )
            )
    return frames


def load_daisee_dataset(
    labels_csv_path: str,
    features_dir: str,
    sample_rate_hz: float = 1.0,
) -> tuple[list[RawSequence], dict[str, int]]:
    """Build RawSequences for every DAiSEE clip that (a) has a mappable
    label and (b) has a corresponding <clip_id>.csv in features_dir.

    Returns (sequences, stats) where stats reports how many clips were
    skipped and why -- always log/print this after calling, since a
    silently-shrinking dataset is exactly the kind of thing that should
    be visible in a training report, not just in a debugger.
    """
    label_map = _read_labels(labels_csv_path)

    stats = {"total_labels": len(label_map), "unmappable": 0, "missing_features": 0, "loaded": 0}
    sequences: list[RawSequence] = []

    for clip_id, label in label_map.items():
        if label is None:
            stats["unmappable"] += 1
            continue

        feature_path = os.path.join(features_dir, f"{clip_id}.csv")
        if not os.path.isfile(feature_path):
            stats["missing_features"] += 1
            continue

        frames = _read_feature_csv(feature_path)
        for frame in frames:
            frame.label = label

        sequences.append(RawSequence(source_id=clip_id, frames=frames, sample_rate_hz=sample_rate_hz))
        stats["loaded"] += 1

    return sequences, stats
