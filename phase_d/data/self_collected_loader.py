"""
self_collected_loader.py
--------------------------
Loads the diary-protocol dataset described in docs/02_TRAINING_GUIDE.txt
Step 2 -- CSV files with one row per timestep and columns:

    timestamp, session_id, participant_id, yaw, pitch, roll, gaze_x,
    gaze_y, eyes_closed, posture_angle, movement_delta, face_conf,
    phone_detected, second_person_detected, look_up_cycle_rate, label

That schema is DELIBERATELY not identical to data/types.py's
FEATURE_COLUMNS: it has three extra columns (session_id, participant_id,
look_up_cycle_rate) that this loader consumes for grouping/context but
does not feed into the model:
  - session_id / participant_id are used here to group rows into
    RawSequences (one sequence per recording session) and are never part
    of the numeric feature vector -- a model that learned to key off
    participant_id would be learning something that doesn't generalize
    to a new student.
  - look_up_cycle_rate is a Phase B *intermediate* signal (used to
    distinguish note-taking from disengagement per Section 6) rather
    than a raw perceptual signal; it's already baked into the
    human-assigned `label` for these clips (a human labeler watching the
    "note-taking" segment writes it down as engaged/idle, not as its own
    class -- see the diary protocol's behaviors 5 and 6, which "feed the
    false-positive test set, not a trained class itself"). Including it
    as a model input here would let the model peek at a hand-computed
    signal that the live watcher's temporal model won't always have in
    exactly this form. It is parsed and validated but intentionally
    dropped before building FeatureFrame objects.

The `label` column is expected to already be one of BEHAVIOR_CLASSES
("engaged" / "idle_present" / "distracted_present") or one of the
false-positive-test-only labels from the diary protocol ("note_taking",
"talking", "eyes_closed_brief", "eyes_closed_prolonged", "phone_present",
"second_person_present", "away") which this loader recognizes but
EXCLUDES from the training set returned by load_self_collected_dataset
(they're diagnostic/false-positive-check data, not training labels) --
see keep_diagnostic_labels to opt into loading them separately for a
false-positive evaluation pass instead.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from .types import BEHAVIOR_CLASSES, FeatureFrame, RawSequence

REQUIRED_COLUMNS = (
    "timestamp",
    "session_id",
    "participant_id",
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
    "look_up_cycle_rate",
    "label",
)

# Labels the diary protocol records that are NOT trained classes -- see
# module docstring. Anything not in BEHAVIOR_CLASSES and not in this set
# is treated as a data-entry error and raises, rather than being
# silently dropped, since a typo'd label should be caught at load time.
DIAGNOSTIC_ONLY_LABELS = {
    "note_taking",
    "talking",
    "eyes_closed_brief",
    "eyes_closed_prolonged",
    "phone_present",
    "second_person_present",
    "away",
}


def _to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def load_self_collected_dataset(
    csv_paths: list[str],
    keep_diagnostic_labels: bool = False,
) -> tuple[list[RawSequence], dict[str, int]]:
    """Load one or more diary-protocol CSVs (typically one per recording
    session, but this also tolerates multiple sessions concatenated in
    one file) and group rows into RawSequences by session_id.

    sample_rate_hz per sequence is inferred from the median timestamp
    gap between consecutive rows of that session -- self-collected
    recordings are less rigidly sampled than the live watcher's fixed
    interval, so trusting a hardcoded 1.0 Hz here would silently
    mis-window the data if a session was actually recorded at, say,
    2 Hz.
    """
    rows_by_session: dict[str, list[dict]] = defaultdict(list)
    stats = {"rows_read": 0, "rows_diagnostic_skipped": 0, "sessions": 0}

    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"self-collected CSV {path!r} missing columns {missing}")
            for row in reader:
                stats["rows_read"] += 1
                label = row["label"].strip()
                if label not in BEHAVIOR_CLASSES and label not in DIAGNOSTIC_ONLY_LABELS:
                    raise ValueError(
                        f"{path!r}: unrecognized label {label!r} -- not a trained class "
                        f"({BEHAVIOR_CLASSES}) or a known diagnostic label "
                        f"({sorted(DIAGNOSTIC_ONLY_LABELS)}). Likely a typo in the CSV."
                    )
                if label in DIAGNOSTIC_ONLY_LABELS and not keep_diagnostic_labels:
                    stats["rows_diagnostic_skipped"] += 1
                    continue
                rows_by_session[row["session_id"]].append(row)

    sequences: list[RawSequence] = []
    for session_id, rows in rows_by_session.items():
        rows.sort(key=lambda r: float(r["timestamp"]))
        timestamps = [float(r["timestamp"]) for r in rows]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
        sample_rate_hz = 1.0 / median_gap if median_gap > 0 else 1.0

        frames = [
            FeatureFrame(
                timestamp=float(r["timestamp"]),
                yaw=float(r["yaw"]),
                pitch=float(r["pitch"]),
                roll=float(r["roll"]),
                gaze_x=float(r["gaze_x"]),
                gaze_y=float(r["gaze_y"]),
                eyes_closed=_to_bool(r["eyes_closed"]),
                posture_angle=float(r["posture_angle"]),
                movement_delta=float(r["movement_delta"]),
                face_conf=float(r["face_conf"]),
                phone_detected=_to_bool(r["phone_detected"]),
                second_person_detected=_to_bool(r["second_person_detected"]),
                label=r["label"].strip(),
            )
            for r in rows
        ]
        sequences.append(RawSequence(source_id=session_id, frames=frames, sample_rate_hz=sample_rate_hz))
        stats["sessions"] += 1

    return sequences, stats
