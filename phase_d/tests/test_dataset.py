"""
test_dataset.py
-----------------
Unit tests for the dataset-agnostic core (data/types.py) and the three
loaders, using only synthetic/hand-built data and temp CSV fixtures --
no DAiSEE download, no self-collected recordings, no camera required to
run these, matching phase_b/phase_c's test-discipline.

Run with:
    pytest tests/test_dataset.py -v
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.daisee_label_mapping import DaiseeLabels, map_daisee_labels
from data.daisee_loader import load_daisee_dataset
from data.self_collected_loader import DIAGNOSTIC_ONLY_LABELS, load_self_collected_dataset
from data.synthetic import generate_synthetic_dataset, generate_synthetic_sequence
from data.types import FEATURE_COLUMNS, FeatureFrame, RawSequence, window_sequence, windows_to_arrays


# ---------------------------------------------------------------------------
# windowing core
# ---------------------------------------------------------------------------

def _make_sequence(labels: list[str], sample_rate_hz: float = 1.0) -> RawSequence:
    frames = [
        FeatureFrame(timestamp=float(i), yaw=float(i), label=label)
        for i, label in enumerate(labels)
    ]
    return RawSequence(source_id="test_seq", frames=frames, sample_rate_hz=sample_rate_hz)


def test_window_sequence_produces_expected_count_non_overlapping():
    """10 one-second frames, 5-second windows, no overlap -> exactly 2 windows."""
    seq = _make_sequence(["engaged"] * 10)
    windows = window_sequence(seq, window_seconds=5)
    assert len(windows) == 2
    assert windows[0].features.shape == (5, len(FEATURE_COLUMNS))


def test_window_sequence_overlapping_stride_produces_more_windows():
    """A stride shorter than the window length produces overlapping
    windows (Section 15's stride-as-hyperparameter behavior)."""
    seq = _make_sequence(["engaged"] * 10)
    non_overlap = window_sequence(seq, window_seconds=5, stride_seconds=5)
    overlap = window_sequence(seq, window_seconds=5, stride_seconds=1)
    assert len(overlap) > len(non_overlap)


def test_window_sequence_drops_short_tail():
    """A sequence shorter than one window produces zero windows, never a
    padded partial window (per types.py's documented design choice)."""
    seq = _make_sequence(["engaged"] * 3)
    windows = window_sequence(seq, window_seconds=5)
    assert windows == []


def test_window_label_is_majority_vote():
    """A window straddling a label change takes the majority label, not
    e.g. the first or last frame's label."""
    labels = ["engaged"] * 3 + ["distracted_present"] * 2  # 3 vs 2 in a 5-frame window
    seq = _make_sequence(labels)
    windows = window_sequence(seq, window_seconds=5)
    assert len(windows) == 1
    assert windows[0].label == "engaged"


def test_window_sequence_drops_unlabeled_frames_by_default():
    seq = _make_sequence(["engaged", "engaged", None, "engaged", "engaged"])  # type: ignore[list-item]
    windows = window_sequence(seq, window_seconds=5)
    assert windows == []


def test_window_sequence_respects_sample_rate_for_duration():
    """window_seconds means real time, not frame count -- at 2 Hz, a
    5-second window should span 10 frames, not 5."""
    seq = _make_sequence(["engaged"] * 20, sample_rate_hz=2.0)
    windows = window_sequence(seq, window_seconds=5)
    assert windows[0].features.shape[0] == 10


def test_windows_to_arrays_shapes_and_dtypes():
    seq = _make_sequence(["engaged"] * 10)
    windows = window_sequence(seq, window_seconds=5)
    x, y = windows_to_arrays(windows)
    assert x.shape == (2, 5, len(FEATURE_COLUMNS))
    assert y.shape == (2,)
    assert x.dtype.name == "float32"
    assert y.dtype.name == "int64"


def test_windows_to_arrays_rejects_inconsistent_lengths():
    seq_a = _make_sequence(["engaged"] * 10)
    seq_b = _make_sequence(["engaged"] * 10, sample_rate_hz=2.0)
    windows = window_sequence(seq_a, window_seconds=5) + window_sequence(seq_b, window_seconds=5)
    try:
        windows_to_arrays(windows)
        assert False, "expected ValueError for mismatched window lengths"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# synthetic generator
# ---------------------------------------------------------------------------

def test_synthetic_dataset_has_all_three_classes_and_is_reproducible():
    ds1 = generate_synthetic_dataset(sequences_per_class=2, duration_seconds=30, seed=42)
    ds2 = generate_synthetic_dataset(sequences_per_class=2, duration_seconds=30, seed=42)
    labels = {seq.frames[0].label for seq in ds1}
    assert labels == {"engaged", "idle_present", "distracted_present"}
    # same seed -> identical first frame values (reproducibility for CI)
    assert ds1[0].frames[0].yaw == ds2[0].frames[0].yaw


def test_synthetic_sequence_is_internally_consistent():
    seq = generate_synthetic_sequence("s1", "engaged", duration_seconds=60, seed=1)
    assert len(seq) == 60
    assert all(f.label == "engaged" for f in seq.frames)
    assert seq.sample_rate_hz == 1.0


# ---------------------------------------------------------------------------
# DAiSEE label mapping
# ---------------------------------------------------------------------------

def test_daisee_mapping_high_engagement_low_negative_affect_is_engaged():
    assert map_daisee_labels(DaiseeLabels(engagement=3, boredom=0, confusion=0, frustration=0)) == "engaged"


def test_daisee_mapping_low_engagement_low_everything_is_idle_present():
    assert map_daisee_labels(DaiseeLabels(engagement=1, boredom=0, confusion=0, frustration=0)) == "idle_present"


def test_daisee_mapping_low_engagement_high_boredom_is_distracted_present():
    assert map_daisee_labels(DaiseeLabels(engagement=0, boredom=3, confusion=0, frustration=0)) == "distracted_present"


def test_daisee_mapping_low_engagement_high_confusion_is_distracted_present():
    assert map_daisee_labels(DaiseeLabels(engagement=1, boredom=0, confusion=2, frustration=0)) == "distracted_present"


def test_daisee_mapping_ambiguous_case_is_unmappable():
    # moderate engagement, moderate confusion: doesn't satisfy any rule
    assert map_daisee_labels(DaiseeLabels(engagement=2, boredom=0, confusion=2, frustration=0)) is None


def test_daisee_labels_rejects_out_of_range_score():
    try:
        DaiseeLabels(engagement=4, boredom=0, confusion=0, frustration=0)
        assert False, "expected ValueError for out-of-range score"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# DAiSEE loader (fixture CSVs, no real dataset)
# ---------------------------------------------------------------------------

def test_daisee_loader_end_to_end_with_fixtures():
    with tempfile.TemporaryDirectory() as tmp:
        features_dir = os.path.join(tmp, "features")
        os.makedirs(features_dir)

        # one mappable, one unmappable, one mappable-but-missing-features clip
        labels_csv = os.path.join(tmp, "labels.csv")
        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ClipID", "Boredom", "Engagement", "Confusion", "Frustration"])
            writer.writerow(["clip_engaged.avi", 0, 3, 0, 0])       # -> engaged
            writer.writerow(["clip_ambiguous.avi", 0, 2, 2, 0])     # -> unmappable
            writer.writerow(["clip_missing.avi", 0, 3, 0, 0])       # -> engaged, but no feature CSV

        with open(os.path.join(features_dir, "clip_engaged.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", *FEATURE_COLUMNS])
            writer.writeheader()
            for t in range(3):
                writer.writerow(
                    {
                        "timestamp": t, "yaw": 0, "pitch": 0, "roll": 0, "gaze_x": 0, "gaze_y": 0,
                        "eyes_closed": "false", "posture_angle": 0, "movement_delta": 0.1,
                        "face_conf": 0.9, "phone_detected": "false", "second_person_detected": "false",
                    }
                )

        sequences, stats = load_daisee_dataset(labels_csv, features_dir)

        assert stats["total_labels"] == 3
        assert stats["unmappable"] == 1
        assert stats["missing_features"] == 1
        assert stats["loaded"] == 1
        assert len(sequences) == 1
        assert sequences[0].source_id == "clip_engaged"
        assert all(f.label == "engaged" for f in sequences[0].frames)


# ---------------------------------------------------------------------------
# self-collected loader (fixture CSV, no real recordings)
# ---------------------------------------------------------------------------

def test_self_collected_loader_groups_by_session_and_infers_sample_rate():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "diary.csv")
        fieldnames = [
            "timestamp", "session_id", "participant_id", "yaw", "pitch", "roll", "gaze_x", "gaze_y",
            "eyes_closed", "posture_angle", "movement_delta", "face_conf", "phone_detected",
            "second_person_detected", "look_up_cycle_rate", "label",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # session A: 0.5s spacing (2 Hz), all "engaged"
            for i in range(4):
                writer.writerow(
                    {
                        "timestamp": i * 0.5, "session_id": "A", "participant_id": "p1", "yaw": 0, "pitch": 0,
                        "roll": 0, "gaze_x": 0, "gaze_y": 0, "eyes_closed": "false", "posture_angle": 0,
                        "movement_delta": 0.1, "face_conf": 0.9, "phone_detected": "false",
                        "second_person_detected": "false", "look_up_cycle_rate": 0, "label": "engaged",
                    }
                )
            # session B: 1s spacing (1 Hz), a diagnostic-only label
            writer.writerow(
                {
                    "timestamp": 0, "session_id": "B", "participant_id": "p1", "yaw": 0, "pitch": 0, "roll": 0,
                    "gaze_x": 0, "gaze_y": 0, "eyes_closed": "false", "posture_angle": 0, "movement_delta": 0.1,
                    "face_conf": 0.9, "phone_detected": "true", "second_person_detected": "false",
                    "look_up_cycle_rate": 0, "label": "phone_present",
                }
            )

        sequences, stats = load_self_collected_dataset([csv_path])

        # session B was entirely diagnostic-only labels -> dropped by default
        assert stats["rows_read"] == 5
        assert stats["rows_diagnostic_skipped"] == 1
        assert len(sequences) == 1
        seq = sequences[0]
        assert seq.source_id == "A"
        assert len(seq.frames) == 4
        assert seq.sample_rate_hz == 2.0  # inferred from 0.5s median gap


def test_self_collected_loader_rejects_unknown_label():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "bad.csv")
        fieldnames = [
            "timestamp", "session_id", "participant_id", "yaw", "pitch", "roll", "gaze_x", "gaze_y",
            "eyes_closed", "posture_angle", "movement_delta", "face_conf", "phone_detected",
            "second_person_detected", "look_up_cycle_rate", "label",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": 0, "session_id": "A", "participant_id": "p1", "yaw": 0, "pitch": 0, "roll": 0,
                    "gaze_x": 0, "gaze_y": 0, "eyes_closed": "false", "posture_angle": 0, "movement_delta": 0.1,
                    "face_conf": 0.9, "phone_detected": "false", "second_person_detected": "false",
                    "look_up_cycle_rate": 0, "label": "totally_not_a_real_label",
                }
            )
        try:
            load_self_collected_dataset([csv_path])
            assert False, "expected ValueError for an unrecognized label"
        except ValueError:
            pass


def test_diagnostic_only_labels_do_not_overlap_trained_classes():
    from data.types import BEHAVIOR_CLASSES

    assert DIAGNOSTIC_ONLY_LABELS.isdisjoint(set(BEHAVIOR_CLASSES))
