"""
test_pipeline.py
------------------
Covers the pieces test_autoencoder.py doesn't: the unlabeled windowing
variant (data/windowing.py), participant grouping from diary-protocol
CSVs (data/participants.py), and the full train_autoencoder.run() CLI
entry point end to end on --dataset synthetic, for all three --mode
values. No real self-collected dataset needed -- fixture CSVs and the
synthetic generators cover it.

Run with:
    pytest tests/test_pipeline.py -v
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from data.participants import group_by_participant, read_session_to_participant_map
from data.reuse import RawSequence
from data.synthetic_fatigue import generate_normal_session
from data.windowing import window_sequence_unlabeled, windows_to_array
from experiments.train_autoencoder import MODES, run


# ---------------------------------------------------------------------------
# unlabeled windowing
# ---------------------------------------------------------------------------

def test_window_sequence_unlabeled_does_not_require_labels():
    seq = generate_normal_session("s1", seconds=90, seed=0)
    assert all(f.label is None for f in seq.frames)  # confirms this really is unlabeled data
    windows = window_sequence_unlabeled(seq, window_seconds=30)
    assert len(windows) == 3


def test_windows_to_array_shape():
    seq = generate_normal_session("s1", seconds=90, seed=0)
    windows = window_sequence_unlabeled(seq, window_seconds=30)
    arr = windows_to_array(windows)
    assert arr.shape[0] == 3
    assert arr.shape[1] == 30


def test_windows_to_array_rejects_empty_list():
    try:
        windows_to_array([])
        assert False, "expected ValueError for empty window list"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# participant grouping
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "timestamp", "session_id", "participant_id", "yaw", "pitch", "roll", "gaze_x", "gaze_y",
    "eyes_closed", "posture_angle", "movement_delta", "face_conf", "phone_detected",
    "second_person_detected", "look_up_cycle_rate", "label",
]


def _write_fixture_row(writer, session_id, participant_id, t=0, label="engaged"):
    writer.writerow(
        {
            "timestamp": t, "session_id": session_id, "participant_id": participant_id, "yaw": 0, "pitch": 0,
            "roll": 0, "gaze_x": 0, "gaze_y": 0, "eyes_closed": "false", "posture_angle": 0,
            "movement_delta": 0.1, "face_conf": 0.9, "phone_detected": "false",
            "second_person_detected": "false", "look_up_cycle_rate": 0, "label": label,
        }
    )


def test_read_session_to_participant_map():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sessions.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            _write_fixture_row(writer, "sessA", "alice")
            _write_fixture_row(writer, "sessB", "bob")
        mapping = read_session_to_participant_map([path])
        assert mapping == {"sessA": "alice", "sessB": "bob"}


def test_read_session_to_participant_map_rejects_conflicting_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sessions.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            _write_fixture_row(writer, "sessA", "alice")
            _write_fixture_row(writer, "sessA", "bob")  # same session, different participant -- data error
        try:
            read_session_to_participant_map([path])
            assert False, "expected ValueError for a session_id mapped to two participants"
        except ValueError:
            pass


def test_group_by_participant():
    seq_a = RawSequence(source_id="sessA", frames=[])
    seq_b = RawSequence(source_id="sessB", frames=[])
    seq_unknown = RawSequence(source_id="sessC", frames=[])
    grouped = group_by_participant([seq_a, seq_b, seq_unknown], {"sessA": "alice", "sessB": "alice"})
    assert set(grouped.keys()) == {"alice", "__unknown__"}
    assert len(grouped["alice"]) == 2
    assert len(grouped["__unknown__"]) == 1


# ---------------------------------------------------------------------------
# full training pipeline, all three modes, synthetic dataset
# ---------------------------------------------------------------------------

def test_run_per_participant_mode_produces_one_model_per_student():
    with tempfile.TemporaryDirectory() as tmp:
        summary = run(dataset="synthetic", mode="per_participant", output_dir=tmp, window_seconds=20, epochs=3)
        assert len(summary["participants"]) == 3  # matches _load_grouped_sequences' 3 synthetic students
        for participant_id in summary["participants"]:
            assert os.path.isfile(os.path.join(tmp, participant_id, "model.pt"))
            assert os.path.isfile(os.path.join(tmp, participant_id, "metrics.json"))


def test_run_pooled_mode_produces_single_shared_model():
    with tempfile.TemporaryDirectory() as tmp:
        summary = run(dataset="synthetic", mode="pooled", output_dir=tmp, window_seconds=20, epochs=3)
        assert "__pooled__" in summary["participants"]
        assert os.path.isfile(os.path.join(tmp, "__pooled__", "model.pt"))
        # pooled mode should NOT also produce per-participant models
        assert len(summary["participants"]) == 1


def test_run_pooled_finetuned_mode_produces_pooled_plus_per_participant():
    with tempfile.TemporaryDirectory() as tmp:
        summary = run(
            dataset="synthetic", mode="pooled_finetuned", output_dir=tmp, window_seconds=20,
            epochs=3, finetune_epochs=2,
        )
        # pooled model + 3 fine-tuned per-student models
        assert "__pooled__" in summary["participants"]
        assert len(summary["participants"]) == 4


def test_run_rejects_unknown_mode():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run(dataset="synthetic", mode="not_a_real_mode", output_dir=tmp)
            assert False, "expected ValueError for an unknown mode"
        except ValueError:
            pass


def test_all_documented_modes_are_exercised_above():
    # a lightweight guard so a future added mode doesn't silently skip
    # test coverage for it.
    assert set(MODES) == {"pooled", "per_participant", "pooled_finetuned"}
