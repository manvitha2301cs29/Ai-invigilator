"""
test_train_synthetic.py
--------------------------
End-to-end pipeline test: windowing -> DataLoader -> model -> training
loop -> metrics/artifacts, entirely on --dataset synthetic, per the
continuation brief's requirement that "the WHOLE pipeline is testable
end-to-end before any real dataset exists." No DAiSEE download, no
self-collected recordings, no GPU required -- runs in a few seconds on
CPU with a handful of epochs.

This test intentionally checks that accuracy is HIGH (>= 0.9) on
synthetic data specifically because synthetic.py's three classes have
deliberately distinct, easy-to-separate signatures (see synthetic.py's
docstring) -- a correctly wired pipeline should have no trouble here.
If this test fails, the bug is almost certainly in the training/model
code, not in the (deliberately easy) data.

Run with:
    pytest tests/test_train_synthetic.py -v
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from experiments.train_temporal_model import RunConfig, run


def _synthetic_config(model: str) -> RunConfig:
    return RunConfig(
        model=model,
        dataset="synthetic",
        window_size=10,
        stride_size=10,
        epochs=8,
        batch_size=8,
        lr=1e-2,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        seed=0,
        val_fraction=0.25,
    )


def test_gru_reaches_high_accuracy_on_synthetic_data():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(_synthetic_config("gru"), tmp)
        assert metrics["accuracy"] >= 0.9, f"expected near-perfect accuracy on easy synthetic data, got {metrics['accuracy']}"


def test_lstm_and_transformer_also_train_without_error():
    """Doesn't require the SAME high bar as GRU (that's checked above),
    just that both comparison architectures run the identical pipeline
    end to end and produce sane output -- proving the shared-interface
    claim actually holds at the training-loop level, not just the
    model-shape level tested in test_models.py."""
    with tempfile.TemporaryDirectory() as tmp:
        for model_name in ("lstm", "transformer"):
            out_dir = os.path.join(tmp, model_name)
            metrics = run(_synthetic_config(model_name), out_dir)
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert metrics["macro_f1"] >= 0.0


def test_run_writes_all_required_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        run(_synthetic_config("gru"), tmp)
        for filename in ("config.json", "metrics.json", "confusion_matrix.png", "best.pt"):
            path = os.path.join(tmp, filename)
            assert os.path.isfile(path), f"missing expected artifact: {filename}"

        with open(os.path.join(tmp, "metrics.json")) as f:
            metrics = json.load(f)
        for key in (
            "accuracy", "macro_precision", "macro_recall", "macro_f1", "confusion_matrix",
            "inference_latency_ms_per_window_cpu", "model_size_mb", "parameter_count",
        ):
            assert key in metrics, f"metrics.json missing required key: {key}"


def test_confusion_matrix_shape_matches_num_classes():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(_synthetic_config("gru"), tmp)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)
        assert metrics["class_names"] == ["engaged", "idle_present", "distracted_present"]


def test_run_is_reproducible_given_same_seed():
    with tempfile.TemporaryDirectory() as tmp:
        m1 = run(_synthetic_config("gru"), os.path.join(tmp, "a"))
        m2 = run(_synthetic_config("gru"), os.path.join(tmp, "b"))
        assert m1["accuracy"] == m2["accuracy"]
