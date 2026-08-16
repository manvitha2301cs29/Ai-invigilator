"""
test_autoencoder.py
---------------------
Per the continuation brief: "prove the autoencoder's reconstruction
error is clearly higher on deliberately anomalous sequences than on
normal ones, using fabricated data, before any real dataset is
involved." Every test here uses synthetic_fatigue.py's fabricated
normal/drifting sessions -- no self-collected recordings required to
run pytest.

Run with:
    pytest tests/test_autoencoder.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.reuse import NUM_FEATURES
from data.synthetic_fatigue import generate_drifting_session, generate_normal_session
from data.windowing import window_sequence_unlabeled, windows_to_array
from models.autoencoder import SequenceAutoencoder, reconstruction_error


def _train_small_model(x: torch.Tensor, epochs: int = 60, lr: float = 5e-3) -> SequenceAutoencoder:
    model = SequenceAutoencoder(NUM_FEATURES, latent_size=8, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def test_reconstruction_error_higher_on_anomalous_tail_of_drifting_session():
    """Train ONLY on the normal-baseline portion of a drifting session
    (as if this were a student's established history), then compare
    reconstruction error on windows from the normal region vs. the
    drifted/fatigued region of the SAME session. The fatigued region
    should reconstruct noticeably worse -- this is the core claim Section
    5 makes about the autoencoder."""
    seq, drift_start = generate_drifting_session(
        "student_a_session_1", normal_seconds=400, fatigued_seconds=200, seed=1
    )

    # Split the RawSequence's frames at drift_start so we can window the
    # normal and fatigued portions independently.
    normal_frames = seq.frames[:drift_start]
    fatigued_frames = seq.frames[drift_start:]

    from data.reuse import RawSequence

    normal_seq = RawSequence(source_id="normal_part", frames=normal_frames, sample_rate_hz=seq.sample_rate_hz)
    fatigued_seq = RawSequence(source_id="fatigued_part", frames=fatigued_frames, sample_rate_hz=seq.sample_rate_hz)

    window_seconds = 30
    train_windows = window_sequence_unlabeled(normal_seq, window_seconds)
    x_train = torch.from_numpy(windows_to_array(train_windows))

    model = _train_small_model(x_train)

    normal_eval_windows = window_sequence_unlabeled(normal_seq, window_seconds)
    fatigued_eval_windows = window_sequence_unlabeled(fatigued_seq, window_seconds)
    x_normal = torch.from_numpy(windows_to_array(normal_eval_windows))
    x_fatigued = torch.from_numpy(windows_to_array(fatigued_eval_windows))

    normal_errors = reconstruction_error(model, x_normal)
    fatigued_errors = reconstruction_error(model, x_fatigued)

    assert fatigued_errors.mean().item() > normal_errors.mean().item(), (
        f"expected higher reconstruction error on the fatigued tail: "
        f"normal={normal_errors.mean().item():.5f} fatigued={fatigued_errors.mean().item():.5f}"
    )
    # a real margin, not a coin-flip-sized difference
    assert fatigued_errors.mean().item() > normal_errors.mean().item() * 1.5


def test_reconstruction_error_low_on_held_out_normal_session():
    """A model trained on one normal session should ALSO reconstruct a
    different, held-out normal session reasonably well (it generalizes
    to "this student's normal pattern," not just to the exact training
    windows) -- the training guide's "reconstruction error should be low
    [on] held-out normal sessions" check."""
    train_seq = generate_normal_session("student_b_session_1", seconds=400, seed=2)
    held_out_seq = generate_normal_session("student_b_session_2", seconds=400, seed=3)

    window_seconds = 30
    x_train = torch.from_numpy(windows_to_array(window_sequence_unlabeled(train_seq, window_seconds)))
    x_held_out = torch.from_numpy(windows_to_array(window_sequence_unlabeled(held_out_seq, window_seconds)))

    model = _train_small_model(x_train)

    train_errors = reconstruction_error(model, x_train)
    held_out_errors = reconstruction_error(model, x_held_out)

    # held-out normal data shouldn't be wildly worse than the training
    # data itself -- generalization, not memorization.
    assert held_out_errors.mean().item() < train_errors.mean().item() * 3.0


def test_reconstruction_error_shape_is_one_scalar_per_window():
    seq = generate_normal_session("student_c", seconds=120, seed=4)
    windows = window_sequence_unlabeled(seq, window_seconds=30)
    x = torch.from_numpy(windows_to_array(windows))
    model = SequenceAutoencoder(NUM_FEATURES, latent_size=8)
    errors = reconstruction_error(model, x)
    assert errors.shape == (x.shape[0],)


def test_autoencoder_forward_preserves_input_shape():
    seq = generate_normal_session("student_d", seconds=60, seed=5)
    windows = window_sequence_unlabeled(seq, window_seconds=30)
    x = torch.from_numpy(windows_to_array(windows))
    model = SequenceAutoencoder(NUM_FEATURES, latent_size=8)
    recon = model(x)
    assert recon.shape == x.shape


def test_gradients_flow_through_autoencoder():
    seq = generate_normal_session("student_e", seconds=60, seed=6)
    windows = window_sequence_unlabeled(seq, window_seconds=30)
    x = torch.from_numpy(windows_to_array(windows))
    model = SequenceAutoencoder(NUM_FEATURES, latent_size=8)
    recon = model(x)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
