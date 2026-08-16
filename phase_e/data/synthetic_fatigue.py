"""
synthetic_fatigue.py
----------------------
Fatigue-specific synthetic data, separate from Phase D's synthetic.py
because it needs a property Phase D's generator has no reason to model:
DRIFT WITHIN a single session, from a "normal" pattern to a fabricated
"FATIGUED-PATTERN" (Section 5: "rising idle/distracted frequency +
posture decline over the last N minutes RELATIVE TO THE STUDENT'S OWN
SESSION BASELINE").

Used by tests/test_autoencoder.py to prove the autoencoder's
reconstruction error is clearly higher on the drifted (anomalous) tail
of a session than on its normal-baseline start, using entirely
fabricated data, before any real per-student history exists -- matching
the same synthetic-data-first discipline Phase D used.
"""

from __future__ import annotations

import numpy as np

from .reuse import FeatureFrame, RawSequence

SAMPLE_RATE_HZ = 1.0


def _normal_frame(t: float, rng: np.random.Generator) -> FeatureFrame:
    """A student's typical, "baseline" working pattern -- mostly
    engaged-like signal with occasional idle stretches, matching what an
    actual mixed study session looks like (not a single pure class, on
    purpose, since Section 5's autoencoder operates on whole sessions,
    not single-class clips)."""
    return FeatureFrame(
        timestamp=t,
        yaw=rng.normal(0, 4),
        pitch=rng.normal(1, 3),
        roll=rng.normal(0, 2),
        gaze_x=rng.normal(0, 0.08),
        gaze_y=rng.normal(0, 0.08),
        eyes_closed=bool(rng.random() < 0.02),
        posture_angle=rng.normal(0, 3),
        movement_delta=rng.normal(0.12, 0.05),
        face_conf=float(np.clip(rng.normal(0.94, 0.02), 0, 1)),
        phone_detected=False,
        second_person_detected=False,
    )


def _fatigued_frame(t: float, rng: np.random.Generator, severity: float) -> FeatureFrame:
    """A drifted-toward-fatigue pattern: severity in [0, 1] interpolates
    between the normal signature and a "rising idle/distracted frequency
    + posture decline" signature (larger, slumping posture_angle; more
    static movement_delta punctuated by distraction spikes; more frequent
    eyes-closed). This is a FABRICATED proxy for the qualitative
    description in Section 5, not a claim about what real fatigue looks
    like in webcam signals -- it exists to give the autoencoder test
    something concretely anomalous relative to _normal_frame's
    distribution."""
    posture_decline = 15.0 * severity  # slumping
    idle_or_distracted_bias = rng.random() < (0.5 * severity)
    return FeatureFrame(
        timestamp=t,
        yaw=rng.normal(0, 4 + 10 * severity),
        pitch=rng.normal(3 + 4 * severity, 3),
        roll=rng.normal(0, 2 + 3 * severity),
        gaze_x=rng.normal(0, 0.08 + 0.3 * severity),
        gaze_y=rng.normal(0, 0.08 + 0.2 * severity),
        eyes_closed=bool(rng.random() < (0.02 + 0.15 * severity)),
        posture_angle=rng.normal(posture_decline, 3),
        movement_delta=rng.normal(0.03, 0.02) if idle_or_distracted_bias else rng.normal(0.3, 0.1),
        face_conf=float(np.clip(rng.normal(0.90 - 0.05 * severity, 0.03), 0, 1)),
        phone_detected=bool(rng.random() < (0.05 * severity)),
        second_person_detected=False,
    )


def generate_drifting_session(
    source_id: str,
    normal_seconds: int = 600,
    fatigued_seconds: int = 300,
    seed: int | None = None,
) -> tuple[RawSequence, int]:
    """One session: `normal_seconds` of baseline pattern, followed by
    `fatigued_seconds` that linearly ramps severity 0 -> 1, matching
    Section 5's "rising ... over the last N minutes" framing (a ramp,
    not an instant step-change). Returns (sequence, drift_start_index)
    so tests/callers know exactly where the "should be anomalous" region
    begins without having to re-derive it."""
    rng = np.random.default_rng(seed)
    frames: list[FeatureFrame] = []
    for t in range(normal_seconds):
        frames.append(_normal_frame(float(t), rng))
    for i in range(fatigued_seconds):
        severity = i / max(1, fatigued_seconds - 1)
        frames.append(_fatigued_frame(float(normal_seconds + i), rng, severity))
    return RawSequence(source_id=source_id, frames=frames, sample_rate_hz=SAMPLE_RATE_HZ), normal_seconds


def generate_normal_session(source_id: str, seconds: int = 600, seed: int | None = None) -> RawSequence:
    """A held-out, entirely-normal session -- used as the "should
    reconstruct well" comparison set, distinct from the sessions an
    autoencoder was trained on, so a low reconstruction error here can't
    be attributed to simple memorization of the exact training frames."""
    rng = np.random.default_rng(seed)
    frames = [_normal_frame(float(t), rng) for t in range(seconds)]
    return RawSequence(source_id=source_id, frames=frames, sample_rate_hz=SAMPLE_RATE_HZ)
