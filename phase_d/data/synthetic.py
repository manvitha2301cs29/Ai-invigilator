"""
synthetic.py
------------
Fabricated RawSequence generator so the WHOLE Phase D pipeline
(windowing -> model -> training loop -> metrics) is testable end-to-end
before DAiSEE access or a self-collected dataset exists -- this is what
the continuation brief calls out explicitly: "a synthetic data generator
so the WHOLE pipeline is testable end-to-end before any real dataset
exists."

Each class gets a distinct, hand-designed signal signature (not just
label noise on top of random data) so a correctly-implemented model
should reach high accuracy on this data quickly -- if it doesn't, that's
a strong signal the bug is in the training/model code, not in a real
dataset's inherent difficulty. This is deliberately a code-correctness
fixture, not a benchmark: reported Phase D metrics for the report should
come from --dataset daisee or --dataset self_collected, never from
--dataset synthetic.
"""

from __future__ import annotations

import numpy as np

from .types import FeatureFrame, RawSequence

SAMPLE_RATE_HZ = 1.0  # one feature frame per second, matching the
# watcher's typical feature-sampling interval mentioned in
# docs/02_TRAINING_GUIDE.txt Step 5 ("if you sample features every 1-2
# seconds...").


def _engaged_frame(t: float, rng: np.random.Generator) -> FeatureFrame:
    """On-screen, minimal drift, eyes open, no phone/second person --
    the "nothing unusual" case."""
    return FeatureFrame(
        timestamp=t,
        yaw=rng.normal(0, 3),
        pitch=rng.normal(0, 3),
        roll=rng.normal(0, 2),
        gaze_x=rng.normal(0, 0.05),
        gaze_y=rng.normal(0, 0.05),
        eyes_closed=bool(rng.random() < 0.02),  # occasional natural blink
        posture_angle=rng.normal(0, 2),
        movement_delta=rng.normal(0.15, 0.05),
        face_conf=float(np.clip(rng.normal(0.95, 0.02), 0, 1)),
        phone_detected=False,
        second_person_detected=False,
        label="engaged",
    )


def _idle_present_frame(t: float, rng: np.random.Generator) -> FeatureFrame:
    """Face on-screen, but nearly motionless for long stretches --
    reading without typing, per the diary protocol's definition."""
    return FeatureFrame(
        timestamp=t,
        yaw=rng.normal(0, 4),
        pitch=rng.normal(2, 3),
        roll=rng.normal(0, 2),
        gaze_x=rng.normal(0, 0.08),
        gaze_y=rng.normal(0, 0.08),
        eyes_closed=bool(rng.random() < 0.03),
        posture_angle=rng.normal(0, 2),
        movement_delta=rng.normal(0.02, 0.01),  # key discriminator: very still
        face_conf=float(np.clip(rng.normal(0.93, 0.03), 0, 1)),
        phone_detected=False,
        second_person_detected=False,
        label="idle_present",
    )


def _distracted_present_frame(t: float, rng: np.random.Generator) -> FeatureFrame:
    """Face present but repeatedly looking away / turned -- key
    discriminator is large, variable yaw and gaze offset."""
    return FeatureFrame(
        timestamp=t,
        yaw=rng.normal(0, 1) * rng.choice([-1, 1]) * rng.uniform(20, 45),
        pitch=rng.normal(0, 6),
        roll=rng.normal(0, 4),
        gaze_x=rng.normal(0, 1) * rng.uniform(0.3, 0.6),
        gaze_y=rng.normal(0, 0.2),
        eyes_closed=bool(rng.random() < 0.02),
        posture_angle=rng.normal(0, 5),
        movement_delta=rng.normal(0.25, 0.1),
        face_conf=float(np.clip(rng.normal(0.85, 0.05), 0, 1)),
        phone_detected=bool(rng.random() < 0.15),
        second_person_detected=bool(rng.random() < 0.05),
        label="distracted_present",
    )


_CLASS_GENERATORS = {
    "engaged": _engaged_frame,
    "idle_present": _idle_present_frame,
    "distracted_present": _distracted_present_frame,
}


def generate_synthetic_sequence(
    source_id: str,
    label: str,
    duration_seconds: int = 300,
    seed: int | None = None,
) -> RawSequence:
    """One synthetic session, entirely one class (a clean fixture --
    class-transition realism is not the point here; window_sequence's
    majority-vote logic is what gets exercised against real, messier
    data in DAiSEE/self-collected runs)."""
    if label not in _CLASS_GENERATORS:
        raise ValueError(f"unknown synthetic class: {label!r}")
    rng = np.random.default_rng(seed)
    gen = _CLASS_GENERATORS[label]
    frames = [gen(float(t), rng) for t in range(duration_seconds)]
    return RawSequence(source_id=source_id, frames=frames, sample_rate_hz=SAMPLE_RATE_HZ)


def generate_synthetic_dataset(
    sequences_per_class: int = 20,
    duration_seconds: int = 300,
    seed: int = 0,
) -> list[RawSequence]:
    """A full synthetic dataset: sequences_per_class sequences of each of
    the three trained classes, each duration_seconds long. Deterministic
    given `seed` so CI/tests get reproducible numbers."""
    rng = np.random.default_rng(seed)
    sequences: list[RawSequence] = []
    for label in _CLASS_GENERATORS:
        for i in range(sequences_per_class):
            seq_seed = int(rng.integers(0, 2**31 - 1))
            sequences.append(
                generate_synthetic_sequence(
                    source_id=f"synthetic_{label}_{i:03d}",
                    label=label,
                    duration_seconds=duration_seconds,
                    seed=seq_seed,
                )
            )
    return sequences
