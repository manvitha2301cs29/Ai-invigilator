"""
types.py
--------
Plain data structures used throughout Phase B. Keeping these separate
from any MediaPipe-specific types is deliberate: it's what lets every
function in this module be tested with fake, hand-built input, with no
webcam, no MediaPipe model files, and no live camera loop required.

This mirrors the "raw_features (JSON: yaw, pitch, roll, gaze_x, gaze_y,
eyes_closed, posture_angle, movement_delta, face_conf, phone_detected,
second_person_detected)" schema from the StateEvent table in the project
data model (Section 8 of AI_Study_Invigilator_Project.txt / the master
coding prompt) -- FrameSignals below is the input side of that pipeline,
FeatureVector is closer to the output/stored side.
"""

from dataclasses import dataclass, field
from enum import Enum


class HeadState(Enum):
    """Coarse head-position classification for a single frame. Deliberately
    NOT the same thing as the project's Engaged/Idle-present/Distracted-
    present/Away taxonomy -- this is a lower-level geometric read that
    Phase C's rule engine (and eventually Phase D's temporal model) will
    combine with other signals to reach those higher-level states."""
    ON_SCREEN = "on_screen"
    DOWN = "down"          # looking down, e.g. at notes
    TURNED = "turned"      # yaw beyond the on-screen envelope, not "down"


@dataclass
class FrameSignals:
    """Raw, single-frame geometric signals -- the output of Layer 1
    (Perception) for one frame, already extracted from MediaPipe's
    landmark/pose/detection results by the watcher's perception module.
    This is intentionally MediaPipe-agnostic: Phase C's real watcher will
    populate this from live MediaPipe output, but tests build it by hand.

    All angles in degrees. Timestamps in seconds (float), not milliseconds,
    for readability in tests -- convert at the call site if needed.
    """
    timestamp: float
    face_present: bool
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    ear: float = 0.30          # eye-aspect-ratio; ~0.30 is a typical
                                # "eyes open" value per Phase A tuning
    posture_angle: float = 0.0  # shoulder-tilt proxy, degrees
    phone_detected: bool = False
    second_person_detected: bool = False


@dataclass
class CalibrationProfile:
    """Per-user calibration, captured once at first run (Section 6 of the
    project doc: "a short one-time CALIBRATION STEP at first use... records
    the student's actual on-screen gaze envelope"). Defaults match Phase
    A's fixed starting constants so the system is usable before a student
    has calibrated, but calibration is what makes wide/multi-monitor
    setups work correctly instead of constantly reading as distraction.
    """
    yaw_envelope_deg: float = 25.0      # on-screen yaw tolerance, +/-
    pitch_down_enter_deg: float = 11.0  # tuned from real Phase A session
                                         # data -- see signal_check.py's
                                         # PITCH_DOWN_THRESHOLD_DEG comment
    pitch_down_exit_deg: float = 8.0    # hysteresis release threshold,
                                         # see signal_check.py's
                                         # PITCH_UP_RELEASE_DEG comment
    ear_closed_threshold: float = 0.22  # tuned from a real session log
                                         # (see conversation history /
                                         # signal_check.py comments) --
                                         # NOT a universal constant, this
                                         # should differ per student,
                                         # especially with glasses


@dataclass
class RollingState:
    """Carries state ACROSS frames -- everything in Section 6 of the
    project doc depends on duration and pattern, not single frames, so
    the feature extractor needs somewhere to remember "was the head down
    a moment ago," "when did eyes first close," etc. One RollingState
    belongs to one continuous monitoring session; Phase C's watcher will
    own exactly one instance of this per active TargetBlock.
    """
    was_looking_down: bool = False
    eyes_closed_since: float | None = None
    look_up_event_timestamps: list = field(default_factory=list)
    phone_since: float | None = None
    second_person_since: float | None = None
    was_turned: bool = False
    turned_since: float | None = None
