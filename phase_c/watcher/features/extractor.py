"""
extractor.py
------------
The Phase B deliverable: a deterministic, pure function that turns one
frame's raw signals (FrameSignals) plus the running session state
(RollingState) into a FeatureVector -- implementing exactly the
duration-gated, pattern-specific rules from Section 6 of
AI_Study_Invigilator_Project.txt ("FALSE-POSITIVE HANDLING").

DESIGN NOTE -- why this takes and returns explicit state instead of being
a class with hidden internal state: it makes every function here trivial
to unit test (build a FrameSignals, build a RollingState, call the
function, assert on the output) without needing a live camera, a running
event loop, or time.sleep() in tests. Phase C's watcher owns one
RollingState per active TargetBlock and threads it through frame by
frame; tests just construct whatever RollingState they need directly.

NOTHING IN THIS FILE TOUCHES A CAMERA OR A MEDIAPIPE MODEL. That's the
point -- this is Layer 2 (Feature Engine) in the project's 8-layer
pipeline, sitting strictly downstream of Layer 1 (Perception).
"""

from dataclasses import dataclass, replace

from .types import CalibrationProfile, FrameSignals, HeadState, RollingState

# ---------------------------------------------------------------------------
# Duration thresholds. These are the concrete, code-level version of the
# SUMMARY TABLE in Section 6 of the project doc. Defaults here match what
# was tuned during Phase A live testing; treat these as starting points,
# not universal constants -- Phase D's evaluation should revisit them
# against real labeled data.
# ---------------------------------------------------------------------------
LOOK_UP_CYCLE_WINDOW_SEC = 30.0     # window for counting look-down->up
                                     # transitions (note-taking signal)
EYES_CLOSED_BRIEF_MAX_SEC = 6 * 60  # below this: not flagged (thinking
                                     # pause). At/above: prolonged-closure
                                     # flag, worded as an inference.
HEAD_TURN_SUSTAINED_SEC = 4.0       # a head turn held this long or longer
                                     # is "sustained" (implies conversation)
                                     # rather than a glance
PHONE_SUSTAINED_SEC = 3.0           # matches the spirit of Phase A's
                                     # duration-gating for phone detection
SECOND_PERSON_SUSTAINED_SEC = 4.0   # matches signal_check.py's
                                     # SECOND_PERSON_DURATION_SEC


@dataclass(frozen=True)
class FeatureVector:
    """The Layer 2 output for a single frame: everything downstream layers
    (Layer 3's temporal model, Layer 4's live monitor) need, with no
    single-frame flags -- every boolean here has already been
    duration-gated according to Section 6's rules.

    Immutable (frozen) deliberately: a FeatureVector represents a fact
    about one instant, and should never be mutated after creation.
    """
    timestamp: float
    face_present: bool

    # Raw geometric values, passed through for the temporal model /
    # storage (matches StateEvent.raw_features in the project schema)
    yaw: float
    pitch: float
    roll: float
    ear: float
    posture_angle: float

    # Derived, duration-gated signals -- these are what Section 6 is
    # actually about
    head_state: HeadState
    on_screen: bool                 # within the calibrated gaze envelope
    looking_down: bool              # raw down-pitch, NOT yet cycle-checked
    look_up_cycle_rate: int         # look-down->up transitions in the
                                     # last LOOK_UP_CYCLE_WINDOW_SEC --
                                     # HIGH rate + looking_down = note-taking
                                     # (do not flag); near-zero rate with
                                     # sustained down-pitch = the "Maybe"
                                     # row in Section 6's summary table
    head_turn_flag: bool            # SUSTAINED turn only -- Section 6:
                                     # "Sustained head turn (talking)"
    eyes_closed: bool
    eyes_closed_duration: float
    eyes_closed_prolonged_flag: bool  # >= EYES_CLOSED_BRIEF_MAX_SEC;
                                       # caller should word this as
                                       # "possible rest/sleep", never fact
    phone_flag: bool                # duration-gated, NOT raw per-frame
                                     # detection
    second_person_flag: bool        # duration-gated, NOT raw per-frame
                                     # detection


def _update_look_up_cycle(
    state: RollingState, looking_down_now: bool, now: float
) -> RollingState:
    """Detects the down->up transition that is the core of the
    note-taking signal (Section 6: "recognized by its look-down/look-up
    cycle... over 10-30 second periods"). Appends a timestamp to
    look_up_event_timestamps whenever the head moves from down to
    on-screen; pruning of old events happens in the rate calculation
    below, not here, so this function stays a simple append."""
    events = list(state.look_up_event_timestamps)
    if state.was_looking_down and not looking_down_now:
        events.append(now)
    return replace(state, was_looking_down=looking_down_now, look_up_event_timestamps=events)


def _look_up_cycle_rate(state: RollingState, now: float) -> int:
    """Counts look-down->up transitions within the trailing window. A
    HIGH rate here, combined with looking_down being True some of the
    time, is exactly the note-taking pattern Section 6 says must NOT be
    flagged as distraction."""
    return sum(
        1 for t in state.look_up_event_timestamps
        if now - t <= LOOK_UP_CYCLE_WINDOW_SEC
    )


def _update_eyes_closed(
    state: RollingState, eyes_closed_now: bool, now: float
) -> tuple[RollingState, float, bool]:
    """Returns (new_state, eyes_closed_duration, prolonged_flag).
    Implements Section 6's brief-vs-prolonged distinction: duration is
    tracked continuously while eyes stay closed, resets to zero the
    instant they open, and only crosses into a flag at
    EYES_CLOSED_BRIEF_MAX_SEC (default 6 minutes)."""
    if eyes_closed_now:
        since = state.eyes_closed_since if state.eyes_closed_since is not None else now
        duration = now - since
        new_state = replace(state, eyes_closed_since=since)
    else:
        duration = 0.0
        new_state = replace(state, eyes_closed_since=None)
    prolonged = duration >= EYES_CLOSED_BRIEF_MAX_SEC
    return new_state, duration, prolonged


def _update_head_turn(
    state: RollingState, turned_now: bool, now: float
) -> tuple[RollingState, bool]:
    """Returns (new_state, sustained_turn_flag). Implements Section 6's
    glance-vs-conversation distinction: yaw outside the calibrated
    envelope must persist for HEAD_TURN_SUSTAINED_SEC before it's
    flagged. A quick glance that returns to on-screen within a second or
    two never crosses this and is correctly NOT flagged."""
    if turned_now:
        since = state.turned_since if state.turned_since is not None else now
        duration = now - since
        new_state = replace(state, was_turned=True, turned_since=since)
    else:
        duration = 0.0
        new_state = replace(state, was_turned=False, turned_since=None)
    sustained = duration >= HEAD_TURN_SUSTAINED_SEC
    return new_state, sustained


def _update_duration_gate(
    since: float | None, active_now: bool, now: float, threshold_sec: float
) -> tuple[float | None, bool]:
    """Generic duration-gate helper shared by phone and second-person
    detection -- both follow the identical pattern (Section 6: "gated by
    the same duration principle as everything else"), so this is
    factored out once rather than duplicated. Returns
    (new_since_value, flag)."""
    if active_now:
        new_since = since if since is not None else now
        flag = (now - new_since) >= threshold_sec
    else:
        new_since = None
        flag = False
    return new_since, flag


def extract_features(
    signals: FrameSignals,
    state: RollingState,
    calibration: CalibrationProfile,
) -> tuple[FeatureVector, RollingState]:
    """The Phase B entry point. Pure function: given the current frame's
    raw signals, the running per-session state, and this student's
    calibration profile, returns (this frame's FeatureVector, the
    updated RollingState to pass into the NEXT call).

    Callers (Phase C's watcher, and every test in this module) are
    expected to thread RollingState through consecutive calls in
    timestamp order -- this function never reads a clock itself or
    reaches out to any global state, which is what makes it possible to
    test entirely with synthetic timestamps.
    """
    now = signals.timestamp

    if not signals.face_present:
        # No face -> no meaningful head/eye signals this frame. Reset
        # the parts of RollingState that only make sense while a face is
        # visible (mirrors signal_check.py's behavior in the same
        # situation), but deliberately do NOT reset phone/second-person
        # tracking, since those can be legitimately detected even when
        # the primary student's own face is briefly not detected.
        state = replace(state, was_looking_down=False, eyes_closed_since=None,
                         was_turned=False, turned_since=None)
        phone_since, phone_flag = _update_duration_gate(
            state.phone_since, signals.phone_detected, now, PHONE_SUSTAINED_SEC
        )
        second_person_since, second_person_flag = _update_duration_gate(
            state.second_person_since, signals.second_person_detected, now,
            SECOND_PERSON_SUSTAINED_SEC,
        )
        state = replace(state, phone_since=phone_since, second_person_since=second_person_since)

        fv = FeatureVector(
            timestamp=now, face_present=False,
            yaw=0.0, pitch=0.0, roll=0.0, ear=0.0, posture_angle=0.0,
            head_state=HeadState.ON_SCREEN, on_screen=False,
            looking_down=False, look_up_cycle_rate=_look_up_cycle_rate(state, now),
            head_turn_flag=False,
            eyes_closed=False, eyes_closed_duration=0.0, eyes_closed_prolonged_flag=False,
            phone_flag=phone_flag, second_person_flag=second_person_flag,
        )
        return fv, state

    # --- Head pose classification ---
    looking_down_now = abs(signals.pitch) > (
        calibration.pitch_down_exit_deg if state.was_looking_down
        else calibration.pitch_down_enter_deg
    )
    on_screen = (not looking_down_now) and (abs(signals.yaw) <= calibration.yaw_envelope_deg)
    turned_now = (not looking_down_now) and (abs(signals.yaw) > calibration.yaw_envelope_deg)

    if looking_down_now:
        head_state = HeadState.DOWN
    elif turned_now:
        head_state = HeadState.TURNED
    else:
        head_state = HeadState.ON_SCREEN

    state = _update_look_up_cycle(state, looking_down_now, now)
    look_up_rate = _look_up_cycle_rate(state, now)

    state, head_turn_flag = _update_head_turn(state, turned_now, now)

    # --- Eyes ---
    eyes_closed_now = signals.ear < calibration.ear_closed_threshold
    state, eyes_closed_duration, eyes_closed_prolonged = _update_eyes_closed(
        state, eyes_closed_now, now
    )

    # --- Phone / second-person (duration-gated the same way regardless
    # of whether the primary student's face is visible, so this logic is
    # shared with the no-face branch above via the same helper) ---
    phone_since, phone_flag = _update_duration_gate(
        state.phone_since, signals.phone_detected, now, PHONE_SUSTAINED_SEC
    )
    second_person_since, second_person_flag = _update_duration_gate(
        state.second_person_since, signals.second_person_detected, now,
        SECOND_PERSON_SUSTAINED_SEC,
    )
    state = replace(state, phone_since=phone_since, second_person_since=second_person_since)

    fv = FeatureVector(
        timestamp=now,
        face_present=True,
        yaw=signals.yaw,
        pitch=signals.pitch,
        roll=signals.roll,
        ear=signals.ear,
        posture_angle=signals.posture_angle,
        head_state=head_state,
        on_screen=on_screen,
        looking_down=looking_down_now,
        look_up_cycle_rate=look_up_rate,
        head_turn_flag=head_turn_flag,
        eyes_closed=eyes_closed_now,
        eyes_closed_duration=eyes_closed_duration,
        eyes_closed_prolonged_flag=eyes_closed_prolonged,
        phone_flag=phone_flag,
        second_person_flag=second_person_flag,
    )
    return fv, state


def is_note_taking_pattern(fv: FeatureVector, min_cycle_rate: int = 2) -> bool:
    """Convenience helper implementing Section 6's core distinction
    directly: 'Sustained head-down WITH a look-up cycle' (note-taking,
    do NOT flag) vs. 'Sustained head-down, no cycle' (falls through to
    the Maybe/distracted-or-fatigue path). This is intentionally a
    separate, named function rather than inlined logic, because it's the
    single most interview-relevant rule in the whole feature engine --
    keeping it as one clearly-named function makes it easy to point to
    and explain on its own.
    """
    return fv.looking_down and fv.look_up_cycle_rate >= min_cycle_rate
