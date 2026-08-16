"""
test_extractor.py
------------------
Unit tests for features/extractor.py. No camera, no MediaPipe models, no
real time delays -- every test builds synthetic FrameSignals with hand-
picked timestamps and asserts on the resulting FeatureVector.

These tests are the literal checklist the master coding prompt asked for
after Phase B:
  "(1) sustained head-down WITH a look-up cycle is NOT flagged,
   (2) sustained head-down with NO look-up cycle eventually falls
       through to a flaggable state,
   (3) a brief head turn is NOT flagged,
   (4) a sustained head turn IS flagged,
   (5) gaze within the calibrated envelope is NOT flagged regardless of
       raw angle."
Each is implemented below as its own test, plus additional coverage for
eyes-closed (brief vs. prolonged) and phone/second-person duration
gating, since those are equally load-bearing parts of Section 6.

Run with:
    pytest test_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.types import CalibrationProfile, FrameSignals, HeadState, RollingState
from features.extractor import extract_features, is_note_taking_pattern


def make_calibration(**overrides):
    """Default calibration profile matching Phase A's tuned constants,
    with the ability to override individual fields per test."""
    defaults = dict(
        yaw_envelope_deg=25.0,
        pitch_down_enter_deg=11.0,
        pitch_down_exit_deg=8.0,
        ear_closed_threshold=0.22,
    )
    defaults.update(overrides)
    return CalibrationProfile(**defaults)


def run_sequence(signal_list, calibration=None):
    """Feeds a list of FrameSignals through extract_features in order,
    threading RollingState between calls exactly as the real watcher
    will. Returns the list of resulting FeatureVectors."""
    calibration = calibration or make_calibration()
    state = RollingState()
    results = []
    for signals in signal_list:
        fv, state = extract_features(signals, state, calibration)
        results.append(fv)
    return results


# ---------------------------------------------------------------------------
# (1) Sustained head-down WITH a look-up cycle -> NOT flagged (note-taking)
# ---------------------------------------------------------------------------
def test_note_taking_pattern_is_not_flagged():
    """Simulates ~24 seconds of realistic note-taking: look down for ~3s,
    glance up for ~1s, repeat several times. This should build up a
    look_up_cycle_rate high enough that is_note_taking_pattern() reads
    True, and the pattern must NOT be conflated with genuine
    disengagement."""
    signals = []
    t = 0.0
    for _cycle in range(5):
        # looking down for 3 seconds (several frames)
        for _ in range(3):
            signals.append(FrameSignals(timestamp=t, face_present=True,
                                         pitch=-18.0, yaw=0.0, ear=0.30))
            t += 1.0
        # glance up for 1 second
        signals.append(FrameSignals(timestamp=t, face_present=True,
                                     pitch=0.0, yaw=0.0, ear=0.30))
        t += 1.0

    results = run_sequence(signals)
    last = results[-1]

    assert last.look_up_cycle_rate >= 2, (
        f"expected several look-up cycles to have been counted, got "
        f"{last.look_up_cycle_rate}"
    )
    # The frame right after a look-up-then-back-down transition should
    # read as looking_down=True but be recognized as note-taking, not a
    # flaggable disengagement pattern.
    down_frame_with_history = results[-2]  # last "looking down" frame
    assert down_frame_with_history.looking_down is True
    assert is_note_taking_pattern(down_frame_with_history) is True


# ---------------------------------------------------------------------------
# (2) Sustained head-down with NO look-up cycle -> falls through to
#     flaggable (the "Maybe" row in Section 6's summary table)
# ---------------------------------------------------------------------------
def test_sustained_head_down_no_cycle_is_not_note_taking():
    """Simulates a head that goes down and simply stays down for a long
    stretch with no return-to-screen glances at all -- Section 6:
    'Sustained head-down, no cycle (head just resting down)' should NOT
    be recognized as the note-taking pattern, since is_note_taking_pattern
    specifically requires a nonzero look_up_cycle_rate."""
    signals = [
        FrameSignals(timestamp=float(i), face_present=True, pitch=-20.0, yaw=0.0, ear=0.30)
        for i in range(40)  # 40 seconds of continuous down-pitch, no glances up
    ]
    results = run_sequence(signals)
    last = results[-1]

    assert last.looking_down is True
    assert last.look_up_cycle_rate == 0, (
        "no look-up events were simulated, so the cycle rate must be zero"
    )
    assert is_note_taking_pattern(last) is False, (
        "sustained down-pitch with zero look-up cycles must NOT be "
        "classified as note-taking -- this is exactly the case Section 6 "
        "says should fall through to distracted/fatigue logic instead"
    )


# ---------------------------------------------------------------------------
# (3) A brief head turn -> NOT flagged
# ---------------------------------------------------------------------------
def test_brief_head_turn_is_not_flagged():
    """A quick glance to the side and back (e.g. checking a second
    monitor) that returns to on-screen within ~2 seconds must NOT trigger
    head_turn_flag, per Section 6: 'a glance... returns to on-screen
    within 1-2 seconds.'"""
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, yaw=0.0, pitch=0.0, ear=0.30),
        FrameSignals(timestamp=0.5, face_present=True, yaw=40.0, pitch=0.0, ear=0.30),
        FrameSignals(timestamp=1.0, face_present=True, yaw=40.0, pitch=0.0, ear=0.30),
        FrameSignals(timestamp=1.5, face_present=True, yaw=0.0, pitch=0.0, ear=0.30),
    ]
    results = run_sequence(signals)

    assert all(fv.head_turn_flag is False for fv in results), (
        "a ~1 second glance must stay under HEAD_TURN_SUSTAINED_SEC and "
        "must never set head_turn_flag"
    )


# ---------------------------------------------------------------------------
# (4) A sustained head turn -> IS flagged
# ---------------------------------------------------------------------------
def test_sustained_head_turn_is_flagged():
    """A head turn held continuously for several seconds (simulating
    talking to someone) must eventually set head_turn_flag, per Section
    6: 'A conversation holds the turned position continuously.'"""
    signals = [
        FrameSignals(timestamp=float(i), face_present=True, yaw=40.0, pitch=0.0, ear=0.30)
        for i in range(8)  # 8 seconds held, well past the 4s threshold
    ]
    results = run_sequence(signals)

    assert results[-1].head_turn_flag is True, (
        "an 8-second continuous head turn must be flagged as sustained"
    )
    # And confirm it does NOT fire immediately on frame 1 -- the point of
    # duration-gating is that it takes time to commit.
    assert results[0].head_turn_flag is False, (
        "head_turn_flag must not fire on the very first frame of a turn "
        "-- that would mean single-frame decisions are being made, which "
        "Section 6 explicitly forbids"
    )


# ---------------------------------------------------------------------------
# (5) Gaze within the calibrated envelope -> NOT flagged regardless of
#     raw angle (wide/multi-monitor support)
# ---------------------------------------------------------------------------
def test_calibrated_envelope_widens_on_screen_zone():
    """A yaw of 35 degrees would be 'turned' under the DEFAULT envelope
    (25 degrees), but must read as on_screen and never set
    head_turn_flag once a student's calibration profile has a wider
    envelope -- this is exactly what Section 6 describes: 'a per-student
    CALIBRATION STEP... records the student's actual on-screen gaze
    envelope, and that envelope -- not a hardcoded angle -- defines
    on-screen for that student's setup.'"""
    wide_monitor_calibration = make_calibration(yaw_envelope_deg=45.0)

    signals = [
        FrameSignals(timestamp=float(i), face_present=True, yaw=35.0, pitch=0.0, ear=0.30)
        for i in range(10)
    ]
    results = run_sequence(signals, calibration=wide_monitor_calibration)

    assert all(fv.on_screen is True for fv in results), (
        "yaw=35 must read as on_screen under a 45-degree calibrated "
        "envelope even though it would fail the default 25-degree one"
    )
    assert all(fv.head_turn_flag is False for fv in results), (
        "a wide-monitor glance within the calibrated envelope must never "
        "accumulate toward the sustained-turn flag"
    )

    # Sanity check: the SAME yaw value under the DEFAULT (narrower)
    # envelope should behave differently, confirming the envelope is
    # actually doing something and this isn't a vacuous test.
    default_results = run_sequence(signals, calibration=make_calibration())
    assert default_results[0].on_screen is False, (
        "yaw=35 should NOT be on_screen under the default 25-degree "
        "envelope -- if this assertion fails, the wide-monitor test above "
        "isn't actually testing anything"
    )


# ---------------------------------------------------------------------------
# Eyes-closed: brief (thinking pause) -> NOT flagged
# ---------------------------------------------------------------------------
def test_brief_eyes_closed_is_not_flagged():
    """A few seconds of closed eyes (a thinking pause) must not trigger
    eyes_closed_prolonged_flag, per Section 6: 'Brief eyes-closed
    (thinking)... NOT flagged. This is normal cognitive behavior.'"""
    signals = [
        FrameSignals(timestamp=float(i), face_present=True, yaw=0.0, pitch=0.0, ear=0.10)
        for i in range(5)  # 5 seconds closed
    ]
    results = run_sequence(signals)

    assert results[-1].eyes_closed is True
    assert results[-1].eyes_closed_prolonged_flag is False, (
        "5 seconds of closed eyes must not cross the ~6-minute prolonged "
        "threshold"
    )


# ---------------------------------------------------------------------------
# Eyes-closed: prolonged -> flagged (as an inference, not a fact -- the
# wording discipline itself is a caller-side concern, but the boolean
# must be correct)
# ---------------------------------------------------------------------------
def test_prolonged_eyes_closed_is_flagged():
    """Eyes closed continuously for >= 6 minutes must set
    eyes_closed_prolonged_flag. Uses coarse per-30-second frames rather
    than one frame per second, purely to keep the test fast -- the
    duration math itself doesn't care about frame density, only about
    elapsed wall-clock time between the first-closed timestamp and now."""
    signals = [
        FrameSignals(timestamp=float(i * 30), face_present=True,
                     yaw=0.0, pitch=0.0, ear=0.10)
        for i in range(15)  # 0s, 30s, 60s, ... up to 420s (7 minutes)
    ]
    results = run_sequence(signals)

    assert results[-1].eyes_closed_duration >= 6 * 60
    assert results[-1].eyes_closed_prolonged_flag is True

    # And confirm it does NOT fire early, e.g. at the 3-minute mark --
    # this guards against an off-by-one or unit-conversion bug that
    # would make the threshold trigger too early.
    three_minute_mark = next(fv for fv in results if fv.eyes_closed_duration >= 180)
    assert three_minute_mark.eyes_closed_prolonged_flag is False, (
        "3 minutes of closed eyes must not yet be flagged as prolonged"
    )


def test_eyes_closed_duration_resets_when_eyes_open():
    """Confirms the duration counter resets to zero immediately when eyes
    open, rather than continuing to climb or leaving a stale high value
    -- important because a bug here would make a series of short blinks
    look identical to one long closure if the counter didn't reset."""
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, ear=0.10),   # closed
        FrameSignals(timestamp=1.0, face_present=True, ear=0.10),   # still closed, 1s
        FrameSignals(timestamp=2.0, face_present=True, ear=0.30),   # open again
        FrameSignals(timestamp=2.5, face_present=True, ear=0.30),   # still open
    ]
    results = run_sequence(signals)

    assert results[1].eyes_closed_duration > 0
    assert results[2].eyes_closed is False
    assert results[2].eyes_closed_duration == 0.0, (
        "duration must reset to exactly 0 the frame eyes reopen"
    )


# ---------------------------------------------------------------------------
# Phone detection: brief -> not flagged, sustained -> flagged
# ---------------------------------------------------------------------------
def test_brief_phone_detection_is_not_flagged():
    """A phone visible for under PHONE_SUSTAINED_SEC (default 3s) must
    not set phone_flag -- Section 6: 'a phone that appears in-frame for
    a single frame... is not flagged.'"""
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, phone_detected=True),
        FrameSignals(timestamp=1.0, face_present=True, phone_detected=True),
    ]
    results = run_sequence(signals)
    assert all(fv.phone_flag is False for fv in results)


def test_sustained_phone_detection_is_flagged():
    """A phone visible continuously for several seconds must eventually
    set phone_flag -- Section 6: 'a phone held/visible for a sustained
    duration... triggers a direct, named warning notification.'"""
    signals = [
        FrameSignals(timestamp=float(i), face_present=True, phone_detected=True)
        for i in range(6)  # 6 seconds, past the 3s threshold
    ]
    results = run_sequence(signals)
    assert results[-1].phone_flag is True
    assert results[0].phone_flag is False, (
        "must not fire on the very first frame"
    )


def test_phone_flag_resets_when_phone_removed():
    """If the phone is put down before the sustained threshold, the
    duration must not carry over to a later, unrelated re-appearance --
    otherwise two brief, separate glimpses could incorrectly sum into a
    false flag."""
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, phone_detected=True),
        FrameSignals(timestamp=1.0, face_present=True, phone_detected=True),
        FrameSignals(timestamp=2.0, face_present=True, phone_detected=False),  # put down
        FrameSignals(timestamp=10.0, face_present=True, phone_detected=True),  # picked up again, much later
        FrameSignals(timestamp=11.0, face_present=True, phone_detected=True),
    ]
    results = run_sequence(signals)
    # Even though there are 4 total "phone_detected=True" frames across
    # the whole sequence, neither individual appearance lasted 3+
    # continuous seconds, so phone_flag must never have fired.
    assert all(fv.phone_flag is False for fv in results), (
        "two separate brief glimpses of a phone, with a gap between them, "
        "must not incorrectly accumulate into a sustained-duration flag"
    )


# ---------------------------------------------------------------------------
# Second-person detection: brief -> not flagged, sustained -> flagged
# ---------------------------------------------------------------------------
def test_brief_second_person_is_not_flagged():
    """Someone briefly walking past in the background must not set
    second_person_flag -- Section 6: 'a second face for a single frame
    (someone briefly walking past in the background)... is NOT
    flagged.'"""
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, second_person_detected=True),
        FrameSignals(timestamp=1.0, face_present=True, second_person_detected=True),
    ]
    results = run_sequence(signals)
    assert all(fv.second_person_flag is False for fv in results)


def test_sustained_second_person_is_flagged():
    """A second person present continuously for several seconds
    (implying they sat down / are talking with the student) must
    eventually set second_person_flag."""
    signals = [
        FrameSignals(timestamp=float(i), face_present=True, second_person_detected=True)
        for i in range(6)  # past the 4s threshold
    ]
    results = run_sequence(signals)
    assert results[-1].second_person_flag is True
    assert results[0].second_person_flag is False


# ---------------------------------------------------------------------------
# Hysteresis: confirms the flicker bug found during real Phase A testing
# stays fixed at the feature-extraction level, not just in the throwaway
# signal_check.py script
# ---------------------------------------------------------------------------
def test_head_down_hysteresis_prevents_flicker_at_boundary():
    """Regression test for the exact issue found in real Phase A testing:
    a pitch value that sits right at the enter-threshold boundary should
    not cause looking_down to flip back and forth every frame. Once
    looking_down becomes True, pitch must fall below the (lower) EXIT
    threshold before it goes False again."""
    calibration = make_calibration(pitch_down_enter_deg=11.0, pitch_down_exit_deg=8.0)
    signals = [
        FrameSignals(timestamp=0.0, face_present=True, pitch=-12.0, yaw=0.0, ear=0.30),  # enters DOWN (>11)
        FrameSignals(timestamp=1.0, face_present=True, pitch=-9.5, yaw=0.0, ear=0.30),   # between 8 and 11:
                                                                                            # should STAY down
                                                                                            # (hysteresis), not flicker
        FrameSignals(timestamp=2.0, face_present=True, pitch=-12.0, yaw=0.0, ear=0.30),  # still down
        FrameSignals(timestamp=3.0, face_present=True, pitch=-6.0, yaw=0.0, ear=0.30),   # below exit (8): now UP
    ]
    results = run_sequence(signals, calibration=calibration)

    assert results[0].looking_down is True
    assert results[1].looking_down is True, (
        "pitch=-9.5 is between the exit threshold (8) and enter threshold "
        "(11) -- with hysteresis this must STAY down rather than flicker "
        "back to on-screen, which is exactly the bug observed in real "
        "Phase A testing before hysteresis was added"
    )
    assert results[2].looking_down is True
    assert results[3].looking_down is False, (
        "pitch=-6.0 is below the exit threshold, so this frame should "
        "finally register as no-longer-looking-down"
    )


# ---------------------------------------------------------------------------
# Face absent (Away) -- confirm no downstream flags apply, and phone/
# second-person tracking is preserved even without the primary face
# ---------------------------------------------------------------------------
def test_face_absent_produces_safe_defaults():
    """When face_present is False (Away), head/eye-derived flags must
    all read as safe defaults rather than stale or garbage values."""
    signals = [FrameSignals(timestamp=0.0, face_present=False)]
    results = run_sequence(signals)
    fv = results[0]

    assert fv.face_present is False
    assert fv.on_screen is False
    assert fv.looking_down is False
    assert fv.head_turn_flag is False
    assert fv.eyes_closed is False
    assert fv.eyes_closed_prolonged_flag is False


def test_phone_detection_still_works_while_primary_face_absent():
    """A phone can be legitimately detected even in a frame where the
    PRIMARY student's face isn't -- e.g. they leaned out of frame but a
    phone is still visible on the desk. This must still duration-gate
    correctly rather than being silently dropped just because
    face_present is False."""
    signals = [
        FrameSignals(timestamp=float(i), face_present=False, phone_detected=True)
        for i in range(6)
    ]
    results = run_sequence(signals)
    assert results[-1].phone_flag is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
