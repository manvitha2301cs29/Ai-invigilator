================================================================================
PHASE B -- FEATURE EXTRACTION MODULE
AI Study Invigilator
================================================================================

WHAT THIS IS
  A deterministic, unit-tested function that turns one frame's raw
  signals (head pose, gaze, eye state, phone/second-person detection)
  into a clean FeatureVector -- implementing exactly the duration-gated,
  pattern-specific false-positive rules from Section 6 of
  AI_Study_Invigilator_Project.txt.

  Corresponds to Phase B in 00_MASTER_CODING_PROMPT.txt and Layer 2
  (Feature Engine) in Section 4 (System Architecture) of the project
  doc.

  Nothing in this folder touches a camera or a MediaPipe model. That's
  deliberate -- Phase A (signal_check.py) proved the raw signals are
  usable; Phase B turns the RULES around those signals into real,
  testable code that Phase C's live watcher will call frame by frame.

FILES IN THIS FOLDER
  features/
    types.py         -- plain data structures (FrameSignals, RollingState,
                       CalibrationProfile, FeatureVector's HeadState enum)
    extractor.py     -- the extract_features() function and its helpers;
                       this is the actual Phase B deliverable
    __init__.py      -- package exports
  tests/
    test_extractor.py -- 16 unit tests, 100% line coverage of extractor.py
  requirements.txt   -- pytest + pytest-cov

WHERE THIS FITS IN THE REAL PROJECT FOLDER STRUCTURE
  Per Section 14 of the project doc, this features/ folder is meant to
  live at watcher/features/ once Phase C scaffolds the full project
  layout. For now it's self-contained so it can be tested in isolation.

--------------------------------------------------------------------------
SETUP (Windows)
--------------------------------------------------------------------------

  1. From this folder (in the same venv you created for Phase A, or a
     fresh one -- this phase does NOT need opencv/mediapipe, only
     pytest):

       pip install -r requirements.txt

--------------------------------------------------------------------------
RUNNING THE TESTS
--------------------------------------------------------------------------

    pytest tests/test_extractor_PHASE_B.py -v

  You should see 16 tests, all PASSED. For a coverage report:

    pytest tests/test_extractor_PHASE_B.py --cov=features --cov-report=term-missing

  This should show 100% coverage on all three files under features/.

--------------------------------------------------------------------------
WHAT EACH TEST PROVES (mapped to Section 6 of the project doc)
--------------------------------------------------------------------------

  test_note_taking_pattern_is_not_flagged
    Sustained head-down WITH a look-up cycle -> NOT flagged.

  test_sustained_head_down_no_cycle_is_not_note_taking
    Sustained head-down with NO look-up cycle -> does NOT get the
    note-taking pass; falls through toward the "Maybe" row in Section
    6's summary table instead.

  test_brief_head_turn_is_not_flagged
  test_sustained_head_turn_is_flagged
    A glance (~1s) is never flagged; a held turn (8s, past the 4s
    threshold) is.

  test_calibrated_envelope_widens_on_screen_zone
    The SAME raw yaw (35 degrees) is on_screen under a wide-monitor
    calibration profile (45-degree envelope) but NOT on_screen under
    the default profile (25-degree envelope) -- proves calibration
    actually changes behavior rather than being a decorative field.

  test_brief_eyes_closed_is_not_flagged
  test_prolonged_eyes_closed_is_flagged
  test_eyes_closed_duration_resets_when_eyes_open
    A few seconds closed is never flagged. 6+ minutes continuously
    closed IS flagged (eyes_closed_prolonged_flag) -- but doesn't fire
    early (checked at the 3-minute mark). Duration resets to exactly 0
    the instant eyes reopen, so separate blinks can't accumulate into a
    false prolonged-closure flag.

  test_brief_phone_detection_is_not_flagged
  test_sustained_phone_detection_is_flagged
  test_phone_flag_resets_when_phone_removed
    Under 3 continuous seconds: never flagged. 6 seconds continuous:
    flagged. Two SEPARATE brief glimpses with a gap between them do NOT
    incorrectly sum into a flag -- each one individually has to be
    sustained.

  test_brief_second_person_is_not_flagged
  test_sustained_second_person_is_flagged
    Same duration-gating pattern as phone detection, per Section 6's
    "A SECOND PERSON ENTERS THE FRAME" case.

  test_head_down_hysteresis_prevents_flicker_at_boundary
    Regression test for the exact bug found during real Phase A
    testing on a live webcam: a pitch value sitting between the enter
    (11 deg) and exit (8 deg) thresholds must NOT flicker the
    looking_down flag back and forth. This test would FAIL against a
    naive single-threshold implementation (verified manually -- a plain
    "abs(pitch) > 11" check produces True/False/True/False on the exact
    input this test uses); it passes here because extractor.py
    implements hysteresis instead.

  test_face_absent_produces_safe_defaults
  test_phone_detection_still_works_while_primary_face_absent
    When the primary student's face isn't detected (Away), head/eye
    flags all read safe defaults -- but phone detection still works
    independently, since a phone can be visible on a desk even if the
    student has leaned out of frame.

--------------------------------------------------------------------------
HOW THIS CONNECTS TO PHASE A
--------------------------------------------------------------------------
  The default CalibrationProfile values in features/types_PHASE_B.py
  (yaw_envelope_deg=25.0, pitch_down_enter_deg=11.0,
  pitch_down_exit_deg=8.0, ear_closed_threshold=0.22) are taken directly
  from what was tuned against a real webcam session in Phase A
  (signal_check.py). If you re-tune signal_check.py's constants further
  on your own machine, update CalibrationProfile's defaults here to
  match, so Phase C's watcher (which will use this module) inherits your
  real tuning rather than Phase A's original guesses.

--------------------------------------------------------------------------
NEXT STEP
--------------------------------------------------------------------------
  Once these tests pass on your machine, move to Phase C: the rule-based
  baseline, the local watcher skeleton, the FastAPI backend, and the
  away-stopwatch + phone/second-person warning state machine. Send the
  Phase C follow-up prompt (full text in 00_MASTER_CODING_PROMPT.txt) to
  continue.
================================================================================
