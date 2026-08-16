"""
signal_check_PHASE_A.py
----------------
PHASE A -- Signal feasibility check.

Purpose: prove, before writing a single line of model or backend code,
that the raw signals we plan to build the whole system on (head pose,
gaze, eyes-closed, posture, phone presence) actually separate different
behaviors in a visually obvious way on a real webcam. No DL, no
database, no backend -- just live numbers on screen.

WHAT TO DO WITH THIS SCRIPT
  Run it, then deliberately act out each behavior for ~15-20 seconds
  while watching the printed values change:
    - Face the screen normally (typing/reading)              -> ENGAGED
    - Look down and write notes, glancing up periodically     -> baseline
      for the look_up_cycle_rate feature (Section 6 of the doc)
    - Turn your head to the side and hold it (simulate talking) -> yaw
      should swing and STAY out of range
    - Glance left/right quickly (simulate a wide monitor)     -> yaw
      should swing and immediately return
    - Close your eyes for a few seconds (thinking)            -> eyes_closed
      should go True briefly
    - Close your eyes and hold (simulate prolonged rest)      -> eyes_closed
      True for an extended period
    - Lean away / get up                                       -> AWAY
    - Hold a phone up in frame                                -> phone_detected
    - Have a second person step into frame briefly, then for a
      sustained period                                        -> second_person_detected,
                                                                   second_person_flag

  If a value doesn't visibly, clearly change between these cases, that's
  a real finding -- it means the geometry/thresholds need work BEFORE
  building Phase B on top of it. Better to find that out now.

USAGE
  python download_models_PHASE_A.py      (once)
  python signal_check_PHASE_A.py

Press 'q' in the video window to quit.

WINDOWS NOTE ON THE CAMERA BACKEND
  cv2.VideoCapture(0) on Windows sometimes takes several seconds to
  open, or opens with the wrong backend and shows a black frame. This
  script explicitly requests the DirectShow backend (cv2.CAP_DSHOW),
  which is typically faster and more reliable to initialize on Windows
  than the default (MSMF) backend. If you still get a black/frozen
  frame, see the troubleshooting note in the README section at the
  bottom of this file.

NOTE ON GPU (RTX 4050 etc.)
  MediaPipe Tasks' Python GPU delegate is NOT available on Windows as
  of this writing -- Google's own docs and issue tracker confirm GPU
  acceleration for this API is currently Ubuntu/Linux-only, requesting
  it on Windows raises NotImplementedError. This script intentionally
  uses CPU only. Two things worth knowing: (1) this isn't a real loss
  for Phase A/B -- Phase A already runs all three models every frame
  and is expected to feel heavier than the eventual watcher, which
  samples every 1-2 seconds, not every frame; (2) the deployed watcher
  (Phase C onward) is designed to run on CPU by default regardless of
  platform anyway, since it has to run continuously in the background
  on whatever machine a student has, GPU or not -- so CPU-only here
  actually matches the real deployment target, not just a Windows
  workaround.
"""

import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
OBJECT_MODEL_PATH = os.path.join(MODELS_DIR, "efficientdet_lite0.tflite")

# ---------------------------------------------------------------------------
# Thresholds -- these are DELIBERATELY rough starting points for Phase A.
# The point of this script is to help you tune them by eye. Phase B will
# turn validated versions of these into the real feature-extraction module.
# ---------------------------------------------------------------------------
EAR_CLOSED_THRESHOLD = 0.22        # eye-aspect-ratio below this = eyes closed.
                                    # NOTE: this is a rough starting point and
                                    # is NOT reliable with glasses -- lens
                                    # reflections and frame edges distort the
                                    # eyelid landmarks this depends on. Watch
                                    # the live eye_aspect_ratio number on
                                    # screen with your own eyes open vs.
                                    # closed and adjust this constant to sit
                                    # roughly halfway between your two
                                    # observed values. This is exactly the
                                    # kind of per-person tuning Phase B's
                                    # calibration step is meant to formalize.
YAW_ON_SCREEN_DEG = 25.0           # default gaze-envelope half-width (Phase B
                                    # will replace this with per-user calibration)
PITCH_DOWN_THRESHOLD_DEG = 11.0    # magnitude of pitch (regardless of sign)
                                    # beyond which we consider the head
                                    # "looking down." Tuned from a real
                                    # session log: observed pitch while
                                    # genuinely looking down ranged from
                                    # about -15 to -22 degrees, and the
                                    # original 15-degree threshold sat
                                    # almost exactly on that boundary,
                                    # causing the flag to flicker True/
                                    # False rapidly instead of committing
                                    # cleanly. 11 degrees gives clear
                                    # separation. If your own sessions
                                    # show a different natural range,
                                    # re-tune this the same way: watch
                                    # the printed pitch min/max while
                                    # deliberately looking down, and set
                                    # the threshold a few degrees inside
                                    # (i.e. smaller than) that range.
PITCH_UP_RELEASE_DEG = 8.0         # HYSTERESIS: once looking_down is True,
                                    # pitch must come back inside +-8 deg
                                    # (not just back under 11) before we
                                    # call it False again. Using a single
                                    # shared threshold for both directions
                                    # is exactly what caused the flicker
                                    # seen in testing -- natural head
                                    # micro-movement near the boundary
                                    # kept crossing back and forth over
                                    # one line. A gap between the "enter"
                                    # and "exit" thresholds (standard
                                    # hysteresis) makes the state commit
                                    # cleanly instead of chattering.
SMOOTHING_WINDOW = 5               # frames to smooth noisy per-frame values
SECOND_PERSON_DURATION_SEC = 4.0   # a second face must be present this long,
                                    # continuously, before we flag it -- a
                                    # single frame of someone walking past
                                    # in the background should NOT trigger
                                    # this. 4s is a starting point for
                                    # Phase A preview purposes; the real
                                    # watcher (Phase C) should validate this
                                    # against real "someone walked past"
                                    # vs. "someone sat down to talk" sessions
                                    # the same way pitch/EAR were tuned above.

# Indices into MediaPipe's 468-point face mesh used for a simple
# eye-aspect-ratio (EAR) calculation. These are standard landmark indices
# for the left eye (from the viewer's perspective, i.e. the subject's
# right eye) -- good enough for Phase A's visual sanity check; Phase B
# should average both eyes.
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133


def eye_aspect_ratio(landmarks, w, h):
    """Rough eye-aspect-ratio: vertical eye opening / horizontal eye width.
    Lower value = eye more closed. This is a simplification for Phase A;
    it uses one eye and 4 points rather than the full 6-point EAR formula,
    which is sufficient to visually confirm open vs. closed separates
    cleanly on screen."""
    top = landmarks[LEFT_EYE_TOP]
    bottom = landmarks[LEFT_EYE_BOTTOM]
    left = landmarks[LEFT_EYE_LEFT]
    right = landmarks[LEFT_EYE_RIGHT]

    top_xy = np.array([top.x * w, top.y * h])
    bottom_xy = np.array([bottom.x * w, bottom.y * h])
    left_xy = np.array([left.x * w, left.y * h])
    right_xy = np.array([right.x * w, right.y * h])

    vertical = np.linalg.norm(top_xy - bottom_xy)
    horizontal = np.linalg.norm(left_xy - right_xy)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def head_pose_from_matrix(transformation_matrix):
    """Extract approximate yaw/pitch/roll (degrees) from MediaPipe Face
    Landmarker's facial_transformation_matrix output. This gives us head
    pose directly from the model's own 3D estimate rather than needing a
    separate solvePnP step in Phase A -- solvePnP is still worth having
    in Phase B as a cross-check / fallback, but this is enough to
    validate the signal here."""
    m = transformation_matrix[:3, :3]
    # standard rotation-matrix -> Euler angle decomposition
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
        roll = math.atan2(m[2, 1], m[2, 2])
    else:
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0
        roll = math.atan2(-m[1, 2], m[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class RollingSmoother:
    """Small helper to smooth a scalar value over the last N frames, so
    single-frame noise doesn't make the printed numbers jump around --
    this mirrors the 'no state from a single frame' principle from the
    project design doc, applied here just to the raw numbers for
    readability."""

    def __init__(self, window=SMOOTHING_WINDOW):
        self.buf = deque(maxlen=window)

    def push(self, value):
        self.buf.append(value)
        return sum(self.buf) / len(self.buf)


def main():
    for path, name in [
        (FACE_MODEL_PATH, "face_landmarker.task"),
        (POSE_MODEL_PATH, "pose_landmarker_lite.task"),
        (OBJECT_MODEL_PATH, "efficientdet_lite0.tflite"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: missing model file {name} at {path}")
            print("Run 'python download_models_PHASE_A.py' first.")
            sys.exit(1)

    # --- Set up MediaPipe tasks ---
    BaseOptions = mp_python.BaseOptions

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        num_faces=2,  # 2, not 1: we still only use face index 0 for the
                      # primary student's head pose/gaze/eyes, but allowing
                      # a second face lets us detect "someone else entered
                      # frame" (second-person detection, see Section 6 of
                      # the project doc) using this same model instead of
                      # a separate detector.
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

    object_options = mp_vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        score_threshold=0.4,
        category_allowlist=["cell phone"],
    )
    object_detector = mp_vision.ObjectDetector.create_from_options(object_options)

    # --- Open the webcam ---
    # CAP_DSHOW: see the Windows note in this file's docstring.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam with CAP_DSHOW, retrying with default backend...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open any webcam. Check camera permissions "
              "(Windows Settings -> Privacy & security -> Camera) and make "
              "sure no other app is currently using it.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    yaw_smoother = RollingSmoother()
    pitch_smoother = RollingSmoother()
    roll_smoother = RollingSmoother()
    ear_smoother = RollingSmoother()

    # Tracks how long eyes have been continuously closed, for a rough
    # live preview of the "brief vs. prolonged eyes-closed" distinction
    # from Section 6 of the project design doc.
    eyes_closed_since = None
    was_eyes_closed = False  # so we can detect the True/False transition
                              # and print it -- you don't need to see the
                              # screen for this, it goes to the terminal

    # Running min/max EAR observed this session, printed periodically to
    # the terminal so you can find YOUR correct threshold after the fact
    # (e.g. by closing your eyes for a few seconds at any point) without
    # needing to read the screen while your eyes are shut.
    ear_min_seen = float("inf")
    ear_max_seen = float("-inf")
    pitch_min_seen = float("inf")
    pitch_max_seen = float("-inf")
    last_console_print = 0.0

    # Second-person duration tracking, mirrors eyes_closed_since above.
    second_person_since = None
    was_second_person = False

    # Tracks look-up-cycle behavior: counts transitions from
    # "looking down" back to "on-screen pitch" within a rolling window,
    # to preview the look_up_cycle_rate feature.
    was_looking_down = False
    look_up_events = deque(maxlen=50)  # (timestamp) of each look-down->up transition

    print("Signal check running. Press 'q' in the video window to quit.")
    print("Try: face screen normally / look down like note-taking / turn head "
          "and hold / quick glances / close eyes briefly / close eyes long / "
          "lean away / hold up a phone.\n")
    print(f"Current EAR_CLOSED_THRESHOLD = {EAR_CLOSED_THRESHOLD}")
    print("Every eyes_closed state change and a running min/max EAR will be "
          "printed HERE in the terminal (not just on the video overlay), so "
          "you can review what happened even with your eyes shut.\n")

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural "looking at self" view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # --- Face + head pose + eyes ---
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_present = len(face_result.face_landmarks) > 0

        yaw = pitch = roll = 0.0
        ear = 0.0
        eyes_closed = False
        looking_down = False

        if face_present:
            landmarks = face_result.face_landmarks[0]
            if face_result.facial_transformation_matrixes:
                matrix = np.array(face_result.facial_transformation_matrixes[0])
                raw_yaw, raw_pitch, raw_roll = head_pose_from_matrix(matrix)
                yaw = yaw_smoother.push(raw_yaw)
                pitch = pitch_smoother.push(raw_pitch)
                roll = roll_smoother.push(raw_roll)

            raw_ear = eye_aspect_ratio(landmarks, w, h)
            ear = ear_smoother.push(raw_ear)
            eyes_closed = ear < EAR_CLOSED_THRESHOLD

            ear_min_seen = min(ear_min_seen, ear)
            ear_max_seen = max(ear_max_seen, ear)

            if eyes_closed != was_eyes_closed:
                state_str = "CLOSED" if eyes_closed else "OPEN"
                print(f"[{time.time() - start_time:6.1f}s] eyes -> {state_str}"
                      f"   (ear={ear:.3f}, threshold={EAR_CLOSED_THRESHOLD})")
            was_eyes_closed = eyes_closed

            if eyes_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
            else:
                eyes_closed_since = None

            # Hysteresis: use a lower bar to ENTER "looking down" (11 deg)
            # and a lower bar still to EXIT it (8 deg) so natural
            # micro-movement near the boundary doesn't flip the flag back
            # and forth every frame -- see PITCH_DOWN_THRESHOLD_DEG /
            # PITCH_UP_RELEASE_DEG comments above for why this was added.
            if was_looking_down:
                looking_down = abs(pitch) > PITCH_UP_RELEASE_DEG
            else:
                looking_down = abs(pitch) > PITCH_DOWN_THRESHOLD_DEG

            if looking_down != was_looking_down:
                state_str = "DOWN" if looking_down else "UP"
                print(f"[{time.time() - start_time:6.1f}s] head -> {state_str}"
                      f"   (pitch={pitch:+.1f} deg, threshold=+-{PITCH_DOWN_THRESHOLD_DEG})")
            if was_looking_down and not looking_down:
                look_up_events.append(time.time())
            was_looking_down = looking_down

            pitch_min_seen = min(pitch_min_seen, pitch)
            pitch_max_seen = max(pitch_max_seen, pitch)
        else:
            eyes_closed_since = None
            was_looking_down = False

        eyes_closed_duration = (
            time.time() - eyes_closed_since if eyes_closed_since else 0.0
        )

        # look_up_cycle_rate preview: how many look-down->up transitions
        # happened in the last 30 seconds
        now = time.time()
        recent_look_ups = [t for t in look_up_events if now - t <= 30]
        look_up_cycle_rate = len(recent_look_ups)

        # --- Pose (posture) ---
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        posture_note = "n/a"
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            # Rough shoulder-line tilt as a posture proxy for Phase A.
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            dx = (right_shoulder.x - left_shoulder.x) * w
            dy = (right_shoulder.y - left_shoulder.y) * h
            shoulder_tilt = math.degrees(math.atan2(dy, dx))
            posture_note = f"shoulder_tilt={shoulder_tilt:.1f} deg"

        # --- Phone detection ---
        object_result = object_detector.detect_for_video(mp_image, timestamp_ms)
        phone_detected = any(
            d.categories[0].category_name == "cell phone"
            for d in object_result.detections
        )

        # --- Second-person detection ---
        # face_result already ran above for the primary student's head
        # pose/eyes; here we just check how many faces it found. A second
        # face in frame, SUSTAINED (not a one-frame flicker from someone
        # briefly passing behind the student), is treated the same way as
        # phone detection and prolonged eye-closure: duration-gated, then
        # a direct, honestly-worded flag -- not "the student is talking,"
        # just "a second person is present," since that's what's actually
        # observable. The watcher (Phase C onward) turns this into the
        # notification: "Stay in a quiet environment -- avoid talking to
        # others, {name}."
        second_person_count = max(0, len(face_result.face_landmarks) - 1)
        second_person_detected = second_person_count > 0

        if second_person_detected != was_second_person:
            state_str = "DETECTED" if second_person_detected else "GONE"
            print(f"[{time.time() - start_time:6.1f}s] second_person -> "
                  f"{state_str}   (face_count={len(face_result.face_landmarks)})")
        was_second_person = second_person_detected

        if second_person_detected:
            if second_person_since is None:
                second_person_since = time.time()
        else:
            second_person_since = None
        second_person_duration = (
            time.time() - second_person_since if second_person_since else 0.0
        )
        # Duration-gated, same principle as everything else in Section 6
        # of the project doc: a second face for a single frame (someone
        # walking past in the background) should NOT be flagged. Only a
        # SUSTAINED second presence should. SECOND_PERSON_DURATION_SEC
        # below is the gate; Phase A just previews the timer, Phase C is
        # where the real threshold and notification get wired up.
        second_person_flag = second_person_duration > SECOND_PERSON_DURATION_SEC

        # --- Determine on-screen gaze status using the default envelope ---
        on_screen = abs(yaw) <= YAW_ON_SCREEN_DEG if face_present else False

        # ---------------- Overlay text on the video frame ----------------
        lines = [
            f"face_present: {face_present}",
            f"yaw: {yaw:+.1f} deg   pitch: {pitch:+.1f} deg   roll: {roll:+.1f} deg",
            f"on_screen (default envelope +-{YAW_ON_SCREEN_DEG:.0f} deg): {on_screen}",
            f"looking_down (note-taking candidate): {looking_down}",
            f"look_up_cycle_rate (last 30s): {look_up_cycle_rate}",
            f"eye_aspect_ratio: {ear:.3f}   eyes_closed: {eyes_closed}",
            f"eyes_closed_duration: {eyes_closed_duration:.1f}s",
            f"posture: {posture_note}",
            f"PHONE DETECTED: {phone_detected}",
            f"SECOND PERSON DETECTED: {second_person_detected}"
            f"  (duration: {second_person_duration:.1f}s, flag: {second_person_flag})",
        ]

        y0 = 24
        for i, line in enumerate(lines):
            color = (0, 255, 0)
            if "PHONE DETECTED: True" in line:
                color = (0, 0, 255)
            elif "SECOND PERSON DETECTED: True" in line and second_person_flag:
                color = (0, 0, 255)
            elif "eyes_closed: True" in line and eyes_closed_duration > 6 * 60:
                color = (0, 0, 255)
            elif "face_present: False" in line:
                color = (0, 165, 255)
            cv2.putText(
                frame, line, (10, y0 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )

        # Periodic console summary (every ~3s) -- gives you the running
        # min/max EAR observed so far without needing to read the screen.
        # Close your eyes for a few seconds at any point during the run;
        # when you're done, the terminal will already have logged both the
        # OPEN and CLOSED transition lines above, and this summary line
        # confirms the full range so you can pick a threshold that sits
        # cleanly between your own min and max.
        if time.time() - last_console_print > 3.0:
            print(f"  ... ear range so far: min={ear_min_seen:.3f}  "
                  f"max={ear_max_seen:.3f}  (pick a threshold roughly "
                  f"halfway between your open and closed values)")
            print(f"  ... pitch range so far: min={pitch_min_seen:+.1f}deg  "
                  f"max={pitch_max_seen:+.1f}deg  (whichever direction moves "
                  f"toward negative or positive when you look DOWN tells us "
                  f"the real sign to threshold on)")
            last_console_print = time.time()

        cv2.imshow("Phase A - Signal Check (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()
    object_detector.close()


if __name__ == "__main__":
    main()

"""
--------------------------------------------------------------------------
README / TROUBLESHOOTING (Windows-specific)
--------------------------------------------------------------------------

INSTALL (run once, inside an activated venv):
  pip install opencv-python mediapipe numpy

RUN:
  python download_models_PHASE_A.py
  python signal_check_PHASE_A.py

WHAT "GOOD" LOOKS LIKE
  - yaw/pitch should swing noticeably and smoothly as you turn/tilt your
    head, and settle back near 0 when facing the screen normally.
  - on_screen should flip to False when you turn to talk, and mostly
    stay True during quick glances (though at the DEFAULT fixed envelope
    used here, a wide-monitor glance MAY incorrectly show False -- that's
    expected and exactly why Phase B adds per-user calibration instead
    of this fixed +-25 degree default).
  - looking_down should go True while writing notes, and
    look_up_cycle_rate should climb as you glance up periodically --
    confirming this signal can differentiate note-taking from checked-out
    disengagement, per Section 6 of the project doc.
  - eyes_closed should flip True within about a second of closing your
    eyes, and eyes_closed_duration should climb steadily if you keep
    them closed.
  - PHONE DETECTED should flip True within a second or two of holding a
    phone up clearly in frame, and back to False when you put it down or
    out of frame.
  - face_present should go False almost immediately when you lean out of
    frame or get up.

IF THE CAMERA WINDOW IS BLACK OR FROZEN (Windows)
  1. Close any other app that might be using the camera (Teams, Zoom,
     Windows Camera app, browser tabs with camera permission).
  2. Confirm Windows camera privacy settings allow desktop apps to use
     the camera: Settings -> Privacy & security -> Camera -> "Let apps
     access your camera" (On) and "Let desktop apps access your camera"
     (On).
  3. If cv2.CAP_DSHOW still fails, try explicitly forcing the Microsoft
     Media Foundation backend instead by changing the VideoCapture line
     in this script to: cv2.VideoCapture(0, cv2.CAP_MSMF)
  4. If you have more than one camera (e.g. a USB webcam plus a laptop
     built-in camera), try index 1 instead of 0:
     cv2.VideoCapture(1, cv2.CAP_DSHOW)

IF MEDIAPIPE FAILS TO IMPORT
  Confirm you're on Python 3.10 or 3.11 (python --version). MediaPipe's
  Windows wheel support is narrower than its Linux/Mac support across
  Python versions -- 3.10/3.11 is the safest choice as of this writing.

IF THE FRAME RATE FEELS VERY SLOW
  This script runs all three models (face, pose, object detector) every
  single frame, which is heavier than the real watcher needs -- the
  actual watcher (Phase C onward) will sample at a much lower rate
  (every 1-2 seconds, not every frame), which is what keeps it cheap
  enough to run continuously in the background. Slowness here is
  expected and not a sign anything is wrong.
--------------------------------------------------------------------------
"""
