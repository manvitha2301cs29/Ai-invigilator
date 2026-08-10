"""
perception_PHASE_C.py
-----------------------
Wraps the three MediaPipe models (Face Landmarker, Pose Landmarker,
Object Detector) exactly as validated in Phase A's
signal_check_PHASE_A.py, but as a clean, reusable class that returns
FrameSignals (Phase B's input type) instead of printing a debug
overlay. This is Layer 1 (Perception) becoming real, watcher-usable
code rather than a one-off visual check script.

The geometry (head-pose-from-matrix, eye-aspect-ratio) is copied
directly from signal_check_PHASE_A.py rather than reimplemented, since
that logic was already tuned and validated against a real webcam in
Phase A -- changing it here would risk silently diverging from what was
actually tested.
"""

import math
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from features.types import FrameSignals

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
OBJECT_MODEL_PATH = os.path.join(MODELS_DIR, "efficientdet_lite0.tflite")

# Same eye-landmark indices used and validated in Phase A.
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133


def _eye_aspect_ratio(landmarks, w, h) -> float:
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


def _head_pose_from_matrix(transformation_matrix) -> tuple[float, float, float]:
    m = transformation_matrix[:3, :3]
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


class Perception:
    """Owns the three MediaPipe models and the webcam. One instance per
    watcher process -- NOT per TargetBlock, since the models themselves
    are expensive to load and should persist across blocks within one
    run of the watcher.

    Use as a context manager:
        with Perception() as p:
            for signals in p.frames():
                ...
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap = None
        self._face_landmarker = None
        self._pose_landmarker = None
        self._object_detector = None
        self._start_time = None

    def __enter__(self):
        for path, name in [
            (FACE_MODEL_PATH, "face_landmarker.task"),
            (POSE_MODEL_PATH, "pose_landmarker_lite.task"),
            (OBJECT_MODEL_PATH, "efficientdet_lite0.tflite"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing model file {name} at {path}. Run "
                    f"download_models_PHASE_A.py (Phase A) first, or copy "
                    f"the models/ folder into watcher/models/."
                )

        BaseOptions = mp_python.BaseOptions

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            num_faces=2,  # primary student + second-person detection,
                          # same as Phase A's tuning
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

        object_options = mp_vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            score_threshold=0.4,
            category_allowlist=["cell phone"],
        )
        self._object_detector = mp_vision.ObjectDetector.create_from_options(object_options)

        # Same CAP_DSHOW-first strategy validated on Windows in Phase A.
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check camera permissions and that "
                "no other app is currently using it."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        import time
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
        if self._face_landmarker is not None:
            self._face_landmarker.close()
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
        if self._object_detector is not None:
            self._object_detector.close()

    def read_one(self) -> FrameSignals | None:
        """Grabs and processes exactly one frame. Returns None if the
        frame grab failed (e.g. camera disconnected) -- callers should
        treat this as "skip this tick, try again next time" rather than
        a fatal error, matching the watcher's general "stay running"
        philosophy (Section 12: this has to run continuously in the
        background without babysitting)."""
        import time

        ok, frame = self._cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        now = time.time()

        face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_present = len(face_result.face_landmarks) > 0

        yaw = pitch = roll = 0.0
        ear = 0.30

        if face_present:
            landmarks = face_result.face_landmarks[0]
            if face_result.facial_transformation_matrixes:
                matrix = np.array(face_result.facial_transformation_matrixes[0])
                yaw, pitch, roll = _head_pose_from_matrix(matrix)
            ear = _eye_aspect_ratio(landmarks, w, h)

        second_person_detected = max(0, len(face_result.face_landmarks) - 1) > 0

        pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        posture_angle = 0.0
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            dx = (right_shoulder.x - left_shoulder.x) * w
            dy = (right_shoulder.y - left_shoulder.y) * h
            posture_angle = math.degrees(math.atan2(dy, dx))

        object_result = self._object_detector.detect_for_video(mp_image, timestamp_ms)
        phone_detected = any(
            d.categories[0].category_name == "cell phone"
            for d in object_result.detections
        )

        return FrameSignals(
            timestamp=now,
            face_present=face_present,
            yaw=yaw, pitch=pitch, roll=roll,
            ear=ear,
            posture_angle=posture_angle,
            phone_detected=phone_detected,
            second_person_detected=second_person_detected,
        )
