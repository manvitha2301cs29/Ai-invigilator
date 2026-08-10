"""
download_models.py
-------------------
Downloads the three pretrained MediaPipe models this project uses:
  1. Face Landmarker  (face + iris + head geometry)
  2. Pose Landmarker  (body/posture landmarks) -- using the "lite"
     variant deliberately: it's the fastest of the three tiers
     (lite/full/heavy), which matters because this will eventually run
     continuously on a student's laptop CPU inside the watcher process.
  3. Object Detector (EfficientDet-Lite0) -- COCO-pretrained, includes
     a "cell phone" class out of the box. No custom training needed.

Run this once before running signal_check.py.

Windows note: these are plain HTTPS downloads via urllib, so no extra
tools (wget/curl) are required -- this will work in a stock Windows
Python install without WSL or Git Bash.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
    ),
}


def download(filename: str, url: str) -> None:
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {filename} already present ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return
    print(f"[downloading] {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"[done] {filename} ({size_mb:.1f} MB) -> {dest}")
    except Exception as e:
        print(f"[FAILED] {filename}: {e}")
        print(
            "  If this keeps failing, download the URL manually in a browser "
            f"and save it to:\n  {dest}"
        )
        raise


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Model directory: {MODELS_DIR}\n")
    for filename, url in MODELS.items():
        download(filename, url)
    print("\nAll models ready. You can now run signal_check.py")


if __name__ == "__main__":
    main()
