"""
extract_daisee_features.py
----------------------------
Offline, one-time script: runs the SAME perception pipeline the live
watcher uses (phase_c/watcher/perception/perception.py's Perception
class, reused directly rather than reimplemented -- see
docs/02_TRAINING_GUIDE.txt Step 1's instruction to "run MediaPipe on
DAiSEE's frames, produce the same feature vector schema your watcher
produces live") over each downloaded DAiSEE clip, and writes one CSV
per clip into --output-dir using data/types.py's FEATURE_COLUMNS
schema. daisee_loader.py reads these CSVs back in -- see its docstring
for the two-step pipeline this script is step 1 of.

NOT covered by Phase D's test suite: this script needs real video files,
the real MediaPipe model files (phase_a/models/), and OpenCV with video
codec support, none of which are available in an automated test
environment. Correctness here is instead exercised indirectly: it reuses
Perception unmodified, and its CSV *output* format is exactly what
daisee_loader.py's tests already validate against hand-built fixture
CSVs.

gaze_x/gaze_y are written as 0.0 -- Phase A/B's perception pipeline does
not yet estimate on-screen gaze position as x/y coordinates (only the
coarse "on_screen" yaw/pitch envelope check in Phase B's extractor.py),
matching the same placeholder status these columns have in
phase_c/backend/main.py's StateEvent ingestion schema today. If a future
phase adds real gaze-point estimation, update BOTH this script and the
live watcher's feature pipeline together, or DAiSEE-trained and
live-deployed models will silently disagree on what gaze_x/gaze_y mean.

movement_delta is computed here as the frame-to-frame Euclidean change
in (yaw, pitch, roll), degrees -- a reasonable proxy for "how much did
the head move" until Phase B's live pipeline defines its own
authoritative movement_delta computation; if that computation is ever
added to the watcher, mirror it here for consistency, per the same
reasoning as the gaze_x/gaze_y note above.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

PHASE_C_WATCHER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "phase_c", "watcher")
)


def _require_phase_c_watcher_on_path() -> None:
    if PHASE_C_WATCHER_DIR not in sys.path:
        sys.path.insert(0, PHASE_C_WATCHER_DIR)


def extract_clip(video_path: str, output_csv_path: str, sample_every_n_frames: int = 1) -> int:
    """Run Perception over one video file, write one CSV row per sampled
    frame. Returns the number of rows written."""
    _require_phase_c_watcher_on_path()
    import cv2  # deferred import: only needed when this function actually runs

    from perception.perception import Perception  # phase_c/watcher/perception

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows = []
    prev_pose = None
    frame_index = 0

    with Perception() as perception:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            signals = perception.process_frame(frame_bgr, timestamp)

            if prev_pose is not None:
                movement_delta = math.sqrt(
                    (signals.yaw - prev_pose[0]) ** 2
                    + (signals.pitch - prev_pose[1]) ** 2
                    + (signals.roll - prev_pose[2]) ** 2
                )
            else:
                movement_delta = 0.0
            prev_pose = (signals.yaw, signals.pitch, signals.roll)

            # ear->eyes_closed threshold matches Phase A's tuned constant
            # (see phase_a/signal_check.py) rather than inventing a new one.
            eyes_closed = signals.ear < 0.22

            rows.append(
                {
                    "timestamp": timestamp,
                    "yaw": signals.yaw,
                    "pitch": signals.pitch,
                    "roll": signals.roll,
                    "gaze_x": 0.0,  # see module docstring
                    "gaze_y": 0.0,  # see module docstring
                    "eyes_closed": int(eyes_closed),
                    "posture_angle": signals.posture_angle,
                    "movement_delta": movement_delta,
                    "face_conf": 1.0 if signals.face_present else 0.0,
                    "phone_detected": int(signals.phone_detected),
                    "second_person_detected": int(signals.second_person_detected),
                }
            )
            frame_index += 1

    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "timestamp", "yaw", "pitch", "roll", "gaze_x", "gaze_y", "eyes_closed",
            "posture_angle", "movement_delta", "face_conf", "phone_detected", "second_person_detected",
        ])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract per-frame features from DAiSEE video clips.")
    p.add_argument("--videos-dir", required=True, help="directory of DAiSEE .avi/.mp4 clips (searched recursively)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-every-n-frames", type=int, default=5, help="subsample for speed; 5 at 30fps ~= 6Hz")
    args = p.parse_args(argv)

    video_paths = sorted(
        glob.glob(os.path.join(args.videos_dir, "**", "*.avi"), recursive=True)
        + glob.glob(os.path.join(args.videos_dir, "**", "*.mp4"), recursive=True)
    )
    if not video_paths:
        print(f"no .avi/.mp4 files found under {args.videos_dir!r}")
        return

    for i, video_path in enumerate(video_paths, 1):
        clip_id = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = os.path.join(args.output_dir, f"{clip_id}.csv")
        try:
            n_rows = extract_clip(video_path, output_csv, args.sample_every_n_frames)
            print(f"[{i}/{len(video_paths)}] {clip_id}: {n_rows} rows -> {output_csv}")
        except Exception as exc:  # noqa: BLE001 -- one bad clip shouldn't kill a multi-hour batch job
            print(f"[{i}/{len(video_paths)}] {clip_id}: FAILED ({exc})")


if __name__ == "__main__":
    main()
