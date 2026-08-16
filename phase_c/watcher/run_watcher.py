"""
run_watcher_PHASE_C.py
------------------------
The actual local watcher process (Section 12 of the project doc: "a
Python process that runs on the student's own machine, owns the webcam
continuously... independent of any browser tab"). This is the Phase C
deliverable that ties everything else in this folder together:

  Perception (perception/perception_PHASE_C.py)
    -> FrameSignals
  Feature extraction (features/extractor_PHASE_C.py, from Phase B)
    -> FeatureVector
  Monitoring state machine (monitoring/monitor_PHASE_C.py)
    -> Notifications + completed events
  Backend client (client_PHASE_C.py)
    -> HTTP calls to the FastAPI backend

Per the master coding prompt's Phase C instructions: "Give me a way to
manually declare a break via a simple CLI command or hotkey for now --
we'll wire up the real UI in Phase G." This uses a background input
thread listening for the 'b' key (toggle break) and 'q' (quit), since a
real desktop-notification UI is explicitly deferred to Phase G.

USAGE (Windows):
    python run_watcher_PHASE_C.py --backend-url http://localhost:8000 --token <your-watcher-token> --name Kyra --duration-min 90

While running:
    press 'b' + Enter  -> toggle a declared break on/off
    press 'q' + Enter  -> end the block and quit
"""

import argparse
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

from features.types import CalibrationProfile, RollingState
from features.extractor import extract_features, is_note_taking_pattern
from monitoring.monitor import (
    TargetBlockMonitorState, tick, start_break, end_break,
)
from client import BackendClient
from perception.perception import Perception

SAMPLE_INTERVAL_SEC = 1.5  # how often to sample the webcam and run the
                            # full perception+feature pipeline. NOT every
                            # frame -- Phase A's script ran every frame
                            # for visual debugging, but the real watcher
                            # must be cheap enough to run continuously in
                            # the background, so it samples on an
                            # interval instead. This is the concrete
                            # implementation of the tradeoff described in
                            # signal_check_PHASE_A.py's own docstring.


class BreakToggle:
    """Tiny thread-safe flag set by the background input-listener thread
    and read by the main loop. Kept separate from the monitor's own
    TargetBlockMonitorState because the input thread and the main
    perception loop run concurrently and must not race on the same
    mutable state object."""
    def __init__(self):
        self._lock = threading.Lock()
        self._toggled = False
        self._quit = False

    def request_toggle(self):
        with self._lock:
            self._toggled = True

    def request_quit(self):
        with self._lock:
            self._quit = True

    def consume_toggle(self) -> bool:
        with self._lock:
            was_toggled = self._toggled
            self._toggled = False
            return was_toggled

    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit


def _input_listener(toggle: BreakToggle):
    """Runs in a background thread, blocking on stdin so the main
    camera loop is never held up waiting for keyboard input."""
    while not toggle.quit_requested():
        try:
            line = input().strip().lower()
        except EOFError:
            break
        if line == "b":
            toggle.request_toggle()
            print("[watcher] break toggle requested")
        elif line == "q":
            toggle.request_quit()
            print("[watcher] quit requested — finishing current tick, then ending block")
            break


def run(args):
    client = BackendClient(args.backend_url, args.token)

    now = datetime.now(timezone.utc)
    planned_end = now + timedelta(minutes=args.duration_min)

    print(f"[watcher] creating target block: {args.duration_min} min, "
          f"{now.isoformat()} -> {planned_end.isoformat()}")
    block = client.create_target_block(
        now.isoformat(), planned_end.isoformat(), args.duration_min
    )
    if block is None:
        print("[watcher] FATAL: could not create target block on backend. "
              "Check --backend-url and --token, and that the backend is running.")
        sys.exit(1)
    block_id = block["id"]
    print(f"[watcher] target block created: {block_id}")

    started = client.start_target_block(block_id)
    if started is None:
        print("[watcher] WARNING: could not mark block as started on backend "
              "(continuing anyway — local monitoring still works)")

    calibration = CalibrationProfile()  # Phase A-tuned defaults; a real
                                          # per-user calibration step is
                                          # a Phase G UI concern
    feature_state = RollingState()
    monitor_state = TargetBlockMonitorState()
    break_toggle = BreakToggle()

    listener_thread = threading.Thread(target=_input_listener, args=(break_toggle,), daemon=True)
    listener_thread.start()

    print(f"[watcher] monitoring started for {args.name}. "
          f"Type 'b' + Enter to toggle a break, 'q' + Enter to end early.\n")

    block_end_time = time.time() + args.duration_min * 60

    try:
        with Perception(camera_index=args.camera_index) as perception:
            while True:
                if break_toggle.quit_requested():
                    print("[watcher] quit requested, ending block")
                    break
                if time.time() >= block_end_time:
                    print("[watcher] planned duration reached, ending block")
                    break

                if break_toggle.consume_toggle():
                    if monitor_state.declared_break_active:
                        monitor_state = end_break(monitor_state)
                        client.post_break_event(
                            block_id,
                            datetime.now(timezone.utc).isoformat(),
                            datetime.now(timezone.utc).isoformat(),
                        )
                        print("[watcher] break ended, resuming monitoring")
                    else:
                        monitor_state = start_break(monitor_state, now=time.time())
                        print("[watcher] break started")

                signals = perception.read_one()
                if signals is None:
                    print("[watcher] frame grab failed, retrying...")
                    time.sleep(SAMPLE_INTERVAL_SEC)
                    continue

                feature_vector, feature_state = extract_features(
                    signals, feature_state, calibration
                )

                monitor_state, notifications = tick(
                    monitor_state,
                    now=signals.timestamp,
                    face_present=signals.face_present,
                    phone_flag=feature_vector.phone_flag,
                    second_person_flag=feature_vector.second_person_flag,
                    student_name=args.name,
                )

                for note in notifications:
                    # Phase C: print to console as a stand-in for a real
                    # OS desktop notification, which is a Phase G concern
                    # (see 00_MASTER_CODING_PROMPT.txt Phase G notes).
                    print(f"\n  >>> [{note.tier.value.upper()}] {note.message}\n")

                # Report the raw state event to the backend. Phase C
                # doesn't yet run the trained temporal classifier
                # (that's Phase D) -- for now, log a simple heuristic
                # state derived directly from the feature vector, which
                # doubles as the "rule-based baseline" the master prompt
                # asked Phase C to include.
                heuristic_state = _heuristic_state(feature_vector, signals.face_present)
                client.post_state_event(
                    block_id,
                    datetime.now(timezone.utc).isoformat(),
                    heuristic_state,
                    yaw=signals.yaw, pitch=signals.pitch, roll=signals.roll,
                    eyes_closed=feature_vector.eyes_closed,
                    posture_angle=signals.posture_angle,
                    face_conf=1.0 if signals.face_present else 0.0,
                    phone_detected=signals.phone_detected,
                    second_person_detected=signals.second_person_detected,
                )

                status_line = (
                    f"[{heuristic_state:>19}] face={signals.face_present} "
                    f"on_screen={feature_vector.on_screen} "
                    f"looking_down={feature_vector.looking_down} "
                    f"eyes_closed={feature_vector.eyes_closed} "
                    f"phone={feature_vector.phone_flag} "
                    f"2nd_person={feature_vector.second_person_flag} "
                    f"break={monitor_state.declared_break_active}"
                )
                print(status_line)

                time.sleep(SAMPLE_INTERVAL_SEC)

        # Close out any still-open events for reporting purposes
        for event in monitor_state.completed_events:
            _report_completed_event(client, block_id, event)

    finally:
        ended = client.end_target_block(block_id)
        if ended:
            print(f"[watcher] block ended, status={ended.get('status')}")
        else:
            print("[watcher] WARNING: could not confirm block end on backend")


def _heuristic_state(fv, face_present: bool) -> str:
    """The Phase C rule-based baseline (master prompt: 'a rule-based
    baseline + local watcher skeleton... using simple thresholds, no
    trained model yet'). This is intentionally simple and will be
    REPLACED by Phase D's trained GRU classifier -- kept here only so
    Phase C produces a genuinely working end-to-end system with
    something meaningful in the 'state' field, and so Phase D has a
    baseline to beat (per Section 15's evaluation methodology)."""
    if not face_present:
        return "away"
    if fv.phone_flag or fv.second_person_flag:
        return "distracted_present"
    if fv.head_turn_flag:
        return "distracted_present"
    if fv.looking_down:
        if is_note_taking_pattern(fv):
            return "engaged"  # note-taking counts as engaged, not idle
        return "idle_present"
    if fv.eyes_closed and fv.eyes_closed_duration > 3.0:
        return "idle_present"
    return "engaged"


def _report_completed_event(client: BackendClient, block_id: str, event):
    from datetime import datetime as dt, timezone as tz
    start_iso = dt.fromtimestamp(event.start, tz=tz.utc).isoformat()
    end_iso = dt.fromtimestamp(event.end, tz=tz.utc).isoformat()
    if event.event_type == "away":
        client.post_away_event(block_id, start_iso, end_iso, event.duration_sec, event.was_notified)
    elif event.event_type == "phone":
        client.post_phone_event(block_id, start_iso, end_iso, event.duration_sec, event.was_notified)
    elif event.event_type == "second_person":
        client.post_second_person_event(block_id, start_iso, end_iso, event.duration_sec, event.was_notified)


def main():
    parser = argparse.ArgumentParser(description="AI Study Invigilator — local watcher (Phase C)")
    parser.add_argument("--backend-url", required=True, help="e.g. http://localhost:8000")
    parser.add_argument("--token", required=True, help="Your watcher_token from account creation")
    parser.add_argument("--name", default="there", help="Used in personalized notifications")
    parser.add_argument("--duration-min", type=int, default=90, help="Target block length in minutes")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
