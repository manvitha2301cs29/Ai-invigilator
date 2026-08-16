"""
client_PHASE_C.py
-------------------
Thin HTTP client the watcher uses to talk to the backend. Kept
deliberately separate from run_watcher_PHASE_C.py so the networking
concerns (auth header, retries, error handling) don't clutter the main
camera loop, and so this module can be unit tested with a mocked
backend if desired later.

Sends ONLY what Section 8's data model specifies -- numeric features and
event records -- never a video frame, per the project's privacy-by-
design principle.
"""

import requests


class BackendClient:
    def __init__(self, base_url: str, watcher_token: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-watcher-token": watcher_token}
        self.timeout = timeout

    def _post(self, path: str, json: dict) -> dict | None:
        try:
            resp = requests.post(
                f"{self.base_url}{path}", json=json,
                headers=self.headers, timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            # Deliberately non-fatal: a watcher losing network for a
            # moment (Wi-Fi hiccup, backend restart) should NOT crash
            # the whole monitoring session -- the student is still being
            # watched locally even if this one event failed to upload.
            # Phase G should add a local retry queue; Phase C just logs
            # and continues.
            print(f"[client] WARNING: POST {path} failed: {e}")
            return None

    def create_target_block(self, planned_start_iso: str, planned_end_iso: str,
                             planned_duration_min: int) -> dict | None:
        return self._post("/target-blocks", {
            "planned_start": planned_start_iso,
            "planned_end": planned_end_iso,
            "planned_duration_min": planned_duration_min,
        })

    def start_target_block(self, block_id: str) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/start", {})

    def end_target_block(self, block_id: str) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/end", {})

    def post_state_event(self, block_id: str, timestamp_iso: str, state: str, **features) -> dict | None:
        payload = {"timestamp": timestamp_iso, "state": state, **features}
        return self._post(f"/target-blocks/{block_id}/state-events", payload)

    def post_away_event(self, block_id: str, start_iso: str, end_iso: str,
                         duration_sec: float, was_notified: bool) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/away-events", {
            "away_start": start_iso, "away_end": end_iso,
            "duration_sec": duration_sec, "was_notified": was_notified,
        })

    def post_phone_event(self, block_id: str, start_iso: str, end_iso: str,
                          duration_sec: float, was_notified: bool) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/phone-events", {
            "detected_start": start_iso, "detected_end": end_iso,
            "duration_sec": duration_sec, "was_notified": was_notified,
        })

    def post_second_person_event(self, block_id: str, start_iso: str, end_iso: str,
                                  duration_sec: float, was_notified: bool) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/second-person-events", {
            "detected_start": start_iso, "detected_end": end_iso,
            "duration_sec": duration_sec, "was_notified": was_notified,
        })

    def post_break_event(self, block_id: str, start_iso: str, end_iso: str | None) -> dict | None:
        return self._post(f"/target-blocks/{block_id}/break-events", {
            "break_start": start_iso, "break_end": end_iso,
        })
