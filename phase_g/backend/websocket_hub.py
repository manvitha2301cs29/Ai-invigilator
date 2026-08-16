"""
websocket_hub.py
------------------
Backs the dashboard's LIVE STATUS screen: "live status via WebSocket
reflecting the watcher's current state including the away-stopwatch and
phone/second-person warning banners" (continuation brief, Phase G).

DESIGN: the watcher (unchanged, per the brief -- "Keep the watcher
running via plain `python run_watcher.py`") gets ONE new, small,
watcher-token-authenticated POST endpoint to push its current status
(routes.py's post_live_status). This module is the in-memory pub/sub
hub that fans each push out to any dashboard browser tabs currently
holding a WebSocket connection open for that target_block_id.

IN-MEMORY, SINGLE-PROCESS CAVEAT (documented, not hidden): connections
are held in a plain dict in this process's memory. This is fine for a
single-instance deployment (Railway/Render's default, per
docs/03_DEPLOYMENT_GUIDE.txt) but will NOT fan out correctly if the
backend is ever horizontally scaled to multiple instances -- a future
scale-up would need a shared pub/sub layer (e.g. Redis) instead. Not
needed for this project's scope; called out here so it isn't a silent
surprise later.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import WebSocket


class LiveStatusHub:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, target_block_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[target_block_id].add(websocket)

    def disconnect(self, target_block_id: str, websocket: WebSocket) -> None:
        self._connections[target_block_id].discard(websocket)
        if not self._connections[target_block_id]:
            self._connections.pop(target_block_id, None)

    async def broadcast(self, target_block_id: str, payload: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections.get(target_block_id, ()):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(target_block_id, ws)


hub = LiveStatusHub()
