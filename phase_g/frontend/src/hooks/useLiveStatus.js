// hooks/useLiveStatus.js
// -----------------------
// Subscribes to the backend's GET /ws/target-blocks/{id}/live-status
// (routes.py) -- the watcher's current state, away-stopwatch, and
// phone/second-person warning flags, pushed by the (unmodified) watcher
// via its own POST to that same block_id and fanned out server-side.
//
// The JWT is passed as a query param (?token=...) since the browser
// WebSocket API cannot set custom headers -- see routes.py's
// ws_live_status docstring for the same note from the other side.

import { useEffect, useRef, useState } from "react";
import { getToken, wsUrl } from "../api/client";

export function useLiveStatus(blockId) {
  const [status, setStatus] = useState(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    if (!blockId) return undefined;

    const token = getToken();
    const socket = new WebSocket(wsUrl(`/ws/target-blocks/${blockId}/live-status?token=${token}`));
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = (event) => {
      try {
        setStatus(JSON.parse(event.data));
      } catch {
        // ignore malformed frames rather than crashing the live view
      }
    };

    return () => socket.close();
  }, [blockId]);

  return { status, connected };
}
