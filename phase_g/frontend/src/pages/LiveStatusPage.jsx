// pages/LiveStatusPage.jsx
// --------------------------
// Screen 2: "live status via WebSocket reflecting the watcher's current
// state including the away-stopwatch and phone/second-person warning
// banners." Since there's no server-side "which block is active right
// now" lookup exposed yet, the student enters/selects the block_id --
// in practice this is the block they just started in their watcher
// (Planner screen's "Go live" link can carry it via router state in a
// fuller build-out; kept as a manual field here to keep this screen's
// scope to exactly what Phase G's WebSocket endpoint provides).

import { useState } from "react";
import { useLiveStatus } from "../hooks/useLiveStatus";

const STATE_LABELS = {
  engaged: "Engaged",
  idle_present: "Idle (present)",
  distracted_present: "Distracted (present)",
  away: "Away",
};

function formatStopwatch(seconds) {
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function LiveStatusPage() {
  const [blockIdInput, setBlockIdInput] = useState("");
  const [activeBlockId, setActiveBlockId] = useState(null);
  const { status, connected } = useLiveStatus(activeBlockId);

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="font-display text-2xl text-ink-50">Live status</h1>
      <p className="mt-1 text-ink-400">
        Shows what your local watcher is currently reporting for an active block.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setActiveBlockId(blockIdInput.trim());
        }}
        className="mt-6 flex items-center gap-2"
      >
        <input
          value={blockIdInput}
          onChange={(e) => setBlockIdInput(e.target.value)}
          placeholder="Active target block ID"
          className="flex-1 rounded-md border border-ink-600 bg-ink-900 px-3 py-2 font-mono text-sm text-ink-50 outline-none focus:border-signal-amber"
        />
        <button
          type="submit"
          className="rounded-md bg-ink-700 px-4 py-2 text-sm font-medium text-ink-50 hover:bg-ink-600"
        >
          Watch
        </button>
      </form>

      {activeBlockId && (
        <div className="mt-8 rounded-xl border border-ink-700 bg-ink-900/50 p-6">
          <div className="flex items-center gap-2 text-sm text-ink-400">
            <span className={`h-2 w-2 rounded-full ${connected ? "bg-verdict-ontrack" : "bg-ink-600"}`} />
            {connected ? "Connected" : "Waiting for connection…"}
          </div>

          {status ? (
            <div className="mt-6 space-y-4">
              <div>
                <div className="text-sm text-ink-400">Current state</div>
                <div className="mt-1 font-display text-2xl text-ink-50">
                  {STATE_LABELS[status.state] || status.state}
                </div>
              </div>

              {status.away_stopwatch_sec > 0 && (
                <div className="rounded-lg border border-signal-amber/40 bg-signal-amber/10 px-4 py-3">
                  <div className="text-sm text-signal-amber">Away stopwatch</div>
                  <div className="font-mono text-xl text-signal-amber">
                    {formatStopwatch(status.away_stopwatch_sec)}
                  </div>
                </div>
              )}

              <div className="flex gap-3">
                {status.phone_flag && (
                  <span className="rounded-full border border-verdict-shorter/40 bg-verdict-shorter/10 px-3 py-1 text-sm text-verdict-shorter">
                    Phone detected
                  </span>
                )}
                {status.second_person_flag && (
                  <span className="rounded-full border border-verdict-shorter/40 bg-verdict-shorter/10 px-3 py-1 text-sm text-verdict-shorter">
                    Second person detected
                  </span>
                )}
              </div>
            </div>
          ) : (
            <p className="mt-6 text-ink-400">No status received yet -- make sure the watcher is running for this block.</p>
          )}
        </div>
      )}
    </div>
  );
}
