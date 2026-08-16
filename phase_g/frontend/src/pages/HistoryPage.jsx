// pages/HistoryPage.jsx
// -----------------------
// Screen 4: "a history/trends screen." Reads GET /target-blocks/history
// (Phase F's BlockRecord history, newest first, with each block's
// per-block verdict from verdict.py -- the exact same thresholds/logic
// that produced the same block's end-of-block verdict).

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import VerdictBadge from "../components/VerdictBadge";

export default function HistoryPage() {
  const [history, setHistory] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .getHistory(30)
      .then(setHistory)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="font-display text-2xl text-ink-50">History &amp; trends</h1>
      <p className="mt-1 text-ink-400">Your most recent completed blocks, newest first.</p>

      {error && <p className="mt-4 text-sm text-verdict-shorter">{error}</p>}

      {history && history.length === 0 && (
        <p className="mt-6 text-ink-400">No completed blocks yet -- finish a study block to see it here.</p>
      )}

      {history && history.length > 0 && (
        <ul className="mt-6 divide-y divide-ink-700 rounded-xl border border-ink-700">
          {history.map((h) => (
            <li key={h.target_block_id} className="flex items-center justify-between gap-4 px-4 py-4">
              <div>
                <div className="text-ink-50">
                  {new Date(h.planned_start).toLocaleDateString(undefined, {
                    weekday: "short",
                    month: "short",
                    day: "numeric",
                  })}
                </div>
                <div className="mt-1 text-sm text-ink-400">
                  {h.planned_duration_min} min planned · {Math.round(h.compliance_pct * 100)}% compliance ·{" "}
                  {h.away_event_count} away event(s)
                </div>
              </div>
              <div className="flex items-center gap-3">
                <VerdictBadge verdict={h.verdict} />
                <Link
                  to={`/summary/${h.target_block_id}`}
                  className="text-sm text-signal-amber hover:underline"
                >
                  Report →
                </Link>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
