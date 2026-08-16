// pages/PlannerPage.jsx
// ----------------------
// Screen 1 of the continuation brief's priority list: "a target-block
// planner screen." Creates a SCHEDULED TargetBlock via
// POST /dashboard/target-blocks (JWT), and lists upcoming/active blocks
// so the student can see their plan for the day.

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";

function defaultTimes() {
  const start = new Date();
  start.setMinutes(start.getMinutes() + 5, 0, 0);
  const end = new Date(start.getTime() + 60 * 60 * 1000);
  return { start, end };
}

function toLocalInputValue(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(
    date.getMinutes()
  )}`;
}

export default function PlannerPage() {
  const { start, end } = defaultTimes();
  const [plannedStart, setPlannedStart] = useState(toLocalInputValue(start));
  const [plannedEnd, setPlannedEnd] = useState(toLocalInputValue(end));
  const [durationMin, setDurationMin] = useState(60);
  const [blocks, setBlocks] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      setBlocks(await api.listUpcomingBlocks());
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.createBlock(
        new Date(plannedStart).toISOString(),
        new Date(plannedEnd).toISOString(),
        Number(durationMin)
      );
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="font-display text-2xl text-ink-50">Plan a study block</h1>
      <p className="mt-1 text-ink-400">
        Set a target block, then run your local watcher during it. Compliance and warnings are tracked
        automatically.
      </p>

      <form onSubmit={handleSubmit} className="mt-8 grid grid-cols-1 gap-4 rounded-xl border border-ink-700 bg-ink-900/50 p-6 sm:grid-cols-3">
        <label className="block">
          <span className="mb-1 block text-sm text-ink-200">Start</span>
          <input
            type="datetime-local"
            value={plannedStart}
            onChange={(e) => setPlannedStart(e.target.value)}
            required
            className="w-full rounded-md border border-ink-600 bg-ink-950 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-sm text-ink-200">End</span>
          <input
            type="datetime-local"
            value={plannedEnd}
            onChange={(e) => setPlannedEnd(e.target.value)}
            required
            className="w-full rounded-md border border-ink-600 bg-ink-950 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-sm text-ink-200">Duration (min)</span>
          <input
            type="number"
            min="5"
            value={durationMin}
            onChange={(e) => setDurationMin(e.target.value)}
            required
            className="w-full rounded-md border border-ink-600 bg-ink-950 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
          />
        </label>

        {error && <p className="sm:col-span-3 text-sm text-verdict-shorter">{error}</p>}

        <button
          type="submit"
          disabled={busy}
          className="sm:col-span-3 justify-self-start rounded-md bg-signal-amber px-4 py-2 font-medium text-ink-950 transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {busy ? "Planning…" : "Add block"}
        </button>
      </form>

      <h2 className="mt-10 font-display text-lg text-ink-50">Upcoming blocks</h2>
      {blocks.length === 0 ? (
        <p className="mt-3 text-ink-400">Nothing planned yet -- add a block above.</p>
      ) : (
        <ul className="mt-3 divide-y divide-ink-700 rounded-xl border border-ink-700">
          {blocks.map((b) => (
            <li key={b.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <div className="text-ink-50">
                  {new Date(b.planned_start).toLocaleString(undefined, {
                    weekday: "short",
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </div>
                <div className="text-sm text-ink-400">{b.planned_duration_min} min · {b.status}</div>
              </div>
              <Link to="/live" className="text-sm text-signal-amber hover:underline">
                Go live →
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
