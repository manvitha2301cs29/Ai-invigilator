// pages/SummaryPage.jsx
// -----------------------
// Screen 3: "a post-block summary screen with the worked/away/break
// breakdown and the LLM-phrased recommendation." Reads
// GET /target-blocks/{id}/recommendation?tone=..., which lazily
// computes Layer 5's summary + Phase F's verdict/trend the first time
// it's called for a given block (backend's routes.py), then phrases it
// via Phase G's LLM layer (or a deterministic fallback sentence).

import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import VerdictBadge from "../components/VerdictBadge";

const TONES = ["neutral", "strict", "encouraging"];

function minutes(sec) {
  return Math.round(sec / 60);
}

export default function SummaryPage() {
  const params = useParams();
  const [blockId, setBlockId] = useState(params.blockId || "");
  const [tone, setTone] = useState("neutral");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function load(targetTone = tone) {
    if (!blockId) return;
    setBusy(true);
    setError("");
    try {
      setResult(await api.getRecommendation(blockId, targetTone));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="font-display text-2xl text-ink-50">End-of-block summary</h1>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <input
          value={blockId}
          onChange={(e) => setBlockId(e.target.value)}
          placeholder="Completed target block ID"
          className="flex-1 min-w-[16rem] rounded-md border border-ink-600 bg-ink-900 px-3 py-2 font-mono text-sm text-ink-50 outline-none focus:border-signal-amber"
        />
        <div className="flex overflow-hidden rounded-md border border-ink-600">
          {TONES.map((t) => (
            <button
              key={t}
              onClick={() => {
                setTone(t);
                load(t);
              }}
              className={`px-3 py-2 text-sm capitalize transition-colors ${
                tone === t ? "bg-signal-amber text-ink-950" : "bg-ink-900 text-ink-200 hover:bg-ink-800"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <button
          onClick={() => load()}
          disabled={busy || !blockId}
          className="rounded-md bg-ink-700 px-4 py-2 text-sm font-medium text-ink-50 hover:bg-ink-600 disabled:opacity-50"
        >
          {busy ? "Loading…" : "Load report"}
        </button>
      </div>

      {error && <p className="mt-4 text-sm text-verdict-shorter">{error}</p>}

      {result && (
        <div className="mt-8 space-y-6">
          <div className="flex items-center gap-3">
            <VerdictBadge verdict={result.block_verdict} />
            <span className="text-sm text-ink-400">
              {Math.round(result.compliance_pct * 100)}% compliance · {result.planned_duration_min} min planned
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Worked" value={`${minutes(result.worked_sec)} min`} />
            <Stat label="Distracted" value={`${minutes(result.distracted_sec)} min`} />
            <Stat label="Away" value={`${minutes(result.away_undeclared_sec)} min`} />
            <Stat label="Breaks" value={`${minutes(result.declared_break_sec)} min`} />
          </div>

          <div className="flex gap-3 text-sm text-ink-400">
            <span>{result.away_event_count} away event(s)</span>
            <span>·</span>
            <span>{result.phone_flag_count} phone flag(s)</span>
            <span>·</span>
            <span>{result.second_person_flag_count} second-person flag(s)</span>
          </div>

          <div className="rounded-xl border border-ink-700 bg-ink-900/50 p-6">
            <div className="text-sm text-ink-400">Report ({result.tone} tone)</div>
            <p className="mt-2 whitespace-pre-line text-ink-50">{result.narrative}</p>
          </div>

          <div className="rounded-lg border border-ink-700 px-4 py-3 text-sm text-ink-200">
            Suggested next schedule:{" "}
            <span className="font-medium text-ink-50">
              {result.schedule_recommendation.replaceAll("_", " ")}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-900/50 px-4 py-3">
      <div className="text-sm text-ink-400">{label}</div>
      <div className="font-display text-xl text-ink-50">{value}</div>
    </div>
  );
}
