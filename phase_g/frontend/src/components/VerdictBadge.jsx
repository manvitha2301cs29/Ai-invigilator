// components/VerdictBadge.jsx
// -----------------------------
// One shared mapping of verdict -> color/label, used everywhere a
// verdict appears (summary screen, history rows) so the same verdict
// always reads the same way across the whole dashboard.

const VERDICT_META = {
  on_track: { label: "On track", className: "bg-verdict-ontrack/15 text-verdict-ontrack border-verdict-ontrack/30" },
  needs_scheduled_breaks: {
    label: "Needs scheduled breaks",
    className: "bg-verdict-breaks/15 text-verdict-breaks border-verdict-breaks/30",
  },
  needs_shorter_blocks: {
    label: "Needs shorter blocks",
    className: "bg-verdict-shorter/15 text-verdict-shorter border-verdict-shorter/30",
  },
};

export default function VerdictBadge({ verdict }) {
  const meta = VERDICT_META[verdict] || { label: verdict, className: "bg-ink-700 text-ink-200 border-ink-600" };
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium ${meta.className}`}>
      {meta.label}
    </span>
  );
}
