"""
session_summary.py
--------------------
Section 4 Layer 5 (SESSION ANALYTICS): "Smooths per-frame states
(majority vote over a small window to remove jitter), splits into
sessions, computes per-block summaries: worked time, distracted time,
away time, break time, compliance %."

This is the one piece of genuinely new backend logic Phase G adds
(everything else in this folder either extends auth or calls Phase
F/the LLM) -- Phase C's watcher/backend round-trip (Phase C's own
README) stops at "events land in Postgres correctly"; nothing before
this phase actually turns a block's raw StateEvent/AwayEvent/
BreakEvent/PhoneEvent/SecondPersonEvent rows into the
TargetBlockSummary row Phase F's recommendation engine reads.

WORKED / DISTRACTED TIME, FROM StateEvent
  StateEvent.state is one of ENGAGED / IDLE_PRESENT / DISTRACTED_PRESENT
  / AWAY (phase_c/backend/models.py's BehaviorState enum). Per-state
  duration is estimated as the gap to the NEXT state_event (or to the
  block's actual_end for the last event) -- the standard way to turn a
  sparse "state changed at time T" event stream into durations.
  AWAY-state StateEvents are deliberately EXCLUDED from this
  duration-accumulation pass: away time is tracked authoritatively by
  the separate AwayEvent stopwatch table (Section 4 Layer 4), and
  summing both would double-count the same wall-clock time.

DECLARED BREAK TIME, FROM BreakEvent
  Sum of (break_end - break_start) for every break with a break_end
  set (an in-progress break with no break_end yet is not counted --
  matches this being called only on a block that has already ended).

COMPLIANCE, PER THE PROJECT DOC
  compliance_pct = worked_sec / planned_duration_sec (Section 4's
  "TargetBlockSummary (computed once a block ends)"; Section 4 Layer 5
  lists compliance % as a direct summary field). Clamped to [0, 1] --
  a block that ran slightly over its planned window (actual_end past
  planned_end) should not report compliance above 100%.
"""

from __future__ import annotations

from datetime import datetime

from bridge import phase_c_models


def _duration_seconds(a: datetime, b: datetime) -> float:
    return max(0.0, (b - a).total_seconds())


def compute_block_summary_fields(block) -> dict:
    """block: a phase_c_models.TargetBlock ORM instance, already loaded
    (relationships accessed) with status COMPLETED and actual_end set.
    Returns a plain dict of the TargetBlockSummary columns EXCEPT
    verdict (Phase F's job, called separately -- see routes.py) and
    target_block_id (the caller already has it)."""
    if block.status != phase_c_models.TargetBlockStatus.COMPLETED or block.actual_end is None:
        raise ValueError("compute_block_summary_fields requires a COMPLETED block with actual_end set")

    end_of_block = block.actual_end

    state_events = sorted(block.state_events, key=lambda e: e.timestamp)
    worked_sec = 0.0
    distracted_sec = 0.0
    for i, event in enumerate(state_events):
        next_timestamp = state_events[i + 1].timestamp if i + 1 < len(state_events) else end_of_block
        duration = _duration_seconds(event.timestamp, next_timestamp)
        if event.state == phase_c_models.BehaviorState.ENGAGED:
            worked_sec += duration
        elif event.state == phase_c_models.BehaviorState.IDLE_PRESENT:
            worked_sec += duration  # Section 5: idle-present counts as working, not distraction
        elif event.state == phase_c_models.BehaviorState.DISTRACTED_PRESENT:
            distracted_sec += duration
        # AWAY: intentionally skipped -- see module docstring.

    away_undeclared_sec = sum(e.duration_sec for e in block.away_events)
    declared_break_sec = sum(
        _duration_seconds(e.break_start, e.break_end) for e in block.break_events if e.break_end is not None
    )

    planned_duration_sec = block.planned_duration_min * 60
    compliance_pct = 0.0 if planned_duration_sec <= 0 else min(1.0, worked_sec / planned_duration_sec)

    return {
        "worked_sec": worked_sec,
        "distracted_sec": distracted_sec,
        "away_undeclared_sec": away_undeclared_sec,
        "declared_break_sec": declared_break_sec,
        "phone_flag_count": len(block.phone_events),
        "second_person_flag_count": len(block.second_person_events),
        "away_event_count": len(block.away_events),
        "compliance_pct": compliance_pct,
    }


def upsert_block_summary(db, block) -> "phase_c_models.TargetBlockSummary":
    """Computes fields via compute_block_summary_fields, applies Phase
    F's per-block verdict (compute_single_block_verdict), and
    writes/updates the TargetBlockSummary row -- idempotent, so calling
    this more than once for the same block (e.g. a retried request)
    safely overwrites rather than duplicating."""
    from bridge import recommendation as rec

    fields = compute_block_summary_fields(block)
    phase_f_verdict = rec.compute_single_block_verdict(
        compliance_pct=fields["compliance_pct"], away_event_count=fields["away_event_count"]
    )
    # phase_f's Verdict is a separate (but value-identical) Enum class
    # from phase_c_models.Verdict -- translate via .value, exactly as
    # recommendation/queries.py's write_verdict docstring specifies,
    # since TargetBlockSummary.verdict is typed against the ORM's own
    # Verdict class.
    orm_verdict = phase_c_models.Verdict(phase_f_verdict.value)

    summary = block.summary
    if summary is None:
        summary = phase_c_models.TargetBlockSummary(target_block_id=block.id)
        db.add(summary)

    for key, value in fields.items():
        setattr(summary, key, value)
    summary.verdict = orm_verdict

    db.commit()
    db.refresh(summary)
    return summary
