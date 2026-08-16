"""
verdict.py
----------
Per-block verdict from a single completed TargetBlockSummary, matching
phase_c/backend/models.py's Verdict enum (on_track / needs_shorter_blocks
/ needs_scheduled_breaks) exactly -- this module's Verdict values are
kept as plain strings equal to that enum's .value strings so callers
that write back to TargetBlockSummary.verdict never need a translation
step.

THRESHOLDS (exact, as specified for this phase -- document any future
change to these numbers here, not just in a commit message, since
they're the single most load-bearing constants in the whole
recommendation engine):

  on_track:              compliance_pct >= 0.85 AND away_event_count <= 1
  needs_shorter_blocks:  compliance_pct <  0.55 OR  away_event_count >= 4
  needs_scheduled_breaks: everything else (the "mixed" middle ground --
                          neither clearly on track nor clearly
                          struggling enough to shrink the block outright)

Evaluated in that ORDER: on_track is checked first, then
needs_shorter_blocks, so a block that happens to satisfy neither's exact
condition (e.g. compliance 0.70, away_event_count 2) falls through to
needs_scheduled_breaks by construction, not as a separate explicit rule.

This is Section 4 Layer 6's "rules/statistics engine (no LLM)" at its
most granular: ONE block in, ONE verdict out, no history, no LLM
involvement, fully deterministic and auditable.
"""

from __future__ import annotations

from enum import Enum

from .types import BlockRecord


class Verdict(str, Enum):
    ON_TRACK = "on_track"
    NEEDS_SHORTER_BLOCKS = "needs_shorter_blocks"
    NEEDS_SCHEDULED_BREAKS = "needs_scheduled_breaks"


ON_TRACK_MIN_COMPLIANCE = 0.85
ON_TRACK_MAX_AWAY_EVENTS = 1
SHORTER_BLOCKS_MAX_COMPLIANCE = 0.55
SHORTER_BLOCKS_MIN_AWAY_EVENTS = 4


def compute_single_block_verdict(compliance_pct: float, away_event_count: int) -> Verdict:
    """Pure function: takes exactly the two numbers the thresholds
    depend on (not a full BlockRecord) so it's trivially unit-testable
    against the documented boundary values without constructing a whole
    fake record for every case."""
    if compliance_pct >= ON_TRACK_MIN_COMPLIANCE and away_event_count <= ON_TRACK_MAX_AWAY_EVENTS:
        return Verdict.ON_TRACK
    if compliance_pct < SHORTER_BLOCKS_MAX_COMPLIANCE or away_event_count >= SHORTER_BLOCKS_MIN_AWAY_EVENTS:
        return Verdict.NEEDS_SHORTER_BLOCKS
    return Verdict.NEEDS_SCHEDULED_BREAKS


def verdict_for_block(block: BlockRecord) -> Verdict:
    """Convenience wrapper around compute_single_block_verdict for
    callers that already have a full BlockRecord (e.g. engine.py)."""
    return compute_single_block_verdict(block.compliance_pct, block.away_event_count)
