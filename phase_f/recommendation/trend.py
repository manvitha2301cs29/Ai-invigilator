"""
trend.py
--------
Section 3.F ("Pattern learning & schedule recommendations") and Section
4 Layer 6: "computes trend ... and a suggested next schedule," looking
"at compliance trends across the student's OWN recent history at a
given block length (not generic population averages)":

    Consistently high compliance         -> suggest extending the block
    Consistently low compliance /
      frequent away-events               -> suggest a shorter block
                                             with a mandatory scheduled
                                             break
    Mixed / improving trend              -> suggest adding one
                                             scheduled break at the
                                             midpoint, keep duration
                                             the same

REQUIRES AGREEMENT ACROSS MULTIPLE RECENT BLOCKS, NOT A SINGLE SESSION:
"Consistently" and "trend" both imply more than one data point -- a
single unusually bad (or unusually good) session must NOT flip the
schedule recommendation on its own. TREND_WINDOW_SIZE recent blocks AT
THE SAME planned_duration_min must unanimously agree on a per-block
verdict (verdict.py) before this module suggests extending or
shrinking; anything short of unanimous agreement (including simply not
having enough history yet) falls through to a more conservative
outcome (ADD_SCHEDULED_BREAK or INSUFFICIENT_HISTORY respectively) --
see compute_schedule_recommendation's docstring for the exact decision
table.

Grouping by planned_duration_min specifically (not just "recent
blocks") is deliberate: a student's compliance at a 25-minute block and
at a 90-minute block are not comparable data points for "should we
extend/shrink THIS block length" -- Section 3.F's phrase "at a given
block length" is the operative constraint.
"""

from __future__ import annotations

from enum import Enum

from .types import BlockRecord
from .verdict import Verdict, verdict_for_block

TREND_WINDOW_SIZE = 3  # how many same-duration recent blocks must agree


class ScheduleRecommendation(str, Enum):
    EXTEND_BLOCK = "extend_block"
    SHRINK_BLOCK_ADD_MANDATORY_BREAK = "shrink_block_add_mandatory_break"
    ADD_SCHEDULED_BREAK_KEEP_DURATION = "add_scheduled_break_keep_duration"
    INSUFFICIENT_HISTORY = "insufficient_history"


def compute_schedule_recommendation(
    recent_blocks: list[BlockRecord],
    reference_planned_duration_min: int | None = None,
    window_size: int = TREND_WINDOW_SIZE,
) -> ScheduleRecommendation:
    """
    recent_blocks: any set of a student's completed blocks, need not be
    pre-filtered or pre-sorted -- this function does both.
    reference_planned_duration_min: which block length to evaluate the
      trend for. Defaults to the most recent block's own duration (the
      common case: "should we change how the block the student JUST did
      is scheduled"), but callers may pass a specific duration to ask
      "what's the trend at 90-minute blocks specifically."

    DECISION TABLE (checked in this order):
      1. Fewer than `window_size` blocks at reference_planned_duration_min
         exist at all -> INSUFFICIENT_HISTORY. This is the "not a single
         session" guarantee: with 1 or 2 data points, no schedule change
         is ever suggested, regardless of how good or bad those blocks
         were.
      2. The `window_size` MOST RECENT such blocks are ALL on_track
         -> EXTEND_BLOCK ("consistently high compliance").
      3. The `window_size` most recent such blocks are ALL
         needs_shorter_blocks -> SHRINK_BLOCK_ADD_MANDATORY_BREAK
         ("consistently low compliance / frequent away-events").
      4. Anything else (mixed verdicts, or unanimous
         needs_scheduled_breaks) -> ADD_SCHEDULED_BREAK_KEEP_DURATION
         ("mixed / improving trend").
    """
    if reference_planned_duration_min is None:
        if not recent_blocks:
            return ScheduleRecommendation.INSUFFICIENT_HISTORY
        latest = max(recent_blocks, key=lambda b: b.planned_start)
        reference_planned_duration_min = latest.planned_duration_min

    same_duration = [b for b in recent_blocks if b.planned_duration_min == reference_planned_duration_min]
    same_duration.sort(key=lambda b: b.planned_start, reverse=True)

    if len(same_duration) < window_size:
        return ScheduleRecommendation.INSUFFICIENT_HISTORY

    window = same_duration[:window_size]
    verdicts = [verdict_for_block(b) for b in window]

    if all(v == Verdict.ON_TRACK for v in verdicts):
        return ScheduleRecommendation.EXTEND_BLOCK
    if all(v == Verdict.NEEDS_SHORTER_BLOCKS for v in verdicts):
        return ScheduleRecommendation.SHRINK_BLOCK_ADD_MANDATORY_BREAK
    return ScheduleRecommendation.ADD_SCHEDULED_BREAK_KEEP_DURATION
