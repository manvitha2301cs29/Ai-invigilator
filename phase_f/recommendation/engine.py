"""
engine.py
---------
Ties verdict.py (single-block) and trend.py (multi-block) into one
evaluate() call, producing the exact structured output the project's
determinism principle requires Phase G's LLM layer to be limited to:
"DETERMINISTIC LOGIC SEPARATE FROM LLM OUTPUT: numbers come from plain
code; the LLM ... only phrases already-computed numbers, never invents
one." RecommendationResult is that boundary artifact -- every field on
it is a plain number, enum, or string computed by this package, and
Phase G's prompt template (phase_g/) will serialize exactly this object
(never raw features, never a TargetBlockSummary ORM row) into the
Claude API call.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trend import ScheduleRecommendation, compute_schedule_recommendation
from .types import BlockRecord
from .verdict import Verdict, verdict_for_block


@dataclass(frozen=True)
class RecommendationResult:
    target_block_id: str
    block_verdict: Verdict
    schedule_recommendation: ScheduleRecommendation
    compliance_pct: float
    away_event_count: int
    worked_sec: float
    distracted_sec: float
    away_undeclared_sec: float
    declared_break_sec: float
    phone_flag_count: int
    second_person_flag_count: int
    planned_duration_min: int


def evaluate(latest_block: BlockRecord, recent_blocks: list[BlockRecord]) -> RecommendationResult:
    """latest_block: the block whose end-of-block report is being
    produced right now. recent_blocks: the student's broader history
    (should include latest_block itself; compute_schedule_recommendation
    handles filtering/sorting internally) used only for the
    trend/schedule recommendation, never for the single-block verdict,
    which depends solely on latest_block's own numbers."""
    block_verdict = verdict_for_block(latest_block)
    schedule_recommendation = compute_schedule_recommendation(
        recent_blocks, reference_planned_duration_min=latest_block.planned_duration_min
    )

    return RecommendationResult(
        target_block_id=str(latest_block.target_block_id),
        block_verdict=block_verdict,
        schedule_recommendation=schedule_recommendation,
        compliance_pct=latest_block.compliance_pct,
        away_event_count=latest_block.away_event_count,
        worked_sec=latest_block.worked_sec,
        distracted_sec=latest_block.distracted_sec,
        away_undeclared_sec=latest_block.away_undeclared_sec,
        declared_break_sec=latest_block.declared_break_sec,
        phone_flag_count=latest_block.phone_flag_count,
        second_person_flag_count=latest_block.second_person_flag_count,
        planned_duration_min=latest_block.planned_duration_min,
    )
