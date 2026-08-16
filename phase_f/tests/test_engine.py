"""
test_engine.py
----------------
Integration test for evaluate(): confirms it correctly wires
verdict.py's per-block result and trend.py's multi-block result into
one RecommendationResult, and that every field on that result is a
plain value (never an ORM object) -- the boundary artifact Phase G's
LLM layer is limited to reading.

Run with:
    pytest tests/test_engine.py -v
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation.engine import evaluate
from recommendation.trend import ScheduleRecommendation
from recommendation.types import BlockRecord
from recommendation.verdict import Verdict

BASE_TIME = datetime(2026, 1, 1, 9, 0, 0)


def _block(days_ago, planned_duration_min=60, compliance_pct=0.9, away_event_count=0):
    return BlockRecord(
        target_block_id=f"block-{days_ago}",
        planned_duration_min=planned_duration_min,
        planned_start=BASE_TIME - timedelta(days=days_ago),
        worked_sec=compliance_pct * planned_duration_min * 60,
        distracted_sec=0.0,
        away_undeclared_sec=0.0,
        declared_break_sec=0.0,
        phone_flag_count=0,
        second_person_flag_count=0,
        away_event_count=away_event_count,
        compliance_pct=compliance_pct,
    )


def test_evaluate_combines_block_verdict_and_schedule_recommendation():
    history = [_block(i, compliance_pct=0.95, away_event_count=0) for i in range(3)]
    latest = history[0]

    result = evaluate(latest, history)

    assert result.block_verdict == Verdict.ON_TRACK
    assert result.schedule_recommendation == ScheduleRecommendation.EXTEND_BLOCK
    assert result.target_block_id == "block-0"
    assert result.compliance_pct == 0.95


def test_evaluate_single_block_history_gives_insufficient_history_trend():
    latest = _block(0, compliance_pct=1.0, away_event_count=0)
    result = evaluate(latest, [latest])
    assert result.block_verdict == Verdict.ON_TRACK
    assert result.schedule_recommendation == ScheduleRecommendation.INSUFFICIENT_HISTORY


def test_evaluate_result_fields_are_plain_values_not_orm_objects():
    """Guards the determinism boundary: nothing on RecommendationResult
    should be an object Phase G could pull extra, uncomputed information
    out of -- every field must be a str/float/int/enum."""
    latest = _block(0)
    result = evaluate(latest, [latest])
    for field_name, value in result.__dict__.items():
        assert isinstance(value, (str, float, int)), f"{field_name} is not a plain value: {type(value)}"
