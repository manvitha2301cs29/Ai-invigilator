"""
test_trend.py
--------------
Per the continuation brief: "the trend logic that requires multiple
recent blocks at the same planned_duration agreeing, not a single
session, before changing the recommendation." Covers: insufficient
history, unanimous-extend, unanimous-shrink, mixed-verdicts fallback,
and the same-planned_duration filtering itself.

Run with:
    pytest tests/test_trend.py -v
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation.trend import (
    TREND_WINDOW_SIZE,
    ScheduleRecommendation,
    compute_schedule_recommendation,
)
from recommendation.types import BlockRecord

BASE_TIME = datetime(2026, 1, 1, 9, 0, 0)


def _block(
    days_ago: int,
    planned_duration_min: int = 60,
    compliance_pct: float = 0.9,
    away_event_count: int = 0,
) -> BlockRecord:
    return BlockRecord(
        target_block_id=f"block-{days_ago}",
        planned_duration_min=planned_duration_min,
        planned_start=BASE_TIME - timedelta(days=days_ago),
        worked_sec=0.0,
        distracted_sec=0.0,
        away_undeclared_sec=0.0,
        declared_break_sec=0.0,
        phone_flag_count=0,
        second_person_flag_count=0,
        away_event_count=away_event_count,
        compliance_pct=compliance_pct,
    )


# ---------------------------------------------------------------------------
# insufficient history -- "not a single session"
# ---------------------------------------------------------------------------

def test_no_history_is_insufficient():
    assert compute_schedule_recommendation([]) == ScheduleRecommendation.INSUFFICIENT_HISTORY


def test_single_excellent_block_does_not_recommend_extending():
    """The core "not a single session" guarantee: one great block alone
    must never trigger EXTEND_BLOCK."""
    blocks = [_block(0, compliance_pct=1.0, away_event_count=0)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.INSUFFICIENT_HISTORY


def test_single_terrible_block_does_not_recommend_shrinking():
    blocks = [_block(0, compliance_pct=0.1, away_event_count=10)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.INSUFFICIENT_HISTORY


def test_one_fewer_than_window_size_is_still_insufficient():
    blocks = [_block(i, compliance_pct=1.0, away_event_count=0) for i in range(TREND_WINDOW_SIZE - 1)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.INSUFFICIENT_HISTORY


# ---------------------------------------------------------------------------
# unanimous agreement required
# ---------------------------------------------------------------------------

def test_unanimous_on_track_recommends_extend():
    blocks = [_block(i, compliance_pct=0.95, away_event_count=0) for i in range(TREND_WINDOW_SIZE)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.EXTEND_BLOCK


def test_unanimous_needs_shorter_blocks_recommends_shrink():
    blocks = [_block(i, compliance_pct=0.3, away_event_count=5) for i in range(TREND_WINDOW_SIZE)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.SHRINK_BLOCK_ADD_MANDATORY_BREAK


def test_one_disagreeing_block_prevents_extend():
    """window_size on_track blocks except the most recent one, which is
    merely middling -- must NOT recommend extending."""
    blocks = [_block(i, compliance_pct=0.95, away_event_count=0) for i in range(1, TREND_WINDOW_SIZE)]
    blocks.append(_block(0, compliance_pct=0.70, away_event_count=1))  # most recent, middling
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.ADD_SCHEDULED_BREAK_KEEP_DURATION


def test_one_disagreeing_block_prevents_shrink():
    blocks = [_block(i, compliance_pct=0.3, away_event_count=5) for i in range(1, TREND_WINDOW_SIZE)]
    blocks.append(_block(0, compliance_pct=0.9, away_event_count=0))  # most recent, great
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.ADD_SCHEDULED_BREAK_KEEP_DURATION


def test_mixed_or_improving_trend_recommends_add_break():
    blocks = [
        _block(2, compliance_pct=0.3, away_event_count=5),
        _block(1, compliance_pct=0.6, away_event_count=2),
        _block(0, compliance_pct=0.9, away_event_count=0),
    ]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.ADD_SCHEDULED_BREAK_KEEP_DURATION


# ---------------------------------------------------------------------------
# same-planned_duration filtering
# ---------------------------------------------------------------------------

def test_blocks_at_a_different_duration_are_ignored():
    """window_size on_track 90-minute blocks should not count toward a
    60-minute trend evaluation."""
    ninety_min_blocks = [
        _block(i, planned_duration_min=90, compliance_pct=0.95, away_event_count=0) for i in range(TREND_WINDOW_SIZE)
    ]
    sixty_min_blocks = [_block(i + 10, planned_duration_min=60, compliance_pct=0.95) for i in range(2)]
    all_blocks = ninety_min_blocks + sixty_min_blocks
    assert (
        compute_schedule_recommendation(all_blocks, reference_planned_duration_min=60)
        == ScheduleRecommendation.INSUFFICIENT_HISTORY
    )
    assert (
        compute_schedule_recommendation(all_blocks, reference_planned_duration_min=90)
        == ScheduleRecommendation.EXTEND_BLOCK
    )


def test_reference_duration_defaults_to_most_recent_blocks_duration():
    blocks = [_block(i, planned_duration_min=45, compliance_pct=0.95, away_event_count=0) for i in range(TREND_WINDOW_SIZE)]
    assert compute_schedule_recommendation(blocks) == ScheduleRecommendation.EXTEND_BLOCK


def test_only_the_most_recent_window_size_blocks_are_considered():
    """An old bad streak followed by a recent good streak (at least
    window_size long) should recommend extending -- old history outside
    the window shouldn't drag the verdict down forever."""
    old_bad = [_block(i + 100, compliance_pct=0.1, away_event_count=10) for i in range(5)]
    recent_good = [_block(i, compliance_pct=0.95, away_event_count=0) for i in range(TREND_WINDOW_SIZE)]
    assert compute_schedule_recommendation(old_bad + recent_good) == ScheduleRecommendation.EXTEND_BLOCK
