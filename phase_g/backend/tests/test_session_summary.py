"""
test_session_summary.py
--------------------------
Unit tests for compute_block_summary_fields() (session_summary.py's
Layer 5 analytics), using an in-memory SQLite database and phase_c's
REAL ORM classes -- no live Postgres needed (SQLAlchemy compiles
phase_c/backend/models.py's Postgres-specific UUID column type down to
a usable SQLite type automatically, the same trick phase_f's
test_queries.py relies on).

Run with:
    pytest tests/test_session_summary.py -v
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import phase_c_models as models
from session_summary import compute_block_summary_fields, upsert_block_summary

T0 = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_user(db):
    user = models.User(
        name="Test Student", email=f"{uuid.uuid4()}@example.com",
        hashed_password="x", watcher_token=uuid.uuid4().hex,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_completed_block(db, user, planned_duration_min=60, actual_end_offset_min=60):
    block = models.TargetBlock(
        user_id=user.id,
        planned_start=T0,
        planned_end=T0 + timedelta(minutes=planned_duration_min),
        planned_duration_min=planned_duration_min,
        status=models.TargetBlockStatus.COMPLETED,
        actual_start=T0,
        actual_end=T0 + timedelta(minutes=actual_end_offset_min),
    )
    db.add(block)
    db.commit()
    db.refresh(block)
    return block


def test_worked_time_sums_engaged_and_idle_present_not_distracted(db_session):
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=10, actual_end_offset_min=10)

    events = [
        (0, models.BehaviorState.ENGAGED),
        (3, models.BehaviorState.IDLE_PRESENT),
        (5, models.BehaviorState.DISTRACTED_PRESENT),
        (8, models.BehaviorState.ENGAGED),
    ]
    for minute, state in events:
        db_session.add(models.StateEvent(target_block_id=block.id, timestamp=T0 + timedelta(minutes=minute), state=state))
    db_session.commit()
    db_session.refresh(block)

    fields = compute_block_summary_fields(block)
    # engaged 0-3 (3min) + idle 3-5 (2min) + engaged 8-10 (2min) = 7min worked
    assert fields["worked_sec"] == pytest.approx(7 * 60)
    # distracted 5-8 (3min)
    assert fields["distracted_sec"] == pytest.approx(3 * 60)


def test_away_state_events_do_not_double_count_with_away_events(db_session):
    """AWAY-state StateEvents must be excluded from worked/distracted
    accumulation -- away time comes ONLY from the AwayEvent table."""
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=10, actual_end_offset_min=10)

    db_session.add(models.StateEvent(target_block_id=block.id, timestamp=T0, state=models.BehaviorState.ENGAGED))
    db_session.add(
        models.StateEvent(target_block_id=block.id, timestamp=T0 + timedelta(minutes=5), state=models.BehaviorState.AWAY)
    )
    db_session.add(
        models.AwayEvent(
            target_block_id=block.id, away_start=T0 + timedelta(minutes=5),
            away_end=T0 + timedelta(minutes=8), duration_sec=180.0,
        )
    )
    db_session.commit()
    db_session.refresh(block)

    fields = compute_block_summary_fields(block)
    assert fields["away_undeclared_sec"] == pytest.approx(180.0)
    assert fields["away_event_count"] == 1
    # engaged 0-5 = 5 min worked; the AWAY-state stretch itself contributes 0
    # to worked_sec/distracted_sec (not double-counted against away_undeclared_sec)
    assert fields["worked_sec"] == pytest.approx(5 * 60)
    assert fields["distracted_sec"] == 0.0


def test_declared_break_seconds_only_counts_closed_breaks(db_session):
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=20, actual_end_offset_min=20)

    db_session.add(
        models.BreakEvent(
            target_block_id=block.id, break_start=T0 + timedelta(minutes=2),
            break_end=T0 + timedelta(minutes=7),
        )
    )
    db_session.add(
        models.BreakEvent(target_block_id=block.id, break_start=T0 + timedelta(minutes=15), break_end=None)
    )  # never resumed/closed -- should not count
    db_session.commit()
    db_session.refresh(block)

    fields = compute_block_summary_fields(block)
    assert fields["declared_break_sec"] == pytest.approx(5 * 60)


def test_compliance_pct_is_worked_over_planned_and_capped_at_one(db_session):
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=10, actual_end_offset_min=15)
    # entirely engaged for the full (overrun) 15 minutes
    db_session.add(models.StateEvent(target_block_id=block.id, timestamp=T0, state=models.BehaviorState.ENGAGED))
    db_session.commit()
    db_session.refresh(block)

    fields = compute_block_summary_fields(block)
    assert fields["compliance_pct"] == 1.0  # capped, not >1.0 despite 15 min worked / 10 min planned


def test_compute_block_summary_fields_rejects_non_completed_block(db_session):
    user = _make_user(db_session)
    block = models.TargetBlock(
        user_id=user.id, planned_start=T0, planned_end=T0 + timedelta(minutes=30),
        planned_duration_min=30, status=models.TargetBlockStatus.ACTIVE,
    )
    db_session.add(block)
    db_session.commit()
    db_session.refresh(block)

    with pytest.raises(ValueError):
        compute_block_summary_fields(block)


def test_upsert_block_summary_writes_a_verdict_matching_phase_f_thresholds(db_session):
    """compliance_pct=0.9 (>=0.85) and away_event_count=0 (<=1) ->
    on_track, per phase_f/recommendation/verdict.py's exact thresholds."""
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=10, actual_end_offset_min=10)
    db_session.add(models.StateEvent(target_block_id=block.id, timestamp=T0, state=models.BehaviorState.ENGAGED))
    db_session.commit()
    db_session.refresh(block)

    summary = upsert_block_summary(db_session, block)
    assert summary.verdict == models.Verdict.ON_TRACK


def test_upsert_block_summary_is_idempotent(db_session):
    user = _make_user(db_session)
    block = _make_completed_block(db_session, user, planned_duration_min=10, actual_end_offset_min=10)
    db_session.add(models.StateEvent(target_block_id=block.id, timestamp=T0, state=models.BehaviorState.ENGAGED))
    db_session.commit()
    db_session.refresh(block)

    first = upsert_block_summary(db_session, block)
    second = upsert_block_summary(db_session, block)
    assert first.target_block_id == second.target_block_id
    # only one summary row should exist for this block
    count = db_session.query(models.TargetBlockSummary).filter_by(target_block_id=block.id).count()
    assert count == 1
