"""
test_queries.py
-----------------
Proves queries.py's SQL matches phase_c/backend/models.py's REAL schema
-- imported directly from that file (not a redefinition/mock of it) --
by running it against a throwaway in-memory SQLite database. SQLite
rather than a live Postgres so this test needs no docker-compose, no
network, and no .env, matching every other phase's test discipline;
SQLAlchemy compiles phase_c's postgresql.dialects UUID column type down
to a usable (if non-native) SQLite type automatically, which is what
makes this possible without touching phase_c/backend/models.py at all.

A full integration test against the actual Postgres instance (per
phase_c/backend/docker-compose.yml) is out of scope for this automated
suite -- see phase_f/README.txt's "OPTIONAL: LIVE POSTGRES CHECK"
section for how to run one manually.

Run with:
    pytest tests/test_queries.py -v
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation.queries import fetch_block_record, fetch_recent_block_summaries, write_verdict
from recommendation.queries import Base, TargetBlock, TargetBlockStatus, TargetBlockSummary, Verdict


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _make_user(session):
    from recommendation.queries import _models  # the loaded phase_c_backend_models module

    user = _models.User(
        name="Test Student",
        email=f"{uuid.uuid4()}@example.com",
        hashed_password="not_a_real_hash",
        watcher_token=str(uuid.uuid4()),
    )
    session.add(user)
    session.flush()
    return user


def _make_completed_block(session, user, days_ago, planned_duration_min=60, compliance_pct=0.9, away_event_count=0):
    start = datetime.now(timezone.utc) - timedelta(days=days_ago)
    block = TargetBlock(
        user_id=user.id,
        planned_start=start,
        planned_end=start + timedelta(minutes=planned_duration_min),
        planned_duration_min=planned_duration_min,
        status=TargetBlockStatus.COMPLETED,
        actual_start=start,
        actual_end=start + timedelta(minutes=planned_duration_min),
    )
    session.add(block)
    session.flush()

    summary = TargetBlockSummary(
        target_block_id=block.id,
        worked_sec=planned_duration_min * 60 * compliance_pct,
        distracted_sec=0.0,
        away_undeclared_sec=0.0,
        declared_break_sec=0.0,
        phone_flag_count=0,
        second_person_flag_count=0,
        away_event_count=away_event_count,
        compliance_pct=compliance_pct,
    )
    session.add(summary)
    session.commit()
    return block, summary


def test_fetch_recent_block_summaries_returns_newest_first(session):
    user = _make_user(session)
    _make_completed_block(session, user, days_ago=5, compliance_pct=0.5)
    _make_completed_block(session, user, days_ago=1, compliance_pct=0.9)
    _make_completed_block(session, user, days_ago=3, compliance_pct=0.7)

    records = fetch_recent_block_summaries(session, user.id, limit=10)

    assert len(records) == 3
    assert [r.compliance_pct for r in records] == [0.9, 0.7, 0.5]  # newest (fewest days ago) first


def test_fetch_recent_block_summaries_ignores_other_users(session):
    user_a = _make_user(session)
    user_b = _make_user(session)
    _make_completed_block(session, user_a, days_ago=1)
    _make_completed_block(session, user_b, days_ago=1)

    records = fetch_recent_block_summaries(session, user_a.id, limit=10)
    assert len(records) == 1


def test_fetch_recent_block_summaries_excludes_blocks_without_a_summary(session):
    """A SCHEDULED or ACTIVE block has no TargetBlockSummary row yet
    (Section 8: summaries are "computed once a block ends") -- the JOIN
    should simply exclude it, not error."""
    user = _make_user(session)
    start = datetime.now(timezone.utc)
    unfinished = TargetBlock(
        user_id=user.id,
        planned_start=start,
        planned_end=start + timedelta(minutes=60),
        planned_duration_min=60,
        status=TargetBlockStatus.ACTIVE,
    )
    session.add(unfinished)
    session.commit()

    records = fetch_recent_block_summaries(session, user.id, limit=10)
    assert records == []


def test_fetch_recent_block_summaries_excludes_abandoned_blocks(session):
    user = _make_user(session)
    start = datetime.now(timezone.utc)
    abandoned = TargetBlock(
        user_id=user.id,
        planned_start=start,
        planned_end=start + timedelta(minutes=60),
        planned_duration_min=60,
        status=TargetBlockStatus.ABANDONED,
    )
    session.add(abandoned)
    session.flush()
    summary = TargetBlockSummary(target_block_id=abandoned.id, compliance_pct=0.0, away_event_count=0)
    session.add(summary)
    session.commit()

    records = fetch_recent_block_summaries(session, user.id, limit=10)
    assert records == []


def test_fetch_block_record_returns_none_when_no_summary(session):
    user = _make_user(session)
    start = datetime.now(timezone.utc)
    block = TargetBlock(
        user_id=user.id, planned_start=start, planned_end=start + timedelta(minutes=60),
        planned_duration_min=60, status=TargetBlockStatus.ACTIVE,
    )
    session.add(block)
    session.commit()

    assert fetch_block_record(session, block.id) is None


def test_fetch_block_record_maps_all_fields(session):
    user = _make_user(session)
    block, summary = _make_completed_block(session, user, days_ago=0, compliance_pct=0.77, away_event_count=2)

    record = fetch_block_record(session, block.id)

    assert record is not None
    assert record.target_block_id == block.id
    assert record.planned_duration_min == 60
    assert record.compliance_pct == 0.77
    assert record.away_event_count == 2


def test_write_verdict_persists_to_summary(session):
    user = _make_user(session)
    block, summary = _make_completed_block(session, user, days_ago=0)

    write_verdict(session, block.id, Verdict.ON_TRACK)

    session.expire_all()
    refreshed = session.get(TargetBlockSummary, block.id)
    assert refreshed.verdict == Verdict.ON_TRACK


def test_write_verdict_raises_for_unknown_block(session):
    with pytest.raises(ValueError):
        write_verdict(session, uuid.uuid4(), Verdict.ON_TRACK)
