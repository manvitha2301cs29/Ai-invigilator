"""
queries.py
----------
The only module in this package that touches the database. Reads
phase_c/backend/models.py's REAL ORM classes (TargetBlock,
TargetBlockSummary) directly -- imported via importlib under a distinct
module name ("phase_c_backend_models") rather than a plain sys.path
import, for the same reason phase_e/data/reuse.py does this: this
package's own recommendation/ folder isn't named `models`, but
phase_c/backend has its OWN `database.py` that also imports a bare
`models` -- loading phase_c/backend's modules under their natural names
risks colliding with anything else on sys.path that also defines a
top-level `models` module (phase_d and phase_e both do, for their
PyTorch model classes). Using a unique alias sidesteps that entirely,
consistent with the precedent set in Phase E.

Every function here takes an already-open SQLAlchemy Session as an
argument rather than opening its own connection -- this keeps
connection/credential management (phase_c/backend/database.py,
.env.example) entirely the caller's responsibility, and is what makes
this module unit-testable against a throwaway in-memory SQLite database
in tests/test_queries.py without needing a live Postgres instance.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from .types import BlockRecord

_PHASE_C_BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "phase_c", "backend")
)


def _load_phase_c_backend_models():
    if "phase_c_backend_models" in sys.modules:
        return sys.modules["phase_c_backend_models"]
    spec = importlib.util.spec_from_file_location(
        "phase_c_backend_models", os.path.join(_PHASE_C_BACKEND_DIR, "models.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase_c_backend_models"] = module
    spec.loader.exec_module(module)
    return module


_models = _load_phase_c_backend_models()

Base = _models.Base
TargetBlock = _models.TargetBlock
TargetBlockSummary = _models.TargetBlockSummary
TargetBlockStatus = _models.TargetBlockStatus
Verdict = _models.Verdict


def _row_to_block_record(target_block: TargetBlock, summary: TargetBlockSummary) -> BlockRecord:
    return BlockRecord(
        target_block_id=target_block.id,
        planned_duration_min=target_block.planned_duration_min,
        planned_start=target_block.planned_start,
        worked_sec=summary.worked_sec,
        distracted_sec=summary.distracted_sec,
        away_undeclared_sec=summary.away_undeclared_sec,
        declared_break_sec=summary.declared_break_sec,
        phone_flag_count=summary.phone_flag_count,
        second_person_flag_count=summary.second_person_flag_count,
        away_event_count=summary.away_event_count,
        compliance_pct=summary.compliance_pct,
    )


def fetch_recent_block_summaries(
    session: Session,
    user_id: UUID | str,
    limit: int = 20,
) -> list[BlockRecord]:
    """The student's `limit` most recent COMPLETED blocks that have a
    summary, newest first. Only status == COMPLETED blocks are
    considered -- an ABANDONED block has no meaningful compliance_pct to
    trend on, and a SCHEDULED/ACTIVE block has no summary row at all yet
    (TargetBlockSummary is "computed once a block ends" per Section 8)."""
    stmt = (
        select(TargetBlock, TargetBlockSummary)
        .join(TargetBlockSummary, TargetBlockSummary.target_block_id == TargetBlock.id)
        .where(TargetBlock.user_id == user_id)
        .where(TargetBlock.status == TargetBlockStatus.COMPLETED)
        .order_by(TargetBlock.planned_start.desc())
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    return [_row_to_block_record(target_block, summary) for target_block, summary in rows]


def fetch_block_record(session: Session, target_block_id: UUID | str) -> BlockRecord | None:
    """A single block's BlockRecord, or None if it has no summary yet
    (e.g. still active) -- callers computing an end-of-block report
    should only call this after the block's summary has been written."""
    stmt = (
        select(TargetBlock, TargetBlockSummary)
        .join(TargetBlockSummary, TargetBlockSummary.target_block_id == TargetBlock.id)
        .where(TargetBlock.id == target_block_id)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    target_block, summary = row
    return _row_to_block_record(target_block, summary)


def write_verdict(session: Session, target_block_id: UUID | str, verdict: "Verdict") -> None:
    """Persists a computed verdict back onto TargetBlockSummary.verdict.
    Takes phase_c_backend_models.Verdict (the ORM enum, re-exported
    above as `Verdict`) rather than recommendation.verdict.Verdict --
    callers (engine.py's result) use their own plain-string Verdict
    enum; translate with Verdict(result.block_verdict.value) at the call
    site, keeping this module's only ORM-facing responsibility to
    reading/writing rows, not owning the recommendation logic itself."""
    summary = session.get(TargetBlockSummary, target_block_id)
    if summary is None:
        raise ValueError(f"no TargetBlockSummary found for target_block_id={target_block_id!r}")
    summary.verdict = verdict
    session.commit()
