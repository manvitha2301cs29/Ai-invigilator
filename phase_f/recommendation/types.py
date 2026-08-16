"""
types.py
--------
BlockRecord: the plain, ORM-free representation of one completed
TargetBlock's summary that verdict.py/trend.py/engine.py operate on.
Deliberately mirrors phase_c/backend/models.py's TargetBlockSummary +
the parent TargetBlock's planned_duration_min (needed for trend.py's
"same planned_duration" grouping) -- but as a plain dataclass, not an
ORM row, so this whole package's core logic has zero database
dependency and can be unit tested with fabricated instances.

queries.py is the ONLY place that converts a real ORM row into a
BlockRecord; everything downstream of that conversion never touches
SQLAlchemy again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class BlockRecord:
    target_block_id: UUID | str
    planned_duration_min: int
    planned_start: datetime
    worked_sec: float
    distracted_sec: float
    away_undeclared_sec: float
    declared_break_sec: float
    phone_flag_count: int
    second_person_flag_count: int
    away_event_count: int
    compliance_pct: float  # 0.0-1.0, NOT 0-100 -- matches
    # TargetBlockSummary.compliance_pct's own convention (worked_sec /
    # planned_duration, a fraction); verdict.py's thresholds assume this.
