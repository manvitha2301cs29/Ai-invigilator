"""
recommendation package
------------------------
Layer 6 of the project doc (Section 4): "A rules/statistics engine (no
LLM) over the student's historical TargetBlock summaries: computes
trend, verdict (on_track / needs_shorter_blocks / needs_scheduled_breaks),
and a suggested next schedule. This layer is fully auditable and
testable on its own."

Split into four small, single-purpose modules:
  types.py    -- BlockRecord: a plain dataclass with NO SQLAlchemy
                 dependency. Everything else in this package operates on
                 BlockRecord, never on ORM objects directly, so
                 verdict.py/trend.py/engine.py can be fully unit tested
                 with fabricated data and stay usable even if the
                 storage layer changes later.
  verdict.py  -- per-block verdict from exact, documented thresholds.
  trend.py    -- multi-block schedule recommendation (Section 3.F /
                 Section 4 Layer 6), requiring AGREEMENT across several
                 recent same-duration blocks before suggesting a
                 schedule change -- never a single session.
  queries.py  -- SQLAlchemy queries against phase_c/backend/models.py's
                 REAL schema, mapping ORM rows into BlockRecord. This is
                 the only module in the package that touches the
                 database.
  engine.py   -- ties verdict.py + trend.py together into one
                 evaluate() call, producing the structured output Phase
                 G's LLM layer will phrase (never invent numbers from).
"""

from .engine import RecommendationResult, evaluate
from .trend import ScheduleRecommendation, compute_schedule_recommendation
from .types import BlockRecord
from .verdict import Verdict, compute_single_block_verdict

__all__ = [
    "BlockRecord",
    "Verdict",
    "compute_single_block_verdict",
    "ScheduleRecommendation",
    "compute_schedule_recommendation",
    "RecommendationResult",
    "evaluate",
]
