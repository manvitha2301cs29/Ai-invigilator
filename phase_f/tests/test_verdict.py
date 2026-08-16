"""
test_verdict.py
-----------------
Exact-threshold unit tests for compute_single_block_verdict, per the
continuation brief's explicit requirement: "compliance >= 0.85 and
away_event_count <= 1 -> on_track; compliance < 0.55 or
away_event_count >= 4 -> needs_shorter_blocks; otherwise ->
needs_scheduled_breaks." Every boundary value is tested on both sides.

Run with:
    pytest tests/test_verdict.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation.verdict import Verdict, compute_single_block_verdict


# ---------------------------------------------------------------------------
# on_track
# ---------------------------------------------------------------------------

def test_on_track_at_exact_boundary():
    assert compute_single_block_verdict(0.85, 1) == Verdict.ON_TRACK


def test_on_track_comfortably_above_boundary():
    assert compute_single_block_verdict(0.95, 0) == Verdict.ON_TRACK


def test_not_on_track_just_below_compliance_boundary():
    assert compute_single_block_verdict(0.849999, 0) != Verdict.ON_TRACK


def test_not_on_track_just_above_away_event_boundary():
    assert compute_single_block_verdict(0.90, 2) != Verdict.ON_TRACK


# ---------------------------------------------------------------------------
# needs_shorter_blocks
# ---------------------------------------------------------------------------

def test_needs_shorter_blocks_from_low_compliance_alone():
    assert compute_single_block_verdict(0.54, 0) == Verdict.NEEDS_SHORTER_BLOCKS


def test_needs_shorter_blocks_compliance_boundary_is_exclusive():
    """compliance_pct < 0.55 -- exactly 0.55 should NOT trigger this
    branch on compliance alone."""
    assert compute_single_block_verdict(0.55, 0) != Verdict.NEEDS_SHORTER_BLOCKS


def test_needs_shorter_blocks_from_away_events_alone():
    assert compute_single_block_verdict(0.95, 4) == Verdict.NEEDS_SHORTER_BLOCKS


def test_needs_shorter_blocks_away_event_boundary_is_inclusive():
    assert compute_single_block_verdict(0.99, 4) == Verdict.NEEDS_SHORTER_BLOCKS
    assert compute_single_block_verdict(0.99, 3) != Verdict.NEEDS_SHORTER_BLOCKS


def test_needs_shorter_blocks_overrides_high_compliance_if_away_events_high():
    """Even excellent compliance should still shrink the block if away
    events are frequent -- the two conditions are OR'd, not AND'd."""
    assert compute_single_block_verdict(0.99, 5) == Verdict.NEEDS_SHORTER_BLOCKS


# ---------------------------------------------------------------------------
# needs_scheduled_breaks (the "everything else" middle ground)
# ---------------------------------------------------------------------------

def test_needs_scheduled_breaks_for_middling_compliance_and_low_away_events():
    assert compute_single_block_verdict(0.70, 1) == Verdict.NEEDS_SCHEDULED_BREAKS


def test_needs_scheduled_breaks_for_high_compliance_but_moderate_away_events():
    """0.90 compliance would satisfy on_track's compliance half, but 2
    away events fails on_track's away_event_count <= 1, and 2 is below
    the needs_shorter_blocks threshold of 4 -- middle ground."""
    assert compute_single_block_verdict(0.90, 2) == Verdict.NEEDS_SCHEDULED_BREAKS


def test_needs_scheduled_breaks_at_compliance_exactly_055():
    assert compute_single_block_verdict(0.55, 1) == Verdict.NEEDS_SCHEDULED_BREAKS


# ---------------------------------------------------------------------------
# precedence: on_track is checked before needs_shorter_blocks
# ---------------------------------------------------------------------------

def test_on_track_condition_checked_before_shorter_blocks_condition():
    """A block satisfying on_track's condition should never fall through
    to needs_shorter_blocks even though the two conditions are checked
    in sequence, not mutually exclusive by construction."""
    assert compute_single_block_verdict(1.0, 0) == Verdict.ON_TRACK
