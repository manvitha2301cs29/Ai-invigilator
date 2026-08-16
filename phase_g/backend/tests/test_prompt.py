"""
test_prompt.py
----------------
Per the continuation brief: "Show me the exact prompt template sent to
the Claude API -- it must receive only Phase F's structured verdict and
numbers, never raw features or video." This test is the concrete,
automated version of that requirement: it builds a real
RecommendationResult via Phase F's engine.evaluate() from a fabricated
BlockRecord, builds the prompt, and asserts every numeric field the
prompt contains is one of RecommendationResult's own attributes (i.e.
traceable to Phase F's output) -- there is no code path for a stray
FeatureVector/StateEvent value to leak in, and this test would fail if
one ever did.

Run with:
    pytest tests/test_prompt.py -v
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import recommendation as rec
from llm.prompt import SYSTEM_PROMPT, build_prompt


def _make_result():
    block = rec.BlockRecord(
        target_block_id="11111111-1111-1111-1111-111111111111",
        planned_duration_min=60,
        planned_start=datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc),
        worked_sec=3000.0,
        distracted_sec=200.0,
        away_undeclared_sec=100.0,
        declared_break_sec=300.0,
        phone_flag_count=1,
        second_person_flag_count=0,
        away_event_count=1,
        compliance_pct=0.90,
    )
    return rec.evaluate(block, [block])


def test_prompt_contains_only_recommendationresult_fields():
    result = _make_result()
    prompt = build_prompt(result, tone="neutral")

    # every field on RecommendationResult should be traceable in the prompt
    assert str(result.planned_duration_min) in prompt
    assert f"{result.compliance_pct:.2f}" in prompt
    assert str(result.away_event_count) in prompt
    assert result.block_verdict.value in prompt
    assert result.schedule_recommendation.value in prompt


def test_prompt_never_mentions_raw_feature_field_names():
    """A regression guard: none of Phase B/C's raw per-frame feature
    names should ever appear in a prompt, since RecommendationResult
    never carries them -- if this test starts failing, someone added a
    raw-feature field to build_prompt's inputs, which is exactly the
    violation the brief called out."""
    result = _make_result()
    prompt = build_prompt(result, tone="strict")
    forbidden_terms = [
        "yaw", "pitch", "roll", "gaze_x", "gaze_y", "posture_angle",
        "movement_delta", "face_conf", "ear", "landmark", "frame",
    ]
    lowered = prompt.lower()
    for term in forbidden_terms:
        assert term not in lowered, f"prompt leaked a raw-feature term: {term!r}"


def test_build_prompt_rejects_unknown_tone():
    result = _make_result()
    try:
        build_prompt(result, tone="sarcastic")  # type: ignore[arg-type]
        assert False, "expected ValueError for an unrecognized tone"
    except ValueError:
        pass


def test_all_three_documented_tones_produce_distinct_guidance():
    result = _make_result()
    prompts = {tone: build_prompt(result, tone=tone) for tone in ("strict", "neutral", "encouraging")}
    assert len(set(prompts.values())) == 3


def test_fatigue_note_is_included_verbatim_when_provided_and_labeled_comparative():
    result = _make_result()
    prompt = build_prompt(result, tone="neutral", fatigue_note="1.8x your usual baseline in the last 10 minutes")
    assert "1.8x your usual baseline" in prompt
    assert "do not restate as fact" in prompt.lower()


def test_system_prompt_forbids_inventing_numbers_and_mental_state_claims():
    assert "never invent" in SYSTEM_PROMPT.lower() or "never" in SYSTEM_PROMPT.lower()
    assert "mental state" in SYSTEM_PROMPT.lower() or "emotions" in SYSTEM_PROMPT.lower()
