"""
prompt.py
---------
Section 4 Layer 7 (LLM REASONING LAYER): "Takes ONLY the structured
output of Layer 6 (verdict, numbers, trend) as input -- never raw video,
never raw per-frame data -- and produces natural-language insights and a
well-phrased recommendation in the configured mentor tone
(strict/neutral/encouraging)."

THE INPUT CONTRACT (enforced, not just documented): build_prompt()'s
ONLY parameter carrying student data is a phase_f
recommendation.engine.RecommendationResult -- a frozen dataclass of
plain numbers/enum values (see phase_f/recommendation/engine.py's
docstring). There is no code path in this file that can reach a
StateEvent, a FeatureVector, a video frame, or any other raw signal --
the function signature itself is the enforcement mechanism, not just a
comment promising restraint.

TONE ("mentor persona") is a second, independent parameter -- one of
strict / neutral / encouraging, Section 3.D's "configurable persona,
applied consistently to both live notifications and end-of-block
summaries." Phase G only wires this into end-of-block/recommendation
phrasing; live in-session notification text (Phase C's monitor.py) is
unchanged by this phase.
"""

from __future__ import annotations

from typing import Literal

ToneName = Literal["strict", "neutral", "encouraging"]

_TONE_GUIDANCE: dict[ToneName, str] = {
    "strict": (
        "Tone: STRICT MENTOR. Be direct and unsentimental. Do not soften "
        "the numbers or add excessive encouragement. Call out slippage "
        "plainly. Still be respectful -- strict means honest, not unkind."
    ),
    "neutral": (
        "Tone: NEUTRAL COACH. Report the numbers plainly and matter-of-factly. "
        "Avoid both harsh criticism and effusive praise -- state what "
        "happened and what it suggests, without editorializing."
    ),
    "encouraging": (
        "Tone: ENCOURAGING COACH. Lead with what went well, frame areas to "
        "improve constructively, and keep the student motivated to try "
        "the next block. Do not sugar-coat the verdict itself -- an "
        "encouraging tone changes the PHRASING, never the underlying "
        "numbers or verdict."
    ),
}

SYSTEM_PROMPT = """You are the reasoning layer of a self-study monitoring tool called \
AI Study Invigilator. You will be given ONLY a structured summary of one \
completed study block -- verdict, numbers, and a schedule recommendation \
that were computed by deterministic, non-LLM code before you ever saw them.

Your job is ONLY to phrase this into a short, natural-language end-of-block \
report for the student. You must follow these rules exactly:

1. NEVER invent, adjust, round in a misleading way, or contradict any number \
   or verdict given to you. Your job is to phrase, not to judge or recompute.
2. NEVER claim to know the student's internal mental state, emotions, or \
   reasons for their behavior (e.g. do not say "you seemed unmotivated" or \
   "you were clearly tired") -- describe observable numbers and patterns only.
3. If a fatigue-pattern anomaly score is provided, describe it as COMPARATIVE \
   to the student's own baseline ("higher than your usual pattern for this \
   time of session"), never as a factual claim about tiredness.
4. Keep the response to 3-5 short sentences: what happened (the numbers), \
   the verdict, and the schedule recommendation, phrased naturally.
5. Follow the requested tone exactly (see the tone instruction below), but \
   the tone changes PHRASING ONLY, never the facts, verdict, or recommendation.
"""


def build_prompt(result, tone: ToneName, fatigue_note: str | None = None) -> str:
    """result: a phase_f recommendation.engine.RecommendationResult
    (duck-typed here rather than importing the class, so this module has
    no hard dependency on phase_f's package layout -- only on the
    attribute names, which are the documented, stable contract).

    fatigue_note: an OPTIONAL, already-computed, plain-language note
    about Phase E's autoencoder reconstruction error for this session
    (e.g. "reconstruction error was 2.1x this student's own baseline in
    the last 10 minutes") -- computed entirely outside this function,
    never a raw reconstruction-error float or model output. Pass None
    if Phase E integration isn't wired up yet for a given deployment;
    the prompt/report works fine without it.

    Returns the single user-turn prompt string; SYSTEM_PROMPT (above) is
    sent as the system prompt in the same API call -- see client.py.
    """
    if tone not in _TONE_GUIDANCE:
        raise ValueError(f"unknown tone {tone!r}; expected one of {list(_TONE_GUIDANCE)}")

    lines = [
        _TONE_GUIDANCE[tone],
        "",
        "STRUCTURED BLOCK DATA (all numbers computed deterministically, upstream of you):",
        f"  planned_duration_min: {result.planned_duration_min}",
        f"  worked_sec: {result.worked_sec:.0f}",
        f"  distracted_sec: {result.distracted_sec:.0f}",
        f"  away_undeclared_sec: {result.away_undeclared_sec:.0f}",
        f"  declared_break_sec: {result.declared_break_sec:.0f}",
        f"  compliance_pct: {result.compliance_pct:.2f}",
        f"  away_event_count: {result.away_event_count}",
        f"  phone_flag_count: {result.phone_flag_count}",
        f"  second_person_flag_count: {result.second_person_flag_count}",
        f"  block_verdict: {result.block_verdict.value}",
        f"  schedule_recommendation: {result.schedule_recommendation.value}",
    ]
    if fatigue_note:
        lines.append(f"  fatigue_pattern_note (already phrased comparatively, do not restate as fact): {fatigue_note}")

    lines += [
        "",
        "Write the end-of-block report now, following all rules above.",
    ]
    return "\n".join(lines)
