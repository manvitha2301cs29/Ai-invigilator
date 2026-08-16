"""
client.py
---------
Thin wrapper around the Anthropic API for phrasing one already-computed
RecommendationResult (see prompt.py for the full input-contract
explanation). This is the ONLY file in this project that calls the
Claude API, and it's backend-only -- never called from the React
frontend directly, matching the project doc's "LLM: Claude API
(Anthropic), backend-only" tech-stack note.

MODEL CHOICE: defaults to "claude-sonnet-5" (configurable via
ANTHROPIC_MODEL) -- a short, cheap, low-latency phrasing task like this
doesn't need a larger/slower model, and this is a backend call made
once per completed block, not a chat loop.
"""

from __future__ import annotations

import os

from anthropic import Anthropic

from .prompt import SYSTEM_PROMPT, ToneName, build_prompt

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_TOKENS = 400

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set -- copy .env.example to .env and fill in a "
                "real key from https://console.anthropic.com before calling phrase_recommendation()"
            )
        _client = Anthropic(api_key=api_key)
    return _client


def phrase_recommendation(result, tone: ToneName, fatigue_note: str | None = None) -> str:
    """result: a phase_f RecommendationResult. Returns the phrased
    end-of-block report as plain text. Raises RuntimeError if
    ANTHROPIC_API_KEY isn't configured -- callers (routes.py) should
    catch this and fall back to a plain, deterministic templated
    sentence (see routes.py's fallback_report()) rather than failing
    the whole request, since the LLM layer is explicitly a phrasing
    convenience, not something the deterministic numbers depend on."""
    prompt = build_prompt(result, tone, fatigue_note)
    client = _get_client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(text_blocks).strip()
