"""
daisee_label_mapping.py
------------------------
The explicit, documented, citable DAiSEE -> project-taxonomy label
mapping required by docs/02_TRAINING_GUIDE.txt Step 1 ("Write this
mapping into a documented function ... so it's citable and defensible
in your report").

DAiSEE labels each clip on FOUR independent 4-level (0-3) ordinal
scales: Engagement, Boredom, Confusion, Frustration. This project needs
ONE label from {engaged, idle_present, distracted_present} (Away is
excluded -- see phase_d/data/types.py's module docstring). The mapping
below is a single, deterministic function of the four DAiSEE scores.

MAPPING RULE (documented per training-guide instructions):
  1. engagement >= 2 AND boredom <= 1 AND confusion <= 1 AND frustration <= 1
       -> "engaged"
     Rationale: high engagement with low negative-affect scores is the
     closest DAiSEE analogue to this project's "actually working" state.

  2. engagement <= 1 AND boredom <= 1
       -> "idle_present"
     Rationale (WEAKEST MAPPING, stated explicitly per the training
     guide): DAiSEE has no "present but idle" class -- a bored-but-not-
     disengaged-looking learner and a genuinely idle one look similar in
     DAiSEE's labeling scheme, since DAiSEE was collected for lecture-
     video engagement, not self-study idleness. Any classifier's
     accuracy specifically on this class should be reported with this
     caveat, not treated as equally trustworthy as the other two.

  3. (engagement <= 1) AND (boredom >= 2 OR confusion >= 2 OR frustration >= 2)
       -> "distracted_present"
     Rationale: low engagement plus a high negative-affect score is the
     best available proxy for a learner who is present but attending to
     something other than the material (visibly disengaged, not just
     quietly idle).

  4. Anything else (e.g. moderate engagement with moderate confusion) is
     UNMAPPABLE and returns None -- callers must drop these clips rather
     than force them into a bucket, per the "state this limitation
     clearly" instruction. This is intentional: silently coercing
     ambiguous clips into a class would quietly bias the dataset in a
     way that's invisible in the final accuracy number.

No "away" mapping exists by design -- DAiSEE assumes a face is always
in frame (per the training guide), so every DAiSEE clip maps to one of
the three classes above or is dropped as unmappable; it never maps to
"away".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DaiseeLabels:
    """The four raw DAiSEE ordinal scores for one clip, each 0-3."""

    engagement: int
    boredom: int
    confusion: int
    frustration: int

    def __post_init__(self) -> None:
        for name, value in (
            ("engagement", self.engagement),
            ("boredom", self.boredom),
            ("confusion", self.confusion),
            ("frustration", self.frustration),
        ):
            if not 0 <= value <= 3:
                raise ValueError(f"DAiSEE {name} score out of range 0-3: {value}")


def map_daisee_labels(labels: DaiseeLabels) -> str | None:
    """Apply the documented mapping rule above. Returns one of
    "engaged" / "idle_present" / "distracted_present", or None if the
    clip is unmappable and should be dropped by the loader."""
    e, b, c, f = labels.engagement, labels.boredom, labels.confusion, labels.frustration

    if e >= 2 and b <= 1 and c <= 1 and f <= 1:
        return "engaged"

    if e <= 1 and b <= 1 and not (b >= 2 or c >= 2 or f >= 2):
        return "idle_present"

    if e <= 1 and (b >= 2 or c >= 2 or f >= 2):
        return "distracted_present"

    return None


def mapping_summary() -> str:
    """Human-readable summary for inclusion in training run logs /
    reports, so every run's output directory can carry a copy of exactly
    which rule version produced its labels."""
    return (
        "DAiSEE->project label mapping v1: "
        "engaged := engagement>=2 & boredom<=1 & confusion<=1 & frustration<=1; "
        "idle_present := engagement<=1 & boredom<=1 & confusion<=1 & frustration<=1 "
        "(weak mapping, no direct DAiSEE analogue); "
        "distracted_present := engagement<=1 & (boredom>=2 | confusion>=2 | frustration>=2); "
        "else -> dropped as unmappable. No 'away' class (DAiSEE assumes face always present)."
    )
