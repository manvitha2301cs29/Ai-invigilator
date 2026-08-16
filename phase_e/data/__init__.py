"""
data package
------------
REUSE DECISION (per the continuation brief's instruction to state which
and why): this phase IMPORTS Phase D's RawSequence/windowing/loader code
directly from ../phase_d/data via a sys.path insert (see reuse.py),
rather than copying those modules into phase_e/.

WHY IMPORT INSTEAD OF COPY:
  - The windowing logic (window_sequence, windows_to_arrays) and the
    FEATURE_COLUMNS schema are exactly the kind of thing that must never
    drift between phases -- if Phase D's windowing definition changes
    (e.g. a bug fix to the majority-vote tie-break), Phase E should get
    that fix automatically, not silently keep using a stale, forked
    copy.
  - Phase E's autoencoder consumes the SAME Window objects Phase D's
    classifier consumes (same feature schema, same window/stride
    semantics) -- reusing the exact same code is what guarantees that,
    rather than two independently-maintained implementations that could
    quietly diverge (e.g. one windowing function fixing an edge case the
    other doesn't).
  - The two phases are developed together, in the same repository, by
    the same person -- there's no deployment reason (e.g. Phase E
    shipping independently without Phase D installed) to prefer a
    self-contained copy here the way there might be for, say, a
    published library.

The only genuinely NEW code in this phase's data/ folder is
synthetic_fatigue.py (a fatigue-specific synthetic generator Phase D has
no reason to know about) and participants.py (grouping sequences by
participant_id, which Phase D's RawSequence deliberately does not carry
-- see participants.py's docstring for why).
"""

from .reuse import (
    BEHAVIOR_CLASSES,
    FEATURE_COLUMNS,
    NUM_FEATURES,
    FeatureFrame,
    RawSequence,
    Window,
    generate_synthetic_dataset,
    load_self_collected_dataset,
    window_sequence,
    windows_to_arrays,
)
from .windowing import UnlabeledWindow, window_sequence_unlabeled, windows_to_array

__all__ = [
    "BEHAVIOR_CLASSES",
    "FEATURE_COLUMNS",
    "NUM_FEATURES",
    "FeatureFrame",
    "RawSequence",
    "Window",
    "generate_synthetic_dataset",
    "load_self_collected_dataset",
    "window_sequence",
    "windows_to_arrays",
    "UnlabeledWindow",
    "window_sequence_unlabeled",
    "windows_to_array",
]
