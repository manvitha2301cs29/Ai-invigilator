"""
participants.py
-----------------
Groups RawSequences by participant_id -- something Phase D's loaders
deliberately DON'T carry on RawSequence itself (see
phase_d/data/self_collected_loader.py's docstring: participant_id is
used for grouping only, never fed to a model, to avoid a model learning
to key off which person a sequence came from).

Phase E needs this grouping for a different reason than Phase D avoids
it: Section 5's FATIGUED-PATTERN definition requires the autoencoder to
be trained "per-student ... comparing a student's current session
pattern to their own established baseline, not to a population norm."
That's an explicit modeling choice at the TRAINING level (which
sequences get pooled into which autoencoder's training set), not a
feature fed into any single window -- so re-deriving the
session_id -> participant_id mapping here, from the same CSVs, for
grouping purposes only, doesn't reintroduce the thing Phase D avoids.

This module re-reads the self-collected CSVs directly (rather than
threading participant_id through RawSequence) specifically so
RawSequence's shape stays identical to Phase D's -- one less thing that
could silently diverge between the two phases' understanding of what a
RawSequence is.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from .reuse import RawSequence


def read_session_to_participant_map(csv_paths: list[str]) -> dict[str, str]:
    """session_id -> participant_id, read directly from the diary-
    protocol CSVs. Raises if a session_id maps to more than one
    participant_id across rows -- that would indicate a data-entry
    error (one recording session should belong to exactly one person)
    that's better caught here than silently producing a mixed-identity
    "baseline."""
    mapping: dict[str, str] = {}
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row["session_id"]
                participant_id = row["participant_id"]
                if session_id in mapping and mapping[session_id] != participant_id:
                    raise ValueError(
                        f"session_id {session_id!r} maps to multiple participant_ids "
                        f"({mapping[session_id]!r} and {participant_id!r}) across the "
                        f"provided CSVs -- check for a copy/paste error in session_id"
                    )
                mapping[session_id] = participant_id
    return mapping


def group_by_participant(
    sequences: list[RawSequence], session_to_participant: dict[str, str]
) -> dict[str, list[RawSequence]]:
    """Group RawSequences (keyed by source_id == session_id, per
    self_collected_loader.py) into {participant_id: [sequences]}.
    Sequences whose session_id has no known participant are collected
    under the key "__unknown__" so callers can decide how to handle them
    (skip, warn, etc.) rather than having them silently vanish."""
    grouped: dict[str, list[RawSequence]] = defaultdict(list)
    for seq in sequences:
        participant_id = session_to_participant.get(seq.source_id, "__unknown__")
        grouped[participant_id].append(seq)
    return dict(grouped)
