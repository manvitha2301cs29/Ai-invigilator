"""
monitor_PHASE_C.py
-------------------
Layer 4 (Live Monitoring & Stopwatch Engine) from Section 4 of
AI_Study_Invigilator_Project.txt. Consumes a stream of FeatureVectors
(Phase B's output, one per sampled frame) for a single active
TargetBlock, and produces:
  - AwayEvent records (the automatic stopwatch)
  - PhoneEvent records
  - SecondPersonEvent records
  - Notification objects (what the watcher should actually show/print)

DESIGN NOTE: this is deliberately NOT the same thing as Phase B's
per-frame duration gating (phone_flag, second_person_flag on
FeatureVector). Phase B answers "has this been going on long enough to
be real" at the SIGNAL level. This module answers "given that, what
EVENT should be recorded and what NOTIFICATION (if any) should fire,
including ESCALATION tiers" at the SESSION level -- e.g. away
notifications get progressively firmer the longer someone is away,
which is a session-level concern, not a per-frame signal.

Everything here is pure/testable the same way Phase B was: no camera,
no real clock, no sleep(). The watcher (run_watcher_PHASE_C.py) is the
only place that calls time.time() and threads real elapsed time through
these functions.
"""

from dataclasses import dataclass, field
from enum import Enum


class NotificationTier(Enum):
    """How urgently a notification should be presented. The watcher maps
    these to actual OS notification calls; this module only decides
    WHICH tier and WHAT text, never how it's displayed."""
    SOFT = "soft"      # e.g. a quiet, dismissible nudge
    FIRM = "firm"       # e.g. a more visually insistent notification
    DIRECT = "direct"   # e.g. phone/second-person warnings -- Section 6:
                         # "the notification is deliberately direct rather
                         # than softly worded"


@dataclass
class Notification:
    tier: NotificationTier
    message: str
    event_type: str  # "away" | "phone" | "second_person"


@dataclass
class MonitorEvent:
    """One completed (closed) event -- an away period, a phone
    appearance, or a second-person appearance that has ENDED. Mirrors
    the AwayEvent / PhoneEvent / SecondPersonEvent tables from the
    project's data model (Section 8): away_start/away_end become
    start/end here, was_notified is tracked too."""
    event_type: str  # "away" | "phone" | "second_person"
    start: float
    end: float
    duration_sec: float
    was_notified: bool


@dataclass
class TargetBlockMonitorState:
    """Everything the monitor needs to remember across calls for ONE
    active TargetBlock. The watcher owns exactly one instance of this
    per active block, exactly like Phase B's RollingState is owned per
    block for feature extraction -- these are separate state objects
    because they track different things at different layers."""
    # Away tracking
    away_since: float | None = None
    away_notified_tiers: set = field(default_factory=set)  # which tiers
                                                              # already fired
                                                              # for the
                                                              # CURRENT away
                                                              # period
    declared_break_active: bool = False  # True if the user tapped
                                          # "Taking a break" -- converts
                                          # what would be an AwayEvent
                                          # into a BreakEvent instead
                                          # (Section 7 design note)

    # Phone tracking
    phone_since: float | None = None
    phone_notified: bool = False

    # Second-person tracking
    second_person_since: float | None = None
    second_person_notified: bool = False

    # Completed events accumulate here for the block; Phase C's backend
    # will persist these as real AwayEvent/PhoneEvent/SecondPersonEvent
    # rows once the API layer exists.
    completed_events: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thresholds -- away escalation tiers per Section 12 / the earlier design
# discussion: "~30s away -> soft nudge... ~2min away -> firmer notice."
# Phone and second-person use the SAME duration floors as Phase B's
# extractor (kept in sync manually here since this module doesn't import
# Phase B's constants directly, to keep it independently testable --
# Phase C's real watcher wiring is what keeps these numbers consistent
# in practice; a future refactor could share one constants module).
# ---------------------------------------------------------------------------
AWAY_SOFT_NUDGE_SEC = 30.0
AWAY_FIRM_NUDGE_SEC = 120.0
PHONE_SUSTAINED_SEC = 3.0
SECOND_PERSON_SUSTAINED_SEC = 4.0


def start_break(state: TargetBlockMonitorState, now: float) -> TargetBlockMonitorState:
    """User explicitly taps 'Taking a break'. Per Section 7's design
    note: if the user is already away (undeclared), this retroactively
    converts the open away period into a declared break rather than
    penalizing them for remembering to declare it a moment late -- this
    handles the realistic case where someone gets up first and only
    marks the break a few seconds later."""
    new_state = TargetBlockMonitorState(
        away_since=None,  # closing out the away period, converted to a break
        away_notified_tiers=set(),
        declared_break_active=True,
        phone_since=state.phone_since,
        phone_notified=state.phone_notified,
        second_person_since=state.second_person_since,
        second_person_notified=state.second_person_notified,
        completed_events=list(state.completed_events),
    )
    return new_state


def end_break(state: TargetBlockMonitorState) -> TargetBlockMonitorState:
    """User taps 'Resume'. Away detection resumes normally on the next
    tick if face is still absent."""
    new_state = TargetBlockMonitorState(
        away_since=None,
        away_notified_tiers=set(),
        declared_break_active=False,
        phone_since=state.phone_since,
        phone_notified=state.phone_notified,
        second_person_since=state.second_person_since,
        second_person_notified=state.second_person_notified,
        completed_events=list(state.completed_events),
    )
    return new_state


def tick(
    state: TargetBlockMonitorState,
    now: float,
    face_present: bool,
    phone_flag: bool,
    second_person_flag: bool,
    student_name: str = "there",
) -> tuple[TargetBlockMonitorState, list[Notification]]:
    """The main entry point, called once per sampled frame (i.e. once
    per Phase B FeatureVector) with the current monitor state and this
    frame's face-presence + already-duration-gated phone/second-person
    flags from Phase B. Returns (new_state, notifications_to_fire_now).

    Note phone_flag/second_person_flag here are PHASE B's outputs (i.e.
    already duration-gated at the signal level) -- this function's own
    PHONE_SUSTAINED_SEC/SECOND_PERSON_SUSTAINED_SEC constants track a
    SEPARATE clock for escalation/event-closing purposes at the session
    level, matching the module docstring's explanation of why these two
    layers are kept distinct.
    """
    notifications: list[Notification] = []
    completed = list(state.completed_events)

    # ---------------- Away tracking ----------------
    away_since = state.away_since
    away_notified_tiers = set(state.away_notified_tiers)

    if state.declared_break_active:
        # On a declared break: never accumulate an away event or fire
        # away notifications, regardless of face presence.
        away_since = None
        away_notified_tiers = set()
    elif not face_present:
        if away_since is None:
            away_since = now
        elapsed = now - away_since

        if elapsed >= AWAY_SOFT_NUDGE_SEC and "soft" not in away_notified_tiers:
            notifications.append(Notification(
                tier=NotificationTier.SOFT,
                message="You've stepped away — mark a break or resume?",
                event_type="away",
            ))
            away_notified_tiers.add("soft")

        if elapsed >= AWAY_FIRM_NUDGE_SEC and "firm" not in away_notified_tiers:
            notifications.append(Notification(
                tier=NotificationTier.FIRM,
                message=f"You've been away {int(elapsed // 60)}+ min. "
                        f"This is being logged as away time.",
                event_type="away",
            ))
            away_notified_tiers.add("firm")
    else:
        if away_since is not None:
            # Was away, now back -- close out the event.
            duration = now - away_since
            completed.append(MonitorEvent(
                event_type="away", start=away_since, end=now,
                duration_sec=duration,
                was_notified=len(away_notified_tiers) > 0,
            ))
        away_since = None
        away_notified_tiers = set()

    # ---------------- Phone tracking ----------------
    phone_since = state.phone_since
    phone_notified = state.phone_notified

    if phone_flag:
        if phone_since is None:
            phone_since = now
        elapsed = now - phone_since
        if elapsed >= PHONE_SUSTAINED_SEC and not phone_notified:
            notifications.append(Notification(
                tier=NotificationTier.DIRECT,
                message=f"Don't use your phone, {student_name}.",
                event_type="phone",
            ))
            phone_notified = True
    else:
        if phone_since is not None:
            duration = now - phone_since
            completed.append(MonitorEvent(
                event_type="phone", start=phone_since, end=now,
                duration_sec=duration, was_notified=phone_notified,
            ))
        phone_since = None
        phone_notified = False

    # ---------------- Second-person tracking ----------------
    second_person_since = state.second_person_since
    second_person_notified = state.second_person_notified

    if second_person_flag:
        if second_person_since is None:
            second_person_since = now
        elapsed = now - second_person_since
        if elapsed >= SECOND_PERSON_SUSTAINED_SEC and not second_person_notified:
            notifications.append(Notification(
                tier=NotificationTier.DIRECT,
                message=f"Stay in a quiet environment — avoid talking to "
                        f"others, {student_name}.",
                event_type="second_person",
            ))
            second_person_notified = True
    else:
        if second_person_since is not None:
            duration = now - second_person_since
            completed.append(MonitorEvent(
                event_type="second_person", start=second_person_since, end=now,
                duration_sec=duration, was_notified=second_person_notified,
            ))
        second_person_since = None
        second_person_notified = False

    new_state = TargetBlockMonitorState(
        away_since=away_since,
        away_notified_tiers=away_notified_tiers,
        declared_break_active=state.declared_break_active,
        phone_since=phone_since,
        phone_notified=phone_notified,
        second_person_since=second_person_since,
        second_person_notified=second_person_notified,
        completed_events=completed,
    )
    return new_state, notifications


def current_away_duration(state: TargetBlockMonitorState, now: float) -> float:
    """Convenience accessor for 'how long has the CURRENT away period
    (if any) been going', used by the dashboard's live status view /
    the watcher's own console output. Returns 0.0 if not currently away."""
    if state.away_since is None:
        return 0.0
    return now - state.away_since
