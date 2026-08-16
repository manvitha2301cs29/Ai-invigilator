"""
test_monitor_PHASE_C.py
------------------------
Unit tests for watcher/monitoring/monitor_PHASE_C.py. Same discipline as
Phase B: synthetic timestamps, no real clock, no camera, no sleep().

Run with:
    pytest tests/test_monitor_PHASE_C.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "watcher"))

from monitoring.monitor import (
    TargetBlockMonitorState, tick, start_break, end_break,
    current_away_duration, NotificationTier,
)


def test_brief_face_absence_does_not_notify():
    """A face-absent reading for only a couple of ticks, well under the
    30s soft-nudge threshold, should not fire any notification."""
    state = TargetBlockMonitorState()
    for t in [0.0, 5.0, 10.0]:
        state, notifications = tick(state, t, face_present=False,
                                     phone_flag=False, second_person_flag=False)
        assert notifications == []


def test_away_soft_nudge_fires_at_30s():
    state = TargetBlockMonitorState()
    state, n1 = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert n1 == []
    state, n2 = tick(state, 31.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert len(n2) == 1
    assert n2[0].tier == NotificationTier.SOFT
    assert n2[0].event_type == "away"


def test_soft_nudge_does_not_refire_every_tick():
    """Once the soft tier has fired, it must not fire again on
    subsequent ticks while still under the firm threshold -- otherwise
    the student gets spammed every frame."""
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, n = tick(state, 35.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert len(n) == 1  # fires once at 35s
    state, n2 = tick(state, 40.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert n2 == [], "soft nudge must not refire on every subsequent tick"


def test_away_firm_nudge_fires_at_120s_and_is_distinct_from_soft():
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, n_soft = tick(state, 31.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert len(n_soft) == 1 and n_soft[0].tier == NotificationTier.SOFT

    state, n_none = tick(state, 60.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert n_none == [], "no new notification between soft and firm thresholds"

    state, n_firm = tick(state, 121.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert len(n_firm) == 1
    assert n_firm[0].tier == NotificationTier.FIRM


def test_returning_closes_away_event_and_resets_tiers():
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, _ = tick(state, 31.0, face_present=False, phone_flag=False, second_person_flag=False)
    # student returns
    state, notifications = tick(state, 40.0, face_present=True, phone_flag=False, second_person_flag=False)

    assert notifications == []
    assert state.away_since is None
    assert len(state.completed_events) == 1
    event = state.completed_events[0]
    assert event.event_type == "away"
    assert event.start == 0.0
    assert event.end == 40.0
    assert event.duration_sec == 40.0
    assert event.was_notified is True  # soft tier fired during this period


def test_away_event_without_notification_recorded_correctly():
    """A short away period that never crossed the soft-nudge threshold
    should still be recorded as a completed event when the student
    returns, just with was_notified=False."""
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, _ = tick(state, 10.0, face_present=True, phone_flag=False, second_person_flag=False)

    assert len(state.completed_events) == 1
    event = state.completed_events[0]
    assert event.duration_sec == 10.0
    assert event.was_notified is False


def test_declared_break_suppresses_away_tracking_entirely():
    """Per Section 7's design note: while on a declared break, no away
    event should accumulate and no away notification should fire,
    regardless of how long face_present stays False."""
    state = TargetBlockMonitorState()
    state = start_break(state, now=0.0)

    for t in [10.0, 50.0, 200.0]:
        state, notifications = tick(state, t, face_present=False,
                                     phone_flag=False, second_person_flag=False)
        assert notifications == [], f"no away notifications during a declared break, got {notifications} at t={t}"

    assert state.away_since is None
    assert state.completed_events == []


def test_start_break_retroactively_converts_open_away_period():
    """The realistic case: student gets up (away starts accumulating),
    THEN remembers to declare a break a few seconds later. Per Section
    7's design note, this should NOT be penalized as an undeclared away
    period once the break is declared."""
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, _ = tick(state, 5.0, face_present=False, phone_flag=False, second_person_flag=False)
    # student remembers to declare the break
    state = start_break(state, now=5.0)

    assert state.away_since is None
    assert state.declared_break_active is True
    # No away event should have been logged as "completed" from the
    # brief pre-declaration window -- it's simply absorbed into the break.
    assert state.completed_events == []


def test_end_break_resumes_normal_away_tracking():
    state = TargetBlockMonitorState()
    state = start_break(state, now=0.0)
    state = end_break(state)
    assert state.declared_break_active is False

    # away tracking should now work normally again
    state, _ = tick(state, 10.0, face_present=False, phone_flag=False, second_person_flag=False)
    state, notifications = tick(state, 41.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert len(notifications) == 1
    assert notifications[0].tier == NotificationTier.SOFT


def test_phone_sustained_fires_direct_warning_with_name():
    state = TargetBlockMonitorState()
    state, n1 = tick(state, 0.0, face_present=True, phone_flag=True, second_person_flag=False,
                      student_name="Kyra")
    assert n1 == []  # not yet sustained

    state, n2 = tick(state, 3.5, face_present=True, phone_flag=True, second_person_flag=False,
                      student_name="Kyra")
    assert len(n2) == 1
    assert n2[0].tier == NotificationTier.DIRECT
    assert n2[0].message == "Don't use your phone, Kyra."
    assert n2[0].event_type == "phone"


def test_phone_warning_does_not_refire_while_still_present():
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=True, phone_flag=True, second_person_flag=False)
    state, n = tick(state, 4.0, face_present=True, phone_flag=True, second_person_flag=False)
    assert len(n) == 1
    state, n2 = tick(state, 5.0, face_present=True, phone_flag=True, second_person_flag=False)
    assert n2 == [], "must not refire every tick while phone stays in frame"


def test_phone_removed_closes_event():
    state = TargetBlockMonitorState()
    state, _ = tick(state, 0.0, face_present=True, phone_flag=True, second_person_flag=False)
    state, _ = tick(state, 4.0, face_present=True, phone_flag=True, second_person_flag=False)
    state, notifications = tick(state, 6.0, face_present=True, phone_flag=False, second_person_flag=False)

    assert notifications == []
    assert len(state.completed_events) == 1
    event = state.completed_events[0]
    assert event.event_type == "phone"
    assert event.was_notified is True


def test_second_person_sustained_fires_direct_warning_with_name():
    state = TargetBlockMonitorState()
    state, n1 = tick(state, 0.0, face_present=True, phone_flag=False, second_person_flag=True,
                      student_name="Kyra")
    assert n1 == []

    state, n2 = tick(state, 4.5, face_present=True, phone_flag=False, second_person_flag=True,
                      student_name="Kyra")
    assert len(n2) == 1
    assert n2[0].tier == NotificationTier.DIRECT
    assert "quiet environment" in n2[0].message
    assert "Kyra" in n2[0].message
    assert n2[0].event_type == "second_person"


def test_phone_and_second_person_tracked_independently():
    """Both can be active at once and must not interfere with each
    other's timers or notifications."""
    state = TargetBlockMonitorState()
    state, notifications = tick(
        state, 0.0, face_present=True, phone_flag=True, second_person_flag=True
    )
    assert notifications == []

    state, notifications = tick(
        state, 5.0, face_present=True, phone_flag=True, second_person_flag=True
    )
    event_types = {n.event_type for n in notifications}
    assert event_types == {"phone", "second_person"}, (
        "both should have independently crossed their own sustained "
        "threshold by t=5.0 and both should notify"
    )


def test_current_away_duration_helper():
    state = TargetBlockMonitorState()
    assert current_away_duration(state, now=100.0) == 0.0

    state, _ = tick(state, 0.0, face_present=False, phone_flag=False, second_person_flag=False)
    assert current_away_duration(state, now=15.0) == 15.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
