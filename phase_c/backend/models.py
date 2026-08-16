"""
models_PHASE_C.py
-------------------
SQLAlchemy 2.x models implementing the exact data model from Section 8
of AI_Study_Invigilator_Project.txt / the master coding prompt, targeting
PostgreSQL specifically (not SQLite) since this project is being built
for real deployment from the start, not local-only development.

Why Postgres from day one instead of SQLite-then-migrate: SQLite is a
single-file, single-writer database -- it works fine for one person
running tests locally, but doesn't hold up behind a public, multi-user
web backend, which is the actual deployment target here. Building
against Postgres from Phase C means what gets tested locally is
architecturally identical to what gets deployed -- no "works on my
SQLite file, breaks in production" surprises later.

Local Postgres for development runs via Docker (see docker-compose.yml
and .env.example in this same folder) rather than a native Windows
install, which is the simplest way to get a real, disposable Postgres
instance on Windows.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Boolean, Integer, Float, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums -- map to native PostgreSQL ENUM types automatically under
# SQLAlchemy 2.x's declarative mapping (confirmed against current
# SQLAlchemy docs: a Python enum.Enum used as a Mapped[...] type creates
# a real `CREATE TYPE ... AS ENUM (...)` in Postgres, not just a VARCHAR
# with an application-level check).
# ---------------------------------------------------------------------------
class TargetBlockStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class BehaviorState(str, enum.Enum):
    ENGAGED = "engaged"
    IDLE_PRESENT = "idle_present"
    DISTRACTED_PRESENT = "distracted_present"
    AWAY = "away"


class Verdict(str, enum.Enum):
    ON_TRACK = "on_track"
    NEEDS_SHORTER_BLOCKS = "needs_shorter_blocks"
    NEEDS_SCHEDULED_BREAKS = "needs_scheduled_breaks"


def _uuid_pk() -> Mapped[uuid.UUID]:
    """Shared primary-key column definition. UUID (not autoincrement
    integer) is used deliberately: this system is designed for many
    independent local watchers (Section 12's deployment architecture)
    pushing events to one shared backend over the network, and UUIDs
    avoid any possibility of primary-key collisions between events
    generated concurrently by different watchers before they've ever
    talked to the central database -- an autoincrement int PK only
    looks safe when there's a single writer, which is exactly the
    assumption this project does not get to make."""
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    # The token a local watcher authenticates with -- see
    # 03_DEPLOYMENT_GUIDE.txt's "CONNECTING THE TWO" section. Generated
    # once from the dashboard's account settings page (Phase G), stored
    # here so the backend can map an incoming watcher's events to the
    # correct user.
    watcher_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    target_blocks: Mapped[list["TargetBlock"]] = relationship(back_populates="user")


class TargetBlock(Base):
    __tablename__ = "target_blocks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    planned_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_duration_min: Mapped[int] = mapped_column(Integer)
    status: Mapped[TargetBlockStatus] = mapped_column(
        default=TargetBlockStatus.SCHEDULED
    )
    actual_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="target_blocks")
    state_events: Mapped[list["StateEvent"]] = relationship(back_populates="target_block", cascade="all, delete-orphan")
    break_events: Mapped[list["BreakEvent"]] = relationship(back_populates="target_block", cascade="all, delete-orphan")
    away_events: Mapped[list["AwayEvent"]] = relationship(back_populates="target_block", cascade="all, delete-orphan")
    phone_events: Mapped[list["PhoneEvent"]] = relationship(back_populates="target_block", cascade="all, delete-orphan")
    second_person_events: Mapped[list["SecondPersonEvent"]] = relationship(back_populates="target_block", cascade="all, delete-orphan")
    summary: Mapped["TargetBlockSummary | None"] = relationship(back_populates="target_block", uselist=False, cascade="all, delete-orphan")


class StateEvent(Base):
    """Fine-grained, continuous log during an active block. raw_features
    intentionally stores only numeric/boolean signal values (matches
    Phase B's FeatureVector) -- NEVER image data, per the project's
    privacy-by-design principle (Section 1)."""
    __tablename__ = "state_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    target_block_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("target_blocks.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state: Mapped[BehaviorState]

    # raw_features flattened into explicit typed columns rather than a
    # JSON blob: this makes it possible to write real SQL aggregate
    # queries (e.g. "average posture_angle where state=distracted_present")
    # without needing Postgres-specific JSON operators, and gives Phase D's
    # training-data export scripts a plain, typed table to query directly.
    yaw: Mapped[float] = mapped_column(Float, default=0.0)
    pitch: Mapped[float] = mapped_column(Float, default=0.0)
    roll: Mapped[float] = mapped_column(Float, default=0.0)
    gaze_x: Mapped[float] = mapped_column(Float, default=0.0)
    gaze_y: Mapped[float] = mapped_column(Float, default=0.0)
    eyes_closed: Mapped[bool] = mapped_column(Boolean, default=False)
    posture_angle: Mapped[float] = mapped_column(Float, default=0.0)
    movement_delta: Mapped[float] = mapped_column(Float, default=0.0)
    face_conf: Mapped[float] = mapped_column(Float, default=0.0)
    phone_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    second_person_detected: Mapped[bool] = mapped_column(Boolean, default=False)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="state_events")


class BreakEvent(Base):
    __tablename__ = "break_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    target_block_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("target_blocks.id"), index=True)
    break_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    break_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    declared: Mapped[bool] = mapped_column(Boolean, default=True)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="break_events")


class AwayEvent(Base):
    __tablename__ = "away_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    target_block_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("target_blocks.id"), index=True)
    away_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    away_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    was_notified: Mapped[bool] = mapped_column(Boolean, default=False)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="away_events")


class PhoneEvent(Base):
    __tablename__ = "phone_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    target_block_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("target_blocks.id"), index=True)
    detected_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    detected_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    was_notified: Mapped[bool] = mapped_column(Boolean, default=False)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="phone_events")


class SecondPersonEvent(Base):
    __tablename__ = "second_person_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    target_block_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("target_blocks.id"), index=True)
    detected_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    detected_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    was_notified: Mapped[bool] = mapped_column(Boolean, default=False)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="second_person_events")


class TargetBlockSummary(Base):
    __tablename__ = "target_block_summaries"

    target_block_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("target_blocks.id"), primary_key=True
    )
    worked_sec: Mapped[float] = mapped_column(Float, default=0.0)
    distracted_sec: Mapped[float] = mapped_column(Float, default=0.0)
    away_undeclared_sec: Mapped[float] = mapped_column(Float, default=0.0)
    declared_break_sec: Mapped[float] = mapped_column(Float, default=0.0)
    phone_flag_count: Mapped[int] = mapped_column(Integer, default=0)
    second_person_flag_count: Mapped[int] = mapped_column(Integer, default=0)
    away_event_count: Mapped[int] = mapped_column(Integer, default=0)
    compliance_pct: Mapped[float] = mapped_column(Float, default=0.0)
    verdict: Mapped[Verdict | None] = mapped_column(nullable=True)

    target_block: Mapped["TargetBlock"] = relationship(back_populates="summary")
