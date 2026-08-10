"""
main_PHASE_C.py
-----------------
Minimal FastAPI backend for Phase C. Covers exactly what the local
watcher needs to report to: authenticate, start/end a target block,
push state events, and close away/phone/second-person events. The full
dashboard-facing read API (history, trends, live WebSocket status) is a
Phase G concern per the master coding prompt's build order -- Phase C's
job is just to prove the watcher <-> backend <-> Postgres round trip
works end to end.

RUN:
    uvicorn main_PHASE_C:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs.
"""

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import get_db, init_db
import models

app = FastAPI(title="AI Study Invigilator API (Phase C)")


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Auth: a single shared-secret watcher_token per user (see models_PHASE_C.py
# User.watcher_token comment). This is intentionally simple for Phase C --
# Phase G should replace/extend this with a real JWT-based flow for the
# dashboard's own login, while keeping a long-lived watcher_token for the
# local watcher specifically, since a human isn't typing a password into
# a background process on every restart.
# ---------------------------------------------------------------------------
def get_current_user(
    x_watcher_token: str = Header(..., description="Per-user watcher auth token"),
    db: Session = Depends(get_db),
) -> models.User:
    user = db.execute(
        select(models.User).where(models.User.watcher_token == x_watcher_token)
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid watcher token")
    return user


# ---------------------------------------------------------------------------
# Pydantic schemas (request/response bodies)
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    watcher_token: str

    class Config:
        from_attributes = True


class TargetBlockCreate(BaseModel):
    planned_start: datetime
    planned_end: datetime
    planned_duration_min: int


class TargetBlockOut(BaseModel):
    id: uuid.UUID
    status: models.TargetBlockStatus
    planned_start: datetime
    planned_end: datetime
    planned_duration_min: int
    actual_start: datetime | None
    actual_end: datetime | None

    class Config:
        from_attributes = True


class StateEventCreate(BaseModel):
    timestamp: datetime
    state: models.BehaviorState
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    eyes_closed: bool = False
    posture_angle: float = 0.0
    movement_delta: float = 0.0
    face_conf: float = 0.0
    phone_detected: bool = False
    second_person_detected: bool = False


class AwayEventCreate(BaseModel):
    away_start: datetime
    away_end: datetime | None = None
    duration_sec: float = 0.0
    was_notified: bool = False


class PhoneEventCreate(BaseModel):
    detected_start: datetime
    detected_end: datetime | None = None
    duration_sec: float = 0.0
    was_notified: bool = False


class SecondPersonEventCreate(BaseModel):
    detected_start: datetime
    detected_end: datetime | None = None
    duration_sec: float = 0.0
    was_notified: bool = False


class BreakEventCreate(BaseModel):
    break_start: datetime
    break_end: datetime | None = None


# ---------------------------------------------------------------------------
# Auth / account endpoints
# ---------------------------------------------------------------------------
@app.post("/users", response_model=UserOut)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Phase C simplification: password is stored hashed but there's no
    login/session endpoint yet -- that's a Phase G dashboard concern.
    This endpoint exists so Phase C can create a test user and get back
    a watcher_token to actually exercise the rest of the API."""
    import hashlib
    existing = db.execute(
        select(models.User).where(models.User.email == payload.email)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = models.User(
        name=payload.name,
        email=payload.email,
        hashed_password=hashlib.sha256(payload.password.encode()).hexdigest(),
        watcher_token=uuid.uuid4().hex,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Target block lifecycle
# ---------------------------------------------------------------------------
@app.post("/target-blocks", response_model=TargetBlockOut)
def create_target_block(
    payload: TargetBlockCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = models.TargetBlock(
        user_id=user.id,
        planned_start=payload.planned_start,
        planned_end=payload.planned_end,
        planned_duration_min=payload.planned_duration_min,
        status=models.TargetBlockStatus.SCHEDULED,
    )
    db.add(block)
    db.commit()
    db.refresh(block)
    return block


@app.post("/target-blocks/{block_id}/start", response_model=TargetBlockOut)
def start_target_block(
    block_id: uuid.UUID,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = _get_owned_block(db, block_id, user)
    block.status = models.TargetBlockStatus.ACTIVE
    block.actual_start = datetime.now(timezone.utc)
    db.commit()
    db.refresh(block)
    return block


@app.post("/target-blocks/{block_id}/end", response_model=TargetBlockOut)
def end_target_block(
    block_id: uuid.UUID,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = _get_owned_block(db, block_id, user)
    block.status = models.TargetBlockStatus.COMPLETED
    block.actual_end = datetime.now(timezone.utc)
    db.commit()
    db.refresh(block)
    return block


def _get_owned_block(db: Session, block_id: uuid.UUID, user: models.User) -> models.TargetBlock:
    block = db.get(models.TargetBlock, block_id)
    if block is None or block.user_id != user.id:
        raise HTTPException(status_code=404, detail="Target block not found")
    return block


# ---------------------------------------------------------------------------
# Event ingestion -- what the watcher actually calls continuously while a
# block is active
# ---------------------------------------------------------------------------
@app.post("/target-blocks/{block_id}/state-events", status_code=201)
def post_state_event(
    block_id: uuid.UUID,
    payload: StateEventCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = _get_owned_block(db, block_id, user)
    event = models.StateEvent(target_block_id=block.id, **payload.model_dump())
    db.add(event)
    db.commit()
    return {"status": "recorded"}


@app.post("/target-blocks/{block_id}/away-events", status_code=201)
def post_away_event(
    block_id: uuid.UUID,
    payload: AwayEventCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The watcher calls this once an away period CLOSES (student
    returned), matching monitor_PHASE_C.py's MonitorEvent -- Phase C
    doesn't stream open/in-progress away periods to the backend, only
    completed ones, keeping this endpoint simple. A live in-progress
    view is a Phase G (WebSocket) concern."""
    block = _get_owned_block(db, block_id, user)
    event = models.AwayEvent(target_block_id=block.id, **payload.model_dump())
    db.add(event)
    db.commit()
    return {"status": "recorded"}


@app.post("/target-blocks/{block_id}/phone-events", status_code=201)
def post_phone_event(
    block_id: uuid.UUID,
    payload: PhoneEventCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = _get_owned_block(db, block_id, user)
    event = models.PhoneEvent(target_block_id=block.id, **payload.model_dump())
    db.add(event)
    db.commit()
    return {"status": "recorded"}


@app.post("/target-blocks/{block_id}/second-person-events", status_code=201)
def post_second_person_event(
    block_id: uuid.UUID,
    payload: SecondPersonEventCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    block = _get_owned_block(db, block_id, user)
    event = models.SecondPersonEvent(target_block_id=block.id, **payload.model_dump())
    db.add(event)
    db.commit()
    return {"status": "recorded"}


@app.post("/target-blocks/{block_id}/break-events", status_code=201)
def post_break_event(
    block_id: uuid.UUID,
    payload: BreakEventCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Called when the user declares a break (start) and again when they
    resume (to set break_end on the same record) -- Phase C keeps this
    simple by having the watcher pass the break's id back on resume;
    Phase G's real UI will wire this more smoothly through the
    dashboard."""
    block = _get_owned_block(db, block_id, user)
    event = models.BreakEvent(target_block_id=block.id, **payload.model_dump())
    db.add(event)
    db.commit()
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# Minimal read endpoint -- just enough to manually verify Phase C works
# end to end without needing a dashboard yet
# ---------------------------------------------------------------------------
@app.get("/target-blocks/{block_id}/summary-debug")
def get_block_debug_summary(
    block_id: uuid.UUID,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """NOT the real TargetBlockSummary computation (that's Layer 5 /
    Phase F's job, per Section 4 of the project doc). This is a Phase C
    convenience endpoint purely for manually confirming data actually
    made it into Postgres correctly -- e.g. via curl or the /docs UI --
    while building the watcher."""
    block = _get_owned_block(db, block_id, user)
    return {
        "block_id": str(block.id),
        "status": block.status,
        "state_event_count": len(block.state_events),
        "away_event_count": len(block.away_events),
        "phone_event_count": len(block.phone_events),
        "second_person_event_count": len(block.second_person_events),
        "break_event_count": len(block.break_events),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
