"""
routes.py
---------
The new, dashboard-facing endpoints Phase G adds on top of Phase C's
existing watcher-facing API (see main.py, which mounts phase_c's router
unmodified alongside this one). Four groups:

  /auth/login              -- JWT dashboard login (auth.py)
  /dashboard/target-blocks -- JWT-authenticated create/list target
                               blocks, for the planner screen. A NEW
                               pair of endpoints, not a duplicate of
                               Phase C's watcher-token-authenticated
                               POST /target-blocks -- see this section's
                               docstring below for why one couldn't just
                               reuse the other.
  /target-blocks/{id}/recommendation
                            -- computes (if needed) + returns the
                               end-of-block report: Phase F's
                               RecommendationResult numbers PLUS an
                               LLM-phrased narrative in the requested
                               tone (with a deterministic fallback if
                               the LLM call fails or no API key is set)
  /target-blocks/history    -- the student's recent completed blocks +
                               verdicts, for the dashboard's
                               history/trends screen
  /target-blocks/{id}/live-status
                            -- POST (watcher-authenticated) to push a
                               status update; GET /ws/... (dashboard-
                               authenticated) WebSocket to receive them
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import JWT_ALGORITHM, JWT_SECRET, authenticate_user, create_access_token, get_current_dashboard_user
from bridge import phase_c_database, phase_c_main, phase_c_models
from bridge import recommendation as rec

import recommendation.queries as rec_queries  # noqa: E402 -- submodule not re-exported by
# recommendation/__init__.py; importable directly once `bridge` above has
# already put phase_f/ on sys.path and loaded the `recommendation` package.
from llm.client import phrase_recommendation
from session_summary import upsert_block_summary
from websocket_hub import hub

router = APIRouter()


# ---------------------------------------------------------------------------
# Dashboard auth
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(phase_c_database.get_db)):
    user = authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return LoginResponse(access_token=create_access_token(str(user.id)))


# ---------------------------------------------------------------------------
# Planner: JWT-authenticated create/list target blocks
# ---------------------------------------------------------------------------
# Phase C's POST /target-blocks (phase_c/backend/main.py) is
# watcher-token-authenticated ONLY -- correct for the watcher itself, but
# unusable directly from the browser dashboard, which authenticates with
# a JWT and may not even have a watcher running yet when the student is
# just planning ahead. A second endpoint under a distinct path
# (/dashboard/target-blocks, not a second handler at the SAME path,
# which would silently collide with Phase C's existing route) is the
# planner screen's actual write path; it creates the exact same
# TargetBlock row Phase C's watcher will later start/end, just via a
# different credential. Reuses Phase C's own TargetBlockCreate/
# TargetBlockOut pydantic schemas rather than redefining them.
@router.post("/dashboard/target-blocks", response_model=phase_c_main.TargetBlockOut)
def create_target_block_from_dashboard(
    payload: phase_c_main.TargetBlockCreate,
    user: phase_c_models.User = Depends(get_current_dashboard_user),
    db: Session = Depends(phase_c_database.get_db),
):
    block = phase_c_models.TargetBlock(
        user_id=user.id,
        planned_start=payload.planned_start,
        planned_end=payload.planned_end,
        planned_duration_min=payload.planned_duration_min,
        status=phase_c_models.TargetBlockStatus.SCHEDULED,
    )
    db.add(block)
    db.commit()
    db.refresh(block)
    return block


@router.get("/dashboard/target-blocks", response_model=list[phase_c_main.TargetBlockOut])
def list_target_blocks_from_dashboard(
    user: phase_c_models.User = Depends(get_current_dashboard_user),
    db: Session = Depends(phase_c_database.get_db),
):
    """Upcoming + in-progress blocks (SCHEDULED/ACTIVE), most recent
    first -- what the planner screen shows so the student can see what's
    already on their schedule before adding more. Completed/abandoned
    blocks belong to the history screen (GET /target-blocks/history),
    not here."""
    from sqlalchemy import select

    stmt = (
        select(phase_c_models.TargetBlock)
        .where(phase_c_models.TargetBlock.user_id == user.id)
        .where(
            phase_c_models.TargetBlock.status.in_(
                [phase_c_models.TargetBlockStatus.SCHEDULED, phase_c_models.TargetBlockStatus.ACTIVE]
            )
        )
        .order_by(phase_c_models.TargetBlock.planned_start.asc())
    )
    return db.execute(stmt).scalars().all()


# ---------------------------------------------------------------------------
# Recommendation (end-of-block report)
# ---------------------------------------------------------------------------
ToneQuery = Literal["strict", "neutral", "encouraging"]


def fallback_report(result) -> str:
    """A deterministic, LLM-free sentence used when the Claude API call
    fails or ANTHROPIC_API_KEY isn't configured -- per the project's
    determinism principle, the NUMBERS and VERDICT must always be
    available to the student even if the phrasing layer is unavailable;
    this is not a degraded product, just a plainer one."""
    return (
        f"You worked {result.worked_sec / 60:.0f} of {result.planned_duration_min} planned minutes "
        f"({result.compliance_pct * 100:.0f}% compliance), with {result.away_event_count} away event(s). "
        f"Verdict: {result.block_verdict.value.replace('_', ' ')}. "
        f"Suggested next schedule: {result.schedule_recommendation.value.replace('_', ' ')}."
    )


@router.get("/target-blocks/{block_id}/recommendation")
def get_recommendation(
    block_id: str,
    tone: ToneQuery = "neutral",
    user: phase_c_models.User = Depends(get_current_dashboard_user),
    db: Session = Depends(phase_c_database.get_db),
):
    try:
        block_uuid = uuid.UUID(block_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Target block not found")
    block = db.get(phase_c_models.TargetBlock, block_uuid)
    if block is None or block.user_id != user.id:
        raise HTTPException(status_code=404, detail="Target block not found")
    if block.status != phase_c_models.TargetBlockStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Block is not completed yet")

    # Lazily compute the summary the first time a recommendation is
    # requested for this block -- see session_summary.py's docstring;
    # idempotent, so a page refresh just re-reads the same numbers.
    if block.summary is None:
        upsert_block_summary(db, block)
        db.refresh(block)

    latest = rec_queries.fetch_block_record(db, block_uuid)
    history = rec_queries.fetch_recent_block_summaries(db, user.id, limit=20)
    result = rec.evaluate(latest, history)

    try:
        narrative = phrase_recommendation(result, tone)
    except Exception:
        # LLM layer is a phrasing convenience only -- never let its
        # failure hide the deterministic numbers from the student.
        narrative = fallback_report(result)

    return {
        "target_block_id": result.target_block_id,
        "block_verdict": result.block_verdict.value,
        "schedule_recommendation": result.schedule_recommendation.value,
        "compliance_pct": result.compliance_pct,
        "away_event_count": result.away_event_count,
        "worked_sec": result.worked_sec,
        "distracted_sec": result.distracted_sec,
        "away_undeclared_sec": result.away_undeclared_sec,
        "declared_break_sec": result.declared_break_sec,
        "phone_flag_count": result.phone_flag_count,
        "second_person_flag_count": result.second_person_flag_count,
        "planned_duration_min": result.planned_duration_min,
        "tone": tone,
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# History / trends
# ---------------------------------------------------------------------------
@router.get("/target-blocks/history")
def get_history(
    limit: int = Query(default=20, le=100),
    user: phase_c_models.User = Depends(get_current_dashboard_user),
    db: Session = Depends(phase_c_database.get_db),
):
    records = rec_queries.fetch_recent_block_summaries(db, user.id, limit=limit)
    return [
        {
            "target_block_id": str(r.target_block_id),
            "planned_start": r.planned_start.isoformat(),
            "planned_duration_min": r.planned_duration_min,
            "compliance_pct": r.compliance_pct,
            "worked_sec": r.worked_sec,
            "distracted_sec": r.distracted_sec,
            "away_event_count": r.away_event_count,
            "verdict": rec.compute_single_block_verdict(r.compliance_pct, r.away_event_count).value,
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# Live status: watcher pushes, dashboard subscribes over WebSocket
# ---------------------------------------------------------------------------
class LiveStatusPush(BaseModel):
    state: phase_c_models.BehaviorState
    away_stopwatch_sec: float = 0.0
    phone_flag: bool = False
    second_person_flag: bool = False


@router.post("/target-blocks/{block_id}/live-status", status_code=202)
async def post_live_status(
    block_id: str,
    payload: LiveStatusPush,
    user: phase_c_models.User = Depends(phase_c_main.get_current_user),  # watcher-token auth, NOT JWT
):
    """Called by the (unmodified) watcher every second or so while a
    block is active -- a lightweight status ping, deliberately separate
    from the durable StateEvent stream Phase C already logs, since this
    is fire-and-forget UI data, not something that needs to survive a
    dashboard reconnect."""
    await hub.broadcast(block_id, payload.model_dump())
    return {"status": "broadcast"}


def _verify_dashboard_ws_token(token: str, db: Session) -> phase_c_models.User:
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        user_uuid = uuid.UUID(claims.get("sub", ""))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.get(phase_c_models.User, user_uuid)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


@router.websocket("/ws/target-blocks/{block_id}/live-status")
async def ws_live_status(websocket: WebSocket, block_id: str, token: str = Query(...)):
    """Dashboard-authenticated (JWT passed as a query param, the
    standard workaround for WebSocket clients that can't set custom
    headers easily -- see phase_g/frontend's live-status hook for how
    the React client constructs this URL, including the ws:// vs wss://
    note from docs/03_DEPLOYMENT_GUIDE.txt Part 1 Step 5)."""
    db_gen = phase_c_database.get_db()
    db = next(db_gen)
    try:
        _verify_dashboard_ws_token(token, db)
    except HTTPException:
        await websocket.close(code=4401)
        return
    finally:
        db.close()

    await hub.connect(block_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # dashboard never sends data; just keeps the socket open
    except WebSocketDisconnect:
        hub.disconnect(block_id, websocket)
