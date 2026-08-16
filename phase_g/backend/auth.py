"""
auth.py
-------
The real JWT-based dashboard login flow, extending Phase C's
watcher_token-only auth per main_PHASE_C.py's own comment: "Phase G
should replace/extend this with a real JWT-based flow for the
dashboard's own login, while keeping a long-lived watcher_token for the
local watcher specifically, since a human isn't typing a password into
a background process on every restart."

TWO SEPARATE CREDENTIALS, BY DESIGN:
  - watcher_token (phase_c/backend/models.py's User.watcher_token):
    long-lived, sent as an X-Watcher-Token header by run_watcher.py on
    every request. Never expires, never used by the browser dashboard.
  - JWT (this file): short-lived, issued at /auth/login after verifying
    an email+password, sent as a Bearer token by the React dashboard.
    Never used by the watcher.
  Mixing these would mean either the watcher needs to handle token
  refresh (awkward for a background process) or the dashboard carries a
  never-expiring credential in browser storage (a real security
  downgrade) -- keeping them separate avoids both.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from bridge import phase_c_database, phase_c_models

JWT_SECRET = os.environ.get("JWT_SECRET", "dev_only_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MINUTES = 60 * 12  # 12 hours -- a dashboard session, not a
# permanent credential; the React app re-prompts login after this.

_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    # Matches phase_c/backend/main.py's existing hashing (sha256) for
    # compatibility with users created via Phase C's /users endpoint --
    # NOTE, called out explicitly for a production deployment: sha256
    # alone (no per-user salt, no deliberate slowness) is a Phase
    # C/G-development-stage simplification, not a hardened password
    # scheme. Swap for passlib's bcrypt/argon2 hashing before any real
    # deployment with real user passwords -- the passlib dependency is
    # already in requirements.txt for exactly this follow-up.
    return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MINUTES)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def authenticate_user(db: Session, email: str, password: str):
    user = db.execute(
        select(phase_c_models.User).where(phase_c_models.User.email == email)
    ).scalar_one_or_none()
    if user is None or user.hashed_password != _hash_password(password):
        return None
    return user


def get_current_dashboard_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(phase_c_database.get_db),
):
    """FastAPI dependency for every dashboard (React-facing) endpoint --
    the JWT analogue of main_PHASE_C.py's get_current_user, which stays
    watcher_token-only and is used only by watcher-facing endpoints."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        # User.id is a UUID column (phase_c/backend/models.py's
        # _uuid_pk()); the JWT "sub" claim is necessarily a plain string
        # (create_access_token below stores str(user.id)), so it must be
        # parsed back into a real uuid.UUID before querying -- passing
        # the raw string through to db.get() fails against a UUID(as_uuid=True)
        # column (works with some backends by accident, not reliably).
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(phase_c_models.User, user_uuid)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user
