"""
test_api_integration.py
--------------------------
End-to-end test of the assembled Phase G app (main.py's `app`, Phase
C's watcher endpoints + this phase's dashboard endpoints together) using
FastAPI's TestClient and a throwaway, file-based SQLite database (via
DATABASE_URL, set before any module in this process imports
phase_c/backend/database.py) -- no live Postgres, no live Claude API
call (the ANTHROPIC_API_KEY-missing fallback path is exercised
deliberately, since that's exactly the environment this test runs in).

WHY A FILE, NOT :memory:, AND NO dependency_overrides/monkeypatching:
phase_c/backend/database.py builds one module-level SQLAlchemy engine
from DATABASE_URL at IMPORT time with no connect_args, and FastAPI runs
each sync path-operation function in a worker-threadpool thread. A
SQLite ":memory:" database is private to the connection that created
it, so different request threads would each see an empty database
unless the engine is patched with StaticPool -- but patching
phase_c_database.SessionLocal (or using app.dependency_overrides) after
this phase's app has already been built was found, empirically and
repeatedly, to make this environment's installed FastAPI version
silently drop newly-included routes (a real, reproduced instability
here, not a design choice). A plain on-disk SQLite file sidesteps both
problems at once: every thread opens its own connection (no
cross-thread object-sharing issue) to the SAME file (so schema/data are
consistently visible across threads), and nothing about the app object
or its dependencies needs to be touched after main.py builds it.

Flow exercised, matching how the real watcher + dashboard interact:
  1. create a user (phase_c's /users)
  2. create + start + push a state event + end a target block, using the
     watcher_token (phase_c's existing watcher-facing endpoints)
  3. log in as that user for a JWT (this phase's /auth/login)
  4. GET the recommendation with the JWT -- confirms Layer 5 (session
     summary), Phase F (verdict/trend), and the LLM-fallback path (no
     API key set) all wire together correctly through one HTTP call.
  5. GET history and confirm the completed block appears.

Run with:
    pytest tests/test_api_integration.py -v
"""

import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JWT_SECRET", "test_secret_do_not_use_in_prod")
os.environ.pop("ANTHROPIC_API_KEY", None)  # force the fallback-report path

# MUST happen before `from bridge import ...` / `import main` below --
# phase_c/backend/database.py reads DATABASE_URL at its own import time
# to build its one module-level engine (see module docstring above for
# why a file, not :memory:, and why this env var rather than a runtime
# patch).
_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), f"phase_g_test_{uuid.uuid4().hex}.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

import pytest
from fastapi.testclient import TestClient

import main as app_main


@pytest.fixture()
def client():
    with TestClient(app_main.app) as c:
        yield c


def _create_user_and_complete_a_block(client: "TestClient") -> tuple[str, str, str]:
    """Returns (email, password, block_id) for a user with one COMPLETED
    target block that has an ENGAGED state event covering the whole
    block. Uses a fresh, uuid-suffixed email per call rather than
    resetting the database between tests -- the on-disk SQLite file
    persists for the whole test session (see module docstring: wiping it
    between tests via a connection-pool reset turned out to be its own
    source of flakiness), so unique emails are what keep tests
    independent instead."""
    email = f"student-{uuid.uuid4().hex}@example.com"
    password = "correct horse battery staple"
    resp = client.post("/users", json={"name": "Test Student", "email": email, "password": password})
    assert resp.status_code == 200, resp.text
    watcher_token = resp.json()["watcher_token"]
    headers = {"X-Watcher-Token": watcher_token}

    resp = client.post(
        "/target-blocks",
        json={
            "planned_start": "2026-08-11T09:00:00Z",
            "planned_end": "2026-08-11T10:00:00Z",
            "planned_duration_min": 60,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    block_id = resp.json()["id"]

    resp = client.post(f"/target-blocks/{block_id}/start", headers=headers)
    assert resp.status_code == 200, resp.text

    resp = client.post(
        f"/target-blocks/{block_id}/state-events",
        json={"timestamp": "2026-08-11T09:00:00Z", "state": "engaged"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text

    resp = client.post(f"/target-blocks/{block_id}/end", headers=headers)
    assert resp.status_code == 200, resp.text

    return email, password, block_id


def test_full_flow_login_and_get_recommendation(client):
    email, password, block_id = _create_user_and_complete_a_block(client)

    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(f"/target-blocks/{block_id}/recommendation", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["block_verdict"] == "on_track"  # full compliance, zero away events
    assert body["planned_duration_min"] == 60
    assert body["compliance_pct"] == pytest.approx(1.0)
    assert "narrative" in body and len(body["narrative"]) > 0
    # no ANTHROPIC_API_KEY in this test env -> deterministic fallback
    # sentence, not an empty/failed response
    assert "Verdict:" in body["narrative"]


def test_recommendation_requires_dashboard_jwt_not_watcher_token(client):
    email, password, block_id = _create_user_and_complete_a_block(client)
    # no Authorization header at all
    resp = client.get(f"/target-blocks/{block_id}/recommendation")
    assert resp.status_code == 401


def test_recommendation_rejects_incomplete_block(client):
    email, password = f"student2-{uuid.uuid4().hex}@example.com", "another good password"
    resp = client.post("/users", json={"name": "Student Two", "email": email, "password": password})
    watcher_token = resp.json()["watcher_token"]
    resp = client.post(
        "/target-blocks",
        json={"planned_start": "2026-08-11T09:00:00Z", "planned_end": "2026-08-11T10:00:00Z", "planned_duration_min": 60},
        headers={"X-Watcher-Token": watcher_token},
    )
    block_id = resp.json()["id"]  # never started/ended -> still SCHEDULED

    resp = client.post("/auth/login", json={"email": email, "password": password})
    token = resp.json()["access_token"]

    resp = client.get(
        f"/target-blocks/{block_id}/recommendation", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 409


def test_login_rejects_wrong_password(client):
    email, password, _ = _create_user_and_complete_a_block(client)
    resp = client.post("/auth/login", json={"email": email, "password": "wrong password"})
    assert resp.status_code == 401


def test_history_includes_completed_block(client):
    email, password, block_id = _create_user_and_complete_a_block(client)
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]

    # trigger summary computation first via the recommendation endpoint
    client.get(f"/target-blocks/{block_id}/recommendation", headers={"Authorization": f"Bearer {token}"})

    resp = client.get("/target-blocks/history", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    history = resp.json()
    assert len(history) == 1
    assert history[0]["target_block_id"] == block_id
    assert history[0]["verdict"] == "on_track"


def test_watcher_endpoints_still_work_unmodified(client):
    """Confirms Phase C's original watcher-facing behavior wasn't
    altered by mounting it inside Phase G's app -- e.g. wrong/missing
    watcher token still 401s exactly as phase_c/tests expects."""
    resp = client.post(
        "/target-blocks",
        json={"planned_start": "2026-08-11T09:00:00Z", "planned_end": "2026-08-11T10:00:00Z", "planned_duration_min": 60},
        headers={"X-Watcher-Token": "not-a-real-token"},
    )
    assert resp.status_code == 401


def test_dashboard_can_plan_a_target_block_via_jwt(client):
    """The planner screen's write path: JWT-authenticated, distinct from
    Phase C's watcher-token-only POST /target-blocks."""
    email = f"planner-{uuid.uuid4().hex}@example.com"
    password = "planner password"
    client.post("/users", json={"name": "Planner", "email": email, "password": password})
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/dashboard/target-blocks",
        json={"planned_start": "2026-08-12T09:00:00Z", "planned_end": "2026-08-12T10:00:00Z", "planned_duration_min": 60},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "scheduled"

    resp = client.get("/dashboard/target-blocks", headers=headers)
    assert resp.status_code == 200, resp.text
    blocks = resp.json()
    assert len(blocks) == 1
    assert blocks[0]["planned_duration_min"] == 60


def test_dashboard_planner_requires_jwt_not_watcher_token(client):
    resp = client.post(
        "/dashboard/target-blocks",
        json={"planned_start": "2026-08-12T09:00:00Z", "planned_end": "2026-08-12T10:00:00Z", "planned_duration_min": 60},
    )
    assert resp.status_code == 401
