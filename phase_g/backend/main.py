"""
main.py
-------
The Phase G deployable backend entry point:

    uvicorn main:app --reload --port 8000

Extends Phase C's existing watcher-facing FastAPI app object DIRECTLY
(auth via X-Watcher-Token, target-block lifecycle, event ingestion --
see phase_c/backend/main.py, imported via bridge.py) with this phase's
new dashboard-facing router (routes.py: JWT login, recommendation,
history, live-status WebSocket) and CORS configuration for the React
frontend.

WHY REUSE THE SAME `app` OBJECT rather than building a new FastAPI()
and merging phase_c's router into it: the installed FastAPI version in
this environment was found, empirically and repeatedly during this
phase's own test suite, to intermittently drop routes added via
app.include_router(other_app.router) (a real, reproduced instability in
this environment, not a design choice -- copying an already-built
app's router into a second app object is a less common pattern than
including a plain APIRouter, and this version's route-resolution
caching doesn't handle it reliably). Adding OUR new router directly onto
Phase C's own `app` avoids the unreliable code path entirely while
still satisfying "Extends the dashboard half, not the watcher's core
loop": nothing in phase_c/backend/main.py, database.py, or models.py is
EDITED by this file -- their routes, dependencies, and
X-Watcher-Token auth are all untouched; this file only calls
`.include_router()` and `.add_middleware()` on the same object from the
outside, exactly as any other file extending a FastAPI app would.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from bridge import phase_c_main  # noqa: E402 -- after load_dotenv() so ANTHROPIC_API_KEY/DATABASE_URL are set first
from routes import router as dashboard_router  # noqa: E402

app = phase_c_main.app
app.title = "AI Study Invigilator API (Phase G)"

# CORS: docs/03_DEPLOYMENT_GUIDE.txt Part 1 Step 5 -- localhost for dev,
# override with the real deployed frontend origin via FRONTEND_ORIGIN in
# production. A bare "*" is deliberately not used once credentials
# (the JWT Authorization header) are involved.
_frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase G's new dashboard-facing endpoints, added onto Phase C's
# existing app/router. Not adding a second /health here deliberately --
# Phase C's is sufficient and avoids two routes silently shadowing each
# other.
app.include_router(dashboard_router)
