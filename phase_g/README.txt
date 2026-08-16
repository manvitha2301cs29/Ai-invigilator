================================================================================
PHASE G -- CLAUDE API INTEGRATION + REACT DASHBOARD + DEPLOYMENT
AI Study Invigilator
================================================================================

WHAT THIS IS
  Section 4 Layer 7 (LLM Reasoning Layer) and the dashboard half of
  Section 4's two-part deployment architecture (Section 12): a Claude
  API integration that phrases Phase F's already-computed
  verdict/numbers in a configurable mentor tone, and a React dashboard
  (target-block planner, live status, post-block summary, history/
  trends) served from a deployed public URL. The watcher (Phase C)
  keeps running exactly as `python run_watcher.py` -- this phase does
  not touch it.

FILES IN THIS FOLDER
  backend/
    bridge.py            -- loads phase_c/backend's main/database/models
                             and phase_f's recommendation package under
                             collision-safe names (importlib, not a
                             plain sys.path import -- see its docstring;
                             this folder ALSO has its own main.py, and
                             phase_c/backend/main.py does a bare
                             `import models`, so naive imports would
                             collide).
    auth.py               -- JWT dashboard login, separate from Phase
                             C's watcher_token (see its docstring for
                             why two credentials, not one).
    session_summary.py    -- Section 4 Layer 5 (Session Analytics): the
                             genuinely new logic in this phase --
                             turns a completed block's raw
                             StateEvent/AwayEvent/BreakEvent/PhoneEvent/
                             SecondPersonEvent rows into the
                             TargetBlockSummary row Phase F reads, then
                             applies Phase F's per-block verdict.
    llm/
      prompt.py            -- SYSTEM_PROMPT + build_prompt(): the exact,
                             enforced input contract -- only a Phase F
                             RecommendationResult (plain numbers/enums)
                             ever reaches the Claude API, never raw
                             features.
      client.py            -- calls the Claude API (model configurable
                             via ANTHROPIC_MODEL); backend-only, per the
                             project's tech stack.
    routes.py              -- /auth/login, /dashboard/target-blocks
                             (planner), /target-blocks/{id}/recommendation,
                             /target-blocks/history,
                             /target-blocks/{id}/live-status (POST, watcher)
                             + /ws/target-blocks/{id}/live-status (WebSocket,
                             dashboard).
    websocket_hub.py       -- in-memory pub/sub fanning live-status
                             pushes out to connected dashboard tabs (see
                             its docstring for the single-instance
                             deployment caveat).
    main.py                -- the deployable entry point; extends Phase
                             C's existing FastAPI app object directly
                             with this phase's routes (see main.py's
                             docstring for why, not app.include_router
                             of a second app -- a real instability found
                             in this environment's FastAPI version while
                             building this phase).
    tests/
      test_prompt.py        -- proves the LLM prompt never contains a
                             raw-feature field name, only
                             RecommendationResult's own fields.
      test_session_summary.py -- Layer 5's worked/distracted/away/break
                             computation and verdict persistence,
                             against phase_c's REAL schema via SQLite.
      test_api_integration.py -- the whole app, end to end, via
                             FastAPI's TestClient (see its docstring for
                             why a file-based SQLite DB, not :memory:
                             with dependency_overrides).
  frontend/
    src/pages/PlannerPage.jsx      -- create + list upcoming target blocks
    src/pages/LiveStatusPage.jsx   -- WebSocket live status view
    src/pages/SummaryPage.jsx      -- end-of-block report + tone picker
    src/pages/HistoryPage.jsx      -- history/trends
    src/pages/LoginPage.jsx        -- sign in / create account (shows
                             the watcher_token once, for pasting into
                             the local watcher's config)
    src/api/client.js       -- fetch wrapper, JWT storage
    src/hooks/useLiveStatus.js -- the live-status WebSocket hook

NON-OBVIOUS DESIGN DECISIONS
  - TWO CREDENTIALS: watcher_token (long-lived, header-based, Phase C)
    for the local watcher; a JWT (short-lived, Bearer, this phase) for
    the browser dashboard. See auth.py's docstring.
  - PLANNER NEEDS ITS OWN ENDPOINT: Phase C's POST /target-blocks is
    watcher-token-only, unusable from the browser. Rather than edit
    Phase C's endpoint (out of scope per the continuation brief) or
    reuse its exact path (would collide), this phase adds
    POST/GET /dashboard/target-blocks, JWT-authenticated, writing the
    exact same TargetBlock row.
  - RECOMMENDATION IS COMPUTED LAZILY: the first GET
    /target-blocks/{id}/recommendation for a completed block computes
    and persists its TargetBlockSummary (Layer 5) if it doesn't exist
    yet; subsequent calls just re-read it. No separate "finalize"
    endpoint needed, and the watcher's client.py is untouched.
  - LLM FAILURE NEVER HIDES THE NUMBERS: if ANTHROPIC_API_KEY is unset
    or the Claude API call fails, /recommendation falls back to a
    plain, deterministic sentence (routes.py's fallback_report()) built
    from the same RecommendationResult -- the numbers and verdict are
    always available even if the phrasing layer isn't.

SETUP -- BACKEND
    cd phase_g/backend
    python -m venv venv
    venv\Scripts\activate            (Windows)  /  source venv/bin/activate (macOS/Linux)
    pip install -r requirements.txt
    copy .env.example to .env, fill in DATABASE_URL (same Postgres
    phase_c uses), ANTHROPIC_API_KEY, JWT_SECRET, FRONTEND_ORIGIN

  RUN THE TESTS FIRST:
    pytest tests/ -v
  WHAT PASSING LOOKS LIKE: all green, no live Postgres or Claude API
  call required (test_api_integration.py uses a throwaway SQLite file;
  test_prompt.py never calls the network; the recommendation endpoint's
  test runs with ANTHROPIC_API_KEY deliberately unset, exercising the
  fallback path).

  RUN THE BACKEND LOCALLY:
    uvicorn main:app --reload --port 8000
  Visit http://localhost:8000/docs -- you should see BOTH Phase C's
  watcher-facing endpoints and this phase's dashboard endpoints on one
  app.

SETUP -- FRONTEND
    cd phase_g/frontend
    npm install
    copy .env.example to .env.local, set VITE_API_URL if not localhost:8000
    npm run dev
  Visit http://localhost:5173 -- create an account (you'll see your
  watcher_token once; paste it into phase_c/watcher's config), then use
  the planner/live/summary/history screens.

DEPLOYING -- PER docs/03_DEPLOYMENT_GUIDE.txt PART 1
  1. POINT AT PRODUCTION POSTGRES: provision a managed Postgres
     instance (Railway/Render both offer this) and set DATABASE_URL on
     the deployed backend -- same schema, no migration needed, since
     Phase C was built against Postgres from the start.
  2. CONTAINERIZE THE BACKEND: create phase_g/backend/Dockerfile:
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . .
        EXPOSE 8000
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
     Note phase_g/backend's requirements.txt is deliberately lean (no
     OpenCV/MediaPipe/PyTorch) -- the backend never touches a camera or
     trains a model.
  3. DEPLOY THE BACKEND: push to GitHub, connect Railway or Render to
     phase_g/backend/Dockerfile, set ANTHROPIC_API_KEY, DATABASE_URL,
     JWT_SECRET, FRONTEND_ORIGIN as environment variables in the
     platform's dashboard. Visit <your-url>/docs to confirm it's live.
  4. BUILD + DEPLOY THE FRONTEND:
        cd phase_g/frontend
        npm run build
     Deploy the resulting dist/ as a static site (Railway/Render
     alongside the backend, or Vercel/Netlify) with
     VITE_API_URL=<your deployed backend URL> set as a build-time env
     var.
  5. HTTPS + CORS: set FRONTEND_ORIGIN on the backend to your deployed
     frontend's actual origin (not localhost) -- main.py's CORS
     middleware reads this directly. Confirm the WebSocket URL upgrades
     to wss:// automatically once served over HTTPS (api/client.js's
     wsUrl() derives the scheme from VITE_API_URL's own protocol,
     specifically to avoid the ws://-vs-wss:// mismatch the deployment
     guide calls out as a common bug).

AT THIS POINT: anyone with your frontend URL can create an account,
plan target blocks, and -- once they run their own local watcher
(phase_c/watcher, pointed at your deployed backend URL, using their own
watcher_token) -- see live status and end-of-block reports. This
satisfies the "everyone with the link can access this" requirement for
the dashboard half of the system per Part 1 of the deployment guide.

NEXT STEP (OPTIONAL)
  Phase H: package phase_c/watcher with PyInstaller as a single
  executable, per docs/03_DEPLOYMENT_GUIDE.txt Part 2 Option B -- only
  worth doing once this phase's watcher + deployed dashboard both work
  end to end.
================================================================================
