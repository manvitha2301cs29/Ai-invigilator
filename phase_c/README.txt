================================================================================
PHASE C -- RULE-BASED BASELINE, LOCAL WATCHER, BACKEND, DATABASE
AI Study Invigilator
================================================================================

WHAT THIS IS
  The first end-to-end working version of the system: a real local
  watcher process that owns your webcam, runs Phase B's feature
  extraction and a rule-based heuristic classifier (the baseline Phase
  D's trained model will later need to beat), tracks the away-stopwatch
  and phone/second-person warnings via a real duration-gated state
  machine, and reports everything to a FastAPI backend backed by real
  PostgreSQL -- not SQLite, since this project is being built for
  production deployment from the start (see PRODUCTION DATABASE NOTE
  below).

  Corresponds to Phase C in 00_MASTER_CODING_PROMPT.txt, and Layers 4
  (Live Monitoring), 5 (Session Analytics, partial), and 8 (Dashboard &
  Storage, backend half only) in Section 4 of
  AI_Study_Invigilator_Project.txt.

PRODUCTION DATABASE NOTE
  This phase uses PostgreSQL, run locally via Docker Desktop, instead
  of SQLite. Why: SQLite is a single-file, single-writer database that
  doesn't hold up behind a public, multi-user backend -- which is the
  actual deployment target for this project (per the "others can use
  it" / shortlisting goal). Building against Postgres from Phase C
  means what you test locally is architecturally identical to what
  gets deployed later (03_DEPLOYMENT_GUIDE.txt Part 1) -- no
  SQLite-to-Postgres migration surprises down the line.

FILES IN THIS PHASE
  backend/
    models_PHASE_C.py          -- SQLAlchemy 2.x models (8 tables,
                                   matches Section 8's data model exactly)
    database_PHASE_C.py        -- Postgres engine/session setup
    main_PHASE_C.py            -- FastAPI app: auth, target-block
                                   lifecycle, event ingestion endpoints
    docker-compose_PHASE_C.yml -- local Postgres via Docker Desktop
    .env.example_PHASE_C       -- copy to .env, fill in real values
    requirements_PHASE_C.txt

  watcher/
    run_watcher_PHASE_C.py     -- the actual entry point; run this
    client_PHASE_C.py          -- HTTP client to the backend
    perception/
      perception_PHASE_C.py    -- MediaPipe wrapper (productionized
                                   from Phase A's signal_check_PHASE_A.py)
    features/
      types_PHASE_C.py         -- carried forward from Phase B
      extractor_PHASE_C.py     -- carried forward from Phase B
    monitoring/
      monitor_PHASE_C.py       -- away-stopwatch, phone/second-person
                                   warning state machine (NEW this phase)
    requirements_PHASE_C.txt

  tests/
    test_monitor_PHASE_C.py    -- 15 unit tests for the monitor state
                                   machine, same discipline as Phase B

--------------------------------------------------------------------------
SETUP (Windows) -- BACKEND
--------------------------------------------------------------------------

  1. Start Docker Desktop (you confirmed it's already installed).

  2. From the backend/ folder, start Postgres:

       docker compose -f docker-compose.yml up -d

     Confirm it's healthy:

       docker compose -f docker-compose.yml ps

     You should see "healthy" in the STATUS column after a few seconds.

  3. Create your venv and install backend dependencies:

       python -m venv venv
       venv\Scripts\activate
       pip install -r requirements_PHASE_C.txt

  4. Copy .env.example_PHASE_C to .env in the same folder, and fill in
     a real JWT_SECRET (the default DATABASE_URL already matches
     docker-compose_PHASE_C.yml, no changes needed there for local dev).

  5. Start the backend:

       uvicorn main_PHASE_C:app --reload --port 8000

     Visit http://localhost:8000/docs -- you should see the interactive
     API docs with 11 endpoints (users, target-blocks, state-events,
     away-events, phone-events, second-person-events, break-events,
     health).

  6. Create a test user (via the /docs UI, or curl):

       curl -X POST http://localhost:8000/users -H "Content-Type: application/json" -d "{\"name\":\"Kyra\",\"email\":\"kyra@example.com\",\"password\":\"testpass123\"}"

     The response includes a "watcher_token" -- copy it, you'll need it
     to run the watcher.

--------------------------------------------------------------------------
SETUP (Windows) -- WATCHER
--------------------------------------------------------------------------

  1. In a SEPARATE terminal/venv (or reuse the Phase A one, since it
     already has opencv/mediapipe installed):

       cd watcher
       python -m venv venv
       venv\Scripts\activate
       pip install -r requirements_PHASE_C.txt

  2. Copy your Phase A models/ folder into watcher/models/ (the same 3
     .task/.tflite files download_models_PHASE_A.py fetched -- no need
     to re-download, just copy the folder):

       xcopy /E /I ..\..\phase_a\models watcher\models

     (adjust the path above to wherever your Phase A models/ folder
     actually is)

--------------------------------------------------------------------------
RUNNING IT END TO END
--------------------------------------------------------------------------

  With the backend running (uvicorn, from the steps above) and Postgres
  up, run the watcher:

    python run_watcher_PHASE_C.py --backend-url http://localhost:8000 --token YOUR_WATCHER_TOKEN --name Kyra --duration-min 5

  (use a short --duration-min like 5 for your first test run, not the
  full 90 -- you can always run it again for a real session once you've
  confirmed it works)

  You should see:
    - "[watcher] target block created: ..."
    - "[watcher] monitoring started for Kyra..."
    - A live status line printed roughly every 1.5 seconds, e.g.:
        [engaged            ] face=True on_screen=True looking_down=False
        eyes_closed=False phone=False 2nd_person=False break=False

  Try:
    - Leaning out of frame for 30+ seconds -> a SOFT away notification
      should print, then a FIRM one past 2 minutes
    - Typing 'b' + Enter -> toggles a declared break (away tracking
      pauses while on a break)
    - Holding a phone up for a few seconds -> a DIRECT "Don't use your
      phone, Kyra." notification should print
    - Typing 'q' + Enter -> ends the block early and exits cleanly

  CONFIRM DATA ACTUALLY LANDED IN POSTGRES:
    Visit http://localhost:8000/docs, use the GET
    /target-blocks/{block_id}/summary-debug endpoint (paste in your
    block_id from the watcher's startup log and your watcher_token as
    the x-watcher-token header) -- you should see non-zero
    state_event_count, and away/phone counts matching whatever you
    triggered during your test run.

  Alternatively, inspect Postgres directly:

    docker exec -it study_invigilator_db psql -U invigilator -d study_invigilator -c "SELECT state, count(*) FROM state_events GROUP BY state;"

--------------------------------------------------------------------------
WHAT "PASSING" PHASE C LOOKS LIKE
--------------------------------------------------------------------------
  - The watcher runs continuously without crashing for at least a few
    minutes.
  - Away detection correctly starts a stopwatch when you leave frame and
    closes it when you return, visible both in the console output and
    in the away_events table.
  - Phone and second-person warnings fire once, not repeatedly, while
    the condition persists, and stop firing once it ends.
  - A declared break ('b') suppresses away notifications while active.
  - Every state_event, away_event, phone_event, and second_person_event
    generated during your test run is actually present in Postgres
    afterward -- not just printed to the console.

--------------------------------------------------------------------------
WHAT'S DELIBERATELY NOT DONE YET (later phases)
--------------------------------------------------------------------------
  - The state classification (_heuristic_state in run_watcher_PHASE_C.py)
    is a simple RULE-BASED baseline, not a trained model. Phase D trains
    the real GRU/LSTM/Transformer classifier and should be evaluated
    against this baseline (Section 15's evaluation methodology).
  - Notifications print to the console, not real OS desktop
    notifications. That's a Phase G concern.
  - Break declaration is a CLI 'b'/'q' keypress, not a real UI. Also
    Phase G.
  - Live in-progress status isn't streamed anywhere except this
    console -- the WebSocket-based live dashboard view is Phase G.
  - Gaze calibration uses Phase A's tuned defaults for everyone, not a
    real per-user calibration step. Also Phase G (or an earlier
    dedicated calibration script if you want it sooner).

--------------------------------------------------------------------------
NEXT STEP
--------------------------------------------------------------------------
  Once Phase C runs end-to-end and you've confirmed data lands in
  Postgres correctly, move to Phase D: the PyTorch training pipeline for
  the real temporal classifier, trained on DAiSEE and your own
  self-collected data. Send the Phase D follow-up prompt (full text in
  00_MASTER_CODING_PROMPT.txt) to continue.
================================================================================
