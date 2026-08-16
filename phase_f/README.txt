================================================================================
PHASE F -- DETERMINISTIC RECOMMENDATION ENGINE
AI Study Invigilator
================================================================================

WHAT THIS IS
  Section 4 Layer 6: "A rules/statistics engine (no LLM) over the
  student's historical TargetBlock summaries: computes trend, verdict
  (on_track / needs_shorter_blocks / needs_scheduled_breaks), and a
  suggested next schedule. This layer is fully auditable and testable
  on its own."

  Pure, deterministic Python -- no PyTorch, no Claude API, no LLM
  anywhere in this folder. Its single output type, RecommendationResult
  (recommendation/engine.py), is the ONLY thing Phase G's LLM layer will
  ever be shown: plain numbers and enum values, never raw features,
  never an ORM row, per the project's "deterministic logic separate
  from LLM output" principle.

FILES IN THIS FOLDER
  recommendation/
    types.py     -- BlockRecord: a plain, ORM-free dataclass mirroring
                     TargetBlock + TargetBlockSummary. Everything in this
                     package except queries.py operates on BlockRecord
                     only, so the core logic has zero database
                     dependency.
    verdict.py    -- compute_single_block_verdict(): the exact
                     thresholds (see its docstring) for on_track /
                     needs_shorter_blocks / needs_scheduled_breaks, given
                     one block's compliance_pct and away_event_count.
    trend.py      -- compute_schedule_recommendation(): requires
                     TREND_WINDOW_SIZE (3) recent blocks AT THE SAME
                     planned_duration_min to unanimously agree before
                     suggesting extend/shrink -- a single session never
                     changes the recommendation.
    queries.py    -- reads phase_c/backend/models.py's REAL ORM classes
                     directly (TargetBlock, TargetBlockSummary) and maps
                     rows into BlockRecord. The only module here that
                     touches a database; takes an already-open
                     SQLAlchemy Session, never manages its own
                     connection.
    engine.py     -- evaluate(): combines verdict.py + trend.py into one
                     RecommendationResult.
  tests/
    test_verdict.py  -- every documented threshold, both sides of every
                        boundary.
    test_trend.py    -- insufficient-history, unanimous-agreement,
                        mixed-trend, and same-duration-filtering cases.
    test_queries.py  -- runs queries.py against phase_c/backend/models.py's
                        REAL schema, using a throwaway in-memory SQLite
                        database (no docker, no live Postgres needed --
                        SQLAlchemy compiles Postgres's UUID column type
                        down to a usable SQLite type automatically).
    test_engine.py   -- confirms evaluate() wires verdict + trend
                        together correctly and that its output is
                        entirely plain values (the LLM-boundary guarantee).

SETUP
    cd phase_f
    python -m venv venv
    venv\Scripts\activate            (Windows)  /  source venv/bin/activate (macOS/Linux)
    pip install -r requirements.txt

RUN THE TESTS
    cd phase_f
    pytest tests/ -v

  WHAT PASSING LOOKS LIKE: all tests green. No database server, no
  docker-compose, no .env needed -- test_queries.py's SQLite fixture is
  entirely self-contained and torn down automatically per test.

OPTIONAL: LIVE POSTGRES CHECK
  test_queries.py intentionally does NOT run against phase_c's actual
  Postgres instance (keeping the automated suite dependency-free, per
  every earlier phase's test discipline). To sanity-check queries.py
  against the real thing once phase_c's backend is running:

    cd ../phase_c/backend
    docker compose up -d
    cd ../../phase_f
    python -c "
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from recommendation.queries import fetch_recent_block_summaries
    engine = create_engine('postgresql+psycopg://<your .env.example connection string>')
    with Session(engine) as s:
        print(fetch_recent_block_summaries(s, user_id='<a real user id>'))
    "

USAGE SKETCH (how Phase G will call this layer)
    from recommendation.queries import fetch_block_record, fetch_recent_block_summaries
    from recommendation.engine import evaluate

    latest = fetch_block_record(session, target_block_id)
    history = fetch_recent_block_summaries(session, latest_user_id, limit=20)
    result = evaluate(latest, history)
    # result (RecommendationResult) is what Phase G's Claude API prompt
    # template serializes -- never `session`, `latest`, or `history`
    # themselves.

NEXT STEP
  Phase G: Claude API integration that phrases exactly this
  RecommendationResult in a configurable tone, plus the React dashboard
  and deployment.
================================================================================
