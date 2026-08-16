================================================================================
AI STUDY INVIGILATOR -- PROJECT LAYOUT
================================================================================

This repo is organized by development phase. Each phase lives in its own
folder with clean file names (no _PHASE_X suffixes needed anymore).

  docs/          Shared project documentation and coding prompts
  phase_a/       Signal feasibility check (live webcam, MediaPipe)
  phase_b/       Feature extraction module (unit-tested rules engine)
  phase_c/       End-to-end baseline (FastAPI backend + local watcher)
  venv/          Shared Python virtual environment (optional; each phase
                 can also use its own venv if you prefer)

--------------------------------------------------------------------------
QUICK START BY PHASE
--------------------------------------------------------------------------

PHASE A -- Webcam signal check
  cd phase_a
  pip install -r requirements.txt
  python download_models.py
  python signal_check.py

PHASE B -- Feature extraction tests
  cd phase_b
  pip install -r requirements.txt
  pytest tests/test_extractor.py -v

PHASE C -- Full end-to-end system
  See phase_c/README.txt for backend (Postgres + FastAPI) and watcher setup.

  Before running the watcher, copy MediaPipe models from Phase A:
    xcopy /E /I phase_a\models phase_c\watcher\models

--------------------------------------------------------------------------
DOCUMENTATION
--------------------------------------------------------------------------

  docs/00_MASTER_CODING_PROMPT.txt   -- build order and follow-up prompts
  docs/AI_Study_Invigilator_Project.txt
  docs/02_TRAINING_GUIDE.txt
  docs/03_DEPLOYMENT_GUIDE.txt

Each phase folder also has its own README.txt with detailed setup steps.
================================================================================
