================================================================================
PHASE E -- FATIGUE-PATTERN AUTOENCODER (ANOMALY DETECTION)
AI Study Invigilator
================================================================================

WHAT THIS IS
  Implements Section 5's FATIGUED-PATTERN: "Rising idle/distracted
  frequency + posture decline over the last N minutes RELATIVE TO THE
  STUDENT'S OWN SESSION BASELINE. Produced by an anomaly / reconstruction
  model (autoencoder), not a hard classifier -- explicitly comparative,
  never an absolute claim about tiredness."

  A GRU sequence-autoencoder is trained to reconstruct a student's own
  feature-window history; at inference time, a window's reconstruction
  error is the anomaly score. High error relative to that student's OWN
  training-time baseline (never a population average) is what a later
  phase reads as "this looks like a drift from this student's normal
  pattern" -- this phase produces the NUMBER only; wording it as a
  student-facing statement is Phase F/G's job, never this one's.

REUSE DECISION (stated per the continuation brief's instruction)
  This phase IMPORTS phase_d/data's RawSequence, FeatureFrame, and
  labeled window_sequence directly via a sys.path insert (see
  data/reuse.py), rather than copying those files. Reasoning is in
  data/__init__.py's docstring: the windowing/schema logic must never
  silently drift between phases, and both phases are developed together
  in one repo.

  The ONE thing genuinely new here (not reused) is
  data/windowing.py's window_sequence_unlabeled(): Phase D's
  window_sequence() requires a majority-vote label and drops unlabeled
  frames, which is correct for Phase D's supervised classifier but wrong
  for this phase's unsupervised autoencoder. See that file's docstring
  for the full reasoning.

FILES IN THIS FOLDER
  data/
    reuse.py                 -- sys.path bridge into phase_d/data.
    windowing.py              -- window_sequence_unlabeled(): the one new
                                 piece of windowing code this phase needed.
    participants.py           -- groups RawSequences by participant_id
                                 (read directly from diary-protocol CSVs,
                                 since Phase D's RawSequence intentionally
                                 doesn't carry it -- see docstring).
    synthetic_fatigue.py      -- fabricated normal + drifting-toward-
                                 fatigued sessions, for testing before any
                                 real per-student history exists.
  models/
    autoencoder.py             -- SequenceAutoencoder (GRU encoder/decoder)
                                 + reconstruction_error().
  experiments/
    train_autoencoder.py       -- CLI: --mode pooled / per_participant /
                                 pooled_finetuned (see file docstring).
  tests/
    test_autoencoder.py        -- the core claim: reconstruction error is
                                 clearly higher on a fabricated fatigued
                                 session tail than on the same session's
                                 normal-baseline start, and stays low on a
                                 held-out normal session.
    test_pipeline.py           -- unlabeled windowing, participant
                                 grouping, and all three --mode values run
                                 end to end on synthetic data.

SETUP
    cd phase_e
    python -m venv venv
    venv\Scripts\activate            (Windows)  /  source venv/bin/activate (macOS/Linux)
    pip install -r requirements.txt

RUN THE TESTS
    cd phase_e
    pytest tests/ -v

  WHAT PASSING LOOKS LIKE: all tests green, in particular
  test_reconstruction_error_higher_on_anomalous_tail_of_drifting_session
  -- reconstruction error on the fabricated "fatigued" tail should be at
  least 1.5x the error on the normal-baseline portion of the same
  fabricated session. This is the "prove it before any real dataset is
  involved" requirement from the continuation brief.

TRAINING ON REAL SELF-COLLECTED DATA
    python experiments/train_autoencoder.py \
      --dataset self_collected --mode per_participant \
      --self-collected-csv ../phase_d/data/my_session_1.csv \
      --self-collected-csv ../phase_d/data/my_session_2.csv \
      --output experiments/runs/autoencoder_v1/

  Needs enough per-participant session history to be worth training on
  independently -- for a brand-new student with only one or two
  sessions, use --mode pooled_finetuned instead so they start from a
  sensible pooled prior rather than an undertrained from-scratch model.

  Each participant's experiments/runs/.../<participant_id>/metrics.json
  records baseline_reconstruction_error_mean/std from their own training
  windows -- THIS is "the student's own session baseline" as a concrete
  number that a later phase can compare a live session's reconstruction
  error against (e.g. flag when a live window's error exceeds
  mean + k*std, entirely deterministic, no LLM involved in that
  comparison).

NEXT STEP
  Phase F: the deterministic recommendation engine, reading
  TargetBlockSummary history from Postgres (phase_c/backend/models.py's
  real schema) -- pure trend/verdict logic, no LLM, no model inference.
================================================================================
