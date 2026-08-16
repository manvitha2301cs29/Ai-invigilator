================================================================================
PHASE D -- TEMPORAL ENGAGEMENT CLASSIFIER (GRU / LSTM / TRANSFORMER)
AI Study Invigilator
================================================================================

WHAT THIS IS
  A PyTorch training pipeline for the temporal classifier that will
  eventually replace/augment Phase C's rule-based state engine: a model
  that looks at a WINDOW of consecutive FeatureVectors (not one frame)
  and predicts one of Engaged / Idle-present / Distracted-present.
  "Away" is deliberately excluded -- it stays a deterministic rule in
  Phase C's monitor.py (Section 4 Layer 3), since it's already reliably
  detectable without a model.

  Corresponds to Phase D of the continuation brief and Section 9/10/15
  (Deep Learning Components / Dataset Strategy / Evaluation Methodology)
  of docs/AI_Study_Invigilator_Project.txt.

  Three architectures share one training script and one CLI so the
  comparison between them (Section 15's ablation) is fair: only
  --model changes between runs, nothing else.

FILES IN THIS FOLDER
  data/
    types.py                  -- FeatureFrame / RawSequence / Window /
                                  window_sequence(): the dataset-agnostic
                                  core every loader below converts INTO.
    synthetic.py               -- fabricated data generator; makes the
                                  whole pipeline testable with no real
                                  dataset (three classes, deliberately
                                  easy to separate -- a correctness
                                  fixture, NOT a benchmark to report).
    daisee_label_mapping.py    -- documented DAiSEE (4-scale ordinal) ->
                                  this project's 3-class mapping.
    daisee_loader.py           -- reads DAiSEE's Labels CSV + per-clip
                                  feature CSVs (produced by
                                  experiments/extract_daisee_features.py)
                                  into RawSequences.
    self_collected_loader.py   -- reads diary-protocol CSVs (schema in
                                  docs/02_TRAINING_GUIDE.txt Step 2) into
                                  RawSequences, grouped by session_id.
  models/
    common.py                  -- build_model() factory + shared
                                  (batch, seq_len, features) ->
                                  (batch, classes) interface.
    gru.py / lstm.py / transformer.py -- the three architectures.
  experiments/
    train_temporal_model.py    -- the CLI entry point (see below).
    export_model.py            -- TorchScript/ONNX export for the watcher.
    extract_daisee_features.py -- offline video -> per-clip-CSV script,
                                  reuses phase_c/watcher/perception/
                                  perception.py directly (NOT reimplemented).
  tests/
    test_dataset.py            -- windowing, synthetic generator, DAiSEE
                                  mapping, both loaders (fixture CSVs).
    test_models.py             -- shared-interface shape/gradient checks
                                  for all three architectures.
    test_train_synthetic.py    -- full pipeline, end to end, on synthetic
                                  data (the "testable before any real
                                  dataset exists" requirement).

SETUP
    cd phase_d
    python -m venv venv
    venv\Scripts\activate            (Windows)  /  source venv/bin/activate (macOS/Linux)
    pip install -r requirements.txt

  extract_daisee_features.py additionally needs opencv-python and
  mediapipe (same versions as phase_a/phase_c/requirements.txt) --
  intentionally NOT in this folder's requirements.txt, since most of
  Phase D (training on synthetic or self_collected data, all tests)
  never touches OpenCV or a video file.

RUN THE TESTS FIRST (matches every earlier phase's workflow)
    cd phase_d
    pytest tests/ -v

  WHAT PASSING LOOKS LIKE: all tests green, in particular
  test_train_synthetic.py's test_gru_reaches_high_accuracy_on_synthetic_data
  (>= 90% accuracy) -- this is the concrete "pipeline works end to end"
  proof the continuation brief asked for, with zero external data.

TRAINING ON A REAL DATASET
  1. DAiSEE:
       a. Download DAiSEE per docs/02_TRAINING_GUIDE.txt Step 1 (external
          to this repo -- follow the dataset maintainers' current access
          process).
       b. Extract features once:
            python experiments/extract_daisee_features.py \
              --videos-dir /path/to/DAiSEE/Videos \
              --output-dir experiments/daisee_features/
       c. Train:
            python experiments/train_temporal_model.py \
              --model gru --dataset daisee --window-size 90 --epochs 30 \
              --daisee-labels-csv /path/to/DAiSEE/Labels/AllLabels.csv \
              --daisee-features-dir experiments/daisee_features/ \
              --output experiments/runs/gru_daisee_w90/

  2. Self-collected (diary protocol, docs/02_TRAINING_GUIDE.txt Step 2):
            python experiments/train_temporal_model.py \
              --model gru --dataset self_collected --window-size 90 \
              --epochs 30 \
              --self-collected-csv data/my_session_1.csv \
              --self-collected-csv data/my_session_2.csv \
              --output experiments/runs/gru_self_collected_w90/

  Repeat with --model lstm and --model transformer, everything else
  identical, for the Section 15 ablation. Then sweep --window-size on
  whichever architecture won. Every run writes config.json,
  metrics.json, confusion_matrix.png, and best.pt into --output.

  SANITY CHECK (per the training guide): compare every run's accuracy
  against Phase C's rule-based baseline on the same held-out data before
  trusting the result.

EXPORTING FOR THE WATCHER
    python experiments/export_model.py \
      --checkpoint experiments/runs/gru_daisee_w90/best.pt \
      --format torchscript \
      --output ../phase_c/watcher/models/temporal_gru.pt

  (Wiring the exported model into run_watcher.py's live loop is future
  work beyond Phase D's scope -- Phase C's monitor still runs on the
  rule-based baseline until that integration happens.)

NEXT STEP
  Once pytest passes on synthetic data (real-dataset training can happen
  on your own timeline separately), move to Phase E -- the fatigue-
  pattern autoencoder, which reuses this phase's RawSequence/windowing
  code directly rather than reimplementing it.
================================================================================
