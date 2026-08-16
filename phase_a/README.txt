================================================================================
PHASE A -- SIGNAL FEASIBILITY CHECK
AI Study Invigilator
================================================================================

WHAT THIS IS
  The very first piece of code for the project. No model training, no
  backend, no database -- just a live webcam window that prints the raw
  signals (head yaw/pitch/roll, eye-openness, look-up-cycle rate,
  posture, phone presence) so you can visually confirm they behave the
  way the project design assumes, before building anything on top of
  them.

  Corresponds to Phase A in 00_MASTER_CODING_PROMPT.txt and Phase A in
  Section 13 (Development Phases / Roadmap) of
  AI_Study_Invigilator_Project.txt.

FILES IN THIS FOLDER
  requirements.txt       -- the 3 packages this phase needs
  download_models.py     -- fetches the 3 pretrained MediaPipe model files
  signal_check.py        -- the live webcam signal-check script itself
  models/               -- created automatically; holds the downloaded
                           .task/.tflite model files (not committed to
                           git -- see .gitignore note below)

--------------------------------------------------------------------------
SETUP (Windows)
--------------------------------------------------------------------------

  1. Open PowerShell or Command Prompt in this folder.

  2. Create and activate a virtual environment (recommended so this
     doesn't collide with any other Python project on your machine):

       python -m venv venv
       venv\Scripts\activate

     Your prompt should now start with (venv).

  3. Install dependencies:

       pip install -r requirements.txt

  4. Download the pretrained models (~15-25 MB total, one-time):

       python download_models.py

     You should see three "[done]" lines. If any fail, the script tells
     you the exact URL to download manually and where to save the file.

  5. Confirm your webcam works and camera permissions are granted:
     Settings -> Privacy & security -> Camera -> ensure both "Let apps
     access your camera" and "Let desktop apps access your camera" are
     turned on.

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------

    python signal_check.py

  A window opens showing your webcam feed with live numbers overlaid.
  Press 'q' with that window focused to quit.

  Work through this checklist, spending ~15-20 seconds on each, and
  watch how the numbers respond:

    [ ] Face the screen normally, as if reading/typing
        -> yaw/pitch near 0, on_screen: True, PHONE DETECTED: False
    [ ] Look down and "write notes" on a real notepad, glancing up
        every few seconds
        -> looking_down flips True while down, look_up_cycle_rate
           should visibly climb as you glance up repeatedly
    [ ] Turn your head to the side and HOLD it (simulating talking to
        someone)
        -> yaw swings past the envelope and STAYS there
    [ ] Glance left/right QUICKLY, returning to center each time
        (simulating a wide-monitor sweep)
        -> yaw swings and immediately returns; on_screen should mostly
           stay True except during the swing itself
    [ ] Close your eyes for 2-3 seconds (thinking pause)
        -> eyes_closed: True briefly, eyes_closed_duration stays low,
           resets to 0 when you open your eyes
    [ ] Close your eyes and hold for 15-20+ seconds (you don't need the
        full 6 minutes to confirm the signal -- just confirm the
        duration counter climbs steadily and doesn't reset on its own)
        -> eyes_closed_duration climbs continuously; text turns red
           once it would cross the 6-minute threshold (won't happen in
           a quick test, but confirms the logic path exists)
    [ ] Lean back out of frame / get up from the desk
        -> face_present: False
    [ ] Hold a phone up clearly in frame for a few seconds
        -> PHONE DETECTED: True (turns red), and back to False once you
           put it down or move it out of frame

--------------------------------------------------------------------------
WHAT COUNTS AS "PASSING" PHASE A
--------------------------------------------------------------------------
  You don't need perfect numbers -- you need each behavior above to
  produce a CLEARLY DIFFERENT, VISUALLY OBVIOUS pattern from the others.
  If, for example, looking_down never seems to trigger even when you're
  clearly looking down at a notepad, or PHONE DETECTED never triggers
  even with a phone held directly in frame, that's a real signal-quality
  issue to fix (usually: lighting, camera angle, or a threshold value in
  signal_check.py that needs adjusting) BEFORE moving to Phase B.

  It's normal and fine if:
    - The default fixed-angle on_screen envelope (+-25 degrees) doesn't
      perfectly match your monitor setup -- that's exactly why Phase B
      replaces it with a per-user calibration step instead of a fixed
      constant.
    - Phone detection is a little slow to trigger (roughly a second or
      so of lag is normal for this model).
    - Numbers jitter slightly frame-to-frame -- they're smoothed over a
      few frames already, but some noise is expected and is handled
      properly starting in Phase B (duration-gating, not just smoothing).

--------------------------------------------------------------------------
NEXT STEP
--------------------------------------------------------------------------
  Once you've confirmed the signals separate cleanly, move to Phase B:
  send the message
    "Phase A looks good, the signals separate cleanly. Start Phase B..."
  (full text in 00_MASTER_CODING_PROMPT.txt) to your AI coding assistant
  to build the real, tested feature-extraction module.

--------------------------------------------------------------------------
.gitignore REMINDER
--------------------------------------------------------------------------
  Add this folder's models/ directory to your project's .gitignore now
  -- these are large binary files you don't want committed:

    watcher/perception/models/*.task
    watcher/perception/models/*.tflite

  (path shown assumes this script's logic gets moved into
  watcher/perception/ per the folder structure in Section 14 of the
  project doc, once Phase C scaffolds the real project layout)
================================================================================
