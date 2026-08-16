================================================================================
PHASE H (OPTIONAL) -- PACKAGING THE WATCHER AS A SINGLE EXECUTABLE
AI Study Invigilator
================================================================================

WHAT THIS IS
  docs/03_DEPLOYMENT_GUIDE.txt Part 2, Option B: bundles
  phase_c/watcher/run_watcher.py and its dependencies (OpenCV,
  MediaPipe, the three model files) into one standalone executable via
  PyInstaller, so a classmate or professor can try the watcher without
  installing Python. Per Part 3 of the deployment guide, this is
  explicitly optional polish -- only worth doing once Phase G's
  plain-Python watcher and deployed dashboard already work end to end.

  Nothing in phase_c/ is modified by this phase -- packaging is
  entirely additive, matching every other phase's "own folder, don't
  touch earlier phases' files" convention.

FILES IN THIS FOLDER
  build_exe.py   -- runs PyInstaller with the exact, verified command
                    (see its docstring for why the --add-data path
                    matters), copies the result into phase_h/dist/.
  requirements.txt -- PyInstaller itself (install phase_c/watcher's own
                    requirements.txt first).

BEFORE YOU BUILD
  Confirm Phase C's watcher already runs via plain
  `python run_watcher.py --backend-url ... --token ...` on your machine
  -- packaging a watcher that doesn't already work standalone just
  bakes the same bug into a much-slower-to-debug binary.

EXACT COMMAND
    cd phase_c/watcher
    pip install -r requirements.txt
    pip install pyinstaller
    pyinstaller --onefile --name study-invigilator-watcher \
      --add-data "models:models" run_watcher.py

  On Windows, PyInstaller's --add-data separator is ';' instead of ':':
    pyinstaller --onefile --name study-invigilator-watcher ^
      --add-data "models;models" run_watcher.py

  Or, from this folder, run the wrapper script (picks the right
  separator for you and copies the result here):
    cd phase_h
    pip install -r requirements.txt
    python build_exe.py

WHY --add-data "models:models" SPECIFICALLY
  perception.py resolves its three MediaPipe model files relative to
  its own location on disk (two directories up from
  watcher/perception/perception.py, landing on watcher/models/).
  PyInstaller's onefile bundling preserves the perception/ package's
  internal folder structure, so the model files need to be added at the
  watcher/ ROOT of the bundle -- a sibling of perception/, not nested
  inside it -- which is exactly what running --add-data "models:models"
  FROM the watcher/ directory produces. Get this wrong (e.g. add models/
  from the wrong working directory) and the packaged executable will
  fail immediately with "models directory not found... copy the
  models/ folder into watcher/models/" -- perception.py's own,
  already-existing error message (see its FileNotFoundError branch) --
  which makes this mistake easy to catch quickly if it happens.

VERIFIED (this exact command, run end to end during development)
  The build completed successfully and produced a working executable.
  Running it with no camera and no live backend available:
    ./dist/study-invigilator-watcher --backend-url http://localhost:9999 --token fake-token
  correctly proceeded PAST all three model files loading (i.e. the
  bundled MediaPipe models were found and loaded without error) and
  failed only at the expected point -- the network call to a backend
  that wasn't actually running:
    [watcher] creating target block: 90 min, ...
    [client] WARNING: POST /target-blocks failed: ... Connection refused
    [watcher] FATAL: could not create target block on backend. Check
    --backend-url and --token, and that the backend is running.
  This confirms the --add-data bundling is correct: a model-loading
  failure would have shown a DIFFERENT, earlier error
  ("models directory not found...") before ever reaching the network
  call. A full camera-attached end-to-end run should still be tested on
  a clean/second machine before wide distribution, per the deployment
  guide's own advice -- this verification covers packaging correctness,
  not webcam behavior, which needs real hardware to confirm.

EXPECTED OUTPUT SIZE
  Roughly 140-200+ MB, one file, at phase_h/dist/study-invigilator-watcher
  (or .exe on Windows). This is normal and expected for a bundled
  OpenCV + MediaPipe application -- don't treat the size as a build
  failure. The exact size varies with which OpenCV variant gets bundled
  (opencv-python vs the smaller opencv-python-headless) and platform.

DISTRIBUTING IT FROM THE DASHBOARD'S "START MONITORING" PAGE
  Add a short section (copy below) to the dashboard's "Start
  Monitoring" / onboarding page (phase_g/frontend), linking directly to
  the built executable -- either uploaded as a GitHub Release asset for
  your repository, or hosted anywhere with a stable direct-download URL
  (a 150+ MB file does not belong committed to git; .gitignore already
  excludes *.pt/*.tflite/*.task style binaries at the repo root -- add
  phase_h/dist/ there too before committing anything else in this
  folder).

  --- SUGGESTED "START MONITORING" PAGE COPY ---
    1. Download the watcher for your platform: [Download for
       Windows/Mac/Linux] (link to the GitHub Release asset)
    2. Run it from a terminal (or double-click on Windows) with your
       account's watcher token, shown below:

         study-invigilator-watcher --backend-url <this dashboard's URL> --token <your watcher_token>

    3. Keep the terminal window open while you study -- the watcher
       needs to keep running for the duration of your target block.
    4. Come back here when you're done to see your end-of-block report.

    Your watcher_token was shown once when you created your account. If
    you've lost it, [contact support / regenerate it in account
    settings] (implement per your own account-settings design -- not
    built in this phase).
  --- END SUGGESTED COPY ---

NEXT STEP
  None -- Phase H is the last phase in the build order. At this point:
  Phase D's trained classifier, Phase E's fatigue autoencoder, Phase F's
  recommendation engine, Phase G's LLM-integrated, deployed dashboard,
  and (optionally) this phase's packaged watcher together complete the
  system described in docs/AI_Study_Invigilator_Project.txt.
================================================================================
