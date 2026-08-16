"""
build_exe.py
------------
docs/03_DEPLOYMENT_GUIDE.txt Part 2 Option B: packages phase_c/watcher's
run_watcher.py as a single executable via PyInstaller, so a classmate or
professor can run the watcher without installing Python or any
dependencies themselves.

VALIDATED COMMAND (this is exactly what this script runs; documented
here so it's runnable by hand too if you'd rather not use the script):

    cd phase_c/watcher
    pyinstaller --onefile --name study-invigilator-watcher \\
      --add-data "models:models" run_watcher.py        (macOS/Linux)
    pyinstaller --onefile --name study-invigilator-watcher \\
      --add-data "models;models" run_watcher.py         (Windows --
      PyInstaller's --add-data separator is OS-specific: ':' on
      macOS/Linux, ';' on Windows)

WHY --add-data "models:models" SPECIFICALLY: perception.py resolves its
three MediaPipe model paths relative to its OWN file location
(MODELS_DIR = dirname(dirname(__file__)) + "/models" -- i.e. two levels
up from watcher/perception/perception.py, landing on watcher/models/).
PyInstaller's onefile bundling preserves the perception/ package's
internal structure, so at runtime the frozen perception.py's __file__
still resolves correctly IF (and only if) models/ was added at the
watcher/ ROOT of the bundle, as a sibling of the perception/ package --
exactly what --add-data "models:models" (run FROM the watcher/
directory) produces. This was verified end to end in development: the
packaged executable was run with no camera/backend available and
correctly reached the network-call stage (i.e. every model file loaded
successfully) rather than failing with a "model file not found" error --
see this phase's README for the exact verification transcript.

EXPECTED OUTPUT SIZE: roughly 140-200+ MB (see README for why the range
is wide -- opencv-python vs opencv-python-headless and which MediaPipe
extras get pulled in both affect this meaningfully). This is normal and
expected for a bundled CV/ML app per the deployment guide -- don't treat
a large file size as a build failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_PHASE_H_DIR = os.path.dirname(os.path.abspath(__file__))
_WATCHER_DIR = os.path.abspath(os.path.join(_PHASE_H_DIR, "..", "phase_c", "watcher"))
_EXE_NAME = "study-invigilator-watcher"


def build() -> str:
    if not os.path.isdir(_WATCHER_DIR):
        raise SystemExit(f"expected phase_c/watcher at {_WATCHER_DIR!r} -- run this from the repo root")

    models_dir = os.path.join(_WATCHER_DIR, "models")
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        raise SystemExit(
            f"{models_dir!r} is empty -- run phase_a/download_models.py (or copy phase_a/models/ into "
            "phase_c/watcher/models/, per phase_c/README.txt) before packaging"
        )

    add_data_sep = ";" if sys.platform.startswith("win") else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        _EXE_NAME,
        "--add-data",
        f"models{add_data_sep}models",
        "run_watcher.py",
    ]

    print(f"Running: {' '.join(cmd)}\n(from {_WATCHER_DIR})")
    subprocess.run(cmd, cwd=_WATCHER_DIR, check=True)

    exe_name = _EXE_NAME + (".exe" if sys.platform.startswith("win") else "")
    built_path = os.path.join(_WATCHER_DIR, "dist", exe_name)
    if not os.path.isfile(built_path):
        raise SystemExit(f"PyInstaller reported success but {built_path!r} wasn't found -- check the log above")

    # Copy the final artifact into phase_h/dist/ (this phase's own
    # output location) rather than leaving it under phase_c/watcher/dist/
    # -- keeps phase_c's folder exactly as Phase C left it, per the
    # "own folder per phase" convention, and gives the dashboard's
    # "Start Monitoring" download link one stable place to point at.
    output_dir = os.path.join(_PHASE_H_DIR, "dist")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, exe_name)
    shutil.copy2(built_path, final_path)

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\nBuilt {final_path} ({size_mb:.0f} MB)")
    return final_path


if __name__ == "__main__":
    build()
