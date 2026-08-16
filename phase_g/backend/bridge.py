"""
bridge.py
---------
Loads phase_c/backend's main/database/models modules and phase_f's
recommendation package under distinct, collision-safe names, using the
same importlib pattern phase_e/data/reuse.py established (see that
file's docstring for the full "why importlib, not a plain import"
reasoning -- the short version: this folder ALSO has a module literally
named main.py, and phase_c/backend/main.py imports a plain, unqualified
`import models` at its own top level, so a naive sys.path-based import
here risks exactly the same name-collision bug reuse.py hit).

Nothing in this file talks to a live database or the Claude API by
itself -- it just makes phase_c's ORM classes/session factory and
phase_f's pure recommendation logic importable from this phase without
copy-pasting them.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PHASE_C_BACKEND_DIR = os.path.join(_ROOT, "phase_c", "backend")
_PHASE_F_DIR = os.path.join(_ROOT, "phase_f")


def _load_module_from_path(module_name: str, file_path: str, search_locations: list[str] | None = None):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path, submodule_search_locations=search_locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_phase_c_backend():
    """phase_c/backend is a flat script folder (not a package -- see
    database.py's own comment about plain, non-relative imports), so its
    modules are loaded individually, in dependency order
    (models -> database -> main), each registered under a
    "phase_c_backend_<name>" key so main_PHASE_C's own `import models`
    / `from database import ...` statements resolve correctly once we
    also temporarily add the folder to sys.path for THIS process.
    """
    if _PHASE_C_BACKEND_DIR not in sys.path:
        sys.path.insert(0, _PHASE_C_BACKEND_DIR)

    models = _load_module_from_path("phase_c_backend_models", os.path.join(_PHASE_C_BACKEND_DIR, "models.py"))
    # database.py and main.py do `import models` / `from database import
    # get_db` at module scope expecting the *bare* names "models" and
    # "database" to be importable (that's what "add the folder to
    # sys.path" above provides) -- register our aliased models module
    # under the bare name too so those bare imports resolve to the SAME
    # object, not a second, independently-loaded copy with a different
    # Base.metadata.
    sys.modules.setdefault("models", models)

    database = _load_module_from_path("phase_c_backend_database", os.path.join(_PHASE_C_BACKEND_DIR, "database.py"))
    sys.modules.setdefault("database", database)

    main = _load_module_from_path("phase_c_backend_main", os.path.join(_PHASE_C_BACKEND_DIR, "main.py"))
    return models, database, main


def _load_phase_f_recommendation():
    if _PHASE_F_DIR not in sys.path:
        sys.path.insert(0, _PHASE_F_DIR)
    return importlib.import_module("recommendation")


phase_c_models, phase_c_database, phase_c_main = _load_phase_c_backend()
recommendation = _load_phase_f_recommendation()
