"""
reuse.py
--------
The actual mechanics of the reuse decision explained in data/__init__.py:
inserts ../../phase_d onto sys.path (once, idempotently) and re-exports
the Phase D symbols this phase needs. Kept as its own tiny module rather
than inlined in __init__.py so the sys.path manipulation is easy to spot
and audit in one place.

This assumes phase_d/ is a sibling directory of phase_e/ (true for both
this repository's checked-out layout and any clone of it) -- if that
ever stops being true, this is the one place to update.

NAME-COLLISION NOTE: this phase's OWN top-level package is also called
`data` (phase_e/data/, this very package). A plain `sys.path.insert` +
`import data.xxx` would silently resolve to phase_e's own `data` package
(Python caches the partially-initialized `phase_e/data` module under the
bare name "data" in sys.modules the moment this __init__ chain starts
running), NOT phase_d's -- a real bug this file hit during development.
The fix is to load phase_d's `data` package under a distinct name
("phase_d_data") via importlib, rather than relying on a plain import
statement to disambiguate two same-named top-level packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_PHASE_D_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phase_d"))
_PHASE_D_DATA_INIT = os.path.join(_PHASE_D_DIR, "data", "__init__.py")


def _load_phase_d_data():
    if "phase_d_data" in sys.modules:
        return sys.modules["phase_d_data"]
    spec = importlib.util.spec_from_file_location(
        "phase_d_data",
        _PHASE_D_DATA_INIT,
        submodule_search_locations=[os.path.join(_PHASE_D_DIR, "data")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase_d_data"] = module
    spec.loader.exec_module(module)
    return module


_phase_d_data = _load_phase_d_data()

# Submodules not re-exported by phase_d/data/__init__.py itself need an
# explicit import so Python registers them as attributes of the
# already-loaded phase_d_data package.
import importlib  # noqa: E402

_daisee_label_mapping = importlib.import_module("phase_d_data.daisee_label_mapping")
_daisee_loader = importlib.import_module("phase_d_data.daisee_loader")
_self_collected_loader = importlib.import_module("phase_d_data.self_collected_loader")
_synthetic = importlib.import_module("phase_d_data.synthetic")
_types = importlib.import_module("phase_d_data.types")

DaiseeLabels = _daisee_label_mapping.DaiseeLabels
map_daisee_labels = _daisee_label_mapping.map_daisee_labels
load_daisee_dataset = _daisee_loader.load_daisee_dataset
load_self_collected_dataset = _self_collected_loader.load_self_collected_dataset
generate_synthetic_dataset = _synthetic.generate_synthetic_dataset
generate_synthetic_sequence = _synthetic.generate_synthetic_sequence

BEHAVIOR_CLASSES = _types.BEHAVIOR_CLASSES
FEATURE_COLUMNS = _types.FEATURE_COLUMNS
LABEL_TO_INDEX = _types.LABEL_TO_INDEX
NUM_CLASSES = _types.NUM_CLASSES
NUM_FEATURES = _types.NUM_FEATURES
FeatureFrame = _types.FeatureFrame
RawSequence = _types.RawSequence
Window = _types.Window
window_sequence = _types.window_sequence
windows_to_arrays = _types.windows_to_arrays
