"""
data package
------------
Dataset-agnostic sequence/windowing core (types.py) plus three loaders
that all produce the same RawSequence objects: synthetic.py (fabricated
data, no external dataset needed), daisee_loader.py (public benchmark,
via daisee_label_mapping.py), and self_collected_loader.py (diary-
protocol CSVs per docs/02_TRAINING_GUIDE.txt Step 2).
"""

from .types import FeatureFrame, RawSequence, Window, window_sequence

__all__ = ["FeatureFrame", "RawSequence", "Window", "window_sequence"]
