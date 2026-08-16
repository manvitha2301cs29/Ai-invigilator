"""
models package
---------------
Three temporal classifiers (GRU, LSTM, Transformer) behind one shared
factory (common.build_model) so train.py never branches on architecture
-- see common.py's docstring for why this matters for Section 15's
ablation.
"""

from .common import MODEL_NAMES, TemporalClassifier, build_model

__all__ = ["MODEL_NAMES", "TemporalClassifier", "build_model"]
