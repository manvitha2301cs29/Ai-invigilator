"""
test_models.py
----------------
Proves the "shared interface" claim in models/common.py's docstring:
every architecture, built via the same build_model() factory with the
same hyperparameters, accepts the same input shape and returns the same
output shape -- the precondition for a fair Section 15 ablation. Pure
CPU, tiny random tensors, no dataset or training loop involved.

Run with:
    pytest tests/test_models.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import MODEL_NAMES, build_model, count_parameters

NUM_FEATURES = 11
NUM_CLASSES = 3
BATCH = 4
SEQ_LEN = 20


def test_all_models_build_and_produce_correct_output_shape():
    for name in MODEL_NAMES:
        model = build_model(name, NUM_FEATURES, NUM_CLASSES, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES), f"{name}: unexpected output shape {logits.shape}"


def test_all_models_produce_raw_logits_not_probabilities():
    """CrossEntropyLoss expects raw logits -- confirm no model applies a
    softmax internally (outputs should be able to fall outside [0,1] and
    not sum to 1 across the class dim)."""
    for name in MODEL_NAMES:
        model = build_model(name, NUM_FEATURES, NUM_CLASSES, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES) * 10  # large inputs to push logits outside [0,1]
        logits = model(x)
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(BATCH), atol=1e-3), (
            f"{name}: output looks like it may already be a probability distribution"
        )


def test_build_model_rejects_unknown_name():
    try:
        build_model("rnn_that_does_not_exist", NUM_FEATURES, NUM_CLASSES)
        assert False, "expected ValueError for an unknown model name"
    except ValueError:
        pass


def test_gradients_flow_for_every_model():
    """A minimal training-loop smoke test: one forward/backward pass
    should populate gradients on every parameter for every architecture,
    catching e.g. an accidentally-detached tensor."""
    for name in MODEL_NAMES:
        model = build_model(name, NUM_FEATURES, NUM_CLASSES, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)
        y = torch.randint(0, NUM_CLASSES, (BATCH,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing_grad, f"{name}: parameters with no gradient: {missing_grad}"


def test_count_parameters_is_positive_and_varies_with_hidden_size():
    small = build_model("gru", NUM_FEATURES, NUM_CLASSES, hidden_size=8, num_layers=1)
    large = build_model("gru", NUM_FEATURES, NUM_CLASSES, hidden_size=64, num_layers=1)
    assert count_parameters(small) > 0
    assert count_parameters(large) > count_parameters(small)


def test_transformer_rejects_hidden_size_not_divisible_by_heads():
    from models.transformer import TransformerClassifier

    try:
        TransformerClassifier(NUM_FEATURES, NUM_CLASSES, hidden_size=10, num_layers=1, num_heads=4)
        assert False, "expected ValueError: 10 is not divisible by 4"
    except ValueError:
        pass
