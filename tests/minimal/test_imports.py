"""Ensure the trimmed package exports the intended public surface."""

import dspy


def test_top_level_symbols():
    assert hasattr(dspy, "RLM")
    assert hasattr(dspy, "GEPA")
    assert hasattr(dspy, "Predict")
    assert hasattr(dspy, "Evaluate")
    assert hasattr(dspy, "bootstrap_trace_data")
    assert hasattr(dspy, "Teleprompter")


def test_evaluate_metrics_available():
    from dspy.evaluate.metrics import answer_exact_match, normalize_text

    assert callable(answer_exact_match)
    assert callable(normalize_text)
