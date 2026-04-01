"""Ensure the trimmed package exports the intended public surface."""

import dspy


def test_top_level_symbols():
    for name in (
        "Signature",
        "InputField",
        "OutputField",
        "Predict",
        "ChainOfThought",
        "Parallel",
        "RLM",
        "GEPA",
        "LM",
    ):
        assert hasattr(dspy, name)

    for name in (
        "ReAct",
        "ProgramOfThought",
        "CodeAct",
        "BestOfN",
        "Refine",
        "MultiChainComparison",
        "KNN",
        "majority",
    ):
        assert not hasattr(dspy, name)


def test_evaluate_metrics_available():
    from dspy.evaluate.metrics import answer_exact_match, normalize_text

    assert callable(answer_exact_match)
    assert callable(normalize_text)
