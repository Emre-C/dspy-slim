"""Ensure the trimmed package exports the intended public surface."""

import dspy
from dspy.clients import configure_cache


def test_top_level_symbols():
    for name in (
        "Signature",
        "InputField",
        "OutputField",
        "Predict",
        "ChainOfThought",
        "Parallel",
        "ReAct",
        "RLM",
        "GEPA",
        "BetterTogether",
        "LM",
    ):
        assert hasattr(dspy, name)

    for name in (
        "ProgramOfThought",
        "CodeAct",
        "BestOfN",
        "Refine",
        "MultiChainComparison",
        "KNN",
        "majority",
        "Teleprompter",
    ):
        assert not hasattr(dspy, name)


def test_evaluate_metrics_available():
    from dspy.evaluate.metrics import answer_exact_match, normalize_text

    assert callable(answer_exact_match)
    assert callable(normalize_text)


def test_configure_cache_rebinds_top_level_cache_aliases():
    original_cache = dspy.cache
    original_dspy_cache = dspy.DSPY_CACHE
    original_clients_cache = dspy.clients.DSPY_CACHE

    try:
        configure_cache(enable_disk_cache=False, enable_memory_cache=True)

        assert dspy.cache is dspy.DSPY_CACHE
        assert dspy.DSPY_CACHE is dspy.clients.DSPY_CACHE
    finally:
        dspy.cache = original_cache
        dspy.DSPY_CACHE = original_dspy_cache
        dspy.clients.DSPY_CACHE = original_clients_cache
