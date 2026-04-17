import pickle

import cloudpickle
import pytest

import dspy
from dspy.clients.cache import Cache


def test_settings_save_and_load_round_trip_supported_defaults(tmp_path):
    settings_path = tmp_path / "settings.pkl"

    with dspy.context(branch_idx=7, rm={"kind": "stub-retriever"}, callbacks=["excluded-callback"]):
        dspy.settings.save(str(settings_path), exclude_keys=["callbacks"])

    loaded = dspy.settings.load(str(settings_path))

    assert loaded["branch_idx"] == 7
    assert loaded["rm"] == {"kind": "stub-retriever"}
    assert "callbacks" not in loaded


def test_cache_memory_persistence_round_trip(tmp_path):
    cache_path = tmp_path / "memory-cache.pkl"
    request = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    value = {"answer": "hello"}

    cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path / "disk-cache"),
    )
    cache.put(request, value)
    cache.save_memory_cache(str(cache_path))
    cache.reset_memory_cache()

    assert cache.get(request) is None

    cache.load_memory_cache(str(cache_path), allow_pickle=True)

    assert cache.get(request) == value


def test_cache_memory_load_requires_allow_pickle(tmp_path):
    cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path / "disk-cache"),
    )

    with pytest.raises(ValueError):
        cache.load_memory_cache(str(tmp_path / "memory-cache.pkl"))


def test_signature_cloudpickle_round_trip():
    class QA(dspy.Signature):
        """Answer the question."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    loaded = pickle.loads(cloudpickle.dumps(QA))

    assert loaded.__name__ == "QA"
    assert list(loaded.input_fields.keys()) == ["question"]
    assert list(loaded.output_fields.keys()) == ["answer"]
    assert loaded.instructions == "Answer the question."


def test_predict_cloudpickle_round_trip():
    class QA(dspy.Signature):
        """Answer the question."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    loaded = pickle.loads(cloudpickle.dumps(dspy.Predict(QA)))

    assert list(loaded.signature.fields.keys()) == ["question", "answer"]
