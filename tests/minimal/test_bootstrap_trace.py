from typing import Any
from unittest import mock

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data


def _make_response(content: str):
    """Build a minimal response object matching OpenAI chat completion shape."""
    return dotdict(
        choices=[dotdict(message=dotdict(content=content, tool_calls=None), finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="test-model",
    )


class SequenceLM(BaseLM):
    """BaseLM subclass that returns a scripted sequence of raw completions."""

    def __init__(self, responses: list[str]):
        super().__init__("test-model", "chat", 0.0, 1000, False)
        self._responses = list(responses)
        self._idx = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        idx = self._idx
        self._idx += 1
        return _make_response(self._responses[idx])


def test_bootstrap_trace_data():
    """Test bootstrap_trace_data function with single dspy.Predict program."""

    class StringToIntSignature(dspy.Signature):
        """Convert a string number to integer"""

        text: str = dspy.InputField()
        number: int = dspy.OutputField()

    program = dspy.Predict(StringToIntSignature)

    dataset = [
        Example(text="one", number=1).with_inputs("text"),
        Example(text="two", number=2).with_inputs("text"),
        Example(text="three", number=3).with_inputs("text"),
        Example(text="four", number=4).with_inputs("text"),
        Example(text="five", number=5).with_inputs("text"),
    ]

    def exact_match_metric(example, prediction, trace=None):
        return example.number == prediction.number

    # 4 successful JSON responses + 1 malformed (triggers AdapterParseError)
    lm = SequenceLM([
        '```json\n{"number": 1}\n```',
        '```json\n{"number": 2}\n```',
        '```json\n{"number": 3}\n```',
        '```json\n{"number": 4}\n```',
        "This is an invalid JSON!",
    ])

    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    results = bootstrap_trace_data(
        program=program,
        dataset=dataset,
        metric=exact_match_metric,
        raise_on_error=False,
        capture_failed_parses=True,
    )

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    successful_count = 0
    failed_count = 0

    for result in results:
        assert "example" in result
        assert "prediction" in result
        assert "trace" in result
        assert "example_ind" in result
        assert "score" in result

        if isinstance(result["prediction"], FailedPrediction):
            failed_count += 1
            assert hasattr(result["prediction"], "completion_text")
            assert hasattr(result["prediction"], "format_reward")
            assert result["prediction"].completion_text == "This is an invalid JSON!"
        else:
            successful_count += 1
            assert hasattr(result["prediction"], "number")

    assert successful_count == 4, f"Expected 4 successful predictions, got {successful_count}"
    assert failed_count == 1, f"Expected 1 failed prediction, got {failed_count}"

    for result in results:
        assert len(result["trace"]) > 0, "Trace should not be empty"
        for trace_entry in result["trace"]:
            assert len(trace_entry) == 3, "Trace entry should have 3 elements"


def test_bootstrap_trace_data_passes_callback_metadata(monkeypatch):
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

    class DummyProgram(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction()

    captured_metadata: dict[str, Any] = {}

    class DummyEvaluate:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, callback_metadata=None, **kwargs):
            captured_metadata["value"] = callback_metadata

            class _Result:
                results: list[Any] = []

            return _Result()

    monkeypatch.setattr(bootstrap_trace_module, "Evaluate", DummyEvaluate)

    bootstrap_trace_module.bootstrap_trace_data(
        program=DummyProgram(),
        dataset=[],
        callback_metadata={"disable_logging": True},
    )

    assert captured_metadata["value"] == {"disable_logging": True}
