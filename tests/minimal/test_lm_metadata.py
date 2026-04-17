"""Tests for LM call metadata propagation (_dspy_lm)."""

import dspy
from dspy import Predict
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.utils.lm_metadata import DSPY_LM_METADATA_KEY


def test_base_lm_chat_completion_includes_metadata():
    class FinishLengthLM(BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return dotdict(
                choices=[
                    dotdict(
                        message=dotdict(content='{"output": "x"}', tool_calls=None),
                        finish_reason="length",
                    )
                ],
                usage=dotdict(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                model="t",
            )

    lm = FinishLengthLM("m", "chat", 0.0, 1000, False)
    out = lm(messages=[{"role": "user", "content": "hi"}])
    assert isinstance(out, list)
    assert isinstance(out[0], dict)
    assert out[0]["text"] == '{"output": "x"}'
    meta = out[0][DSPY_LM_METADATA_KEY]
    assert meta["truncated"] is True
    assert meta["finish_reason"] == "length"


def test_adapter_prediction_lm_metadata_roundtrip():
    from tests.minimal.helpers.dummies import DummyLM

    lm = DummyLM([{"reasoning": "r", "code": "print(1)", "output": "ok"}])
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    pred = Predict("input -> reasoning, code, output")(input="x")
    meta = pred.lm_metadata
    assert meta is not None
    assert meta.get("truncated") is False
    assert "usage" in meta
