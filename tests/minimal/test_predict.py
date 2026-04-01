import dspy
from dspy import Predict, Signature
from tests.minimal.helpers.dummies import DummyLM


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions


def test_reset_method():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "modified"
    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_lm_after_dump_and_load_state():
    predict_instance = Predict("input -> output")
    lm = dspy.BaseLM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
    )
    predict_instance.lm = lm
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm.model == "openai/gpt-4o-mini"
    assert new_instance.lm.kwargs["temperature"] == 1
    assert new_instance.lm.kwargs["max_tokens"] == 100


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}])
    dspy.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"
