import pytest

import dspy
from dspy import ChainOfThought, Predict, Signature
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.utils.exceptions import AdapterParseError
from tests.minimal.helpers.dummies import DummyLM


def _make_response(content: str):
    return dotdict(
        choices=[dotdict(message=dotdict(content=content, tool_calls=None), finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="test-model",
    )


class JSONThenChatLM(BaseLM):
    def __init__(self):
        super().__init__("test-model", "chat", 0.0, 1000, False)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _make_response("{'path': file_path, 'text': all_texts[0]}")
        return _make_response("[[ ## output ## ]]\nrecovered answer\n\n[[ ## completed ## ]]")


class ChatThenJSONLM(BaseLM):
    def __init__(self):
        super().__init__("test-model", "chat", 0.0, 1000, False)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _make_response("not ChatAdapter-formatted output")
        return _make_response('{"output": "recovered answer"}')


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

def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}], adapter=ChatAdapter())
    dspy.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"


def test_supported_top_level_compatibility_exports_are_available():
    assert dspy.Tool is not None
    assert dspy.ToolCalls is not None
    assert dspy.Adapter is not None
    assert dspy.ChatAdapter is not None
    assert dspy.asyncify is not None
    assert dspy.load is not None
    assert dspy.streamify is not None
    assert dspy.track_usage is not None


def test_chain_of_thought_keeps_upstream_reasoning_prefix():
    chain = ChainOfThought("question -> answer")
    reasoning = chain.predict.signature.output_fields["reasoning"]

    assert reasoning.json_schema_extra["prefix"] == "Reasoning: Let's think step by step in order to"


def test_predict_warns_for_extra_fields_and_ignores_them(capfd):
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}], adapter=ChatAdapter())
    dspy.configure(lm=lm)

    result = predict_instance(input="test input", extra="ignored")
    _, err = capfd.readouterr()

    assert result.output == "test output"
    assert "Input contains fields not in signature" in err


def test_predict_warns_for_input_type_mismatch(capfd):
    class TypedSignature(dspy.Signature):
        count: int = dspy.InputField()
        result: str = dspy.OutputField()

    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}], adapter=ChatAdapter())
    dspy.configure(lm=lm)

    predict_instance(count="not an int")
    _, err = capfd.readouterr()

    assert "Type mismatch for field 'count': expected int" in err


def test_predict_can_disable_type_mismatch_warnings(capfd):
    class TypedSignature(dspy.Signature):
        count: int = dspy.InputField()
        result: str = dspy.OutputField()

    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}], adapter=ChatAdapter())
    dspy.configure(lm=lm)

    with dspy.context(warn_on_type_mismatch=False):
        predict_instance(count="not an int")
    _, err = capfd.readouterr()

    assert "Type mismatch for field 'count'" not in err


def test_predict_load_sanitizes_unsafe_lm_state_by_default(capfd, tmp_path):
    predict_instance = Predict("input -> output")
    predict_instance.lm = dspy.LM("openai/gpt-4o-mini")
    state_path = tmp_path / "predict-state.json"
    predict_instance.save(state_path)

    loaded = Predict("input -> output")
    state = predict_instance.dump_state()
    state["lm"]["api_base"] = "http://override.local/v1"
    state["lm"]["model_list"] = [{"model_name": "openai/gpt-4o-mini"}]

    loaded.load_state(state)
    _, err = capfd.readouterr()

    assert loaded.lm is not None
    assert "api_base" not in loaded.lm.kwargs
    assert "model_list" not in loaded.lm.kwargs
    assert "Ignoring unsafe LM config key(s) during state load" in err


def test_predict_load_can_preserve_unsafe_lm_state_with_opt_in(tmp_path):
    predict_instance = Predict("input -> output")
    predict_instance.lm = dspy.LM("openai/gpt-4o-mini")
    state_path = tmp_path / "predict-state.json"
    predict_instance.save(state_path)

    state = predict_instance.dump_state()
    state["lm"]["base_url"] = "http://override.local/v1"

    loaded = Predict("input -> output")
    loaded.load_state(state, allow_unsafe_lm_state=True)

    assert loaded.lm is not None
    assert loaded.lm.kwargs["base_url"] == "http://override.local/v1"


def test_predict_load_from_file_sanitizes_unsafe_lm_state_by_default(tmp_path):
    predict_instance = Predict("input -> output")
    predict_instance.lm = dspy.LM("openai/gpt-4o-mini")
    state_path = tmp_path / "predict-state.json"
    predict_instance.save(state_path)

    import orjson

    state = orjson.loads(state_path.read_bytes())
    state["lm"]["base_url"] = "http://override.local/v1"
    state_path.write_bytes(orjson.dumps(state))

    loaded = Predict("input -> output")
    loaded.load(state_path)

    assert loaded.lm is not None
    assert "base_url" not in loaded.lm.kwargs


def test_predict_load_from_file_can_preserve_unsafe_lm_state_with_opt_in(tmp_path):
    predict_instance = Predict("input -> output")
    predict_instance.lm = dspy.LM("openai/gpt-4o-mini")
    state_path = tmp_path / "predict-state.json"
    predict_instance.save(state_path)

    import orjson

    state = orjson.loads(state_path.read_bytes())
    state["lm"]["base_url"] = "http://override.local/v1"
    state_path.write_bytes(orjson.dumps(state))

    loaded = Predict("input -> output")
    loaded.load(state_path, allow_unsafe_lm_state=True)

    assert loaded.lm is not None
    assert loaded.lm.kwargs["base_url"] == "http://override.local/v1"


def test_predict_defaults_to_chat_adapter_with_json_fallback():
    predict_instance = Predict("input -> output")
    lm = ChatThenJSONLM()
    dspy.configure(lm=lm, adapter=None)

    result = predict_instance(input="test input")

    assert result.output == "recovered answer"
    assert lm.calls == 2


def test_chat_adapter_can_disable_json_fallback():
    predict_instance = Predict("input -> output")
    lm = ChatThenJSONLM()
    dspy.configure(lm=lm, adapter=ChatAdapter(use_json_adapter_fallback=False))

    try:
        predict_instance(input="test input")
        raise AssertionError("Expected AdapterParseError")
    except AdapterParseError:
        pass

    assert lm.calls == 1


def test_json_adapter_remains_single_shot_on_parse_failure():
    predict_instance = Predict("input -> output")
    lm = JSONThenChatLM()
    dspy.configure(lm=lm, adapter=JSONAdapter())

    try:
        predict_instance(input="test input")
        raise AssertionError("Expected AdapterParseError")
    except AdapterParseError:
        pass

    assert lm.calls == 1


def test_predict_records_lm_usage_when_enabled():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "tracked output"}], adapter=ChatAdapter())

    with dspy.context(lm=lm, track_usage=True):
        result = predict_instance(input="test input")

    assert result.output == "tracked output"
    assert result.get_lm_usage() == {
        "dummy": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def test_map_named_predictors_replaces_nested_predictors():
    class NestedModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.steps = {"first": Predict("question -> answer")}

    program = NestedModule()
    original = program.steps["first"]

    returned = program.map_named_predictors(
        lambda predictor: Predict(predictor.signature, **predictor.get_config())
    )

    assert returned is program
    assert isinstance(program.steps["first"], Predict)
    assert program.steps["first"] is not original
    assert program.steps["first"].signature == original.signature


def test_named_sub_modules_walks_breadth_first_and_can_skip_compiled_children():
    class Inner(dspy.Module):
        def __init__(self):
            super().__init__()
            self.gamma = Predict("question -> answer")

    class Outer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.alpha = Predict("question -> answer")
            self.beta = Inner()

    program = Outer()

    assert [name for name, _ in program.named_sub_modules()] == [
        "self",
        "self.alpha",
        "self.beta",
        "self.beta.gamma",
    ]

    program.beta._compiled = True
    assert [name for name, _ in program.named_sub_modules(skip_compiled=True)] == [
        "self",
        "self.alpha",
        "self.beta",
    ]


def test_tool_compatibility_shims_remain_available():
    def add(x: int, y: int) -> int:
        return x + y

    tool = dspy.Tool(add)
    tool_call = dspy.ToolCalls.from_dict_list([{"name": "add", "args": {"x": 2, "y": 3}}]).tool_calls[0]

    assert tool.format_as_litellm_function_call() == tool.format_as_openai_function_call()
    assert tool_call.execute([tool]) == 5

    with pytest.raises(NotImplementedError):
        dspy.Tool.from_langchain(object())


def test_save_program_round_trips_via_top_level_load(tmp_path):
    predict_instance = Predict("question -> answer")

    predict_instance.save(tmp_path, save_program=True)
    loaded = dspy.load(tmp_path, allow_pickle=True)

    assert isinstance(loaded, Predict)
    assert loaded.dump_state() == predict_instance.dump_state()
