import pydantic
import pytest

from dspy.clients import lm as lm_module
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict


class NestedPayload(pydantic.BaseModel):
    text: str


class ResponsePayload(pydantic.BaseModel):
    message: NestedPayload
    usage: dict[str, int]


class OutputSchema(pydantic.BaseModel):
    answer: str


class DummyLM(BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError


def test_normalize_openai_object_converts_nested_models_to_dotdict():
    payload = ResponsePayload(message=NestedPayload(text="hello"), usage={"total_tokens": 3})

    normalized = lm_module._normalize_openai_object(payload)

    assert normalized.message.text == "hello"
    assert normalized.usage.total_tokens == 3


def test_build_chat_response_format_converts_pydantic_model_class():
    response_format = lm_module._build_chat_response_format(OutputSchema)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "OutputSchema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["title"] == "OutputSchema"


def test_convert_chat_request_to_responses_request_preserves_message_boundaries():
    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
        "reasoning_effort": "low",
        "response_format": OutputSchema,
    }

    converted = lm_module._convert_chat_request_to_responses_request(request)

    assert converted["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "Be terse."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]
    assert converted["reasoning"] == {"effort": "low", "summary": "auto"}
    assert converted["text"]["format"]["type"] == "json_schema"
    assert converted["text"]["format"]["name"] == "OutputSchema"


def test_build_client_kwargs_defaults_openrouter_base_url(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    request, client_kwargs = lm_module._build_client_kwargs(
        {"model": "openrouter/qwen/qwen3.6-plus-preview:free"},
        num_retries=4,
    )

    assert request["model"] == "openrouter/qwen/qwen3.6-plus-preview:free"
    assert client_kwargs["api_key"] == "test-key"
    assert client_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert client_kwargs["max_retries"] == 4


def test_context_window_detection_matches_provider_error_shapes():
    class FakeBadRequest(Exception):
        def __init__(self):
            self.body = {"error": {"code": "context_length_exceeded", "message": "Context length exceeded."}}

    assert lm_module._looks_like_context_window_exceeded(FakeBadRequest()) is True


@pytest.mark.parametrize(
    "message",
    [
        dotdict(tool_calls=None),
        dotdict(content=None, tool_calls=None),
    ],
)
def test_process_completion_tolerates_missing_message_content(message):
    lm = DummyLM("dummy")
    response = dotdict(
        choices=[dotdict(message=message, finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="dummy",
    )

    assert lm._process_completion(response, {}) == [""]
