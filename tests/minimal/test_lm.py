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


class _AsyncCollector:
    def __init__(self):
        self.events = []

    async def send(self, event):
        self.events.append(event)


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
    class FakeBadRequestError(Exception):
        def __init__(self):
            self.body = {"error": {"code": "context_length_exceeded", "message": "Context length exceeded."}}

    assert lm_module._looks_like_context_window_exceeded(FakeBadRequestError()) is True


@pytest.mark.parametrize(
    "message",
    [
        dotdict(tool_calls=None),
        dotdict(content=None, tool_calls=None),
    ],
)
def test_process_completion_tolerates_missing_message_content(message):
    from dspy.utils.lm_metadata import DSPY_LM_METADATA_KEY

    lm = DummyLM("dummy")
    response = dotdict(
        choices=[dotdict(message=message, finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="dummy",
    )

    out = lm._process_completion(response, {})
    assert isinstance(out[0], dict)
    assert out[0]["text"] == ""
    assert DSPY_LM_METADATA_KEY in out[0]
    assert out[0][DSPY_LM_METADATA_KEY]["truncated"] is False


def test_stream_openai_chat_completion_requests_usage_and_reconstructs_response(monkeypatch):
    chunks = [
        dotdict(
            model="gpt-4o-mini",
            choices=[dotdict(delta=dotdict(content="Hel", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        dotdict(
            model="gpt-4o-mini",
            choices=[dotdict(delta=dotdict(content="lo", tool_calls=None), finish_reason="stop")],
            usage=None,
        ),
        dotdict(
            model="gpt-4o-mini",
            choices=[],
            usage=dotdict(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        ),
    ]
    captured_requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = dotdict(completions=dotdict(create=self.create))

        def create(self, **kwargs):
            captured_requests.append(kwargs)
            return iter(chunks)

    monkeypatch.setattr(lm_module, "OpenAI", FakeClient)

    collector = _AsyncCollector()
    caller_predict = object()
    with lm_module.dspy.context(send_stream=collector, caller_predict=caller_predict, track_usage=True):
        response = lm_module._stream_openai_chat_completion(
            request={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            num_retries=2,
        )

    assert captured_requests[0]["stream_options"] == {"include_usage": True}
    assert response.choices[0].message.content == "Hello"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 5
    assert all(event.predict_id == id(caller_predict) for event in collector.events)


def test_stream_openai_responses_completion_requests_usage_and_reconstructs_response(monkeypatch):
    events = [
        dotdict(type="response.output_text.delta", delta="Hel"),
        dotdict(type="response.output_text.delta", delta="lo"),
        dotdict(
            type="response.completed",
            response=dotdict(
                output=[dotdict(type="message", content=[dotdict(text="Hello")])],
                usage=dotdict(prompt_tokens=2, completion_tokens=3, total_tokens=5),
                status="completed",
                model="gpt-5",
            ),
        ),
    ]
    captured_requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            self.responses = dotdict(create=self.create)

        def create(self, **kwargs):
            captured_requests.append(kwargs)
            return iter(events)

    monkeypatch.setattr(lm_module, "OpenAI", FakeClient)

    collector = _AsyncCollector()
    caller_predict = object()
    with lm_module.dspy.context(send_stream=collector, caller_predict=caller_predict, track_usage=True):
        response = lm_module._stream_openai_responses_completion(
            request={"model": "openai/gpt-5", "messages": [{"role": "user", "content": "hello"}]},
            num_retries=2,
        )

    assert captured_requests[0]["stream_options"] == {"include_usage": True}
    assert response.output[0].content[0].text == "Hello"
    assert response.usage.total_tokens == 5
    assert all(event.predict_id == id(caller_predict) for event in collector.events)
