# ruff: noqa: E402

import logging
import os
import re
import warnings
from typing import Any, Literal

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"openai\._compat",
)

import pydantic
from openai import AsyncOpenAI, BadRequestError, OpenAI

import dspy
from dspy.clients.cache import request_cache
from dspy.dsp.utils.utils import dotdict
from dspy.utils.exceptions import ContextWindowExceededError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)

_REASONING_MODEL_PATTERN = re.compile(
    r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$"
)
_RESPONSE_FORMAT_PARAMS = {"response_format"}


def _normalize_openai_object(value: Any) -> Any:
    if isinstance(value, pydantic.BaseModel):
        return _normalize_openai_object(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return dotdict({key: _normalize_openai_object(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_normalize_openai_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_openai_object(item) for item in value)
    return value


def _normalize_openai_response(response: Any) -> dotdict:
    normalized = _normalize_openai_object(response)
    if not isinstance(normalized, dotdict):
        normalized = dotdict(normalized)
    if normalized.get("usage") is None:
        normalized["usage"] = dotdict()
    normalized["cache_hit"] = False
    return normalized


def _is_pydantic_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, pydantic.BaseModel)


def _build_chat_response_format(response_format: Any) -> Any:
    if _is_pydantic_model_type(response_format):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": response_format.model_json_schema(),
            },
        }
    return response_format


def _build_responses_response_format(response_format: Any) -> Any:
    if _is_pydantic_model_type(response_format):
        return {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": response_format.model_json_schema(),
        }
    return response_format


def _provider_name_from_model(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"

def _provider_model_name(model: str) -> str:
    provider = _provider_name_from_model(model)
    if provider in {"openai", "openrouter"} and "/" in model:
        return model.split("/", 1)[1]
    return model


def _provider_env_name(provider: str, suffix: str) -> str:
    return f"{provider.upper()}_{suffix}"


def _resolve_api_key(provider: str, request: dict[str, Any]) -> str | None:
    api_key = request.pop("api_key", None)
    if api_key is not None:
        return api_key
    return os.getenv(_provider_env_name(provider, "API_KEY"))


def _resolve_base_url(provider: str, request: dict[str, Any]) -> str | None:
    base_url = request.pop("base_url", None) or request.pop("api_base", None)
    if base_url is not None:
        return base_url

    env_base_url = os.getenv(_provider_env_name(provider, "API_BASE")) or os.getenv(
        _provider_env_name(provider, "BASE_URL")
    )
    if env_base_url:
        return env_base_url

    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"

    return None


def _build_client_kwargs(request: dict[str, Any], num_retries: int) -> tuple[dict[str, Any], dict[str, Any]]:
    request = dict(request)
    provider = _provider_name_from_model(request["model"])

    client_kwargs = {
        "api_key": _resolve_api_key(provider, request),
        "max_retries": num_retries,
    }
    if base_url := _resolve_base_url(provider, request):
        client_kwargs["base_url"] = base_url
    if organization := request.pop("organization", None):
        client_kwargs["organization"] = organization
    if project := request.pop("project", None):
        client_kwargs["project"] = project

    return request, client_kwargs


def _extract_error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)


def _looks_like_context_window_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")

    if code in {"context_length_exceeded", "string_above_max_length"}:
        return True

    message = _extract_error_message(exc).lower()
    phrases = (
        "context length",
        "context window",
        "maximum context length",
        "maximum context window",
        "prompt is too long",
        "too many tokens",
        "reduce the length",
    )
    return any(phrase in message for phrase in phrases)


def _convert_chat_request_to_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)

    if "messages" in request:
        input_items = []
        for msg in request.pop("messages"):
            content = msg.get("content")
            if isinstance(content, str):
                content_blocks = [{"type": "input_text", "text": content}]
            elif isinstance(content, list):
                content_blocks = [_convert_content_item_to_responses_format(item) for item in content]
            else:
                content_blocks = []

            input_items.append({"role": msg.get("role", "user"), "content": content_blocks})
        request["input"] = input_items

    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    if "max_completion_tokens" in request:
        request["max_output_tokens"] = request.pop("max_completion_tokens")
    elif "max_tokens" in request:
        request["max_output_tokens"] = request.pop("max_tokens")

    if "response_format" in request:
        response_format = _build_responses_response_format(request.pop("response_format"))
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}

    return request


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {
            "type": "input_image",
            "image_url": image_url,
        }
    if item.get("type") == "text":
        return {
            "type": "input_text",
            "text": item.get("text", ""),
        }
    if item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }

    return item


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }


def _streaming_context() -> tuple[Any | None, int | None]:
    stream = dspy.settings.send_stream
    caller_predict = dspy.settings.caller_predict
    caller_predict_id = id(caller_predict) if caller_predict else None
    return stream, caller_predict_id


def _add_stream_usage_if_needed(request: dict[str, Any]) -> None:
    if not dspy.settings.track_usage:
        return
    request["stream_options"] = {**request.get("stream_options", {}), "include_usage": True}


def _merge_tool_call_delta(tool_calls: dict[int, dict[str, Any]], delta_tool_calls: list[Any] | None) -> None:
    if not delta_tool_calls:
        return

    for delta in delta_tool_calls:
        index = getattr(delta, "index", 0)
        merged = tool_calls.setdefault(index, {"id": None, "type": None, "function": {"name": "", "arguments": ""}})
        if getattr(delta, "id", None):
            merged["id"] = delta.id
        if getattr(delta, "type", None):
            merged["type"] = delta.type
        function = getattr(delta, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            merged["function"]["name"] += function.name
        if getattr(function, "arguments", None):
            merged["function"]["arguments"] += function.arguments


def _build_streamed_chat_response(chunks: list[dotdict], fallback_model: str) -> dotdict:
    content_parts: list[str] = []
    finish_reason = "stop"
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = dotdict()
    model = fallback_model

    for chunk in chunks:
        if chunk.get("model"):
            model = chunk.model
        if chunk.get("usage"):
            usage = dotdict(chunk.usage)

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or dotdict()
        if delta.get("content") is not None:
            content_parts.append(delta.content)
        _merge_tool_call_delta(tool_calls, delta.get("tool_calls"))
        if choice.get("finish_reason"):
            finish_reason = choice.finish_reason

    message = dotdict(content="".join(content_parts), tool_calls=[dotdict(item) for item in tool_calls.values()] or None)
    return dotdict(
        choices=[dotdict(message=message, finish_reason=finish_reason)],
        usage=usage,
        model=model,
        cache_hit=False,
    )


def _stream_openai_chat_completion(request: dict[str, Any], num_retries: int):
    from dspy.streaming.messages import sync_send_to_stream

    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    if "response_format" in request:
        request["response_format"] = _build_chat_response_format(request["response_format"])
    _add_stream_usage_if_needed(request)
    request["stream"] = True

    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = OpenAI(**client_kwargs)

    stream, caller_predict_id = _streaming_context()
    chunks: list[dotdict] = []
    for raw_chunk in client.chat.completions.create(extra_headers=headers, **request):
        chunk = _normalize_openai_object(raw_chunk)
        if not isinstance(chunk, dotdict):
            chunk = dotdict(chunk)
        if caller_predict_id is not None:
            chunk.predict_id = caller_predict_id
        chunks.append(chunk)
        if stream is not None:
            sync_send_to_stream(stream, chunk)

    return _build_streamed_chat_response(chunks, request["model"])


async def _astream_openai_chat_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    if "response_format" in request:
        request["response_format"] = _build_chat_response_format(request["response_format"])
    _add_stream_usage_if_needed(request)
    request["stream"] = True

    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = AsyncOpenAI(**client_kwargs)

    stream, caller_predict_id = _streaming_context()
    chunks: list[dotdict] = []
    async for raw_chunk in await client.chat.completions.create(extra_headers=headers, **request):
        chunk = _normalize_openai_object(raw_chunk)
        if not isinstance(chunk, dotdict):
            chunk = dotdict(chunk)
        if caller_predict_id is not None:
            chunk.predict_id = caller_predict_id
        chunks.append(chunk)
        if stream is not None:
            await stream.send(chunk)

    return _build_streamed_chat_response(chunks, request["model"])


def _build_streamed_responses_result(text_parts: list[str], final_response: dotdict | None, fallback_model: str) -> dotdict:
    if final_response is not None:
        normalized = _normalize_openai_response(final_response)
        if normalized.get("model") is None:
            normalized["model"] = fallback_model
        return normalized

    return dotdict(
        output=[dotdict(type="message", content=[dotdict(text="".join(text_parts))])],
        usage=dotdict(),
        model=fallback_model,
        cache_hit=False,
    )


def _stream_openai_responses_completion(request: dict[str, Any], num_retries: int):
    from dspy.streaming.messages import sync_send_to_stream

    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    request = _convert_chat_request_to_responses_request(request)
    _add_stream_usage_if_needed(request)
    request["stream"] = True

    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = OpenAI(**client_kwargs)

    stream, caller_predict_id = _streaming_context()
    text_parts: list[str] = []
    final_response = None
    for raw_event in client.responses.create(extra_headers=headers, **request):
        event = _normalize_openai_object(raw_event)
        if not isinstance(event, dotdict):
            event = dotdict(event)
        if caller_predict_id is not None:
            event.predict_id = caller_predict_id
        if event.get("type") == "response.output_text.delta" and event.get("delta"):
            text_parts.append(event.delta)
        elif event.get("type") == "response.completed" and event.get("response"):
            final_response = event.response
        if stream is not None:
            sync_send_to_stream(stream, event)

    return _build_streamed_responses_result(text_parts, final_response, request["model"])


async def _astream_openai_responses_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    request = _convert_chat_request_to_responses_request(request)
    _add_stream_usage_if_needed(request)
    request["stream"] = True

    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = AsyncOpenAI(**client_kwargs)

    stream, caller_predict_id = _streaming_context()
    text_parts: list[str] = []
    final_response = None
    async for raw_event in await client.responses.create(extra_headers=headers, **request):
        event = _normalize_openai_object(raw_event)
        if not isinstance(event, dotdict):
            event = dotdict(event)
        if caller_predict_id is not None:
            event.predict_id = caller_predict_id
        if event.get("type") == "response.output_text.delta" and event.get("delta"):
            text_parts.append(event.delta)
        elif event.get("type") == "response.completed" and event.get("response"):
            final_response = event.response
        if stream is not None:
            await stream.send(event)

    return _build_streamed_responses_result(text_parts, final_response, request["model"])


class LM(BaseLM):
    """Language model for chat or responses via an OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list | None = None,
        num_retries: int = 3,
        use_developer_role: bool = False,
        **kwargs,
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.use_developer_role = use_developer_role
        self._warned_zero_temp_rollout = False

        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        if _REASONING_MODEL_PATTERN.match(model_family):
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise ValueError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                    "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)"
                )
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)

        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    @property
    def _provider_name(self) -> str:
        return _provider_name_from_model(self.model)

    @property
    def supports_function_calling(self) -> bool:
        return self.model_type in {"chat", "responses"} and self._provider_name in {"openai", "openrouter"}

    @property
    def supports_reasoning(self) -> bool:
        model_family = self.model.split("/")[-1].lower() if "/" in self.model else self.model.lower()
        return self._provider_name == "openai" and bool(_REASONING_MODEL_PATTERN.match(model_family))

    @property
    def supports_response_schema(self) -> bool:
        return self.model_type in {"chat", "responses"} and self._provider_name == "openai"

    @property
    def supported_params(self) -> set[str]:
        return set(_RESPONSE_FORMAT_PARAMS) if self.supports_response_schema else set()

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _get_cached_completion_fn(self, completion_fn, cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)
        return completion_fn

    def dump_state(self):
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key != "api_key"}
        return {
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": self.num_retries,
            "use_developer_role": self.use_developer_role,
            **filtered_kwargs,
        }

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        try:
            request = dict(model=self.model, messages=messages, **kwargs)
            if dspy.settings.send_stream is not None:
                if self.model_type == "chat":
                    results = _stream_openai_chat_completion(request=request, num_retries=self.num_retries)
                elif self.model_type == "responses":
                    results = _stream_openai_responses_completion(request=request, num_retries=self.num_retries)
                else:
                    raise ValueError(f"Unsupported model_type={self.model_type}")
            else:
                if self.model_type == "chat":
                    completion = openai_chat_completion
                elif self.model_type == "responses":
                    completion = openai_responses_completion
                else:
                    raise ValueError(f"Unsupported model_type={self.model_type}")

                completion = self._get_cached_completion_fn(completion, cache)
                results = completion(request=request, num_retries=self.num_retries)
        except BadRequestError as e:
            if _looks_like_context_window_exceeded(e):
                raise ContextWindowExceededError(model=self.model) from e
            raise

        self._check_truncation(results)
        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        try:
            request = dict(model=self.model, messages=messages, **kwargs)
            if dspy.settings.send_stream is not None:
                if self.model_type == "chat":
                    results = await _astream_openai_chat_completion(request=request, num_retries=self.num_retries)
                elif self.model_type == "responses":
                    results = await _astream_openai_responses_completion(request=request, num_retries=self.num_retries)
                else:
                    raise ValueError(f"Unsupported model_type={self.model_type}")
            else:
                if self.model_type == "chat":
                    completion = aopenai_chat_completion
                elif self.model_type == "responses":
                    completion = aopenai_responses_completion
                else:
                    raise ValueError(f"Unsupported model_type={self.model_type}")

                completion = self._get_cached_completion_fn(completion, cache)
                results = await completion(request=request, num_retries=self.num_retries)
        except BadRequestError as e:
            if _looks_like_context_window_exceeded(e):
                raise ContextWindowExceededError(model=self.model) from e
            raise

        self._check_truncation(results)
        return results

    def _check_truncation(self, results):
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
            max_tokens = self.kwargs.get("max_tokens", self.kwargs.get("max_completion_tokens"))
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={max_tokens}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )


def openai_chat_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    if "response_format" in request:
        request["response_format"] = _build_chat_response_format(request["response_format"])
    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(extra_headers=headers, **request)
    return _normalize_openai_response(response)


async def aopenai_chat_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    if "response_format" in request:
        request["response_format"] = _build_chat_response_format(request["response_format"])
    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = AsyncOpenAI(**client_kwargs)
    response = await client.chat.completions.create(extra_headers=headers, **request)
    return _normalize_openai_response(response)


def openai_responses_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    request = _convert_chat_request_to_responses_request(request)
    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = OpenAI(**client_kwargs)
    response = client.responses.create(extra_headers=headers, **request)
    return _normalize_openai_response(response)


async def aopenai_responses_completion(request: dict[str, Any], num_retries: int):
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    request = _convert_chat_request_to_responses_request(request)
    request, client_kwargs = _build_client_kwargs(request, num_retries)
    request["model"] = _provider_model_name(request["model"])
    client = AsyncOpenAI(**client_kwargs)
    response = await client.responses.create(extra_headers=headers, **request)
    return _normalize_openai_response(response)
