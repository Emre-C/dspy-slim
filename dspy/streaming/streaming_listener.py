import inspect
import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any

import jiter

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from dspy.primitives.module import Module

ADAPTER_SUPPORT_STREAMING = (ChatAdapter, JSONAdapter)


class StreamListener:
    """Capture streamed tokens for a specific predictor output field."""

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name
        self.allow_reuse = allow_reuse

        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self.json_adapter_state = {"field_accumulated_messages": ""}
        self.adapter_identifiers = {
            "ChatAdapter": {
                "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
                "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
                "start_indicator": "[",
                "end_pattern_prefixes": ["[", "[[", "[[ ", "[[ #", "[[ ##"],
                "end_pattern_contains": "[[ ##",
            },
            "JSONAdapter": {
                "start_identifier": f'"{self.signature_field_name}":',
                "end_identifier": re.compile(r"\w*\"(,|\s*})"),
                "start_indicator": '"',
                "end_pattern_prefixes": ['"', '",', '" ', '"}'],
                "end_pattern_contains": "}",
            },
        }

    def _get_streaming_adapter_name(self) -> str:
        adapter = settings.adapter
        if adapter is None or (isinstance(adapter, ChatAdapter) and not isinstance(adapter, JSONAdapter)):
            return "ChatAdapter"
        if isinstance(adapter, JSONAdapter):
            return "JSONAdapter"
        raise ValueError(
            "Unsupported adapter for streaming: "
            f"{type(adapter).__name__}, please use one of the following adapters: "
            f"{', '.join(adapter_type.__name__ for adapter_type in ADAPTER_SUPPORT_STREAMING)}"
        )

    def _buffered_message_end_with_start_identifier(self, concat_message: str, start_identifier: str) -> bool:
        for index in range(len(concat_message)):
            if start_identifier.startswith(concat_message[len(concat_message) - index - 1 :]):
                return True
        return False

    def _could_form_end_identifier(self, concat_message: str, adapter_name: str) -> bool:
        adapter_config = self.adapter_identifiers[adapter_name]
        end_pattern_prefixes = adapter_config.get("end_pattern_prefixes", [])
        end_pattern_contains = adapter_config.get("end_pattern_contains")

        if any(concat_message.endswith(prefix) for prefix in end_pattern_prefixes):
            return True

        if end_pattern_contains and end_pattern_contains in concat_message:
            return True

        return False

    def receive(self, chunk: Any):
        adapter_name = self._get_streaming_adapter_name()
        start_identifier = self.adapter_identifiers[adapter_name]["start_identifier"]
        end_identifier = self.adapter_identifiers[adapter_name]["end_identifier"]
        start_indicator = self.adapter_identifiers[adapter_name]["start_indicator"]

        if self.stream_end:
            if self.allow_reuse:
                self.stream_end = False
                self.cache_hit = False
                self.field_start_queue = []
                self.field_end_queue = Queue()
                self.json_adapter_state["field_accumulated_messages"] = ""
                self.stream_start = False
            else:
                return None

        if (
            self._output_type
            and inspect.isclass(self._output_type)
            and issubclass(self._output_type, Type)
            and self._output_type.is_streamable()
        ):
            if parsed_chunk := self._output_type.parse_stream_chunk(chunk):
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    parsed_chunk,
                    is_last_chunk=self.stream_end,
                )

        try:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is None:
                return None
        except Exception:
            return None

        if chunk_message and start_identifier in chunk_message and adapter_name != "JSONAdapter":
            message_after_start_identifier = chunk_message[
                chunk_message.find(start_identifier) + len(start_identifier) :
            ]
            if re.search(end_identifier, message_after_start_identifier):
                self.cache_hit = True
                self.stream_start = True
                self.stream_end = True
                return None

        if len(self.field_start_queue) == 0 and not self.stream_start and start_indicator in chunk_message:
            self.field_start_queue.append(chunk_message)
            return None

        if len(self.field_start_queue) > 0 and not self.stream_start:
            self.field_start_queue.append(chunk_message)
            concat_message = "".join(self.field_start_queue)

            if start_identifier in concat_message:
                self.stream_start = True
                self.field_start_queue = []
                value_start_index = concat_message.find(start_identifier) + len(start_identifier)
                chunk_message = concat_message[value_start_index:].lstrip()

                if adapter_name == "JSONAdapter":
                    self.json_adapter_state["field_accumulated_messages"] += "{" + start_identifier

            elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
                return None
            else:
                self.field_start_queue = []
                return None

        if self.stream_start and chunk_message:
            self.field_end_queue.put(chunk_message)

            token = None
            concat_message = "".join(self.field_end_queue.queue).strip()

            if not self._could_form_end_identifier(concat_message, adapter_name):
                token = self.flush()
            elif self.field_end_queue.qsize() > 10:
                token = self.field_end_queue.get()

            if adapter_name == "JSONAdapter":
                return self._json_adapter_handle_stream_chunk(token, chunk_message)
            return self._default_handle_stream_chunk(token, end_identifier)

        return None

    def _json_adapter_handle_stream_chunk(self, token: str | None, chunk_message: str) -> StreamResponse | None:
        self.json_adapter_state["field_accumulated_messages"] += chunk_message
        if self.json_adapter_state["field_accumulated_messages"].rstrip().endswith("}"):
            try:
                jiter.from_json(self.json_adapter_state["field_accumulated_messages"].encode("utf-8"))
                self.stream_end = True
                last_token = self.flush()
                right_curly_bracket_index = last_token.rfind("}")
                if right_curly_bracket_index == -1:
                    right_curly_bracket_index = len(last_token)
                token = token + last_token[:right_curly_bracket_index] if token else last_token[:right_curly_bracket_index]
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    token,
                    is_last_chunk=self.stream_end,
                )
            except ValueError:
                pass

        try:
            parsed = jiter.from_json(
                self.json_adapter_state["field_accumulated_messages"].encode("utf-8"),
                partial_mode="trailing-strings",
            )
            if len(parsed) > 1:
                self.stream_end = True
                last_token = self.flush()

                next_field_name = next(key for key in parsed if key != self.signature_field_name)
                last_token_index = last_token.find(next_field_name)
                if last_token_index == -1:
                    last_token_index = len(last_token)
                token = token + last_token[:last_token_index] if token else last_token[:last_token_index]
        except ValueError:
            pass

        if token or self.stream_end:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )
        return None

    def _default_handle_stream_chunk(self, token: str | None, end_identifier: re.Pattern[str]) -> StreamResponse | None:
        concat_message = "".join(self.field_end_queue.queue).strip()

        if re.search(end_identifier, concat_message):
            self.stream_end = True
            last_token = self.flush()
            token = token + last_token if token else last_token
            token = token.rstrip()

        if token or self.stream_end:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )
        return None

    def flush(self) -> str:
        last_tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()

        adapter_name = self._get_streaming_adapter_name()
        if adapter_name == "JSONAdapter":
            return last_tokens

        boundary_index = last_tokens.find("[[")
        if boundary_index == -1:
            boundary_index = len(last_tokens)
        return last_tokens[:boundary_index]

    def finalize(self) -> StreamResponse | None:
        if self.stream_end or not self.stream_start:
            return None

        self.stream_end = True
        if self.field_end_queue.qsize() == 0:
            return None

        token = self.flush()
        if token:
            return StreamResponse(self.predict_name, self.signature_field_name, token, is_last_chunk=True)
        return None

    @property
    def _output_type(self) -> type | None:
        try:
            return self.predict.signature.output_fields[self.signature_field_name].annotation
        except Exception:
            return None


def find_predictor_for_stream_listeners(
    program: "Module", stream_listeners: list[StreamListener]
) -> dict[int, list[StreamListener]]:
    predictors = program.named_predictors()

    field_name_to_named_predictor = {}
    for listener in stream_listeners:
        if listener.predict:
            continue
        field_name_to_named_predictor[listener.signature_field_name] = None

    for name, predictor in predictors:
        for field_name in predictor.signature.output_fields:
            if field_name not in field_name_to_named_predictor:
                continue
            if field_name_to_named_predictor[field_name] is not None:
                raise ValueError(
                    f"Signature field {field_name} is not unique in the program; specify the predictor explicitly."
                )
            field_name_to_named_predictor[field_name] = (name, predictor)

    predict_id_to_listener = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
            predict_id_to_listener[id(listener.predict)].append(listener)
            continue
        if listener.signature_field_name not in field_name_to_named_predictor:
            raise ValueError(
                f"Signature field {listener.signature_field_name} is not a field of any predictor in the program."
            )
        listener.predict_name, listener.predict = field_name_to_named_predictor[listener.signature_field_name]
        predict_id_to_listener[id(listener.predict)].append(listener)
    return predict_id_to_listener
