"""ReplayLM — deterministic LM that replays pre-recorded responses.

Used for golden-transcript testing: record real LM responses once,
then replay them in CI to test the full pipeline without API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict


def _should_validate_turn(entry: dict[str, Any]) -> bool:
    return entry.get("prompt") is not None or bool(entry.get("messages"))


def _dump_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _non_system_messages(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []
    return [message for message in messages if not isinstance(message, dict) or message.get("role") != "system"]


def _system_roles(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []
    return [message.get("role") for message in messages if isinstance(message, dict) and message.get("role") == "system"]


def _messages_match(expected: Any, actual: Any) -> bool:
    if expected == actual:
        return True
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if _non_system_messages(expected) != _non_system_messages(actual):
        return False

    expected_system_roles = _system_roles(expected)
    if not expected_system_roles:
        return True

    return expected_system_roles == _system_roles(actual) and len(expected) == len(actual)


class ReplayLM(BaseLM):
    """Language model that replays pre-recorded outputs in order.

    Each call to forward() pops the next output from the queue and returns
    it as an OpenAI-compatible chat completion response.

    Usage:
        lm = ReplayLM(outputs=['{"answer": "Paris"}', '{"answer": "42"}'])
        dspy.configure(lm=lm)
        # Now Predict/ChainOfThought calls will get these outputs in order.
    """

    def __init__(
        self,
        outputs: list[str] | None = None,
        *,
        transcripts: list[dict[str, Any]] | None = None,
    ):
        super().__init__("replay-lm", "chat", 0.0, 1000, False)
        if outputs is not None and transcripts is not None:
            raise ValueError("ReplayLM accepts either outputs or transcripts, not both.")
        if outputs is None and transcripts is None:
            raise ValueError("ReplayLM requires outputs or transcripts.")

        if transcripts is not None:
            self._queue = [
                {
                    "output": entry["output"],
                    "prompt": entry.get("prompt"),
                    "messages": entry.get("messages"),
                    "validate": _should_validate_turn(entry),
                }
                for entry in transcripts
            ]
        else:
            self._queue = [
                {"output": output, "prompt": None, "messages": None, "validate": False}
                for output in outputs or []
            ]
        self._cursor: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayLM":
        """Load from a golden transcript JSON file.

        The file should be an array of objects with an "output" key:
        [{"messages": [...], "output": "...", "dataset": "...", "example_idx": 0}, ...]
        """
        data = json.loads(Path(path).read_text())
        return cls(transcripts=data)

    @classmethod
    def from_transcripts(cls, entries: list[dict[str, Any]]) -> "ReplayLM":
        """Load from a list of golden transcript entry dicts."""
        return cls(transcripts=entries)

    @property
    def remaining(self) -> int:
        """Number of outputs remaining in the queue."""
        return len(self._queue) - self._cursor

    @property
    def exhausted(self) -> bool:
        """Whether all outputs have been consumed."""
        return self._cursor >= len(self._queue)

    def _assert_expected_call(self, *, prompt: Any, messages: Any, turn: dict[str, Any]) -> None:
        if not turn["validate"]:
            return

        expected_prompt = turn["prompt"]
        expected_messages = turn["messages"]
        if prompt == expected_prompt and _messages_match(expected_messages, messages):
            return

        raise RuntimeError(
            "ReplayLM payload mismatch at turn "
            f"{self._cursor}: expected prompt={_dump_payload(expected_prompt)}, "
            f"messages={_dump_payload(expected_messages)}; got prompt={_dump_payload(prompt)}, "
            f"messages={_dump_payload(messages)}."
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        if self._cursor >= len(self._queue):
            raise RuntimeError(
                f"ReplayLM exhausted: all {len(self._queue)} recorded outputs consumed."
            )

        turn = self._queue[self._cursor]
        self._assert_expected_call(prompt=prompt, messages=messages, turn=turn)
        output = turn["output"]
        self._cursor += 1

        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=output, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="replay-lm",
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)
