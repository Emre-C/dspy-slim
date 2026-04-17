import asyncio
from typing import Any

from pydantic import BaseModel

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.exceptions import ContextWindowExceededError
from tests.minimal.helpers.dummies import DummyLM


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: dict[str, str]


class InvitationSignature(dspy.Signature):
    participant_name: str = dspy.InputField(desc="The name of the participant to invite")
    event_info: CalendarEvent = dspy.InputField(desc="The information about the event")
    invitation_letter: str = dspy.OutputField(desc="The invitation letter to be sent to the participant")


def _invitation_answers() -> list[dict[str, Any]]:
    return [
        {
            "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
            "next_tool_name": "write_invitation_letter",
            "next_tool_args": {
                "participant_name": "Alice",
                "event_info": {
                    "name": "Science Fair",
                    "date": "Friday",
                    "participants": {"Alice": "female", "Bob": "male"},
                },
            },
        },
        {
            "next_thought": "I have the invitation letter, so I can finish now.",
            "next_tool_name": "finish",
            "next_tool_args": {},
        },
        {
            "reasoning": "I used the tool result to prepare the final invitation letter.",
            "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
        },
    ]


def _event_info() -> CalendarEvent:
    return CalendarEvent(
        name="Science Fair",
        date="Friday",
        participants={"Alice": "female", "Bob": "male"},
    )


def test_react_runs_sync_tool_loop_with_typed_args():
    def write_invitation_letter(participant_name: str, event_info: CalendarEvent) -> str | None:
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    with dspy.context(lm=DummyLM(_invitation_answers(), adapter=ChatAdapter())):
        outputs = react(participant_name="Alice", event_info=_event_info())

    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."
    assert outputs.trajectory == {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have the invitation letter, so I can finish now.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }


def test_react_runs_async_tool_loop_with_typed_args():
    async def write_invitation_letter(participant_name: str, event_info: CalendarEvent) -> str | None:
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    async def run():
        with dspy.context(lm=DummyLM(_invitation_answers(), adapter=ChatAdapter())):
            return await react.acall(participant_name="Alice", event_info=_event_info())

    outputs = asyncio.run(run())

    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."
    assert outputs.trajectory["tool_name_0"] == "write_invitation_letter"
    assert outputs.trajectory["observation_0"] == "It's my honor to invite Alice to event Science Fair on Friday"
    assert outputs.trajectory["tool_name_1"] == "finish"


def test_react_truncates_trajectory_after_context_window_overflow():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    call_count = 0

    def mock_react(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return dspy.Prediction(
                next_thought=f"Thought {call_count}",
                next_tool_name="echo",
                next_tool_args={"text": f"Text {call_count}"},
            )
        if call_count == 3:
            raise ContextWindowExceededError()
        return dspy.Prediction(next_thought="Final thought", next_tool_name="finish", next_tool_args={})

    react.react = mock_react
    react.extract = lambda **kwargs: dspy.Prediction(output_text="Final output")

    result = react(input_text="test input")

    assert "thought_0" not in result.trajectory
    assert result.trajectory["thought_1"] == "Thought 2"
    assert result.trajectory["tool_name_2"] == "finish"
    assert result.output_text == "Final output"


def _metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    return 0.0


def test_gepa_compile_with_react_exposes_predictors(monkeypatch):
    import gepa as gepa_pkg

    captured_kwargs: dict[str, Any] = {}

    class StubResult:
        def __init__(self, best_candidate: dict[str, str]):
            self.best_candidate = best_candidate

    def fake_optimize(**kwargs):
        captured_kwargs.update(kwargs)
        return StubResult(kwargs["seed_candidate"])

    monkeypatch.setattr(gepa_pkg, "optimize", fake_optimize)

    def lookup_weather(city: str) -> str:
        return f"{city} is sunny"

    student = dspy.ReAct("question -> answer", tools=[lookup_weather])
    example = dspy.Example(question="What is the weather in Tokyo?", answer="Tokyo is sunny").with_inputs("question")

    optimizer = dspy.GEPA(
        metric=_metric,
        reflection_lm=DummyLM([{"new_instruction": "Unused."}]),
        max_metric_calls=1,
    )
    program = optimizer.compile(student, trainset=[example], valset=[example])

    assert isinstance(program, dspy.ReAct)
    assert set(captured_kwargs["seed_candidate"]) == {"react", "extract.predict"}
    assert "lookup_weather" in captured_kwargs["seed_candidate"]["react"]
    assert captured_kwargs["seed_candidate"]["extract.predict"] == student.extract.predict.signature.instructions
