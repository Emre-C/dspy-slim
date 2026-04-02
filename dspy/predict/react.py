import logging
import traceback
from typing import Any, Callable, Literal

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.exceptions import ContextWindowExceededError

logger = logging.getLogger(__name__)


class ReAct(Module):
    def __init__(self, signature: str | type[Signature], tools: list[Callable[..., Any] | Tool], max_iters: int = 20):
        """Tool-using DSPy module implementing a standard ReAct loop."""
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        normalized_tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in normalized_tools}

        inputs = ", ".join(f"`{name}`" for name in signature.input_fields)
        outputs = ", ".join(f"`{name}`" for name in signature.output_fields)
        instructions = [f"{signature.instructions}\n"] if signature.instructions else []
        instructions.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. "
                "And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information "
                f"for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, "
                "and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        self.tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=(
                "Marks the task as complete. That is, signals that all information for producing the outputs, "
                f"i.e. {outputs}, are now available to be extracted."
            ),
            args={},
        )

        for index, tool in enumerate(self.tools.values(), start=1):
            instructions.append(f"({index}) {tool}")
        instructions.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            Signature({**signature.input_fields}, "\n".join(instructions))
            .append("trajectory", InputField(), type_=str)
            .append("next_thought", OutputField(), type_=str)
            .append("next_tool_name", OutputField(), type_=Literal[tuple(self.tools.keys())])
            .append("next_tool_args", OutputField(), type_=dict[str, Any])
        )
        fallback_signature = Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", InputField(), type_=str)

        self.react = Predict(react_signature)
        self.extract = ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]) -> str:
        adapter = settings.adapter or ChatAdapter()
        trajectory_signature = Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def forward(self, **input_args: Any) -> Prediction:
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)

        for index in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning("Ending the trajectory: Agent failed to select a valid tool: %s", _fmt_exc(err))
                break

            trajectory[f"thought_{index}"] = pred.next_thought
            trajectory[f"tool_name_{index}"] = pred.next_tool_name
            trajectory[f"tool_args_{index}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{index}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:  # pragma: no cover - exact traceback text varies.
                trajectory[f"observation_{index}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args: Any) -> Prediction:
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)

        for index in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning("Ending the trajectory: Agent failed to select a valid tool: %s", _fmt_exc(err))
                break

            trajectory[f"thought_{index}"] = pred.next_thought
            trajectory[f"tool_name_{index}"] = pred.next_tool_name
            trajectory[f"tool_args_{index}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{index}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:  # pragma: no cover - exact traceback text varies.
                trajectory[f"observation_{index}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module: Module, trajectory: dict[str, Any], **input_args: Any):
        current_trajectory = trajectory
        for _ in range(3):
            try:
                return module(**input_args, trajectory=self._format_trajectory(current_trajectory))
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                current_trajectory = self.truncate_trajectory(current_trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    async def _async_call_with_potential_trajectory_truncation(
        self, module: Module, trajectory: dict[str, Any], **input_args: Any
    ):
        current_trajectory = trajectory
        for _ in range(3):
            try:
                return await module.acall(**input_args, trajectory=self._format_trajectory(current_trajectory))
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                current_trajectory = self.truncate_trajectory(current_trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    def truncate_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        """Drop the oldest tool step from the prompt trajectory."""
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)
        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
