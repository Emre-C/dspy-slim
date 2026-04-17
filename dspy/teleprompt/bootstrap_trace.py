import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, TypedDict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    callback_metadata: dict[str, Any] | None = None,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        # Tracebacks are suppressed: bootstrap_trace already catches and logs
        # format failures individually; surfacing raw tracebacks would duplicate
        # noise without adding diagnostic value.
        provide_traceback=False,
        # A generous error budget (10× dataset size) lets bootstrapping survive
        # sporadic LM formatting failures without aborting the entire run, while
        # still bounding runaway retry loops.
        max_errors=len(dataset) * 10,
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    # Use `object.__getattribute__` to bypass the custom hook `Module.__getattribute__` so that we avoid
    # the warning that `forward` is not accessed through `__call__`.
    original_forward = object.__getattribute__(program, "forward")

    def patched_forward(program_to_use: Module, **kwargs):
        with dspy.context(trace=[]):
            try:
                return original_forward(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program_to_use.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score)
                        * (len(present) / max(len(expected), 1)),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following "
                        "the adapter's formatting."
                    )

                return failed_pred, trace

    program.forward = MethodType(patched_forward, program)

    try:
        results = evaluator(
            program,
            metric=wrapped_metric,
            callback_metadata=callback_metadata,
        ).results
    finally:
        program.forward = original_forward

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace = prediction
        except ValueError as ve:
            # During bootstrapping the LLM response may not follow dspy formatting,
            # leading to a ValueError when unpacking (prediction, trace).
            # This fork treats warn-and-skip as the final behavior: capturing
            # malformed responses for negative-score training (upstream GRPO
            # proposal) is out of scope since fine-tuning is not a supported surface.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following "
                "dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data
