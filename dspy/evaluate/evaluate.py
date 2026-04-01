import logging
from typing import Any, Callable

import tqdm

import dspy
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)


class EvaluationResult(Prediction):
    """Result of an evaluation: aggregate score and per-example rows."""

    def __init__(self, score: float, results: list[tuple["dspy.Example", "dspy.Example", Any]]):
        super().__init__(score=score, results=results)

    def __repr__(self):
        return f"EvaluationResult(score={self.score}, results=<list of {len(self.results)} results>)"


class Evaluate:
    """Run a metric over a devset in parallel (used by GEPA and manual eval)."""

    def __init__(
        self,
        *,
        devset: list["dspy.Example"],
        metric: Callable | None = None,
        num_threads: int | None = None,
        display_progress: bool = False,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        failure_score: float = 0.0,
        callback_metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        if kwargs.get("return_outputs") is not None:
            raise ValueError(
                "`return_outputs` is no longer supported. Results are always returned inside `EvaluationResult.results`."
            )
        # Legacy / internal flags (e.g. `return_all_scores` from GEPA) — ignored in minimal Evaluate.
        _ = callback_metadata
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score

    @with_callbacks
    def __call__(
        self,
        program: "dspy.Module",
        metric: Callable | None = None,
        devset: list["dspy.Example"] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        callback_metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress

        if callback_metadata:
            logger.debug("Evaluate is called with callback metadata: %s", callback_metadata)

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=(self.max_errors if self.max_errors is not None else dspy.settings.max_errors),
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)
            return prediction, score

        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results, strict=False)]
        ncorrect, ntotal = sum(score for *_, score in results), len(devset)

        logger.info("Average Metric: %s / %s (%s%%)", ncorrect, ntotal, round(100 * ncorrect / ntotal, 1))

        return EvaluationResult(
            score=round(100 * ncorrect / ntotal, 2),
            results=results,
        )
