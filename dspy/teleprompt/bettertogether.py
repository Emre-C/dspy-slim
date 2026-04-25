import inspect
import logging
import random
from typing import Any, Callable

from dspy.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)


class BetterTogether(Teleprompter):
    """Sequence explicit optimizers and keep the best candidate on a validation set."""

    STRAT_SEP = " -> "

    def __init__(self, metric: Callable, **optimizers: Teleprompter):
        self.metric = metric
        if not optimizers:
            raise ValueError(
                "BetterTogether in dspy-slim requires explicit optimizers, e.g. "
                "BetterTogether(metric=metric, gepa=GEPA(...))."
            )
        for key, optimizer in optimizers.items():
            if not hasattr(optimizer, "compile"):
                raise TypeError(f"Optimizer '{key}' must define compile(), got {type(optimizer).__name__}")
        self.optimizers = optimizers

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | list[Module] | None = None,
        valset: list[Example] | None = None,
        num_threads: int | None = None,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        seed: int | None = None,
        valset_ratio: float = 0.1,
        shuffle_trainset_between_steps: bool = True,
        strategy: str | None = None,
        optimizer_compile_args: dict[str, dict[str, Any]] | None = None,
    ) -> Module:
        student = student.deepcopy()
        trainset, valset = self._prepare_trainset_and_valset(trainset, valset, valset_ratio)
        parsed_strategy = self._prepare_strategy(strategy)
        optimizer_compile_args = self._prepare_optimizer_compile_args(optimizer_compile_args)

        rng = random.Random(seed)
        candidates: list[dict[str, Any]] = []
        flag_compilation_error_occurred = False

        baseline_score = self._evaluate_on_valset(
            student,
            valset,
            num_threads=num_threads,
            max_errors=max_errors,
            provide_traceback=provide_traceback,
        )
        self._add_candidate(candidates, student, "", baseline_score)

        current = student
        for step_index, step_code in enumerate(parsed_strategy):
            optimizer = self.optimizers[step_code]
            if shuffle_trainset_between_steps:
                rng.shuffle(trainset)

            try:
                current = self._run_optimizer(
                    optimizer,
                    current,
                    trainset=trainset,
                    teacher=teacher,
                    valset=valset,
                    compile_args=optimizer_compile_args.get(step_code, {}),
                )
            except Exception:
                logger.exception("BetterTogether step %s (%s) failed", step_index + 1, step_code)
                flag_compilation_error_occurred = True
                break

            score = self._evaluate_on_valset(
                current,
                valset,
                num_threads=num_threads,
                max_errors=max_errors,
                provide_traceback=provide_traceback,
            )
            self._add_candidate(candidates, current, self.STRAT_SEP.join(parsed_strategy[: step_index + 1]), score)

        if valset:
            ordered_candidates = sorted(
                enumerate(candidates),
                key=lambda item: ((item[1]["score"] if item[1]["score"] is not None else float("-inf")), -item[0]),
                reverse=True,
            )
            best = ordered_candidates[0][1]
            candidate_programs = [candidate for _, candidate in ordered_candidates]
        else:
            best = candidates[-1]
            candidate_programs = candidates

        compiled = best["program"]
        compiled.candidate_programs = candidate_programs
        compiled.flag_compilation_error_occurred = flag_compilation_error_occurred
        compiled._compiled = True
        return compiled

    def _prepare_trainset_and_valset(
        self,
        trainset: list[Example],
        valset: list[Example] | None,
        valset_ratio: float,
    ) -> tuple[list[Example], list[Example] | None]:
        if not trainset:
            raise ValueError("trainset cannot be empty")
        if valset_ratio < 0 or valset_ratio >= 1:
            raise ValueError(f"valset_ratio must be in range [0, 1), got {valset_ratio}")

        trainset = trainset[:]
        if valset is not None:
            return trainset, valset
        if valset_ratio == 0:
            return trainset, None

        num_val_examples = int(valset_ratio * len(trainset))
        if num_val_examples == 0:
            return trainset, None
        return trainset[num_val_examples:], trainset[:num_val_examples]

    def _prepare_strategy(self, strategy: str) -> list[str]:
        if strategy is None:
            strategy = self.STRAT_SEP.join(self.optimizers)
        if not strategy.strip():
            raise ValueError("strategy cannot be empty")
        parsed = strategy.split(self.STRAT_SEP)
        invalid = [step for step in parsed if step not in self.optimizers]
        if invalid:
            raise ValueError(
                f"Strategy contains invalid optimizer keys: {invalid}. Valid keys are: {list(self.optimizers.keys())}"
            )
        return parsed

    def _prepare_optimizer_compile_args(
        self,
        optimizer_compile_args: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not optimizer_compile_args:
            return {}

        for optimizer_key, compile_args in optimizer_compile_args.items():
            if optimizer_key not in self.optimizers:
                raise ValueError(
                    f"Invalid optimizer key '{optimizer_key}'. Valid keys are: {list(self.optimizers.keys())}"
                )
            if "student" in compile_args:
                raise ValueError(
                    f"'student' is not allowed in optimizer_compile_args for optimizer '{optimizer_key}'."
                )
            valid_params = inspect.signature(self.optimizers[optimizer_key].compile).parameters
            invalid_args = set(compile_args) - set(valid_params)
            if invalid_args:
                raise ValueError(
                    f"Invalid compile arguments for optimizer '{optimizer_key}': {sorted(invalid_args)}. "
                    f"{type(self.optimizers[optimizer_key]).__name__}.compile() accepts: {list(valid_params.keys())}"
                )
        return optimizer_compile_args

    def _run_optimizer(
        self,
        optimizer: Teleprompter,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | list[Module] | None,
        valset: list[Example] | None,
        compile_args: dict[str, Any],
    ) -> Module:
        potential_args = {
            "trainset": trainset,
            "teacher": teacher,
            "valset": valset,
            **compile_args,
        }
        accepted_params = set(inspect.signature(optimizer.compile).parameters)
        filtered_args = {key: value for key, value in potential_args.items() if key in accepted_params}
        student._compiled = False
        return optimizer.compile(student, **filtered_args)

    def _evaluate_on_valset(
        self,
        program: Module,
        valset: list[Example] | None,
        *,
        num_threads: int | None,
        max_errors: int | None,
        provide_traceback: bool | None,
    ) -> float | None:
        if not valset:
            return None

        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=num_threads,
            max_errors=max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )
        return evaluate(program).score

    def _add_candidate(self, candidates: list[dict[str, Any]], program: Module, strategy: str, score: float | None) -> None:
        candidates.append(
            {
                "score": score,
                "program": program.deepcopy(),
                "strategy": strategy,
            }
        )
