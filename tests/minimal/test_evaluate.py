import threading

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from tests.minimal.helpers.dummies import DummyLM


def new_example(question, answer):
    """Helper function to create a new example."""
    return dspy.Example(
        question=question,
        answer=answer,
    ).with_inputs("question")


def test_evaluate_initialization():
    devset = [new_example("What is 1+1?", "2")]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    assert ev.devset == devset
    assert ev.metric == answer_exact_match
    assert ev.num_threads is None
    assert not ev.display_progress


def test_evaluate_call():
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    assert program(question="What is 1+1?").answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = ev(program)
    assert score.score == 100.0


def test_evaluate_single_thread_runs_in_main_thread():
    """Evaluate with num_threads=1 should run in the main thread."""
    dspy.configure(
        lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}})
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]

    execution_threads = []

    original_metric = answer_exact_match

    def tracking_metric(example, prediction, **kwargs):
        execution_threads.append(threading.current_thread())
        return original_metric(example, prediction, **kwargs)

    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=tracking_metric,
        display_progress=False,
        num_threads=1,
    )
    result = ev(program)
    assert result.score == 100.0
    assert all(t is threading.main_thread() for t in execution_threads)

