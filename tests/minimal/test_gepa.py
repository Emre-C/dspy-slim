import threading
from typing import Any

import pytest
from gepa import EvaluationBatch

import dspy
from dspy import Example
from dspy.clients.base_lm import BaseLM
from dspy.predict import Predict
from tests.minimal.helpers.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


class DictDummyLM(BaseLM):
    def __init__(self, history):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.history = {}
        for m in history:
            self.history[hash(repr(m["messages"]))] = m

    def __call__(self, prompt=None, messages=None, **kwargs):
        assert hash(repr(messages)) in self.history, f"Message {messages} not found in history"
        m = self.history[hash(repr(messages))]
        return m["outputs"]


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=example.output == prediction.output, feedback="Wrong answer.")


def bad_metric(example, prediction):
    return 0.0


@pytest.mark.parametrize("reflection_minibatch_size, batch, expected_callback_metadata", [
    (None, [], {"metric_key": "eval_full"}),
    (None, [Example(input="What is the color of the sky?", output="blue")], {"metric_key": "eval_full"}),
    (1, [], {"disable_logging": True}),
    (1, [
        Example(input="What is the color of the sky?", output="blue"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!"),
    ], {"metric_key": "eval_full"}),
])
def test_gepa_adapter_disables_logging_on_minibatch_eval(monkeypatch, reflection_minibatch_size, batch, expected_callback_metadata):
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module
    from dspy.teleprompt.gepa import gepa_utils

    class DummyModule(dspy.Module):
        def forward(self, **kwargs):  # pragma: no cover - stub forward
            return dspy.Prediction()

    adapter = gepa_utils.DspyAdapter(
        student_module=SimpleModule("input -> output"),
        metric_fn=simple_metric,
        feedback_map={},
        failure_score=0.0,
        reflection_minibatch_size=reflection_minibatch_size,
    )

    captured_kwargs: dict[str, Any] = {}

    def dummy_bootstrap_trace_data(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(bootstrap_trace_module, "bootstrap_trace_data", dummy_bootstrap_trace_data)
    monkeypatch.setattr(
        gepa_utils.DspyAdapter,
        "build_program",
        lambda self, candidate: DummyModule(),
    )

    adapter.evaluate(batch=batch, candidate={}, capture_traces=True)

    assert captured_kwargs["callback_metadata"] == expected_callback_metadata


def test_metric_requires_feedback_signature():
    reflection_lm = DictDummyLM([])
    with pytest.raises(TypeError):
        dspy.GEPA(metric=bad_metric, reflection_lm=reflection_lm, max_metric_calls=1)


def any_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    return 0.0


def test_gepa_compile_with_track_usage_no_tuple_error(caplog):
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")]

    task_lm = DummyLM([{"answer": "mock answer 1"}])
    reflection_lm = DummyLM([{"new_instruction": "Something new."}])

    compiled_container: dict[str, Any] = {}
    exc_container: dict[str, BaseException] = {}

    def run_compile():
        try:
            with dspy.context(lm=task_lm, track_usage=True):
                optimizer = dspy.GEPA(metric=any_metric, reflection_lm=reflection_lm, max_metric_calls=3)
                compiled_container["prog"] = optimizer.compile(student, trainset=trainset, valset=trainset)
        except BaseException as e:
            exc_container["e"] = e

    t = threading.Thread(target=run_compile, daemon=True)
    t.start()
    t.join(timeout=120.0)

    assert not t.is_alive(), "GEPA.compile did not complete within timeout."
    assert "'tuple' object has no attribute 'set_lm_usage'" not in caplog.text

    if "e" in exc_container:
        pytest.fail(f"GEPA.compile raised unexpectedly: {exc_container['e']}")

    if "prog" not in compiled_container:
        pytest.fail("GEPA.compile did return a program.")


class TwoPredictorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.a = Predict("input -> answer_a")
        self.b = Predict("input -> answer_b")

    def forward(self, **kwargs):
        return self.a(**kwargs)


def predictor_feedback(*args, **kwargs):
    return dspy.Prediction(score=1.0, feedback="Looks good.")


def test_gepa_reflective_dataset_skips_predictors_without_trace_examples(caplog):
    from dspy.teleprompt.gepa import gepa_utils

    student = TwoPredictorModule()
    adapter = gepa_utils.DspyAdapter(
        student_module=student,
        metric_fn=simple_metric,
        feedback_map={"a": predictor_feedback, "b": predictor_feedback},
        failure_score=0.0,
    )

    example = Example(input="What is the color of the sky?", output="blue").with_inputs("input")
    prediction = dspy.Prediction(answer_a="blue")
    eval_batch = EvaluationBatch(
        outputs=[prediction],
        scores=[1.0],
        trajectories=[{
            "example_ind": 0,
            "example": example,
            "prediction": prediction,
            "trace": [(student.a, {"input": "What is the color of the sky?"}, prediction)],
            "score": 1.0,
        }],
    )

    dataset = adapter.make_reflective_dataset({}, eval_batch, ["a", "b"])

    assert list(dataset.keys()) == ["a"]
    assert "No valid reflective examples found" not in caplog.text


def test_gepa_propose_new_texts_returns_empty_when_no_components_have_reflective_data():
    from dspy.teleprompt.gepa import gepa_utils

    adapter = gepa_utils.DspyAdapter(
        student_module=SimpleModule("input -> output"),
        metric_fn=simple_metric,
        feedback_map={},
        reflection_lm=DummyLM([{"new_instruction": "Unused."}]),
        failure_score=0.0,
    )

    result = adapter.propose_new_texts(
        candidate={"predictor": "Current instruction"},
        reflective_dataset={},
        components_to_update=["predictor"],
    )

    assert result == {}


def test_gepa_adapter_evaluate_preserves_subscores(monkeypatch):
    from dspy.teleprompt.gepa import gepa_utils

    adapter = gepa_utils.DspyAdapter(
        student_module=SimpleModule("input -> output"),
        metric_fn=simple_metric,
        feedback_map={},
        failure_score=0.0,
    )

    class DummyEvaluate:
        def __init__(self, **kwargs):
            pass

        def __call__(self, program):
            return dspy.Prediction(
                results=[
                    (
                        None,
                        dspy.Prediction(output="blue"),
                        dspy.Prediction(score=1.0, feedback="ok", subscores={"accuracy": 1.0}),
                    )
                ]
            )

    monkeypatch.setattr(gepa_utils, "Evaluate", DummyEvaluate)

    result = adapter.evaluate(
        batch=[Example(input="What is the color of the sky?", output="blue")],
        candidate={},
        capture_traces=False,
    )

    assert result.scores == [1.0]
    assert result.objective_scores == [{"accuracy": 1.0}]


def test_gepa_reflective_dataset_warns_once_on_score_mismatch(capsys):
    from dspy.teleprompt.gepa import gepa_utils

    student = TwoPredictorModule()
    adapter = gepa_utils.DspyAdapter(
        student_module=student,
        metric_fn=simple_metric,
        feedback_map={"a": lambda *args, **kwargs: dspy.Prediction(score=0.0, feedback="Needs work.")},
        failure_score=0.0,
    )

    example = Example(input="What is the color of the sky?", output="blue").with_inputs("input")
    prediction = dspy.Prediction(answer_a="blue")
    eval_batch = EvaluationBatch(
        outputs=[prediction],
        scores=[1.0],
        trajectories=[{
            "example_ind": 0,
            "example": example,
            "prediction": prediction,
            "trace": [(student.a, {"input": "What is the color of the sky?"}, prediction)],
            "score": 1.0,
        }],
    )

    adapter.make_reflective_dataset({}, eval_batch, ["a"])
    assert "The score returned by the metric with pred_name is different" in capsys.readouterr().err

    adapter.make_reflective_dataset({}, eval_batch, ["a"])
    assert "The score returned by the metric with pred_name is different" not in capsys.readouterr().err
    assert adapter.warn_on_score_mismatch is False


def test_gepa_compile_forwards_new_optimize_kwargs(monkeypatch):
    import gepa as gepa_pkg

    captured_kwargs: dict[str, Any] = {}

    class StubResult:
        def __init__(self):
            self.best_candidate = {"predictor": "Updated instruction"}

    def fake_optimize(**kwargs):
        captured_kwargs.update(kwargs)
        return StubResult()

    monkeypatch.setattr(gepa_pkg, "optimize", fake_optimize)

    optimizer = dspy.GEPA(
        metric=any_metric,
        reflection_lm=DummyLM([{"new_instruction": "Something new."}]),
        max_metric_calls=1,
        candidate_selection_strategy="top_k_pareto",
        use_mlflow=True,
        mlflow_tracking_uri="sqlite:///tmp/mlflow.db",
        mlflow_experiment_name="gepa-tests",
        raise_on_exception=False,
    )
    program = optimizer.compile(
        SimpleModule("input -> output"),
        trainset=[Example(input="What is the color of the sky?", output="blue").with_inputs("input")],
        valset=[Example(input="What is the color of the sky?", output="blue").with_inputs("input")],
    )

    assert program.predictor.signature.instructions == "Updated instruction"
    assert captured_kwargs["candidate_selection_strategy"] == "top_k_pareto"
    assert captured_kwargs["mlflow_tracking_uri"] == "sqlite:///tmp/mlflow.db"
    assert captured_kwargs["mlflow_experiment_name"] == "gepa-tests"
    assert captured_kwargs["raise_on_exception"] is False
