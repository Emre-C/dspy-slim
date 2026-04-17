import asyncio
import threading

import pytest

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from tests.minimal.helpers.dummies import DummyLM


def test_parallel_module():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ],
        adapter=ChatAdapter(),
    )
    dspy.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output")

            self.parallel = dspy.Parallel(num_threads=2)

        def forward(self, input):
            return self.parallel(
                [
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                ]
            )

    output = MyModule()(dspy.Example(input="test input").with_inputs("input"))

    expected_outputs = {f"test output {i}" for i in range(1, 6)}
    assert {r.output for r in output} == expected_outputs


def test_parallel_preserves_usage_tracking_per_result():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
        ],
        adapter=ChatAdapter(),
    )

    predictor = dspy.Predict("input -> output")
    parallel = dspy.Parallel(num_threads=2)

    with dspy.context(lm=lm, track_usage=True):
        results = parallel(
            [
                (predictor, {"input": "first"}),
                (predictor, {"input": "second"}),
            ]
        )

    assert {result.output for result in results} == {"test output 1", "test output 2"}
    assert all(
        result.get_lm_usage() == {
            "dummy": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        for result in results
    )


def test_asyncify_cancels_without_waiting_for_blocked_worker():
    started = threading.Event()
    release = threading.Event()

    class BlockingProgram(dspy.Module):
        def forward(self):
            started.set()
            release.wait(timeout=1.0)
            return "done"

    async_program = dspy.asyncify(BlockingProgram())

    async def run_and_cancel():
        task = asyncio.create_task(async_program())
        for _ in range(50):
            if started.is_set():
                break
            await asyncio.sleep(0.01)

        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)

    try:
        asyncio.run(run_and_cancel())
    finally:
        release.set()
