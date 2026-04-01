import dspy
from tests.minimal.helpers.dummies import DummyLM


def test_parallel_module():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
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

