import dspy

from dspy.teleprompt.teleprompt import Teleprompter


class IncrementProgram(dspy.Module):
    def __init__(self, value: int = 0):
        super().__init__()
        self.value = value

    def forward(self, dummy: str | None = None):
        return dspy.Prediction(value=self.value)


class AddValueOptimizer(Teleprompter):
    def __init__(self, amount: int):
        self.amount = amount

    def compile(self, student, *, trainset, **kwargs):
        student = student.deepcopy()
        student.value += self.amount
        return student


def score_value(example, prediction, trace=None):
    return float(prediction.value)


def test_better_together_sequences_optimizers_and_keeps_best_candidate():
    optimizer = dspy.BetterTogether(
        metric=score_value,
        plus_one=AddValueOptimizer(1),
        plus_two=AddValueOptimizer(2),
    )
    student = IncrementProgram()
    dataset = [dspy.Example(dummy="ignored", answer="ignored").with_inputs("dummy")]

    compiled = optimizer.compile(
        student,
        trainset=dataset,
        valset=dataset,
        strategy="plus_one -> plus_two",
    )

    assert compiled.value == 3
    assert compiled.candidate_programs[0]["strategy"] == "plus_one -> plus_two"
    assert compiled.candidate_programs[0]["score"] == 300.0
    assert compiled.flag_compilation_error_occurred is False
