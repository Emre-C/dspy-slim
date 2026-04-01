import json
import random
from collections import defaultdict
from typing import Any

from dspy.adapters.chat_adapter import FieldInfoWithName, field_header_pattern
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.signatures.field import OutputField


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _normalize_row(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    if n < 1e-10:
        return v
    return [x / n for x in v]


class DummyLM(BaseLM):
    """Dummy language model for unit tests."""

    def __init__(
        self,
        answers: list[dict[str, Any]] | dict[str, dict[str, Any]],
        follow_examples: bool = False,
        reasoning: bool = False,
        adapter=None,
    ):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples
        self.reasoning = reasoning

        self.adapter = adapter if adapter is not None else JSONAdapter()

    def _use_example(self, messages):
        fields = defaultdict(int)
        for message in messages:
            if "content" in message:
                if ma := field_header_pattern.match(message["content"]):
                    fields[message["content"][ma.start() : ma.end()]] += 1
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]

        final_input = messages[-1]["content"].split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            if any(field in output["content"] for field in output_fields) and final_input in input["content"]:
                return output["content"]

    def _format_answer_fields(self, field_names_and_values: dict[str, Any]):
        if isinstance(self.adapter, JSONAdapter):
            return json.dumps(field_names_and_values, ensure_ascii=False)
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=OutputField()): value
            for field_name, value in field_names_and_values.items()
        }
        try:
            return self.adapter.format_field_with_value(fields_with_values, role="assistant")
        except TypeError:
            return self.adapter.format_field_with_value(fields_with_values)

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        choices = []
        for _ in range(kwargs.get("n", 1)):
            if self.follow_examples:
                current_output = self._use_example(messages)
            elif isinstance(self.answers, dict):
                current_output = next(
                    (self._format_answer_fields(v) for k, v in self.answers.items() if k in messages[-1]["content"]),
                    "No more responses",
                )
            else:
                current_output = self._format_answer_fields(next(self.answers, {"answer": "No more responses"}))

            message = dotdict(content=current_output, tool_calls=None)
            if self.reasoning:
                message.reasoning_content = "Some reasoning"
            choices.append(dotdict(message=message, finish_reason="stop"))

        return dotdict(
            choices=choices,
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def get_convo(self, index):
        return self.history[index]["messages"], self.history[index]["outputs"]


class DummyVectorizer:
    """Simple n-gram bag-of-vectors (for tests; no numpy)."""

    def __init__(self, max_length=100, n_gram=2):
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        h = 1
        for coeff, c in zip(self.coeffs, gram, strict=False):
            h = h * coeff + ord(c)
            h %= self.P
        return h % self.max_length

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            grams = [text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)]
            vec = [0.0] * self.max_length
            for gram in grams:
                vec[self._hash(gram)] += 1.0
            m = sum(vec) / len(vec) if vec else 0.0
            vec = [v - m for v in vec]
            out.append(_normalize_row(vec))
        return out


def dummy_rm(passages=()) -> callable:
    if not passages:

        def inner(query: str, *, k: int, **kwargs):
            raise ValueError("No passages defined")

        return inner
    max_length = max(map(len, passages)) + 100
    vectorizer = DummyVectorizer(max_length)
    passage_vecs = vectorizer(passages)

    def inner(query: str, *, k: int, **kwargs):
        assert k <= len(passages)
        query_vec = vectorizer([query])[0]
        scores = [_dot(pv, query_vec) for pv in passage_vecs]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [dotdict(long_text=passages[i]) for i in order]

    return inner
