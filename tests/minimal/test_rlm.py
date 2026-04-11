"""
Tests for the RLM (Recursive Language Model) module.

Test organization:
- Unit tests (no Deno required): MockInterpreter, RLM formatting, signatures
- Integration tests (@pytest.mark.deno): PythonInterpreter with Deno
"""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from dspy.adapters.types.tool import Tool
from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from tests.minimal.helpers.replay_lm import ReplayLM
from tests.mock_interpreter import MockInterpreter

_RLM_REPLAY_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "spec" / "fixtures" / "rlm_replay.json"

# ============================================================================
# Test Helpers and Factories
# ============================================================================


def make_mock_predictor(responses: list[dict], async_mode: bool = False):
    """Factory for mock predictors with scripted responses.

    Args:
        responses: List of dicts with keys like 'reasoning', 'code'.
        async_mode: If True, returns a predictor with acall() instead of __call__().
    """

    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def _next_response(self):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**result)

        def __call__(self, **kwargs):
            return self._next_response()

        async def acall(self, **kwargs):
            return self._next_response()

    return MockPredictor()


@contextmanager
def dummy_lm_context(responses: list[dict]):
    """Context manager for DummyLM setup."""
    import dspy
    from tests.minimal.helpers.dummies import DummyLM

    lm = DummyLM(responses)
    with dspy.context(lm=lm):
        yield lm


# Common test tools
def echo_tool(text: str = "") -> str:
    """Echo the input text."""
    return f"Echo: {text}"


def add_tool(a: int = 0, b: int = 0) -> str:
    """Add two numbers."""
    return str(a + b)


def multiply_tool(a: int = 0, b: int = 0) -> str:
    """Multiply two numbers."""
    return str(a * b)


def _load_shared_rlm_replay_cases() -> list[dict]:
    return json.loads(_RLM_REPLAY_FIXTURE_PATH.read_text())["cases"]


def _python_interpreter_response(raw: object):
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported interpreter response fixture: {raw!r}")
    if "submit" in raw:
        return FinalOutput(raw["submit"])
    if "error" in raw:
        return CodeInterpreterError(str(raw["error"]))
    raise ValueError(f"Unsupported interpreter response fixture: {raw!r}")

# ============================================================================
# Unit Tests: MockInterpreter
# ============================================================================


class TestMockInterpreter:
    """Unit tests for MockInterpreter."""

    def test_scripted_responses(self):
        """Test that MockInterpreter returns scripted responses in order."""
        mock = MockInterpreter(responses=["first", "second", "third"])
        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

    def test_returns_final_output_result(self):
        """Test that MockInterpreter can return FinalOutput."""
        mock = MockInterpreter(responses=["exploring", FinalOutput("42")])
        assert mock.execute("print(len(data))") == "exploring"
        result = mock.execute("SUBMIT('42')")
        assert isinstance(result, FinalOutput)
        assert result.output == "42"

    def test_raises_exception_from_responses(self):
        """Test that MockInterpreter raises exceptions from responses."""
        mock = MockInterpreter(responses=["ok", CodeInterpreterError("undefined variable")])
        assert mock.execute("code1") == "ok"
        with pytest.raises(CodeInterpreterError, match="undefined variable"):
            mock.execute("code2")

    def test_records_call_history(self):
        """Test that MockInterpreter records call history for test assertions."""
        mock = MockInterpreter(responses=["resp"])
        mock.execute("print(1)", variables={"x": 10})
        assert mock.call_history == [("print(1)", {"x": 10})]


# ============================================================================
# Unit Tests: RLM Module (no interpreter needed)
# ============================================================================


class TestRLMInitialization:
    """Tests for RLM module initialization."""

    def test_basic_initialization(self):
        """Test RLM module initializes correctly with signature."""
        rlm = RLM("context, query -> answer", max_iterations=5)
        assert rlm.max_iterations == 5
        assert rlm.generate_action is not None
        assert rlm.extract is not None
        assert rlm.tools == {}  # No user tools provided
        assert "context" in rlm.signature.input_fields
        assert "query" in rlm.signature.input_fields
        assert "answer" in rlm.signature.output_fields

    def test_custom_signature(self):
        """Test RLM with custom signature."""
        rlm = RLM("document, question -> summary, key_facts", max_iterations=5)
        assert "document" in rlm.signature.input_fields
        assert "question" in rlm.signature.input_fields
        assert "summary" in rlm.signature.output_fields
        assert "key_facts" in rlm.signature.output_fields

    def test_custom_tools(self):
        """Test RLM with custom tools."""
        def custom_tool(x: str = "") -> str:
            return x.upper()

        rlm = RLM("context -> answer", max_iterations=5, tools=[custom_tool])
        assert "custom_tool" in rlm.tools
        assert len(rlm.tools) == 1  # Only user tools, not internal llm_query/llm_query_batched

    @pytest.mark.parametrize("tool_name", ["invalid-name", "123start"])
    def test_tool_validation_invalid_identifier(self, tool_name):
        """Test RLM rejects tool names that aren't valid Python identifiers."""
        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, name=tool_name)
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            RLM("context -> answer", tools=[tool])

    @pytest.mark.parametrize("tool_name", ["llm_query", "SUBMIT", "print"])
    def test_tool_validation_reserved_names(self, tool_name):
        """Test RLM rejects tool names that conflict with built-in functions."""
        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, name=tool_name)
        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM("context -> answer", tools=[tool])

    @pytest.mark.parametrize("invalid_value", ["not a function", 123])
    def test_tool_validation_not_callable(self, invalid_value):
        """Test RLM rejects tools that aren't callable."""
        with pytest.raises(TypeError, match="must be callable"):
            RLM("context -> answer", tools=[invalid_value])

    def test_tools_dict_rejected(self):
        """Test RLM rejects dict format for tools with helpful error."""
        def my_tool() -> str:
            return "result"

        with pytest.raises(TypeError, match="tools must be a list, not a dict"):
            RLM("context -> answer", tools={"my_tool": my_tool})

    def test_optional_parameters(self):
        """Test RLM optional parameters and their defaults."""
        import dspy

        # Test defaults
        rlm = RLM("context -> answer")
        assert rlm.max_llm_calls == 50
        assert rlm.sub_lm is None
        assert rlm._interpreter is None

        # Test custom values
        mock = MockInterpreter()
        mock_lm = dspy.BaseLM("openai/gpt-4o-mini")
        rlm = RLM("context -> answer", max_llm_calls=100, sub_lm=mock_lm, interpreter=mock)
        assert rlm.max_llm_calls == 100
        assert rlm.sub_lm is mock_lm
        assert rlm._interpreter is mock

    def test_forward_validates_required_inputs(self):
        """Test that forward() raises ValueError for missing required inputs."""
        mock = MockInterpreter(responses=["result"])

        # Single missing input
        rlm = RLM("context, query -> answer", max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError, match="Missing required input"):
            rlm.forward(context="some context")  # Missing 'query'

        # Multiple missing inputs - all should be reported
        rlm = RLM("a, b, c -> answer", max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError) as exc_info:
            rlm.forward(a="only a")  # Missing 'b' and 'c'
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_batched_query_errors_have_clear_markers(self):
        """Test that errors in llm_query_batched are prefixed with [ERROR]."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock()
        mock_lm.side_effect = RuntimeError("LM failed")

        rlm = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm._make_llm_tools()

        results = tools["llm_query_batched"](prompts=["test prompt"])
        assert len(results) == 1
        assert results[0].startswith("[ERROR]")
        assert "LM failed" in results[0]

    def test_tools_call_counter_is_thread_safe(self):
        """Test that the LLM call counter is thread-safe for concurrent llm_query_batched calls.

        The call counter must be protected by a lock since llm_query_batched uses
        ThreadPoolExecutor for concurrent execution.
        """
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock

        mock_lm = MagicMock()
        mock_lm.return_value = ["response"]

        rlm = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm._make_llm_tools()

        call_count = [0]
        errors = []

        def make_call():
            try:
                tools["llm_query"](prompt="test")
                call_count[0] += 1
            except RuntimeError as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_call) for _ in range(10)]
            for f in futures:
                f.result()

        assert call_count[0] == 10, f"Expected 10 successful calls, got {call_count[0]}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
            tools["llm_query"](prompt="one more")


class TestRLMCodeFenceParsing:
    """Tests for robust fenced-code extraction."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Standard python fence
            ("```python\nprint(1)\n```", "print(1)"),
            ("```py\nx = 1\nprint(x)\n```", "x = 1\nprint(x)"),
            # Bare fence (no language tag)
            ("```\nprint('no lang')\n```", "print('no lang')"),
            # No fences at all
            ("not fenced code", "not fenced code"),
            # Text before fence (preamble is skipped)
            ("I'll inspect first.\n```python\nprint('hello')\n```\nThen I will submit.", "print('hello')"),
            # Text after closing fence (ignored)
            ("```python\nprint(1)\n```\nsome trailing text", "print(1)"),
            # Unclosed fence (just return the body)
            ("```python\nprint('oops')", "print('oops')"),
            # Double fences (outer decorative ```)
            ("```\n```python\nprint(1)\n```\n```", "print(1)"),
            ("```\n```\nprint(2)\n```\n```", "print(2)"),
        ],
    )
    def test_strip_code_fences(self, raw, expected):
        assert _strip_code_fences(raw) == expected

    def test_strip_code_fences_rejects_non_python_lang(self):
        with pytest.raises(SyntaxError, match="json"):
            _strip_code_fences('```json\n{"a": 1}\n```')


class TestRLMFormatting:
    """Tests for RLM formatting helpers."""

    def test_format_history(self):
        """Test history formatting using REPLHistory."""
        history = REPLHistory()
        history = history.append(reasoning="Need to check the data", code="print(1)", output="1")
        history = history.append(reasoning="Now calculate", code="x = 2", output="")
        formatted = history.format()
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "print(1)" in formatted
        assert "Need to check" in formatted

    def test_format_history_empty(self):
        """Test history formatting with empty history."""
        history = REPLHistory()
        formatted = history.format()
        assert "have not interacted with the REPL" in formatted

    def test_action_signature_has_iteration_field(self):
        """Test action signature includes iteration input field."""
        rlm = RLM("context -> answer")
        action_sig = rlm.generate_action.signature
        assert "iteration" in action_sig.input_fields

    def test_format_output(self):
        """Test output formatting."""
        rlm = RLM("context -> answer")
        formatted = rlm._format_output("output text")
        assert "output text" in formatted

    def test_format_output_empty(self):
        """Test output formatting with empty output."""
        rlm = RLM("context -> answer")
        formatted = rlm._format_output("")
        assert "no output" in formatted.lower()

    def test_format_output_passthrough(self):
        """Test that _format_output passes through non-empty output without truncation."""
        rlm = RLM("context -> answer", max_output_chars=100)
        long_output = "a" * 200
        formatted = rlm._format_output(long_output)
        assert formatted == long_output

    def test_format_variable_info_string(self):
        """Test variable info formatting for string value using REPLVariable."""
        var = REPLVariable.from_value("context", "Hello world", preview_chars=5)
        formatted = var.format()
        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "11" in formatted  # length
        assert "He" in formatted  # head
        assert "ld" in formatted  # tail
        assert "..." in formatted  # truncation indicator

    def test_format_variable_info_dict(self):
        """Test variable info formatting for dict value using REPLVariable."""
        var = REPLVariable.from_value("data", {"key": "value"})
        formatted = var.format()
        assert "Variable: `data`" in formatted
        assert "Type: dict" in formatted
        assert "key" in formatted

    def test_build_variables_multiple(self):
        """Test building multiple variables."""
        rlm = RLM("context, query -> answer")
        variables = rlm._build_variables(
            context="Hello world",
            query="What is this?"
        )
        assert len(variables) == 2
        formatted = "\n\n".join(v.format() for v in variables)
        assert "Variable: `context`" in formatted
        assert "Variable: `query`" in formatted
        assert "Hello world" in formatted
        assert "What is this?" in formatted


class TestREPLTypes:
    """Tests for the REPL type classes."""

    def test_repl_history_immutability(self):
        """Test that REPLHistory.append() returns new instance."""
        h1 = REPLHistory()
        h2 = h1.append(code="print(1)", output="1")
        assert len(h1) == 0  # Original unchanged
        assert len(h2) == 1  # New has entry

    def test_repl_history_len_iter_bool(self):
        """Test REPLHistory list-like interface."""
        h = REPLHistory()
        assert len(h) == 0
        assert not bool(h)

        h = h.append(code="x = 1", output="")
        h = h.append(code="x = 2", output="")
        assert len(h) == 2
        assert bool(h)

        codes = [e.code for e in h]
        assert codes == ["x = 1", "x = 2"]

    def test_repl_entry_format(self):
        """Test REPLEntry formatting."""
        entry = REPLEntry(reasoning="test reason", code="print(1)", output="1")
        formatted = entry.format(index=0)
        assert "Step 1" in formatted
        assert "test reason" in formatted
        assert "print(1)" in formatted
        assert "1" in formatted

    def test_repl_entry_format_truncation(self):
        """Test REPLEntry.format() truncates with head+tail and shows true length."""
        output = "a" * 100 + "b" * 100
        entry = REPLEntry(code="print('a' + 'b')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        # Head and tail preserved
        assert "a" * 50 in formatted
        assert "b" * 50 in formatted
        assert "100 characters omitted" in formatted
        # True original length shown in header
        assert "200 chars" in formatted

    def test_repl_entry_format_no_truncation(self):
        """Test REPLEntry.format() passes short output through without truncation."""
        output = "a" * 50
        entry = REPLEntry(code="print('a')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        assert output in formatted
        assert "omitted" not in formatted

    def test_repl_history_threads_max_output_chars(self):
        """Test REPLHistory carries max_output_chars through append()."""
        h = REPLHistory(max_output_chars=50)
        h2 = h.append(code="print(1)", output="a" * 100)
        assert h2.max_output_chars == 50
        # Formatting should truncate at 50 chars
        formatted = h2.format()
        assert "50 characters omitted" in formatted

    def test_repl_variable_from_value(self):
        """Test REPLVariable.from_value() factory."""
        var = REPLVariable.from_value("test", "hello world")
        assert var.name == "test"
        assert var.type_name == "str"
        assert var.total_length == 11
        assert "hello world" in var.preview

    def test_repl_variable_truncation(self):
        """Test REPLVariable preview shows head and tail."""
        var = REPLVariable.from_value("big", "a" * 500 + "b" * 500, preview_chars=50)
        assert var.preview.startswith("a" * 25)
        assert var.preview.endswith("b" * 25)
        assert "..." in var.preview

    def test_repl_variable_with_field_info(self):
        """Test REPLVariable includes desc and constraints from field_info."""
        import dspy

        # Create a field with description and constraints
        field = dspy.InputField(desc="The user's question", ge=0, le=100)

        var = REPLVariable.from_value("query", "What is 2+2?", field_info=field)
        assert var.desc == "The user's question"
        assert "greater than or equal to" in var.constraints

        # Verify format includes the metadata
        formatted = var.format()
        assert "Description: The user's question" in formatted
        assert "Constraints:" in formatted

    def test_repl_variable_without_field_info(self):
        """Test REPLVariable works without field_info."""
        var = REPLVariable.from_value("data", [1, 2, 3])
        assert var.desc == ""
        assert var.constraints == ""

        # Format should not include empty desc/constraints lines
        formatted = var.format()
        assert "Description:" not in formatted
        assert "Constraints:" not in formatted

    def test_build_variables_includes_field_metadata(self):
        """Test _build_variables passes field_info to REPLVariable."""
        import dspy

        class QASig(dspy.Signature):
            """Answer questions."""
            context: str = dspy.InputField(desc="Background information")
            question: str = dspy.InputField(desc="The question to answer")
            answer: str = dspy.OutputField()

        rlm = RLM(QASig, max_iterations=3)
        variables = rlm._build_variables(context="Some text", question="What?")

        # Find the context variable
        context_var = next(v for v in variables if v.name == "context")
        assert context_var.desc == "Background information"

        question_var = next(v for v in variables if v.name == "question")
        assert question_var.desc == "The question to answer"


class TestRLMCallMethod:
    """Tests for RLM __call__ method."""

    def test_call_is_alias_for_forward(self):
        """Test that __call__ is an alias for forward()."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = rlm(query="What is the answer?")
        assert result.answer == "42"


class TestRLMMaxIterationsFallback:
    """Tests for max_iterations reached and extract fallback."""

    def test_max_iterations_triggers_extract(self):
        """Test that reaching max_iterations uses extract fallback."""
        mock = MockInterpreter(responses=[
            "exploring...",
            "still exploring...",
            "more exploring...",
        ])
        rlm = RLM("query -> answer", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore 1", "code": "print('exploring')"},
            {"reasoning": "Explore 2", "code": "print('exploring')"},
            {"reasoning": "Explore 3", "code": "print('exploring')"},
        ])
        # Mock the extract predictor to return a value
        rlm.extract = make_mock_predictor([
            {"answer": "extracted_answer"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "extracted_answer"
        assert result.final_reasoning == "Extract forced final output"


class TestRLMToolExceptions:
    """Tests for tool exception handling."""

    def test_tool_exception_returns_error_in_output(self):
        """Test that tool exceptions are caught and returned as errors."""
        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        mock = MockInterpreter(responses=[
            CodeInterpreterError("RuntimeError: Tool failed!"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock, tools=[failing_tool])
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Call tool", "code": "failing_tool()"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"

    def test_runtime_error_history_uses_stripped_code(self):
        """Runtime execution failures should preserve stripped code in history."""
        mock = MockInterpreter(responses=[
            CodeInterpreterError("NameError: name 'x' is not defined"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Will fail", "code": "```python\nprint(x)\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"
        first_step = result.trajectory[0]
        assert first_step["code"] == "print(x)"

    def test_syntax_error_from_execute_is_recoverable(self):
        """SyntaxError from interpreter.execute should be surfaced as an iteration error."""
        mock = MockInterpreter(responses=[
            SyntaxError("invalid syntax"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Bad code", "code": "```python\ndef incomplete(\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error] invalid syntax")

    def test_syntax_error_from_strip_code_fences_is_recoverable(self):
        """SyntaxError raised by _strip_code_fences (e.g. non-Python fence tag) should be recoverable."""
        mock = MockInterpreter(responses=[
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Wrong language", "code": "```bash\nls -la\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error]")


class TestRLMDynamicSignature:
    """Tests for the dynamically built RLM signatures."""

    def test_action_signature_structure(self):
        """Test action signature has required fields and instructions."""
        rlm = RLM("document, question -> summary, answer")
        action_sig = rlm.generate_action.signature

        # Required input/output fields
        assert "variables_info" in action_sig.input_fields
        assert "repl_history" in action_sig.input_fields
        assert "reasoning" in action_sig.output_fields
        assert "code" in action_sig.output_fields

        # Instructions mention key tools and variables
        instructions = action_sig.instructions
        assert "llm_query" in instructions
        assert "llm_query_batched" in instructions
        assert "SUBMIT" in instructions
        assert "`document`" in instructions
        assert "`question`" in instructions
        assert "`summary`" in instructions
        assert "`answer`" in instructions

    def test_paper_instruction_appendix_qwen_coder(self):
        rlm = RLM("context, query -> answer", paper_instruction_appendix="qwen_coder")
        assert "200k" in rlm.generate_action.signature.instructions
        assert "llm_query" in rlm.generate_action.signature.instructions

    def test_paper_instruction_appendix_qwen_small(self):
        rlm = RLM("context, query -> answer", paper_instruction_appendix="qwen_small")
        ins = rlm.generate_action.signature.instructions
        assert "32k" in ins
        assert "10k-15k" in ins

    def test_paper_instruction_appendix_invalid(self):
        with pytest.raises(ValueError, match="paper_instruction_appendix"):
            RLM("context, query -> answer", paper_instruction_appendix="typo")

    def test_extract_signature_structure(self):
        """Test extract signature has required fields for all outputs."""
        rlm = RLM("document, question -> summary, key_facts, confidence")
        extract_sig = rlm.extract.signature
        assert "variables_info" in extract_sig.input_fields
        assert "repl_history" in extract_sig.input_fields
        assert "summary" in extract_sig.output_fields
        assert "key_facts" in extract_sig.output_fields
        assert "confidence" in extract_sig.output_fields


class TestSharedRLMReplayFixtures:
    """Replay shared RLM scenarios through the Python port."""

    @pytest.mark.parametrize("case", _load_shared_rlm_replay_cases(), ids=lambda case: str(case["id"]))
    def test_shared_fixture(self, case):
        import dspy

        python_case = case["python"]
        budget = case.get("budget") or {}
        lm = ReplayLM(outputs=[json.dumps(output) for output in python_case["lm_outputs"]])
        dspy.configure(lm=lm)

        interpreter = MockInterpreter(
            responses=[_python_interpreter_response(item) for item in python_case["interpreter_responses"]],
        )
        rlm = RLM(
            case["signature"],
            max_iterations=int(budget.get("max_iterations", 20)),
            max_llm_calls=int(budget.get("max_llm_calls", 50)),
            interpreter=interpreter,
        )

        result = rlm(**case["inputs"])
        expected = case["expected"]

        assert result.answer == expected["answer"]
        assert result.final_reasoning == expected["python_final_reasoning"]
        assert expected["python_output_contains"] in result.trajectory[0]["output"]
        assert lm.exhausted
