import dspy
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.tool import Tool
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_adapter_parse_start(self, call_id, instance, inputs):
        self.events.append(("adapter_parse_start", inputs))

    def on_adapter_parse_end(self, call_id, outputs, exception=None):
        self.events.append(("adapter_parse_end", outputs, exception))

    def on_tool_start(self, call_id, instance, inputs):
        self.events.append(("tool_start", inputs))

    def on_tool_end(self, call_id, outputs, exception=None):
        self.events.append(("tool_end", outputs, exception))


def increment(value: int) -> int:
    return value + 1


def test_adapter_parse_callbacks_do_not_require_top_level_adapter_export():
    callback = RecordingCallback()
    adapter = JSONAdapter()

    with dspy.context(callbacks=[callback]):
        parsed = adapter.parse(dspy.Signature("question -> answer"), '{"answer": "ok"}')

    assert not hasattr(dspy, "Adapter")
    assert parsed == {"answer": "ok"}
    assert [event[0] for event in callback.events] == ["adapter_parse_start", "adapter_parse_end"]


def test_tool_callbacks_do_not_require_top_level_tool_export():
    callback = RecordingCallback()
    tool = Tool(increment)

    with dspy.context(callbacks=[callback]):
        result = tool(value=1)

    assert not hasattr(dspy, "Tool")
    assert result == 2
    assert [event[0] for event in callback.events] == ["tool_start", "tool_end"]
