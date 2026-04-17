import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.dsp.utils.utils import dotdict


def _chat_chunk(content: str):
    return dotdict(choices=[dotdict(delta=dotdict(content=content))])


def test_stream_listener_captures_chat_field_and_marks_last_chunk():
    listener = dspy.streaming.StreamListener(signature_field_name="answer", predict_name="predict")

    with dspy.context(adapter=ChatAdapter()):
        assert listener.receive(_chat_chunk("[[ ## answer ## ]]\nHel")) is None
        chunk = listener.receive(_chat_chunk("lo"))
        final_chunk = listener.receive(_chat_chunk("\n\n[[ ## completed ## ]]"))

    assert chunk.predict_name == "predict"
    assert chunk.signature_field_name == "answer"
    assert chunk.chunk == "Hello"
    assert chunk.is_last_chunk is False
    assert final_chunk.chunk == ""
    assert final_chunk.is_last_chunk is True


def test_stream_listener_supports_chat_adapter_subclasses():
    class DerivedChatAdapter(ChatAdapter):
        pass

    listener = dspy.streaming.StreamListener(signature_field_name="answer", predict_name="predict")

    with dspy.context(adapter=DerivedChatAdapter()):
        assert listener.receive(_chat_chunk("[[ ## answer ## ]]\nHel")) is None
        chunk = listener.receive(_chat_chunk("lo"))
        final_chunk = listener.receive(_chat_chunk("\n\n[[ ## completed ## ]]"))

    assert chunk.chunk == "Hello"
    assert chunk.is_last_chunk is False
    assert final_chunk.is_last_chunk is True


def test_stream_listener_detects_partial_end_identifiers_for_kept_adapters():
    listener = dspy.streaming.StreamListener(signature_field_name="answer")

    assert listener._could_form_end_identifier("some text [[ ##", "ChatAdapter") is True
    assert listener._could_form_end_identifier("hello world", "ChatAdapter") is False
    assert listener._could_form_end_identifier('some text "', "JSONAdapter") is True
    assert listener._could_form_end_identifier("hello world", "JSONAdapter") is False


def test_stream_listener_finalizes_json_adapter_buffers():
    listener = dspy.streaming.StreamListener(signature_field_name="answer", predict_name="predict")

    with dspy.context(adapter=JSONAdapter()):
        assert listener.receive(_chat_chunk('{"ans')) is None
        chunk = listener.receive(_chat_chunk('wer": "Hel'))
        assert listener.receive(_chat_chunk('lo"')) is None
        final_chunk = listener.finalize()

    assert chunk.predict_name == "predict"
    assert chunk.signature_field_name == "answer"
    assert chunk.chunk == '"Hel'
    assert chunk.is_last_chunk is False
    assert final_chunk.chunk == 'lo"'
    assert final_chunk.is_last_chunk is True
