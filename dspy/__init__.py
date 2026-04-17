from dspy.evaluate import Evaluate
from dspy.predict import ChainOfThought, Parallel, Predict, RLM, ReAct
from dspy.primitives import BaseModule, Example, Module, Prediction
from dspy.signatures import (
    InputField,
    OutputField,
    Signature,
    SignatureMeta,
    ensure_signature,
    infer_prefix,
    make_signature,
)
from dspy.teleprompt import GEPA, bootstrap_trace_data

from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, Tool, ToolCalls
from dspy.clients import BaseLM, DSPY_CACHE, LM
from dspy.streaming.streamify import streamify
from dspy.utils.asyncify import asyncify
from dspy.utils.exceptions import ContextWindowExceededError
from dspy.utils.logging_utils import configure_dspy_loggers
from dspy.utils.saving import load
from dspy.utils.usage_tracker import track_usage

from dspy.dsp.utils.settings import settings
from dspy.__metadata__ import __author__, __author_email__, __description__, __name__, __url__, __version__

configure_dspy_loggers(__name__)

configure = settings.configure
context = settings.context
cache = DSPY_CACHE

__all__ = [
    "BaseLM",
    "BaseModule",
    "ChainOfThought",
    "ChatAdapter",
    "GEPA",
    "Evaluate",
    "Example",
    "InputField",
    "Adapter",
    "JSONAdapter",
    "LM",
    "Module",
    "OutputField",
    "Parallel",
    "Prediction",
    "Predict",
    "ReAct",
    "RLM",
    "Signature",
    "SignatureMeta",
    "Tool",
    "ToolCalls",
    "asyncify",
    "bootstrap_trace_data",
    "load",
    "configure",
    "context",
    "ensure_signature",
    "infer_prefix",
    "make_signature",
    "ContextWindowExceededError",
    "streamify",
    "track_usage",
]
