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

from dspy.adapters import JSONAdapter
from dspy.clients import BaseLM, DSPY_CACHE, LM
from dspy.utils.exceptions import ContextWindowExceededError
from dspy.utils.logging_utils import configure_dspy_loggers

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
    "GEPA",
    "Evaluate",
    "Example",
    "InputField",
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
    "bootstrap_trace_data",
    "configure",
    "context",
    "ensure_signature",
    "infer_prefix",
    "make_signature",
    "ContextWindowExceededError",
]
