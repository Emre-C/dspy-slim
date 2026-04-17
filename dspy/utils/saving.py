import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

try:  # pragma: no cover - optional dependency path.
    import cloudpickle as _pickle_module
except ImportError:  # pragma: no cover - exercised when cloudpickle is absent.
    _pickle_module = pickle

if TYPE_CHECKING:
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)


def get_pickle_module():
    return _pickle_module


def get_dependency_versions():
    import dspy

    pickle_version = getattr(_pickle_module, "__version__", "stdlib")
    if pickle_version != "stdlib":
        pickle_version = ".".join(str(pickle_version).split(".")[:2])

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": dspy.__version__,
        "pickle_module": pickle_version,
    }


def load(path: str, allow_pickle: bool = False) -> "Module":
    if not allow_pickle:
        raise ValueError(
            "Loading with pickle is not allowed. Please set `allow_pickle=True` if you trust the source."
        )

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"The path '{path_obj}' does not exist.")

    with open(path_obj / "metadata.json", "rb") as handle:
        metadata = orjson.loads(handle.read())

    dependency_versions = get_dependency_versions()
    saved_dependency_versions = metadata["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions.get(key) != saved_version:
            logger.warning(
                "There is a mismatch of %s version between saved model and current environment. "
                "Saved with `%s`, current `%s`.",
                key,
                saved_version,
                dependency_versions.get(key),
            )

    with open(path_obj / "program.pkl", "rb") as handle:
        return _pickle_module.load(handle)
