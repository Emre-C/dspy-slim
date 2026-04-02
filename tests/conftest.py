import copy
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    # Match scripts/e2e_runner.py and RLM integration tests: Deno is often installed outside default PATH.
    for p in (Path.home() / ".deno" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


@pytest.fixture(autouse=True)
def clear_settings():
    yield
    import dspy
    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG))


@pytest.fixture
def lm_for_test():
    import os

    model = os.environ.get("LM_FOR_TEST", None)
    if model is None:
        pytest.skip("LM_FOR_TEST is not set in the environment variables")
    return model
