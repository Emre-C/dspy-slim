import copy

import pytest


@pytest.fixture(autouse=True)
def clear_settings():
    yield
    import dspy
    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def lm_for_test():
    import os

    model = os.environ.get("LM_FOR_TEST", None)
    if model is None:
        pytest.skip("LM_FOR_TEST is not set in the environment variables")
    return model
