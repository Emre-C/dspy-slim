import sys

import dspy


def get_dependency_versions():
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": dspy.__version__,
    }
