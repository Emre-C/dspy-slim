"""Offline sandbox E2E: ``rlm_long_context_validation.py --synthetic`` (Deno + Pyodide)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from tests.minimal.helpers.workspace_scripts import minimal_dspy_root


@pytest.mark.e2e
@pytest.mark.deno
def test_rlm_long_context_synthetic_sandbox_script():
    root = minimal_dspy_root()
    if root is None:
        pytest.skip("Not in minimal_dspy layout (no scripts/ + dspy-slim/ parent).")
    script = root / "scripts" / "rlm_long_context_validation.py"
    dspy_slim = root / "dspy-slim"
    if not script.is_file():
        pytest.skip(f"Shared script is not present: {script}")
    if not shutil.which("deno"):
        pytest.skip("Deno not on PATH")

    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--target-chars",
            "32000",
        ],
        cwd=str(dspy_slim),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[sandbox]" in r.stdout or "SANDBOX" in r.stdout
    assert "[rlm] Skipped" in r.stdout
