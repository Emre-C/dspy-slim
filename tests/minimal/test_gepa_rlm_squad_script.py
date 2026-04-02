"""Smoke test: `scripts/gepa_rlm_squad.py --dry-run` loads SQuAD and exits 0."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.minimal.helpers.workspace_scripts import minimal_dspy_root

pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")


@pytest.mark.network
def test_gepa_rlm_squad_dry_run_script():
    root = minimal_dspy_root()
    if root is None:
        pytest.skip("Not in minimal_dspy layout (no scripts/ + dspy-slim/ parent).")
    script = root / "scripts" / "gepa_rlm_squad.py"
    dspy_slim = root / "dspy-slim"
    if not script.is_file():
        pytest.skip(f"Shared script is not present: {script}")
    r = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--train", "2", "--val", "1"],
        cwd=str(dspy_slim),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[data]" in r.stdout
    assert "[dry-run]" in r.stdout
