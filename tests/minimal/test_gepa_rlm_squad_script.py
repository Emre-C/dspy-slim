"""Smoke test: `scripts/gepa_rlm_squad.py --dry-run` loads SQuAD and exits 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "gepa_rlm_squad.py"
DSPY_SLIM = Path(__file__).resolve().parents[2]


pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")


def test_gepa_rlm_squad_dry_run_script():
    assert SCRIPT.is_file(), f"Missing {SCRIPT}"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--train", "2", "--val", "1"],
        cwd=str(DSPY_SLIM),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[data]" in r.stdout
    assert "[dry-run]" in r.stdout
