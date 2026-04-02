"""Resolve the parent ``minimal_dspy`` workspace (``scripts/`` + ``dspy-slim/``) from any test path."""

from __future__ import annotations

from pathlib import Path


def minimal_dspy_root() -> Path | None:
    """Return the repo root that contains ``scripts/`` and ``dspy-slim/``, or ``None``."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        squad = ancestor / "scripts" / "gepa_rlm_squad.py"
        slim = ancestor / "dspy-slim" / "pyproject.toml"
        if squad.is_file() and slim.is_file():
            return ancestor
    return None
