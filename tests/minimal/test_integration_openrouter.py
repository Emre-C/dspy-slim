"""End-to-end checks against OpenRouter (skipped unless OPENROUTER_API_KEY is set)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

try:
    from dspy.clients.lm import LM as _OpenAICompatibleLM

    _HAS_LM = True
except Exception:
    _HAS_LM = False


def _load_env() -> None:
    env_path = None
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".env"
        if candidate.is_file():
            env_path = candidate
            break
    if env_path is None:
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _deno_path() -> str | None:
    for p in (Path.home() / ".deno" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("deno")


@pytest.fixture(scope="module")
def openrouter_configured():
    if not _HAS_LM:
        pytest.skip("dspy.LM is unavailable")
    _load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set (e.g. in repo-root .env)")
    return True


def test_predict_live(openrouter_configured):
    import dspy

    lm = _OpenAICompatibleLM(
        "openrouter/qwen/qwen3.6-plus-preview:free",
        temperature=0.5,
        max_tokens=256,
        cache=False,
    )
    dspy.configure(lm=lm)
    out = dspy.Predict("question -> answer")(question="Reply with the single word: OK")
    assert hasattr(out, "answer")
    assert len(str(out.answer)) > 0


@pytest.mark.deno
def test_rlm_live(openrouter_configured):
    if not _deno_path():
        pytest.skip("Deno not installed (brew install deno)")

    import dspy

    lm = _OpenAICompatibleLM(
        "openrouter/qwen/qwen3.6-plus-preview:free",
        temperature=0.4,
        max_tokens=768,
        cache=False,
    )
    dspy.configure(lm=lm)
    rlm = dspy.RLM(
        "context, query -> answer",
        max_iterations=6,
        max_llm_calls=12,
        paper_instruction_appendix="qwen_coder",
    )
    out = rlm(
        context="The code is XYZZY-42.",
        query="What is the code? Use Python and SUBMIT(answer=...).",
    )
    assert "XYZZY" in str(out.answer).upper() or "42" in str(out.answer)
