# dspy-slim

A deliberately minimal fork of [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) — keeping only what matters for recursive/agentic workflows and prompt optimization.

**~75% of the Python removed** — the entire codebase is under 10k lines. It fits comfortably in an LLM context window, which means AI coding tools can reason about and generate DSPy code far more reliably. The small surface also makes it practical to maintain ports to other languages like TypeScript and Rust.

**Fork:** [github.com/Emre-C/dspy-slim](https://github.com/Emre-C/dspy-slim) · **Upstream:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) · **General DSPy docs:** [dspy.ai](https://dspy.ai)

## Installation

```bash
pip install git+https://github.com/Emre-C/dspy-slim.git
```

The `dspy` name on PyPI belongs to upstream. This fork installs as `import dspy` from git or a path/editable install.

## Supported surface

| Module | Role |
|--------|------|
| `dspy.Signature` / `InputField` / `OutputField` | First-class authoring contracts for declarative LM programs |
| `dspy.Predict` | Core predictor — the building block everything else composes |
| `dspy.ChainOfThought` | Lightweight sugar over `Predict` (not a separate architectural pillar) |
| `dspy.ReAct` | Tool-loop agent built from `Predict` + `ChainOfThought` extraction; sync and async (`acall`) |
| `dspy.Parallel` | Concurrent execution of predictors |
| `dspy.RLM` | Recursive Language Model scaffold — code execution, sub-LM calls, symbolic recursion (see `rlm.md`) |
| `dspy.GEPA` | Prompt optimization via the [GEPA](https://arxiv.org/abs/2507.19457) reflective evolution algorithm |
| `dspy.LM` | Language model client — OpenAI-compatible APIs only (chat + Responses), OpenRouter via the same path |

Support utilities (`dspy.Evaluate`, `dspy.JSONAdapter`, `dspy.bootstrap_trace_data`) remain for implementation needs but are not product pillars.

## What's removed

**Modules:** `ProgramOfThought`, `CodeAct`, `BestOfN`, `Refine`, `MultiChainComparison`, `KNN`, `majority`, module serialization helpers, async wrapper utilities, legacy adapter types (`dspy.Code`, `dspy.Reasoning`), `dspy.Teleprompter`.

**Integrations:** LiteLLM, MCP, LangChain, Weaviate, Optuna, legacy text-completion mode.

**Infrastructure:** In-repo docs build (MkDocs), upstream full test matrix, upstream release/publish workflows.

## Architecture decisions

- **`dspy.LM` does not depend on LiteLLM.** The LM transport is a direct OpenAI-compatible client; OpenRouter is supported through its OpenAI-compatible API. Only chat-style and Responses-style APIs — no legacy text-completion mode.
- **GEPA is a package dependency, not vendored.** Integration uses the published `gepa[dspy]==0.1.1` package. The DSPy wrapper tracks the API of the pinned version, not unreleased upstream kwargs. Local adapter hardening (skip predictors with no reflective examples, empty proposals instead of crashes, fallback to module-level scoring on mismatch) is preserved across syncs.
- **`dspy.Tool` is not a top-level re-export.** Import from `dspy.predict.Tool` or `dspy.adapters.types`. Tool schemas depend on `pydantic` and `jsonschema` (both explicit dependencies).
- **ReAct exposes inner predictors to GEPA.** `named_predictors()` yields both `react` and `extract.predict`, so `GEPA.compile` can target the agent loop and extraction step separately.
- **Internal imports use concrete types.** Runtime dispatch and typing import from owning modules (`ChatAdapter`, `Tool`, `Predict`, etc.), never from broad `dspy` top-level re-exports.
- **RLM sandbox is intentionally narrow.** Typed `SUBMIT`, tool bridging, and large-variable injection are supported; host file mounts, env passthrough, and sandbox network config are omitted unless reintroduced for a validated need.
- **`jsonschema` is an explicit dependency** because DSPy imports it directly (not relied on transitively).

## Monorepo context

This repo typically lives inside a `minimal_dspy/` workspace alongside **Korbex** (an agentic exploration product that consumes DSPy). See [`DSPY_KORBEX.md`](../DSPY_KORBEX.md) for the responsibility split. Standalone clones work — tests skip cleanly when parent-workspace scripts are absent.

Shared scripts (from `minimal_dspy/`):

| Script | Purpose |
|--------|---------|
| `scripts/sync_dspy_upstream.sh` | One-command upstream merge |
| `scripts/run_e2e.sh` | Full E2E (add `--live` + `OPENROUTER_API_KEY` for GEPA+RLM API smoke tests) |
| `scripts/gepa_rlm_squad.py` | SQuAD smoke-test, tuned for practical runs |
| `scripts/rlm_long_context_validation.py` | Long-context RLM validation |
| `run_minimal_tests.sh` | Quick regression from workspace root |

## Fork workflow

Detailed remote setup and merge procedure are in [`FORK.md`](FORK.md).

| Task | Command |
|------|---------|
| Sync upstream | `./scripts/sync_dspy_upstream.sh` (from `minimal_dspy/`) |
| Minimal tests (workspace) | `./run_minimal_tests.sh` (from `minimal_dspy/`) |
| Minimal tests (standalone) | `uv sync --dev --python .venv/bin/python --extra dev` then `uv run --python .venv/bin/python --module pytest tests/minimal -vv` |
| Full E2E | `./scripts/run_e2e.sh [--live]` (from `minimal_dspy/`) |
| CI | `.github/workflows/minimal_fork_tests.yml` — runs `tests/minimal` only |
| Tag workflow | `.github/workflows/build_and_release.yml` — validates builds; no PyPI publish |

Prefer **frequent, small** merges from upstream over rare large ones. After each merge, run the minimal test suite. If upstream restores workflow files (e.g. full `run_tests.yml`), rename them so they don't run.

## Lessons learned

- **Wrapper compat > speculative features.** Support a smaller published GEPA surface reliably rather than advertising kwargs that only exist in unreleased upstream code. Before widening a wrapper, check the *installed* package signature.
- **Dependency upgrades flush hidden transitives.** During the GEPA upgrade, `jsonschema` had to become explicit because DSPy imported it directly.
- **Top-level `import dspy` is not a safe internal boundary.** Internal dispatch on `Adapter`, `Tool`, `Evaluate`, or `BaseLM` must import concrete classes from their owning modules.
- **`uv` must bind to the intended interpreter.** In multi-workspace layouts, use `uv sync --dev --python .venv/bin/python --extra dev` and `uv run --python .venv/bin/python ...` to avoid workspace ambiguity.
- **README records durable truths, not migration noise.** Prefer decisions, constraints, and validated behavior over one-off implementation details.

## How to make future decisions

When extending this fork, prefer the smallest change that:

1. keeps the existing user-facing DSPy API stable,
2. improves confidence in the benchmarked and tested paths,
3. avoids introducing broad new abstraction layers unless they solve a real problem in this fork,
4. documents the reason for divergence from upstream when divergence is intentional.

Update this README when **priorities**, **technical decisions**, **lessons learned**, or **validated state** materially change — not for every commit.

## Citation

**[Jul'25] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**
**[Jun'24] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**
**[Oct'23] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**

```bibtex
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
