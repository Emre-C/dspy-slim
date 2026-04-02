## DSPy: _Programming_—not prompting—Foundation Models

**Documentation:** [DSPy Docs](https://dspy.ai/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)


----

DSPy is the framework for _programming—rather than prompting—language models_. It allows you to iterate fast on **building modular AI systems** and offers algorithms for **optimizing their prompts and weights**, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**. Learn more via our [official documentation site](https://dspy.ai/) or meet the community, seek help, or start contributing via this GitHub repo and our [Discord server](https://discord.gg/XCGy2WDCQB).


## Documentation: [dspy.ai](https://dspy.ai)


**Please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**


## Installation


```bash
pip install dspy
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```

**This fork ([dspy-slim](https://github.com/Emre-C/dspy-slim)):**

```bash
pip install git+https://github.com/Emre-C/dspy-slim.git
```

## This fork: enduring goals, decisions, and current state

This repository is being maintained as a deliberately minimal, robust DSPy fork. The goal is not to mirror every upstream feature immediately. The goal is to keep a small set of important capabilities working well, understand the system deeply, and make changes that improve reliability rather than increasing surface area for its own sake.

### Priorities (explicit order)

1. **Maximum compatibility with upstream DSPy** — Keep the same import path (`import dspy`) and package layout where possible; merge [`stanfordnlp/dspy`](https://github.com/stanfordnlp/dspy) regularly; avoid gratuitous renames or wide refactors that make every upstream sync a merge war.
2. **Minimal maintenance** — Keep the diff against upstream small and intentional; document deliberate divergence instead of accumulating silent drift.
3. **Branding and positioning** — Third. Clarity matters (`dspy-slim`, this README); a separate “product story” does not.

### Naming and lineage

- **This repo is [dspy-slim](https://github.com/Emre-C/dspy-slim)** on GitHub. It is a **fork** for a reduced surface area, **not** the official [`stanfordnlp/dspy`](https://github.com/stanfordnlp/dspy) project.
- **Upstream** (source of ongoing DSPy innovation) is `stanfordnlp/dspy`. Day-to-day git remotes, merge workflow, and install-from-git notes are documented in **`FORK.md`** in this repository.

### What we are trying to do
 
 - Keep a minimal DSPy codebase that is easy to reason about and safe to modify.
 - Preserve the parts of DSPy that matter for recursive / agentic workflows, especially `dspy.Signature`, `dspy.RLM`, `dspy.ReAct`, and `dspy.GEPA`, while keeping the core authoring experience ergonomic with `dspy.Predict`, `dspy.ChainOfThought`, and `dspy.Parallel`.
 - Prefer architectural clarity and operational robustness over broad provider abstraction.
 - Stay close enough to upstream DSPy to benefit from its ideas, while allowing intentional divergence when the minimal fork has different priorities.
 
 ### Enduring technical decisions

 - `BaseLM` is the abstraction boundary for language models in this fork.
 - `dspy.LM` should not depend on LiteLLM in this fork.
 - The current LM transport is a direct OpenAI-compatible client implementation, with OpenRouter supported through its OpenAI-compatible API.
- `dspy.LM` in this fork intentionally supports only chat-style and Responses-style OpenAI-compatible APIs, not legacy text-completion mode.
 - We preserve the existing DSPy programming model wherever possible instead of introducing a new LM interface just for the fork.
 - `dspy.Signature` plus `dspy.InputField` / `dspy.OutputField` are first-class stable authoring contracts in this fork.
- The intentionally supported top-level authoring surface is `dspy.Signature`, `dspy.InputField`, `dspy.OutputField`, `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.Parallel`, `dspy.RLM`, `dspy.GEPA`, and `dspy.LM`.
- Support utilities such as `dspy.Evaluate`, `dspy.JSONAdapter`, and `dspy.bootstrap_trace_data` may remain for compatibility or implementation needs, but they are not the product pillars of the fork.
 - `dspy.ChainOfThought` is treated as lightweight compatibility sugar over `dspy.Predict`, not as a separate architectural pillar.
 - **`dspy.ReAct`** is a first-class supported module again in this fork (upstream `dspy/predict/react.py` semantics): a tool loop built from `dspy.Predict` plus `dspy.ChainOfThought` for the final extraction step. Pass plain callables or `Tool` instances; async tool execution uses `Tool.acall` when you call `dspy.ReAct.acall(...)`. Implementation imports concrete types (`ChatAdapter`, `Tool`, `Predict`, etc.) rather than relying on broad top-level `dspy` re-exports.
 - **`Tool` and ReAct:** `dspy.Tool` is not a top-level re-export here; use `dspy.predict.Tool` (re-export of `dspy.adapters.types.Tool`) or import from `dspy.adapters.types`. Tool schemas depend on `pydantic` and `jsonschema`, both explicit dependencies in this fork.
 - **GEPA + ReAct:** `GEPA.compile` discovers predictors via `named_predictors()`; a `dspy.ReAct` student exposes both inner predictors (`react` and `extract.predict`), so prompt optimization can target the agent loop and the extraction step separately.
- Non-core helper surfaces such as module serialization helpers, async wrapper utilities, and legacy adapter convenience types may be intentionally omitted when they are not required for the supported surface.
- Upstream modules such as `ProgramOfThought`, `CodeAct`, `BestOfN`, `Refine`, `MultiChainComparison`, `KNN`, and `majority` remain intentionally omitted unless a clear benchmarked need emerges in this fork.
 - Capability claims should be conservative. We only advertise native features when the backend path is stable enough to support them reliably.
 - Benchmark and smoke-test paths should favor fast, practical defaults. Larger-budget runs should be explicit rather than the default.
 - Warning spam and “non-blocking” runtime issues are still considered bugs if they degrade confidence in the system.
 - Internal runtime code should import concrete implementation types directly for dispatch and typing logic instead of depending on broad top-level `dspy` re-exports.
- The default RLM sandbox path in this fork is intentionally narrow: typed `SUBMIT`, tool bridging, and large-variable injection are supported, while optional host file mounts, env passthrough, and sandbox network configuration are omitted unless reintroduced for a validated need.
 - GEPA integration is intentionally handled through the published `gepa[dspy]` package rather than by vendoring GEPA core into this fork.
 - The DSPy GEPA wrapper should track the API of the published GEPA package we actually pin, not unreleased kwargs seen on the upstream repository `main` branch.
 - Local GEPA adapter hardening is part of the fork’s intended behavior and should be preserved across upstream syncs. In particular:
  - predictors with no usable reflective examples should be skipped rather than crashing the run,
  - empty reflective datasets should produce empty proposal updates rather than hard failure,
  - predictor-level score mismatches should warn once and fall back to module-level scoring.
 - When DSPy runtime behavior depends on a package at import time, that dependency should be declared explicitly rather than relied on transitively.
 - The **in-repo documentation build** (MkDocs site, full `docs/` tree) and **most of the upstream integration test suite** are intentionally not carried here; use **[dspy.ai](https://dspy.ai)** for general DSPy documentation and **`tests/minimal`** here for regression coverage of the supported surface.
 - The in-repo docs workflow is intentionally disabled; durable fork knowledge lives primarily in this README and `FORK.md`, while general framework documentation lives at **[dspy.ai](https://dspy.ai)**.

 ### Enduring lessons learned

 - A cloned upstream repository and the published package for the same version label may not expose the exact same callable surface. Before widening a wrapper API, check the installed package signature directly.
 - Wrapper compatibility is more valuable than speculative feature exposure. It is better to support a smaller published GEPA surface reliably than to advertise kwargs that only exist in unreleased upstream code.
 - Dependency upgrades are a good time to flush out hidden transitive dependencies. During the GEPA upgrade, `jsonschema` had to become an explicit DSPy dependency because imports relied on it directly.
 - GEPA integration quality depends on preserving evaluation structure, not just scalar scores. If DSPy metrics expose subscores, the adapter should carry them through as GEPA `objective_scores`.
 - Top-level `import dspy` convenience for users is not a safe internal dependency boundary. When internal code needs to dispatch on `Adapter`, `Tool`, `Evaluate`, or `BaseLM`, import those concrete classes from their owning modules instead of assuming top-level re-exports exist.
 - In a multi-workspace or monorepo environment, `uv` commands should bind explicitly to the intended interpreter. In this repo, the most reliable local test path is `uv sync --dev --python .venv/bin/python --extra dev` followed by `uv run --python .venv/bin/python --module pytest ...`.
 - Standalone-clone behavior matters. Tests that rely on parent-workspace scripts or assets should skip cleanly when those files are absent instead of failing by assuming the monorepo layout.
 - Optional dependency groups are product claims. If integrations such as MCP, LangChain, Weaviate, Optuna, or alternate packaging flows are intentionally omitted from the slim build, their extras should be removed instead of left behind as stale metadata.
- Release automation should reflect the real publishing policy. Validation-only tag flows and disabled docs publishing are preferable to carrying upstream publishing or docs-subtree machinery on the active path when the fork does not use them.
 - README notes in this section should record durable operating truths, not temporary migration noise. Prefer decisions, constraints, and validated behavior over one-off implementation details.
 - **Upstream sync:** Prefer **frequent, small** merges from `upstream/main` over rare huge ones. After each merge, run the minimal test suite (`tests/minimal`). Concentrate fork-specific edits in as few files as practical so merges stay predictable.
 - **PyPI vs git:** The **`dspy`** distribution on PyPI is upstream’s. For this fork, **`pip install git+https://github.com/Emre-C/dspy-slim.git`** or a path/editable install is the canonical route. The tag workflow in this fork validates builds by default; the PyPI publish job remains intentionally gated off unless you deliberately adopt a fork-side publishing strategy.
 - **CI scope:** Continuous integration here runs **only** `tests/minimal` (see `.github/workflows/minimal_fork_tests.yml`), including the offline synthetic RLM sandbox check when the monorepo layout and Deno are present. The upstream full test matrix is intentionally **not** run as-is (the old workflow file is kept under a non-workflow filename so upstream merges do not re-enable a failing full suite by accident).
 - **GitHub `main` vs local history:** If the fork’s remote `main` ever diverges from the slim tree you intend to publish (for example, right after creating the fork on GitHub), reconciling may require a **one-time** deliberate `git push --force-with-lease` once local `main` is the canonical minimal fork—then return to normal fast-forward pushes.
 - **Monorepo layout:** When this package lives inside the parent **`minimal_dspy`** workspace next to Korbex, shared scripts such as `gepa_rlm_squad.py` and `rlm_long_context_validation.py` live under the **parent** `scripts/` directory; paths below assume that layout. Consumers who clone **only** `dspy-slim` should copy or adapt those scripts locally if needed.

### Upstream sync, tests, and packaging (reference)

 | Topic | Where it lives |
 |--------|----------------|
 | Remotes (`origin` = this fork, `upstream` = stanfordnlp), merge workflow | `FORK.md` |
 | One-command merge from upstream (run from parent `minimal_dspy/`) | `../scripts/sync_dspy_upstream.sh` (override directory with env `DSPY_SLIM` if needed) |
 | Minimal tests from parent workspace | `../run_minimal_tests.sh` |
 | Full E2E (pytest + optional live API smoke) from parent workspace | `../scripts/run_e2e.sh` or `../scripts/e2e_runner.py`; add `--live` with `OPENROUTER_API_KEY` and Deno for GEPA+RLM API checks |
 | Minimal tests inside this repo only | `uv sync --dev --python .venv/bin/python --extra dev` then `uv run --python .venv/bin/python --module pytest tests/minimal -vv` |
| Tag workflow behavior in this fork | `.github/workflows/build_and_release.yml` (builds and validates tags; no fork-side publish step) |
 | Docs workflow behavior in this fork | `.github/workflows/docs-push.yml` (manual no-op; in-repo docs publishing intentionally disabled) |

 ### Current validated state

 - The GEPA + RLM benchmark path is working in this fork.
 - The SQuAD smoke-test script lives at `../scripts/gepa_rlm_squad.py` (relative to this folder when inside `minimal_dspy`) and is tuned for practical test runs instead of unnecessarily expensive defaults.
 - The long-context validation script lives at `../scripts/rlm_long_context_validation.py`.
 - The LM backend no longer depends on LiteLLM for `dspy.LM`.
- The LM backend intentionally omits legacy text-completion mode and targets chat/responses APIs only.
 - The OpenRouter path is exercised through the OpenAI-compatible client path.
- The top-level import surface asserted by `tests/minimal/test_imports.py` currently includes `dspy.Signature`, `dspy.InputField`, `dspy.OutputField`, `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.Parallel`, `dspy.RLM`, `dspy.GEPA`, and `dspy.LM`.
- Intentionally omitted upstream modules such as `ProgramOfThought`, `CodeAct`, `BestOfN`, `Refine`, `MultiChainComparison`, `KNN`, and `majority` are asserted to remain absent from the slim top-level surface.
- Non-core compatibility helpers such as `dspy.Teleprompter`, module save/load helpers, async wrapper utilities, and legacy `dspy.Code` / `dspy.Reasoning` adapter types are intentionally omitted.
 - `gepa[dspy]` is currently pinned to `0.1.1`.
 - The current DSPy GEPA wrapper exposes the published GEPA `0.1.1` surfaces we rely on directly, including:
  - `candidate_selection_strategy` values `pareto`, `current_best`, `epsilon_greedy`, and `top_k_pareto`,
  - `mlflow_tracking_uri`,
  - `mlflow_experiment_name`,
  - `raise_on_exception`.
 - The wrapper does not currently expose tracker “attach existing run” kwargs, because the published `gepa==0.1.1` package used by this fork does not accept them even though similar options appear in the upstream repository.
 - The local GEPA adapter preserves DSPy-specific resilience behavior around missing reflective examples and score mismatch handling.
 - The local GEPA adapter forwards metric `subscores` into GEPA `objective_scores` when they are present.
 - `jsonschema` is an explicit dependency in this fork because DSPy imports it directly.
 - Focused minimal regression coverage exists for imports, callback dispatch, bootstrap trace, GEPA, RLM, parallel execution, metrics, evaluate, predict, the parent-workspace SQuAD smoke-test wrapper, and ReAct (`tests/minimal/test_react.py`: sync/async tool loop, context-window trajectory truncation, GEPA compile compatibility).
 - Adapter and tool callback dispatch is covered by `tests/minimal/test_callback_dispatch.py` so the runtime does not silently depend on omitted top-level exports such as `dspy.Adapter` or `dspy.Tool`.
 - The parent-workspace SQuAD smoke-test wrapper skips cleanly in standalone clones when the shared script is not present.
 - Package metadata no longer advertises optional extras for removed integrations such as MCP, LangChain, Weaviate, or Optuna.
 - The minimal CI workflow binds `uv` to the repo `.venv` explicitly to avoid workspace ambiguity when running `pytest` as a project tool.
 - The tag workflow validates built distributions against `tests/minimal`, and the in-repo docs workflow is intentionally disabled.
 - Some upstream release-support files may still remain on disk under `.github/.internal_dspyai/` or `.github/workflow_scripts/`, but they are inactive leftovers rather than part of the active slim-fork workflow.
 - **Downstream products** (e.g. Korbex) that need strict alignment can depend on this fork via a path or editable install instead of PyPI `dspy`; see the parent workspace’s `DSPY_KORBEX.md` for the responsibility split between library and product layer.

 ### How to make future decisions

When extending this fork, prefer the smallest change that:

1. keeps the existing user-facing DSPy API stable,
2. improves confidence in the benchmarked and tested paths,
3. avoids introducing broad new abstraction layers unless they solve a real problem in this fork,
4. documents the reason for divergence from upstream when divergence is intentional.

### Maintenance note

This section is intended to be durable. Update it when **priorities**, **technical decisions**, **lessons learned**, or **validated state** materially change—not for every commit. The **reference table** (`FORK.md`, scripts, test commands) should stay accurate when those entry points move. Do not use this README as a day-by-day changelog.




## 📜 Citation & Reading More

If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai).

If you're looking to understand the underlying research, this is a set of our papers:

**[Jul'25] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**       
**[Jun'24] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**       
**[Oct'23] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**     
[Jul'24] [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)     
[Jun'24] [Prompts as Auto-Optimized Training Hyperparameters](https://arxiv.org/abs/2406.11706)    
[Feb'24] [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)         
[Jan'24] [In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178)       
[Dec'23] [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)   
[Dec'22] [Demonstrate-Search-Predict: Composing Retrieval & Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024.pdf)

To stay up to date or learn more, follow [@DSPyOSS](https://twitter.com/DSPyOSS) on Twitter or the DSPy page on LinkedIn.

The **DSPy** logo is designed by **Chuyi Zhang**.

If you use DSPy or DSP in a research paper, please cite our work as follows:

```
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
```

<!-- You can also read more about the evolution of the framework from Demonstrate-Search-Predict to DSPy:

* [**DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines**](https://arxiv.org/abs/2312.13382)   (Academic Paper, Dec 2023) 
* [**DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**](https://arxiv.org/abs/2310.03714) (Academic Paper, Oct 2023) 
* [**Releasing DSPy, the latest iteration of the framework**](https://twitter.com/lateinteraction/status/1694748401374490946) (Twitter Thread, Aug 2023)
* [**Releasing the DSP Compiler (v0.1)**](https://twitter.com/lateinteraction/status/1625231662849073160)  (Twitter Thread, Feb 2023)
* [**Introducing DSP**](https://twitter.com/lateinteraction/status/1617953413576425472)  (Twitter Thread, Jan 2023)
* [**Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP**](https://arxiv.org/abs/2212.14024.pdf) (Academic Paper, Dec 2022) -->

