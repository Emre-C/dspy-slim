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

### What we are trying to do

- Keep a minimal DSPy codebase that is easy to reason about and safe to modify.
- Preserve the parts of DSPy that matter for recursive / agentic workflows, especially `dspy.RLM` and `dspy.GEPA`, while keeping the core authoring experience ergonomic with `dspy.Predict` and `dspy.ChainOfThought`.
- Prefer architectural clarity and operational robustness over broad provider abstraction.
- Stay close enough to upstream DSPy to benefit from its ideas, while allowing intentional divergence when the minimal fork has different priorities.

### Enduring technical decisions

- `BaseLM` is the abstraction boundary for language models in this fork.
- `dspy.LM` should not depend on LiteLLM in this fork.
- The current LM transport is a direct OpenAI-compatible client implementation, with OpenRouter supported through its OpenAI-compatible API.
- We preserve the existing DSPy programming model wherever possible instead of introducing a new LM interface just for the fork.
- The intentionally supported built-in module surface is currently `dspy.Predict`, `dspy.ChainOfThought`, `dspy.Parallel`, and `dspy.RLM`.
- `dspy.ChainOfThought` is treated as lightweight compatibility sugar over `dspy.Predict`, not as a separate architectural pillar.
- Upstream modules such as `ReAct`, `ProgramOfThought`, `CodeAct`, `BestOfN`, `Refine`, `MultiChainComparison`, `KNN`, and `majority` remain intentionally omitted unless a clear benchmarked need emerges in this fork.
- Capability claims should be conservative. We only advertise native features when the backend path is stable enough to support them reliably.
- Benchmark and smoke-test paths should favor fast, practical defaults. Larger-budget runs should be explicit rather than the default.
- Warning spam and “non-blocking” runtime issues are still considered bugs if they degrade confidence in the system.
- GEPA integration is intentionally handled through the published `gepa[dspy]` package rather than by vendoring GEPA core into this fork.
- The DSPy GEPA wrapper should track the API of the published GEPA package we actually pin, not unreleased kwargs seen on the upstream repository `main` branch.
- Local GEPA adapter hardening is part of the fork’s intended behavior and should be preserved across upstream syncs. In particular:
  - predictors with no usable reflective examples should be skipped rather than crashing the run,
  - empty reflective datasets should produce empty proposal updates rather than hard failure,
  - predictor-level score mismatches should warn once and fall back to module-level scoring.
- When DSPy runtime behavior depends on a package at import time, that dependency should be declared explicitly rather than relied on transitively.

### Enduring lessons learned

- A cloned upstream repository and the published package for the same version label may not expose the exact same callable surface. Before widening a wrapper API, check the installed package signature directly.
- Wrapper compatibility is more valuable than speculative feature exposure. It is better to support a smaller published GEPA surface reliably than to advertise kwargs that only exist in unreleased upstream code.
- Dependency upgrades are a good time to flush out hidden transitive dependencies. During the GEPA upgrade, `jsonschema` had to become an explicit DSPy dependency because imports relied on it directly.
- GEPA integration quality depends on preserving evaluation structure, not just scalar scores. If DSPy metrics expose subscores, the adapter should carry them through as GEPA `objective_scores`.
- README notes in this section should record durable operating truths, not temporary migration noise. Prefer decisions, constraints, and validated behavior over one-off implementation details.

### Current validated state

- The GEPA + RLM benchmark path is working in this fork.
- The SQuAD smoke-test script lives at `../scripts/gepa_rlm_squad.py` and is tuned for practical test runs instead of unnecessarily expensive defaults.
- The long-context validation script lives at `../scripts/rlm_long_context_validation.py`.
- The LM backend no longer depends on LiteLLM for `dspy.LM`.
- The OpenRouter path is exercised through the OpenAI-compatible client path.
- The user-facing built-in modules currently exported by this fork are `dspy.Predict`, `dspy.ChainOfThought`, `dspy.Parallel`, and `dspy.RLM`.
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
- The minimal test suite and the OpenRouter integration checks are passing at the time of this update.
- Focused GEPA validation is passing, including `tests/minimal/test_gepa.py`, `tests/minimal/test_imports.py`, and `tests/minimal/test_gepa_rlm_squad_script.py`.

### How to make future decisions

When extending this fork, prefer the smallest change that:

1. keeps the existing user-facing DSPy API stable,
2. improves confidence in the benchmarked and tested paths,
3. avoids introducing broad new abstraction layers unless they solve a real problem in this fork,
4. documents the reason for divergence from upstream when divergence is intentional.

### Maintenance note

This section is intended to be durable. Update it when an enduring decision changes, when a core architectural direction changes, or when the validated state of the fork materially changes. Do not treat it as a changelog for every small edit.




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

