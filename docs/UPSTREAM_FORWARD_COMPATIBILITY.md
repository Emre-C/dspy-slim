# Upstream Forward Compatibility

This document defines how to use upstream post-release commits to test whether `dspy-slim` is staying slim in a way that still makes kept-surface upstream improvements easy to absorb.

It complements:

- [`README.md`](../README.md) for the supported product surface and maintenance workflow
- [`UPSTREAM_COMPATIBILITY_PLAN.md`](UPSTREAM_COMPATIBILITY_PLAN.md) for repo-level compatibility policy
- [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) for the stable-release compatibility audit

## Why This Exists

`dspy-slim` now has a completed behavioral-compatibility audit against upstream DSPy `3.1.3`.

That stable audit answers one question:

- are upstream DSPy users on the kept surface able to treat this fork as plug-and-play today?

It does not fully answer a second question:

- is the fork slim in a way that keeps future upstream improvements easy to merge?

Those are related but different maintenance problems.

## Two Reference Modes

### Stable Compatibility Mode

Use the latest audited upstream stable release as the reference when the repo is making claims about:

- user-facing interchangeability
- README statements about current compatibility
- the compatibility matrix and regression slice
- whether a local difference is a drift from the supported stable contract

For the current audit cycle, that anchor is upstream DSPy `3.1.3` at `4ef729d2`.

### Forward-Compatibility Mode

Use upstream `main` after the stable tag as the reference when testing:

- mergeability between releases
- whether kept-surface refactors upstream still fit the slim fork cleanly
- whether the fork is staying operationally slimmer without becoming a sibling framework
- whether a future stable sync is likely to be small or noisy

Forward-compatibility work should not overwrite the stable compatibility claim unless the repo intentionally re-audits and adopts a new stable anchor.

## Scope Filter

The forward-compatibility drill should stay scoped to the same kept surfaces used by the compatibility charter.

### Primary Kept-Surface Lane

Review upstream changes under:

- `dspy/__init__.py`
- `dspy/adapters/`
- `dspy/clients/`
- `dspy/evaluate/`
- `dspy/predict/` except the `RLM`-specific divergence lane below
- `dspy/primitives/` except the `RLM`-specific divergence lane below
- `dspy/signatures/`
- `dspy/streaming/`
- `dspy/teleprompt/gepa/`
- `dspy/utils/`
- `pyproject.toml`

### Separate `RLM` Divergence Lane

Treat these as a second pass, not as the primary measure of fork slimness:

- `dspy/predict/rlm.py`
- `dspy/primitives/python_interpreter.py`
- `dspy/primitives/repl_types.py`
- `dspy/primitives/runner.js`

`RLM` is part of the kept surface, but this repo already carries approved intentional divergence there for truncation-aware finalization and related reliability work. Using `RLM` churn as the first mergeability score would blur the line between accidental drift and explicit fork policy.

### Skip Removed-Surface Noise

Do not let these dominate the exercise:

- removed predictors or optimizers
- removed integrations such as LiteLLM, MCP, LangChain, retrievers, or Optuna-specific wiring
- docs-only or CI-only churn
- unreleased upstream package-surface changes that do not match the installed stable dependency versions this fork intentionally targets

## Success Criteria

The fork is handling upstream well when most kept-surface upstream changes fall into one of these buckets:

- already absorbed locally by previous compatibility work
- small, low-risk backports on the supported surface
- clearly irrelevant because they only touch removed subsystems
- clearly separate because they belong to an intentional divergence lane

The exercise is going badly when upstream changes repeatedly force the fork to:

- reintroduce removed infrastructure just to stay compatible
- special-case fork-only semantics on previously upstream-shaped surfaces
- touch many unrelated files for a single kept-surface improvement
- reinterpret docs because the fork no longer matches either stable upstream or deliberate fork policy

## Repeatable Workflow

1. Fetch upstream refs in the comparison clone.
2. Choose a stable tag and a cutoff commit or date on `main`.
3. Generate a kept-surface-only log and diff.
4. Classify each relevant upstream change as one of:
   - already absorbed
   - next merge candidate
   - intentional skip
   - separate `RLM` divergence follow-up
5. Backport the smallest high-leverage candidates first.
6. Run focused tests on the touched contract surface.
7. Update this document and the compatibility matrix if the local recommendation changes.

Example command shape using the existing upstream mirror in `../tmp/dspy`:

```bash
git -C ../tmp/dspy fetch origin main --tags
git -C ../tmp/dspy rev-list -n 1 --before='2026-04-16 00:00' origin/main
git -C ../tmp/dspy log --oneline 3.1.3..a2b01f34 -- \
  dspy/__init__.py \
  dspy/adapters \
  dspy/clients \
  dspy/evaluate \
  dspy/predict \
  dspy/primitives \
  dspy/signatures \
  dspy/streaming \
  dspy/teleprompt/gepa \
  dspy/utils \
  pyproject.toml
git -C ../tmp/dspy diff --stat 3.1.3..a2b01f34 -- \
  dspy/__init__.py \
  dspy/adapters \
  dspy/clients \
  dspy/evaluate \
  dspy/predict \
  dspy/primitives \
  dspy/signatures \
  dspy/streaming \
  dspy/teleprompt/gepa \
  dspy/utils \
  pyproject.toml
```

## First Forward-Compatibility Pass

### Window

- Stable anchor: upstream DSPy `3.1.3` at `4ef729d2`
- Cutoff commit for this pass: upstream `main` at `a2b01f34` on 2026-04-15
- Purpose: test forward mergeability on kept surfaces without changing the repo's current stable compatibility claim

### Already Absorbed Or Effectively Covered

| Upstream commit(s) | Surface | Local assessment | Why it matters |
|---|---|---|---|
| `fed54d0a` | `JSONAdapter.acall` forwarding `use_native_function_calling` | Already present locally. | Confirms async adapter behavior remains upstream-shaped on a subtle but real correctness boundary. |
| `b833bc55` | `AdapterParseError` on empty LM responses | Already present locally. | Avoids silent `None`-shaped predictions when the LM returns empty content. |
| `af2a955f` | `Predict.load_state()` / `BaseModule.load()` unsafe LM state keys | Now present locally. | Restores the upstream hardening boundary that strips serialized endpoint overrides unless the caller explicitly opts into trusted unsafe LM state. |
| `7e02cc8e` | `InputField` / `OutputField` deprecation warnings for `prefix`, `format`, and `parser` | Already present locally. | Keeps warning behavior on supported signature-authoring surfaces aligned with current upstream. |
| `3278e0f8` | Duplicate input/output field-name rejection in `Signature` | Already present locally. | Preserves an upstream validation boundary that catches ambiguous signatures early. |
| `35613ab5` | Guarded subclass checks for generic annotations in adapter utils | Already present locally. | Avoids runtime crashes when tool or field annotations use generics such as `list[str]`. |
| `c3424073` | `ParallelExecutor` runs on the main thread when `num_threads=1` | Already present locally. | Keeps `Evaluate` and related execution paths compatible with single-thread callers that require main-thread execution. |
| `a3874e34` | JSON serialization uses `ensure_ascii=False` | Already present locally. | Maintains correct non-ASCII handling for JSONAdapter and related JSON utilities. |
| `fcb648de` | `Signature` cloudpickle serialization on Python 3.14 | Now present locally. | Keeps `Signature` and `Predict` cloudpickle round-trips working after the fork reintroduced cloudpickle-backed save/load helpers. |
| `71493dba`, `9cdb0aac` | Input type validation, `settings.warn_on_type_mismatch`, and the matching `typeguard==4.4.3` pin | Now present locally as an intentional forward-compatibility adoption beyond the original `3.1.3` stable audit. | Aligns the kept `Predict` input-warning boundary with current upstream `main` while keeping the stable compatibility matrix separate as a historical audit artifact. |
| `c8b3ebed`, `0d390ad2`, `d0b89320`, `94062912` | `BaseLM` capability properties and DSPy-owned `ContextWindowExceededError` | Already present locally. | This is one of the strongest mergeability signals: adapters and error handling remained easy to align without reintroducing LiteLLM coupling. |

### Next Merge Candidates

| Upstream commit(s) | Surface | Current local status | Recommended action | Priority |
|---|---|---|---|---|
| `16255432` | `Image` type `verify=` support | Not currently prioritized. This is additive and low-signal for the main slimness question. | Revisit only if image transport becomes an active kept-surface concern. | P3 |

### Intentional Skip Or Separate Lane

| Upstream commit(s) | Surface | Local decision | Why |
|---|---|---|---|
| `30ebe34f`, `66e33399` | LiteLLM import-time and version-cap handling | Skip. | `dspy-slim` removed LiteLLM, so these are not meaningful mergeability signals for the kept surface. |
| `3dde431b` | `SemanticF1` / `CompleteAndGrounded` return `Prediction` | Skip for now. | These metrics are outside the fork's current supported product pillars and are not required to judge core kept-surface slimness. |
| `d8fae34b`, `1ff12dd0`, `3109c61f`, `a8041dae`, `295e2b35` | `RLM` / interpreter / REPL hardening | Separate `RLM` divergence lane. | Useful work, but it should be reviewed after the core kept-surface lane because this repo already carries intentional `RLM` divergence. |
| `43bf2c59` | Optuna optional dependency | Skip. | Optuna stays out of scope for the fork. |
| `b3ee96fa` | GEPA package bump to upstream repo state | Skip as a default merge candidate. | This fork intentionally audits GEPA against the published stable package surface, not arbitrary upstream repo drift. |

## Recommended Cadence

Run this exercise:

- after a cluster of upstream kept-surface commits lands on `main`
- before the next upstream stable release if you want to keep the eventual sync small
- whenever the repo needs evidence that the slim fork is still easy to maintain

Keep the stable-release matrix as the user-facing compatibility artifact, and use this document as the maintainability and mergeability artifact between releases.
