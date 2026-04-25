# Upstream Forward Compatibility

This document defines how to use upstream post-release commits to test whether `dspy-slim` is staying slim in a way that still makes kept-surface upstream improvements easy to absorb.

It complements:

- [`README.md`](../README.md) for the supported product surface and maintenance workflow
- [`UPSTREAM_COMPATIBILITY_PLAN.md`](UPSTREAM_COMPATIBILITY_PLAN.md) for repo-level compatibility policy
- [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) for the stable-release compatibility audit

## Why This Exists

`dspy-slim` now has a completed behavioral-compatibility audit against upstream DSPy `3.2.0`.

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

For the current audit cycle, that anchor is upstream DSPy `3.2.0` at `d3a890c0`.

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
- `dspy/teleprompt/` for the kept `GEPA` and `BetterTogether` surfaces
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
git -C ../tmp/dspy rev-list -n 1 --before='2026-04-24 00:00' origin/main
git -C ../tmp/dspy log --oneline 3.2.0..origin/main -- \
  dspy/__init__.py \
  dspy/adapters \
  dspy/clients \
  dspy/evaluate \
  dspy/predict \
  dspy/primitives \
  dspy/signatures \
  dspy/streaming \
  dspy/teleprompt \
  dspy/utils \
  pyproject.toml
git -C ../tmp/dspy diff --stat 3.2.0..origin/main -- \
  dspy/__init__.py \
  dspy/adapters \
  dspy/clients \
  dspy/evaluate \
  dspy/predict \
  dspy/primitives \
  dspy/signatures \
  dspy/streaming \
  dspy/teleprompt \
  dspy/utils \
  pyproject.toml
```

## Current Baseline

- Stable anchor: upstream DSPy `3.2.0` at `d3a890c0`
- Purpose: measure how noisy the next upstream sync is likely to be on the kept surface without changing the current stable compatibility claim
- Historical note: the earlier `3.1.3`-anchored forward-compatibility pass is now superseded by the `3.2.0` stable sync and should not be used as the current mergeability baseline

## What To Record In Each New Pass

For each forward-compatibility run after `3.2.0`, classify relevant upstream commits as one of:

- already absorbed locally
- next merge candidate
- intentional skip because the change only touches removed subsystems
- separate `RLM` divergence follow-up

Record only the kept-surface deltas that would materially affect the next sync. Do not restate the whole stable audit here.

## Recommended Cadence

Run this exercise:

- after a cluster of upstream kept-surface commits lands on `main`
- before the next upstream stable release if you want to keep the eventual sync small
- whenever the repo needs evidence that the slim fork is still easy to maintain

Keep the stable-release matrix as the user-facing compatibility artifact, and use this document as the maintainability and mergeability artifact between releases.
