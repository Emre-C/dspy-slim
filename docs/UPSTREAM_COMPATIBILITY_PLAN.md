# Upstream Compatibility Plan

This document defines the operating plan for keeping `dspy-slim` behaviorally faithful to upstream DSPy on the surfaces that this fork still supports, while remaining operationally slimmer internally.

It complements:

- [`README.md`](../README.md) for the supported product surface and fork goals
- [`FORK.md`](FORK.md) for merge and sync workflow
- [`UPSTREAM_FORWARD_COMPATIBILITY.md`](UPSTREAM_FORWARD_COMPATIBILITY.md) for post-release `main` triage and mergeability drills
- [`UPSTREAM_ISSUE_DRAFTS.md`](UPSTREAM_ISSUE_DRAFTS.md) and [`UPSTREAM_PR_SUBMISSION_GUIDE.md`](UPSTREAM_PR_SUBMISSION_GUIDE.md) for careful upstream contribution work
- [`SELF_IMPROVEMENT_EVAL_ISSUES.md`](SELF_IMPROVEMENT_EVAL_ISSUES.md) for concrete drift discoveries and upstreamable bugs

## Goal

`dspy-slim` should behave like upstream DSPy on the user-visible surfaces that this fork chooses to keep.

The repo may still be slimmer by:

- removing unsupported upstream subsystems entirely
- simplifying internal implementation structure
- narrowing infrastructure and packaging
- keeping experimental work isolated and explicit

The repo should not silently become a sibling framework with different default semantics unless that divergence is deliberate, documented, and easy for users to opt into or out of.

## Definitions

### Supported Surface

The subset of DSPy that this fork intends users to rely on. Today that includes the public symbols and flows described in [`README.md`](../README.md), especially `Predict`, adapters used by kept modules, `ReAct`, `RLM`, `GEPA`, `LM`, and the tooling/types those paths expose.

### Behavioral Compatibility

For the supported surface, the following should match upstream unless explicitly documented otherwise:

- public import paths and top-level exports
- default control-flow semantics
- fallback and retry behavior
- user-visible output shapes
- error and warning boundaries when they affect user decisions or extension points

### Operational Slimming

The fork may differ from upstream in ways that do not change supported-surface semantics, such as:

- fewer modules and integrations
- smaller dependency footprint
- simpler helper structure
- reduced CI and packaging scope
- leaner tests focused on the supported surface

### Intentional Divergence

A behavior difference is acceptable only when all of the following are true:

- it is useful for this fork's goals
- it is documented in repo docs
- it is reflected in tests
- it is not accidentally used as evidence for an upstream issue or PR

## Decision Rules

1. The latest audited upstream stable release is the default reference for compatibility claims on the supported surface.
2. Upstream `main` is the reference for forward-compatibility drills, mergeability checks, and upstream contribution prep.
3. Unsupported upstream features may remain removed, but that removal must be explicit.
4. Default behavior matters as much as public API names.
5. Internal simplification is encouraged only after compatibility is preserved or re-established.
6. Experimental improvements should be opt-in until they are either upstreamed or intentionally adopted as fork policy.
7. Before treating a local finding as an upstream bug, verify current upstream behavior directly.

## Priority Surfaces

### Tier 1: Contract Surfaces

These surfaces should be audited first and kept closest to upstream.

- Top-level exports and import paths for supported symbols
- Adapter defaults and fallback policy
- `Predict` orchestration semantics
- `RLM` tool-facing behavior and core control flow
- Direct LM output shape and metadata flow when user-visible

### Tier 2: User Guidance Surfaces

- README claims about supported behavior
- examples and snippets
- error messages and migration notes
- upstream issue and PR preparation docs

### Tier 3: Internal Slimming Surfaces

- helper layout
- module factoring
- performance-oriented simplifications
- test organization and CI shape

Tier 3 may diverge much more freely as long as Tier 1 remains compatible.

## Work Plan

### Phase 0: Charter And Artifacts

- Maintain this plan as the repo-level statement of compatibility policy.
- Keep a compatibility matrix that records upstream contract, local behavior, and the chosen action for each high-leverage surface.
- Link these docs from [`README.md`](../README.md) so they are part of normal maintenance, not hidden thread context.

### Phase 1: Capture The Upstream Contract

For each Tier 1 surface:

- inspect the current upstream source directly from a fresh upstream clone or a verified remote read
- record the exact upstream files where the behavior lives
- avoid inferring upstream behavior from this fork
- update the compatibility matrix before making refactors that change user-visible semantics

This step exists specifically to prevent local drift from being mistaken for upstream truth.

### Phase 2: Add A Narrow Compatibility Test Slice

Before major refactors, add a small set of regression tests that encode supported-surface upstream behavior.

The initial slice should cover:

- top-level import/export parity for supported symbols such as `dspy.Tool`
- default adapter selection in `Predict`
- adapter fallback behavior and LM call-count semantics
- one `RLM` smoke test covering tool-facing or orchestration behavior that should remain upstream-faithful
- direct `lm()` output shape only if that shape is treated as part of the supported surface

Use the command shape that already worked reliably in this checkout:

```bash
uv run --extra dev python -m pytest <targeted tests>
```

### Phase 3: Refactor In Compatibility Order

#### 3.1 Public API Parity

First align imports and top-level exports for supported symbols. Users form their mental model of the library from `import dspy`, so this is the highest-leverage place to remove accidental drift.

#### 3.2 Adapter Defaults And Fallback Policy

Next align default adapter resolution and fallback behavior. Hidden retries, parse-failure propagation, and default adapter choice are behavioral semantics, not internal cleanup.

#### 3.3 `Predict` Orchestration

Audit and align:

- default adapter path
- LM precedence rules
- warning versus error boundaries
- tracing behavior when user-visible
- handling of extra or missing input fields when that affects callers

#### 3.4 `RLM` Boundary

Separate `RLM` work into:

- upstream-faithful reliability improvements that should remain strong upstream candidates
- fork-only experimental behavior that should be explicit and non-default until adopted intentionally

#### 3.5 Docs And Examples

After code changes, update:

- README claims
- examples
- issue/PR drafts that rely on behavioral comparisons
- migration notes for any intentional divergence that remains

### Phase 4: Verification Discipline

For each refactor slice:

1. run the targeted compatibility tests first
2. run nearby minimal tests for the changed area
3. if a change affects upstream contribution work, re-check the corresponding upstream source before finalizing the local interpretation

Prefer focused verification over broad churn, but do not merge behavior changes without tests on the affected contract surface.

### Phase 5: Ongoing Audit Cadence

Re-run the Tier 1 audit:

- after each upstream sync that touches adapters, `Predict`, `RLM`, LM plumbing, or exports
- before opening new upstream issues or PRs based on local findings
- when README or examples start describing behavior in a way that may differ from upstream

Maintain the post-release `main` triage in [`UPSTREAM_FORWARD_COMPATIBILITY.md`](UPSTREAM_FORWARD_COMPATIBILITY.md):

- between stable releases when you want to measure mergeability instead of restating the current compatibility guarantee
- after clusters of upstream kept-surface commits land on `main`
- before the next stable release if you want the eventual sync to stay small and predictable

## Immediate Execution Order

1. Maintain the compatibility matrix in [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md).
2. Add compatibility tests for top-level exports and adapter defaults.
3. Resolve top-level `dspy.Tool` parity for the supported surface.
4. Resolve default adapter and fallback-policy parity between `Predict` and `ChatAdapter`.
5. Re-audit `RLM` against upstream after the adapter and export surfaces are stable.
6. Update README statements that currently describe drift as if it were settled fork policy.

## Non-Goals

- full API parity with every upstream subsystem that this fork intentionally removed
- bulk-copying upstream internals when a smaller implementation preserves the same supported behavior
- treating all local improvements as candidates to become default fork behavior immediately

## Success Criteria

The plan is succeeding when:

- supported-surface imports and defaults no longer surprise upstream DSPy users
- local bug reports can be classified cleanly as upstream bugs, fork bugs, or intentional fork policy
- upstream issue and PR preparation no longer depends on remembering undocumented fork caveats
- internal slimming remains possible without reintroducing public behavioral drift
