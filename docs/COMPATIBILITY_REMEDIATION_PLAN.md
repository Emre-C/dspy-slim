# Compatibility remediation plan (archival)

**Audit:** 2026-04-15 vs upstream DSPy 3.1.3 (see [`AUDIT_METHODOLOGY.md`](AUDIT_METHODOLOGY.md)).

## Current status

The **action list below was written during the audit.** Most Tier-3 “gap” items called out then are now **implemented**; treat [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) as the **living** source of truth for compatibility state.

As of the matrix refresh linked from that file:

- **`BaseModule.named_sub_modules()`** — implemented (see matrix row).
- **`Settings.save()` / `Settings.load()`** and related settings keys — implemented (see matrix).
- **Docstrings** on `Settings.configure()` / `context()` — parity work either done or tracked in matrix; verify against current [`dspy/dsp/utils/settings.py`](../dspy/dsp/utils/settings.py).

Remaining **policy** choices (intentional slim vs parity) for extra upstream-only surfaces — e.g. LiteLLM client wiring, optional type exports beyond the kept surface — stay documented in [`README.md`](../README.md) and the matrix, not in this archival note.

## Why this file remains

It preserves the audit-era **rationale** and **upstream code references** that were used to justify small backports. For line-by-line audit artifacts, see [`COMPATIBILITY_AUDIT_REPORT.md`](COMPATIBILITY_AUDIT_REPORT.md).

## Original prioritized actions (historical)

Phase 1 was “none — Tier 1 already compatible.” Phase 2 listed:

1. Add `named_sub_modules()` to `BaseModule`
2. Add `Settings.save()` / `load()`
3. Align `Settings` docstrings with upstream examples

Phase 3 listed README/matrix decisions for type exports and LiteLLM — still **decision documentation**, not necessarily “bugs.”

## Testing (still valid)

```bash
uv run --extra dev python -m pytest tests/minimal/test_predict.py tests/minimal/test_parallel.py -q
```

Full slice: `uv run --extra dev python -m pytest tests/minimal -q`

## Rollback

All listed remediations were additive; no special rollback section is required for current `main`.
