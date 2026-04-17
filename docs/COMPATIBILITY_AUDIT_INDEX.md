# Compatibility audit index (2026-04-15)

**Scope:** dspy-slim vs upstream DSPy **3.1.3** (pinned mirror under `../tmp/dspy` at audit time).

**Verdict:** No **blocking** Tier-1 contract gaps on the kept surface; remaining differences are intentional slimming, documented divergences (e.g. LM metadata / RLM), or support utilities — see [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) for the **current** checklist.

---

## Where to read

| Document | Role |
|----------|------|
| [`AUDIT_SUMMARY.md`](AUDIT_SUMMARY.md) | Executive summary of the 2026-04-15 audit |
| [`COMPATIBILITY_AUDIT_REPORT.md`](COMPATIBILITY_AUDIT_REPORT.md) | Per-surface detail (sub-agent breakdown) |
| [`COMPATIBILITY_REMEDIATION_PLAN.md`](COMPATIBILITY_REMEDIATION_PLAN.md) | Archival: audit-era follow-ups; matrix supersedes “open” status |
| [`AUDIT_METHODOLOGY.md`](AUDIT_METHODOLOGY.md) | How the audit was run, limitations, pinned commits |
| [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) | **Living** compatibility table (update this when code or upstream changes) |
| [`UPSTREAM_COMPATIBILITY_PLAN.md`](UPSTREAM_COMPATIBILITY_PLAN.md) | Ongoing policy for compatibility vs slimming |

**Framing note:** The detailed report’s “drift” count includes **intentional** removals and **non-blocking** items. “Zero blocking gaps” in the summary refers to **Tier-1 contract surfaces** on the **kept** product API.

---

## Maintainers

1. Before adopting a new upstream **stable** anchor, re-run a comparison and update the matrix.
2. Before upstream PRs from local findings, run `./scripts/run_e2e.sh --live` when API-backed paths change.
3. Quarterly or after large merges: smoke `tests/minimal` and skim the matrix for stale rows.

Ground truth for comparisons: pinned upstream checkout (see `AUDIT_METHODOLOGY.md`).

---

## Files in this directory

Audit bundle (this folder):

- `COMPATIBILITY_AUDIT_INDEX.md` — this file  
- `AUDIT_SUMMARY.md`  
- `COMPATIBILITY_AUDIT_REPORT.md`  
- `COMPATIBILITY_REMEDIATION_PLAN.md`  
- `AUDIT_METHODOLOGY.md`  

Related: `UPSTREAM_COMPATIBILITY_PLAN.md`, `UPSTREAM_COMPATIBILITY_MATRIX.md`, `UPSTREAM_FORWARD_COMPATIBILITY.md`, root [`README.md`](../README.md).
