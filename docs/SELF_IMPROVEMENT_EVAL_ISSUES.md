# Self-improvement eval — DSPy findings (dspy-slim)

This file records **DSPy-side** issues from the cross-repo self-improvement evaluation. Integration-only fixes (Korbex adapters, eval harness, cache accounting, provenance prompts) live in the **Korbex** workspace, not here.

For upstream contribution workflow, use [`UPSTREAM_ISSUE_DRAFTS.md`](UPSTREAM_ISSUE_DRAFTS.md) and [`UPSTREAM_PR_SUBMISSION_GUIDE.md`](UPSTREAM_PR_SUBMISSION_GUIDE.md).

---

## DSPY-01 — Adapter default/fallback drift was local compatibility drift

- **Status:** Fixed locally; closed as non-upstream bug.
- **Resolution:** `dspy-slim` matches upstream-shaped defaults again (`Predict` → `ChatAdapter`, optional strict `ChatAdapter(use_json_adapter_fallback=False)`). Fork-only `JSONWithChatFallbackAdapter` removed.
- **Tests:** [`tests/minimal/test_predict.py`](../tests/minimal/test_predict.py).
- **Upstream PR:** Optional policy discussion only; not part of the remaining eval-failure narrative.

---

## DSPY-02 — Top-level `dspy.Tool` export drift was local compatibility drift

- **Status:** Fixed locally; closed as non-upstream bug.
- **Resolution:** Top-level re-exports for `Tool`, `ToolCalls`, and kept adapters restored.
- **Tests:** [`tests/minimal/test_predict.py`](../tests/minimal/test_predict.py), [`tests/minimal/test_rlm.py`](../tests/minimal/test_rlm.py).

---

## DSPY-03 — RLM finalization and prompt projection

- **Status:** Implemented in this fork; candidate for upstream as reliability work.
- **Scope:** Finalization mode, urgency-aware iteration labels, prompt-history compaction, truncation-triggered finalization (see [`dspy/predict/rlm.py`](../dspy/predict/rlm.py), [`dspy/primitives/repl_types.py`](../dspy/primitives/repl_types.py)).
- **Tests:** [`tests/minimal/test_rlm.py`](../tests/minimal/test_rlm.py) (`TestHistoryCompaction`, `TestRLMBudgetManagement`).

---

## DSPY-04 — LM call metadata propagation

- **Status:** Implemented in this fork; candidate for upstream.
- **Scope:** Provider-agnostic metadata on completion dicts and `Prediction.lm_metadata` (see [`dspy/clients/base_lm.py`](../dspy/clients/base_lm.py), [`dspy/adapters/base.py`](../dspy/adapters/base.py), [`dspy/primitives/prediction.py`](../dspy/primitives/prediction.py), [`dspy/utils/lm_metadata.py`](../dspy/utils/lm_metadata.py)).
- **Note:** Direct `lm()` callers may see `list[dict]` instead of collapsed `list[str]`; treat as a deliberate contract change until upstream aligns.
- **Tests:** [`tests/minimal/test_lm_metadata.py`](../tests/minimal/test_lm_metadata.py), [`tests/minimal/test_lm.py`](../tests/minimal/test_lm.py).

---

## DSPY-05 — Bootstrap trace `format_reward` divided lists instead of lengths

- **Status:** Fixed locally; candidate for upstream (one-line fix).
- **Location:** [`dspy/teleprompt/bootstrap_trace.py`](../dspy/teleprompt/bootstrap_trace.py).

---

## Open follow-ups (DSPy upstream)

- Treat **DSPY-03**, **DSPY-04**, and **DSPY-05** as the coherent upstream stack: truncation visible to modules → `RLM` finalization → adjacent bootstrap reward fix.
- Keep **DSPY-01** and **DSPY-02** out of the “remaining eval failure” story; they are documented compatibility drift, covered by [`UPSTREAM_COMPATIBILITY_MATRIX.md`](UPSTREAM_COMPATIBILITY_MATRIX.md) and minimal tests.
