# Upstream PR Submission Guide

This guide is intentionally conservative. The goal is to submit two useful upstream PRs that feel easy for maintainers to review, not to push your local `dspy-slim` patch stack upstream wholesale.

## Read This First

- Do not open the PRs from this `dspy-slim` checkout.
- This repo has diverged from `stanfordnlp/dspy`, and at least one local issue (`dspy.Tool` guidance) does not reproduce on upstream `main`.
- This repo also contains intentional local reliability work around LM metadata propagation and truncation-aware `RLM` finalization; treat it as proof-of-concept context, not as the upstream baseline.
- Of the two upstream PRs below, only the RLM reliability stack is part of the remaining self-improvement-eval upstream narrative. The `ChatAdapter` fallback PR is still worth sending, but it should be framed as a separate compatibility-policy discussion.
- Use this repo only as a proof-of-concept reference.
- Make the actual PRs from a fresh fork of `stanfordnlp/dspy`.

## Submit In This Order

1. PR 1: adapter fallback boundary
2. PR 2: RLM reliability

That order matters.

- PR 1 is small, easy to explain, and has a clean migration story.
- PR 2 is valuable, but it touches a broader framework surface and includes a breaking-change discussion.

## One-Time Setup

1. Fork `https://github.com/stanfordnlp/dspy` on GitHub.
2. Clone your fork locally.

```bash
git clone git@github.com:<your-github-name>/dspy.git ~/codebase/dspy-upstream
cd ~/codebase/dspy-upstream
git remote add upstream https://github.com/stanfordnlp/dspy.git
git fetch upstream
```

3. Create the recommended environment.

```bash
uv venv --python 3.10
uv sync --extra dev
pre-commit install
uv run --extra dev python -m pytest tests/predict
```

4. Open the two issues from [UPSTREAM_ISSUE_DRAFTS.md](UPSTREAM_ISSUE_DRAFTS.md).
5. Wait for maintainer acknowledgment before opening PR 2.
6. You can prepare PR 1 sooner, but still reference the issue in the PR body.

## PR 1: Adapter Fallback Boundary

### What This PR Should Actually Be Upstream

In upstream DSPy, the behavior lives in `ChatAdapter`, not `Predict`.

This is a separate adapter-boundary policy change, not part of the remaining self-improvement-eval failure narrative.

Your upstream PR should therefore be framed as:

- make `ChatAdapter` fallback opt-in
- keep compatibility behavior available explicitly
- add strict-default and opt-in regression tests

Do not recreate the old local `JSONWithChatFallbackAdapter` patch mechanically. That fork-only adapter was removed once `dspy-slim` returned to upstream-compatible `ChatAdapter` defaults, and it was never part of upstream DSPy's public surface.

### Exact Steps

1. Start from clean upstream `main`.

```bash
cd ~/codebase/dspy-upstream
git fetch upstream
git switch -c emre/chatadapter-fallback-optin upstream/main
```

2. Reproduce the current behavior locally before changing code.

- Confirm that one `Predict(...)` call can trigger two LM calls when `ChatAdapter` falls back to `JSONAdapter`.
- Save that reproducer as a focused regression test.

3. Implement the smallest acceptable change.

- Preferred options, in order:
  - keep the current mechanism but make it opt-in by default
  - if maintainers prefer, move the retry into a clearly named compatibility adapter
- Do not widen the scope.
- Do not touch unrelated `Predict` behavior.

4. Add exactly two high-leverage tests.

- strict default: parse failure propagates, no hidden retry
- compatibility path: fallback still works when explicitly enabled

5. Run only the checks you need first.

```bash
uv run --extra dev python -m pytest <the test file you changed>
uv run --extra dev ruff check dspy/adapters/chat_adapter.py <the test file you changed>
pre-commit run --files dspy/adapters/chat_adapter.py <the test file you changed>
```

6. If those pass, run the nearby test area.

```bash
uv run --extra dev python -m pytest tests/predict tests/adapters
```

7. Commit with a plain, boring message.

```bash
git add dspy/adapters/chat_adapter.py <test-file>
git commit -m "Make ChatAdapter fallback opt-in"
```

8. Push your branch.

```bash
git push -u origin emre/chatadapter-fallback-optin
```

9. Open the PR as a Draft.

- Base repo: `stanfordnlp/dspy`
- Base branch: `main`
- Head repo: your fork
- Head branch: `emre/chatadapter-fallback-optin`

### Suggested PR Title

`Make ChatAdapter JSON fallback opt-in`

### Suggested PR Body

```md
## Summary

`ChatAdapter` currently catches broad exceptions and silently retries with `JSONAdapter` by default. This makes parse failures implicit and can hide the first formatting failure behind a second LM call.

This PR makes that fallback opt-in and adds regression coverage for both the strict default and the explicit compatibility path.

This is intentionally framed as a small adapter-boundary policy change, not as part of the remaining RLM reliability stack from the self-improvement eval.

## Migration

If you relied on the old behavior, enable the compatibility path explicitly.

## Testing

- `uv run --extra dev python -m pytest <test file>`
- `uv run --extra dev python -m pytest tests/predict tests/adapters`

## AI Disclosure

I used Amp as an assisted coding tool for code reading, test drafting, and PR text refinement. I verified the behavior locally and wrote the final issue/PR text myself.

Prompts used:
- "Inspect the current upstream adapter flow and verify where fallback retry actually happens."
- "Help me draft a minimal reproducer for hidden adapter fallback."
- "Help me tighten the PR description and migration note."
```

### After You Open It

- If a maintainer says "please keep the existing flag and just flip the default," do exactly that.
- If a maintainer says "please avoid a new adapter class," do not fight for it.
- If they ask for a smaller diff, reduce scope immediately.

## PR 2: RLM Reliability

### Current Readiness

This is worth upstreaming, but only after one more careful hardening pass.

This is the PR that should carry the remaining self-improvement-eval upstream story.

Before you open this PR, make sure the upstream version does all of the following:

- propagates LM truncation metadata without leaking it into user-facing signature outputs
- gives `RLM` a structured way to react to truncation
- includes a targeted test for truncation-triggered finalization
- includes a targeted test for the `bootstrap_trace` partial-parse reward bug
- includes a clear migration note if direct `lm()` callers stop receiving `list[str]`

Also: do not include the local `dspy.Tool` error-message change in this PR. That issue does not reproduce on upstream `main`.

### Exact Steps

1. Wait for a maintainer to reply to the issue.

- If they want it split, split it.
- If they are comfortable with one PR, keep the narrative tight and the commits clean.

2. Start a fresh branch from upstream `main`.

```bash
cd ~/codebase/dspy-upstream
git fetch upstream
git switch -c emre/rlm-truncation-state upstream/main
```

3. Port the change intentionally, not by bulk-copying files from `dspy-slim`.

Focus on the current upstream files:

- `dspy/clients/base_lm.py`
- `dspy/adapters/base.py`
- `dspy/primitives/prediction.py`
- `dspy/predict/rlm.py`
- `dspy/primitives/repl_types.py`
- `dspy/teleprompt/bootstrap_trace.py`
- new helper file only if still justified on upstream

4. Keep the implementation story simple.

- metadata transport first
- `RLM` consumes that metadata second
- `bootstrap_trace` bug fix last

5. Add the tests maintainers are most likely to ask for.

- chat-completion metadata propagation
- responses-API metadata propagation if that path is affected
- metadata does not leak into user-visible `Prediction` fields
- truncation causes `RLM` to switch into finalization-oriented prompting
- `bootstrap_trace` partial-parse reward uses lengths, not lists

6. Run focused checks first.

```bash
uv run --extra dev python -m pytest <the RLM/LM test files you changed>
uv run --extra dev ruff check <the files you changed>
pre-commit run --files <the files you changed>
```

7. Then run the surrounding test areas.

```bash
uv run --extra dev python -m pytest tests/predict tests/teleprompt
```

8. Commit with a narrow message.

```bash
git add <changed files>
git commit -m "Expose LM truncation state to RLM"
```

9. Push the branch.

```bash
git push -u origin emre/rlm-truncation-state
```

10. Open this PR as a Draft first.

### Suggested PR Title

`Expose LM truncation state so RLM can finalize gracefully`

### Suggested PR Body

```md
## Summary

This PR makes LM truncation a framework-visible state so `RLM` can react when an action step is cut off instead of spending its remaining turns on exploratory steps.

It does that by propagating provider-agnostic LM metadata through the framework and then using that signal in `RLM`'s late-iteration path. It also fixes a small adjacent bug in `bootstrap_trace_data`, where the partial-parse reward path currently divides two lists instead of their lengths.

## Breaking Change

If you call `lm(...)` directly and currently rely on text-only outputs collapsing to `list[str]`, this PR changes that behavior. The migration path is to read `output["text"]` from each completion dict.

## Testing

- `uv run --extra dev python -m pytest <targeted test files>`
- `uv run --extra dev python -m pytest tests/predict tests/teleprompt`

## AI Disclosure

I used Amp as an assisted coding tool for code reading, test drafting, and PR text refinement. I verified the behavior locally and wrote the final issue/PR text myself.

Prompts used:
- "Inspect the current upstream LM / adapter / RLM path and verify whether truncation is visible to module code."
- "Help me draft a minimal reproducer for truncation visibility and the bootstrap partial-parse crash."
- "Help me tighten the PR description and migration note for the LM output shape change."
```

### After You Open It

- Be ready for maintainers to ask for a split. If they do, split without argument.
- If they push back on the breaking output-shape change, offer to separate metadata transport from `RLM` behavior.
- Keep every review reply short, specific, and technical.

## Review Etiquette

- Keep the PR descriptions short and specific.
- Do not tell the Korbex backstory unless someone asks how you found the bug.
- Do not mention AI in a vague way. Follow their policy exactly: tool name, what it helped with, and the prompts.
- Do not let any tool submit the PR for you.
- Be able to explain every changed line without consulting notes.

## Final Checklist Before Clicking Submit

- issues opened first
- branch based on current upstream `main`
- diff is minimal and scoped
- tests pass locally
- `pre-commit` passes on changed files
- AI disclosure included
- prompts included
- migration note included when behavior changes
- PR opened as Draft first
