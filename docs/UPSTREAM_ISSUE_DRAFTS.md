# Upstream DSPy Issue Drafts

These drafts are based on:

- the current upstream contribution guide on `stanfordnlp/dspy` `main` as of 2026-04-13
- direct inspection of current upstream source files
- the local proof-of-concept patches and tests in this workspace

## What To Open

Open exactly two upstream issues.

- Do open one issue for the RLM truncation / metadata visibility problem, with the `bootstrap_trace` partial-parse crash included as a small adjacent bug. This is the remaining direct upstream-value issue stack from the self-improvement eval.
- Do open one issue for the adapter fallback policy, but frame it around `ChatAdapter`, not `Predict`. Treat that as a separate compatibility-policy discussion discovered during the upstream audit, not as an unresolved eval failure.
- Do not open a separate issue for the `dspy.Tool` error message. Current upstream `main` still exports `dspy.Tool` at top level, so that local `dspy-slim` issue does not reproduce upstream.
- Do not treat current `ToolCall.execute(list[Tool])` coercion gaps or async-tool sync-conversion behavior as unresolved local fork drift. The current upstream helper code has the same semantics, so any change there should be handled as a separate upstream helper-surface discussion rather than silently improved only in `dspy-slim`.
- In `dspy-slim`, the truncation metadata transport and `RLM` finalization behavior already exist as intentional local reliability work. Use this repo as a proof of concept only, not as evidence that upstream already has the same boundary.

## Issue 1

### Suggested Title

`Expose LM truncation to modules so RLM can finalize gracefully`

### Why This Is Worth Upstreaming

- It is a framework-level gap, not a product-specific tweak.
- Current upstream DSPy logs truncation, but does not expose it to module code.
- `RLM` therefore cannot react when its action step is cut off by `finish_reason="length"`.
- There is also a real adjacent bug in `bootstrap_trace_data`: the partial-parse reward path divides two lists instead of their lengths, which raises `TypeError`.

### Paste-Ready Draft

```md
I found an RLM reliability gap that seems worth fixing at the framework layer.

## Summary

Right now truncation is visible as a warning in `dspy/clients/lm.py`, but it is not propagated through the normal `BaseLM` / adapter / `Prediction` path. That means modules like `RLM` cannot tell that an action step was cut off with `finish_reason="length"`; they just receive the already-collapsed text and keep going.

In practice this means RLM can waste its remaining iterations on exploratory turns after a truncated step instead of switching into a finalization-oriented turn.

While tracing this, I also found a small adjacent bug in `bootstrap_trace_data`: the partial-parse reward path currently computes `present / expected` where both values are lists, which raises `TypeError` whenever `parsed_result` is non-empty.

## Current upstream locations

- `dspy/clients/lm.py`: `_check_truncation()` only logs a warning
- `dspy/clients/base_lm.py`: `_process_completion()` collapses text-only outputs to `list[str]` and drops `finish_reason`
- `dspy/predict/rlm.py`: no structured truncation signal to react to
- `dspy/teleprompt/bootstrap_trace.py`: partial-parse reward uses `(present / expected)` instead of a length ratio

## Minimal repro for the truncation visibility gap

```python
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict


class FinishLengthLM(BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content='{"answer": "partial"}', tool_calls=None),
                    finish_reason="length",
                )
            ],
            usage=dotdict(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            model="test-model",
        )


lm = FinishLengthLM("test-model", "chat", 0.0, 1000, False)
print(lm(messages=[{"role": "user", "content": "hi"}]))
# The truncation warning is logged, but the returned output does not expose
# a structured `finish_reason` / `truncated` signal to the caller.
```

## Proposed direction

My current thought is:

1. propagate provider-agnostic LM metadata (for example `truncated`, `finish_reason`, `usage`) through the framework
2. keep that metadata outside the user signature fields
3. let `RLM` use it to switch into a finalization-oriented prompt path when an action step is truncated
4. fix the `bootstrap_trace_data` length-ratio bug in the same PR unless you'd prefer it split out

One caveat: if the best implementation requires `BaseLM.__call__` to always return dict outputs instead of collapsing to `list[str]`, I would call that out as a breaking change and include a short migration note for direct `lm()` callers.

If this direction makes sense, I'm happy to open a focused PR.
```

### Notes For You

- Post this issue first and wait for a maintainer signal before opening the PR.
- If they ask to split it, the clean fallback split is:
  - LM metadata transport
  - RLM finalization behavior
  - `bootstrap_trace` bug fix
- This is the issue that should stay tied to the self-improvement-eval narrative.

## Issue 2

### Suggested Title

`Make ChatAdapter JSON fallback opt-in instead of a silent default retry`

### Why This Is Worth Upstreaming

- This is an abstraction-boundary problem in upstream DSPy today.
- The retry is currently in `ChatAdapter`, not `Predict`.
- `ChatAdapter.__call__` and `acall` catch broad exceptions and silently retry with `JSONAdapter` by default.
- That makes parse failures harder to observe in teleprompt / bootstrap / eval flows.
- It is a small, clean change with a clear migration story, which makes it the best first PR.

### Paste-Ready Draft

```md
I found a behavior in the current adapter stack that seems worth making explicit.

## Summary

`Predict` itself is clean here, but `ChatAdapter` currently catches broad exceptions and silently retries with `JSONAdapter` by default (`use_json_adapter_fallback=True`).

That means one logical `Predict(...)` call can hide an initial parse failure and make a second LM call without the caller opting into that policy. In flows that want to observe formatting failures directly (for example teleprompt / bootstrap / evaluation), that makes the adapter boundary less explicit than it could be.

## Current upstream location

- `dspy/adapters/chat_adapter.py`: `__call__` and `acall` catch broad exceptions and retry with `JSONAdapter`

## Minimal repro

```python
import dspy
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict


def response(text: str):
    return dotdict(
        choices=[dotdict(message=dotdict(content=text, tool_calls=None), finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="test-model",
    )


class CountingLM(BaseLM):
    def __init__(self):
        super().__init__("test-model", "chat", 0.0, 1000, False)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return response("not ChatAdapter-formatted output")
        return response('{"output": "recovered"}')


lm = CountingLM()
dspy.configure(lm=lm)
pred = dspy.Predict("input -> output")(input="x")

print(pred.output)
print(lm.calls)
# recovered
# 2
```

## Proposed direction

I think this fallback should be opt-in rather than default. I could implement that in whichever shape you prefer:

- flip the existing `use_json_adapter_fallback` default to `False`, or
- move the retry into a clearly named compatibility adapter / wrapper

Either way, I would include regression tests for:

- strict default behavior (parse failures propagate)
- explicit compatibility behavior (fallback still available when requested)

If this sounds aligned with how you want the adapter boundary to work, I'm happy to send a PR.
```

### Notes For You

- This is the best PR to send first.
- Keep the issue short.
- Do not mention `Predict` in the title; upstream `Predict` is not where the fallback currently lives.
- Keep it framed as a separate adapter-boundary policy discussion, not as part of the remaining eval-failure stack.
