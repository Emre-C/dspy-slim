"""Provider-agnostic LM call metadata attached to completion dicts.

The key ``DSPY_LM_METADATA_KEY`` is reserved and must not collide with
signature output field names. Adapters merge it into parsed prediction dicts
so modules (e.g. RLM) can react to truncation without scraping ``lm.history``.
"""

from __future__ import annotations

from typing import Any

# Reserved key on raw LM output dicts and on parsed adapter dicts.
DSPY_LM_METADATA_KEY = "_dspy_lm"


def build_chat_completion_metadata(
    response: Any,
    *,
    choice_index: int = 0,
) -> dict[str, Any]:
    """Metadata for OpenAI-style chat completions (choices[].finish_reason)."""
    usage: dict[str, Any] = {}
    if getattr(response, "usage", None) is not None:
        try:
            usage = dict(response.usage)
        except (TypeError, ValueError):
            usage = {}

    choices = getattr(response, "choices", None) or []
    if not choices or choice_index >= len(choices):
        return {"truncated": False, "finish_reason": None, "usage": usage}

    c = choices[choice_index]
    fr = getattr(c, "finish_reason", None)
    if fr is None and isinstance(c, dict):
        fr = c.get("finish_reason")

    truncated = fr == "length"
    return {"truncated": truncated, "finish_reason": fr, "usage": usage}


def build_responses_api_metadata(response: Any) -> dict[str, Any]:
    """Metadata for OpenAI Responses API (status / incomplete_details)."""
    usage: dict[str, Any] = {}
    if getattr(response, "usage", None) is not None:
        try:
            usage = dict(response.usage)
        except (TypeError, ValueError):
            usage = {}

    status = getattr(response, "status", None)
    incomplete = getattr(response, "incomplete_details", None)
    reason = None
    if incomplete is not None:
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
        else:
            reason = getattr(incomplete, "reason", None)

    truncated = status == "incomplete" and reason in (
        "max_output_tokens",
        "max_tokens",
    )
    if truncated:
        finish_reason = "length"
    elif status == "completed":
        finish_reason = "stop"
    else:
        finish_reason = None
    return {
        "truncated": truncated,
        "finish_reason": finish_reason,
        "responses_status": status,
        "incomplete_reason": reason,
        "usage": usage,
    }
