"""Compatibility helpers for xAI Chat Completions models."""

from collections.abc import Sequence
from typing import Any


XAI_REASONING_EFFORT = {
    "grok-4.3": "none",
    "grok-4.3-latest": "none",
    "grok-latest": "none",
    "grok-4.5": "low",
}

RETIRED_XAI_MODEL_REPLACEMENTS = {
    "grok-4-1-fast-non-reasoning": "grok-4.3",
    "grok-2-vision-1212": "grok-4.3",
}


def normalize_xai_model(model: str | None) -> str:
    """Replace model slugs retired by xAI while preserving custom models."""
    selected_model = model or "grok-4.3"
    return RETIRED_XAI_MODEL_REPLACEMENTS.get(selected_model, selected_model)


def build_xai_chat_payload(
    *,
    model: str,
    messages: Sequence[dict[str, Any]],
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Build a low-latency xAI Chat Completions request payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "stream": stream,
    }

    reasoning_effort = XAI_REASONING_EFFORT.get(model)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    return payload


def xai_chat_timeout(model: str, configured_timeout: str | None = None) -> int:
    """Return a longer timeout for models that cannot disable reasoning."""
    if configured_timeout:
        return int(configured_timeout)
    return 120 if model == "grok-4.5" else 45
