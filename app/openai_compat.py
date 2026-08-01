"""Compatibility helpers for OpenAI Chat Completions models."""

from collections.abc import Sequence
from typing import Any


# These picker models support reasoning_effort="none" and are intended to
# behave as low-latency text models in this application.
NON_REASONING_OPENAI_MODELS = frozenset(
    {
        "gpt-5.6-luna",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
    }
)


def build_openai_chat_payload(
    *,
    model: str,
    messages: Sequence[dict[str, Any]],
    max_completion_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Build the shared low-latency Chat Completions request payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_completion_tokens": max_completion_tokens,
        "stream": stream,
    }

    if model in NON_REASONING_OPENAI_MODELS:
        payload["reasoning_effort"] = "none"

    return payload
