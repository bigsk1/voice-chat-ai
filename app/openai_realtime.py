"""Compatibility helpers for OpenAI's GA Realtime API."""

from __future__ import annotations


DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2.1"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
OPENAI_REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"

OPENAI_REALTIME_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "marin",
        "sage",
        "shimmer",
        "verse",
    }
)


def normalize_openai_realtime_model(model: str | None) -> str:
    """Move known retired Realtime model IDs to their current GA replacements."""
    candidate = (model or "").strip()
    if not candidate:
        return DEFAULT_OPENAI_REALTIME_MODEL

    lowered = candidate.lower()
    if lowered.startswith(("gpt-4o-mini-realtime", "gpt-realtime-mini")):
        return "gpt-realtime-2.1-mini"
    if lowered.startswith(("gpt-4o-realtime", "gpt-realtime-2025")) or lowered == "gpt-realtime":
        return DEFAULT_OPENAI_REALTIME_MODEL
    return candidate


def normalize_openai_realtime_voice(voice: str | None) -> str:
    """Return a current Realtime voice, falling back to OpenAI's recommended voice."""
    candidate = (voice or "").strip().lower()
    if candidate in OPENAI_REALTIME_VOICES:
        return candidate
    return DEFAULT_OPENAI_REALTIME_VOICE


def build_openai_realtime_session(model: str | None, voice: str | None) -> dict:
    """Build the GA session object sent with a WebRTC SDP offer."""
    return {
        "type": "realtime",
        "model": normalize_openai_realtime_model(model),
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                }
            },
            "output": {
                "voice": normalize_openai_realtime_voice(voice),
            },
        },
    }
