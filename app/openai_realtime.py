"""Compatibility helpers for OpenAI's GA Realtime API."""

from __future__ import annotations

import re
from urllib.parse import quote, urljoin, urlparse


DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2.1"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
OPENAI_REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
DEFAULT_REALTIME_PROVIDER = "openai"
DEFAULT_LOCAL_REALTIME_CALLS_URL = "http://127.0.0.1:8765/v1/realtime/calls"
DEFAULT_LOCAL_REALTIME_VOICE = "Aiden"
DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD = 0.6
DEFAULT_LOCAL_REALTIME_SILENCE_MS = 1000

LOCAL_REALTIME_VOICES = (
    "Aiden",
    "Ryan",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
)
LOCAL_REALTIME_VOICE_LOOKUP = {
    voice.casefold(): voice for voice in LOCAL_REALTIME_VOICES
}

REALTIME_PROVIDERS = frozenset({"openai", "local"})
LOCAL_REALTIME_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~-]+$")

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


def normalize_realtime_provider(provider: str | None) -> str:
    """Return a supported Realtime backend name."""
    candidate = (provider or DEFAULT_REALTIME_PROVIDER).strip().lower()
    if candidate not in REALTIME_PROVIDERS:
        return DEFAULT_REALTIME_PROVIDER
    return candidate


def normalize_local_realtime_url(url: str | None) -> str:
    """Validate the configured local GA-compatible WebRTC calls endpoint."""
    candidate = (url or DEFAULT_LOCAL_REALTIME_CALLS_URL).strip().rstrip("/")
    parsed = urlparse(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1/realtime/calls")
    ):
        raise ValueError(
            "LOCAL_REALTIME_URL must be an http(s) URL ending in /v1/realtime/calls"
        )
    return candidate


def normalize_local_realtime_voice(voice: str | None) -> str:
    """Return a canonical Qwen CustomVoice speaker supported by the server."""
    candidate = (voice or "").strip().casefold()
    return LOCAL_REALTIME_VOICE_LOOKUP.get(candidate, DEFAULT_LOCAL_REALTIME_VOICE)


def normalize_local_realtime_vad_threshold(value: str | float | None) -> float:
    """Return a useful Silero speech threshold for the local Realtime server."""
    try:
        candidate = float(value) if value is not None else DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD
    except (TypeError, ValueError):
        return DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD
    if not 0.0 <= candidate <= 1.0:
        return DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD
    return candidate


def normalize_local_realtime_silence_ms(value: str | int | None) -> int:
    """Return a safe end-of-turn silence duration in milliseconds."""
    try:
        candidate = int(value) if value is not None else DEFAULT_LOCAL_REALTIME_SILENCE_MS
    except (TypeError, ValueError):
        return DEFAULT_LOCAL_REALTIME_SILENCE_MS
    if not 100 <= candidate <= 5000:
        return DEFAULT_LOCAL_REALTIME_SILENCE_MS
    return candidate


def extract_local_realtime_call_id(location: str | None, calls_url: str) -> str | None:
    """Extract a safe call ID from a local server Location response header."""
    if not location:
        return None

    normalized_calls_url = normalize_local_realtime_url(calls_url)
    absolute_location = urljoin(f"{normalized_calls_url}/", location)
    base = urlparse(normalized_calls_url)
    target = urlparse(absolute_location)
    prefix = f"{base.path.rstrip('/')}/"

    if (
        target.scheme != base.scheme
        or target.netloc != base.netloc
        or target.query
        or target.fragment
        or not target.path.startswith(prefix)
    ):
        return None

    call_id = target.path[len(prefix):]
    if not call_id or "/" in call_id or not LOCAL_REALTIME_CALL_ID_PATTERN.fullmatch(call_id):
        return None
    return call_id


def build_local_realtime_call_url(calls_url: str, call_id: str) -> str:
    """Build a safe local hangup URL from a validated call ID."""
    if not LOCAL_REALTIME_CALL_ID_PATTERN.fullmatch(call_id):
        raise ValueError("Invalid local Realtime call ID")
    return f"{normalize_local_realtime_url(calls_url)}/{quote(call_id, safe='')}"


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
