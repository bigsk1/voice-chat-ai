from typing import Optional

from core import (
    transcribe_audio_bytes as core_transcribe_audio_bytes,
    generate_response_text as core_generate_response_text,
    synthesize_text as core_synthesize_text,
)

async def transcribe_audio_bytes(audio_bytes: bytes, api_key: str | None = None, model: str = None) -> str:
    """Wrapper for backward compatibility."""
    return await core_transcribe_audio_bytes(audio_bytes, api_key=api_key, model=model)

async def generate_response_text(user_text: str, api_key: str | None = None) -> str:
    """Wrapper for backward compatibility."""
    return await core_generate_response_text(user_text, api_key=api_key)

async def synthesize_text(text: str, api_key: str | None = None) -> bytes:
    """Wrapper for backward compatibility."""
    return await core_synthesize_text(text, api_key=api_key)
