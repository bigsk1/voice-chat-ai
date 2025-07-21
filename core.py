import os
from typing import Optional
from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MODEL_TTS = os.getenv("OPENAI_MODEL_TTS", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
LANGUAGE = os.getenv("LANGUAGE", "en")


async def transcribe_audio_bytes(audio_bytes: bytes, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    """Transcribe audio bytes using OpenAI API."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = AsyncOpenAI(api_key=key)
    model_name = model or OPENAI_TRANSCRIPTION_MODEL
    result = await client.audio.transcriptions.create(
        model=model_name,
        file=("audio.wav", audio_bytes, "audio/wav"),
        language=LANGUAGE,
    )
    return result.text


async def generate_response_text(user_text: str, api_key: Optional[str] = None) -> str:
    """Generate chat response using OpenAI Chat API."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = AsyncOpenAI(api_key=key)
    completion = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": user_text}],
    )
    return completion.choices[0].message.content.strip()


async def synthesize_text(text: str, api_key: Optional[str] = None) -> bytes:
    """Synthesize speech using OpenAI TTS API and return audio bytes."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = AsyncOpenAI(api_key=key)
    response = await client.audio.speech.create(
        model=OPENAI_MODEL_TTS,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="wav",
        language=LANGUAGE,
    )
    return await response.read()
