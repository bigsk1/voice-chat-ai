import os
from typing import Optional
import openai
from openai import AsyncOpenAI

# 環境変数読み込み
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MODEL_TTS = os.getenv("OPENAI_MODEL_TTS", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
LANGUAGE = os.getenv("LANGUAGE", "en")


def _set_api_base_from_env(env_var: str, suffixes: list[str]) -> None:
    """
    環境変数 env_var から取得した URL を openai.api_base に設定する。
    suffixes のいずれかが末尾に含まれる場合はその部分を切り落とす。
    """
    base = os.getenv(env_var, "https://api.openai.com/v1").rstrip('/')  # 末尾のスラッシュを削除
    for s in suffixes:
        if base.endswith(s):
            base = base[: -len(s)]
            break
    openai.api_base = base


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """Transcribe audio bytes using OpenAI Whisper transcription API."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    # Whisper Endpoint: remove /chat/completions if present
    _set_api_base_from_env("OPENAI_BASE_URL", ["/chat/completions", "/audio/transcriptions"])
    openai.api_key = key

    client = AsyncOpenAI()
    model_name = model or OPENAI_TRANSCRIPTION_MODEL
    result = await client.audio.transcriptions.create(
        model=model_name,
        file=("audio.wav", audio_bytes, "audio/wav"),
        language=LANGUAGE,
    )
    return result.text


async def generate_response_text(
    user_text: str,
    api_key: Optional[str] = None
) -> str:
    """Generate chat response using OpenAI Chat completions API."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    # Chat Endpoint: ensure base ends at /v1
    _set_api_base_from_env("OPENAI_BASE_URL", ["/chat/completions"])
    openai.api_key = key

    client = AsyncOpenAI()
    completion = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": user_text}],
    )
    return completion.choices[0].message.content.strip()


async def synthesize_text(
    text: str,
    api_key: Optional[str] = None
) -> bytes:
    """Synthesize speech using OpenAI TTS API and return audio bytes."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    # TTS Endpoint: remove /audio/speech if present
    _set_api_base_from_env("OPENAI_TTS_URL", ["/audio/speech"])
    openai.api_key = key

    client = AsyncOpenAI()
    response = await client.audio.speech.create(
        model=OPENAI_MODEL_TTS,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="wav",
        language=LANGUAGE,
    )
    return await response.read()
