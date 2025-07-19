import os
import tempfile
from typing import Optional
from fastapi.logger import logger
from ..transcription import transcribe_with_whisper, transcribe_with_openai_api
from ..app_logic import current_transcription_model, use_local_whisper, characters_folder, adjust_prompt
from ..app import (
    analyze_mood,
    chatgpt_streamed,
    sanitize_response,
    open_file,
    openai_text_to_speech,
    elevenlabs_text_to_speech,
)
from ..app_logic import (
    save_conversation_history,
    save_character_specific_history,
)
from ..shared import conversation_history, get_current_character

async def transcribe_audio_bytes(audio_bytes: bytes, api_key: str | None = None, model: str = None) -> str:
    logger.info(f"[transcribe_audio_bytes] モデル: {current_transcription_model}, use_local: {use_local_whisper}, model: {model}")
    """Transcribe uploaded audio bytes using the configured method."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        if use_local_whisper:
            text = transcribe_with_whisper(tmp_path)
        else:
            #text = await transcribe_with_openai_api(tmp_path, current_transcription_model, api_key)
            text = await transcribe_with_openai_api(tmp_path, model or current_transcription_model, api_key)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text

async def generate_response_text(user_text: str, api_key: str | None = None) -> str:
    """Generate a chat response for the given text without playing audio."""
    current_character = get_current_character()
    character_folder = os.path.join("characters", current_character)
    character_prompt_file = os.path.join(character_folder, f"{current_character}.txt")

    base_system_message = open_file(character_prompt_file)
    mood = analyze_mood(user_text)
    mood_prompt = adjust_prompt(mood)

    chatbot_response = chatgpt_streamed(user_text, base_system_message, mood_prompt, conversation_history, api_key)
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": chatbot_response})

    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    if is_story_character:
        save_character_specific_history(conversation_history, current_character)
    else:
        save_conversation_history(conversation_history)

    sanitized = sanitize_response(chatbot_response)
    return sanitized

async def synthesize_text(text: str, api_key: str | None = None) -> bytes:
    """Generate speech audio for the given text and return it as bytes."""
    provider = os.getenv("TTS_PROVIDER", "openai")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name
    try:
        if provider == "elevenlabs":
            await elevenlabs_text_to_speech(text, output_path)
        else:
            await openai_text_to_speech(text, output_path, api_key)
        with open(output_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass
    return data
