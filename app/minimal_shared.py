"""
Shared resources used across the application
"""

import os
from dotenv import load_dotenv
import logging
import json
import re
import aiohttp
from pathlib import Path
import io
import wave
import requests
import tempfile

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
BLUE = '\033[94m'
RESET_COLOR = '\033[0m'

load_dotenv()

# WebSocket clients
clients = set()
active_client_status = {}  # Track status of websocket clients
client_api_keys = {}

# Shared state variables
current_character = os.getenv("CHARACTER_NAME")  # Get from .env
conversation_active = False
conversation_history = []
continue_conversation = False  # Added missing variable
characters_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "characters")


MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'openai')
CHARACTER_NAME = os.getenv('CHARACTER_NAME', 'wizard')
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
OPENAI_TTS_URL = os.getenv('OPENAI_TTS_URL', 'https://api.openai.com/v1/audio/speech')
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', 'alloy')
OPENAI_MODEL_TTS = os.getenv('OPENAI_MODEL_TTS', 'gpt-4o-mini-tts')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions')
XAI_API_KEY = os.getenv('XAI_API_KEY')
XAI_MODEL = os.getenv('XAI_MODEL', 'grok-2-1212')
XAI_BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-7-sonnet-20250219')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_TTS_VOICE = os.getenv('ELEVENLABS_TTS_VOICE')
ELEVENLABS_TTS_MODEL = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_multilingual_v2')
KOKORO_BASE_URL = os.getenv('KOKORO_BASE_URL', 'http://localhost:8880/v1')
KOKORO_TTS_VOICE = os.getenv('KOKORO_TTS_VOICE', 'af_bella')
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))
VOICE_SPEED = os.getenv('VOICE_SPEED', '1.0')
XTTS_NUM_CHARS = int(os.getenv('XTTS_NUM_CHARS', 255))
LANGUAGE = os.getenv('LANGUAGE', 'en')

# Functions to get and set shared state
def get_current_character():
    """Get the current character."""
    global current_character # noqa: F824
    return current_character

def set_current_character(character):
    """Set the current character."""
    global current_character # noqa: F824
    current_character = character

def is_conversation_active():
    """Check if a conversation is active."""
    global conversation_active # noqa: F824
    return conversation_active

def set_conversation_active(active):
    """Set the conversation active state."""
    global conversation_active # noqa: F824
    conversation_active = active

def add_client(client):
    """Add a client to the set of connected clients."""
    clients.add(client)
    active_client_status[client] = True

def remove_client(client):
    """Remove a client from the set of connected clients."""
    clients.discard(client)
    if client in active_client_status:
        del active_client_status[client]
    if client in client_api_keys:
        del client_api_keys[client]

def is_client_active(client):
    """Check if a client is active."""
    return active_client_status.get(client, False)

def set_client_inactive(client):
    """Mark a client as inactive."""
    active_client_status[client] = False

def set_client_api_key(client, api_key):
    client_api_keys[client] = api_key

def get_client_api_key(client):
    return client_api_keys.get(client)

def clear_conversation_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []

def load_character_prompt(character_name):
    """
    Load the character prompt from the character's text file.
    
    Args:
        character_name (str): The name of the character folder.
        
    Returns:
        str: The character prompt text.
    """
    try:
        character_file_path = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        if not os.path.exists(character_file_path):
            return None
            
        with open(character_file_path, 'r', encoding='utf-8') as file:
            character_prompt = file.read()
            
            
        return character_prompt
    except Exception as e:
        print(f"Error loading character prompt: {e}")
        return None

async def transcribe_audio_bytes(audio_bytes: bytes, api_key: str | None = None, model: str = None) -> str:
    """Transcribe uploaded audio bytes using the configured method."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        text = await transcribe_with_openai_api(tmp_path, model , api_key)
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
        await openai_text_to_speech(text, output_path, api_key)
        with open(output_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass
    return data


# Function to save conversation history to a file
def save_conversation_history(conversation_history):
    history_file = "conversation_history.txt"
    try:
        with open(history_file, "w", encoding="utf-8") as file:
            for message in conversation_history:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n")
    except Exception as e:
        logging.error(f"Error saving conversation history: {e}")
        return {"status": "error", "message": str(e)}
    return {"status": "success"}


def save_character_specific_history(history, character_name):
    """
    Save conversation history to a character-specific file for story/game characters.
    Only to be used for characters with names starting with story_ or game_
    
    Args:
        history: The conversation history to save
        character_name: The name of the character
        
    Returns:
        dict: Status of the operation
    """
    try:
        # Only process for story/game characters
        if not character_name.startswith("story_") and not character_name.startswith("game_"):
            print(f"Not a story/game character: {character_name}, using global history instead")
            return save_conversation_history(history)
            
        # Create character-specific history file path
        character_dir = os.path.join(characters_folder, character_name)
        history_file = os.path.join(character_dir, "conversation_history.txt")
        
        print(f"Saving character-specific history for {character_name}")
        
        with open(history_file, "w", encoding="utf-8") as file:
            for message in history:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n\n")  # Extra newline for readability
                
        print(f"Saved {len(history)} messages to character-specific history file")
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error saving character-specific history: {e}")
        return {"status": "error", "message": str(e)}

def adjust_prompt(mood):
    """Load mood-specific prompts from the character's prompts.json file."""
    # Import with alias to avoid potential shadowing issues
    #from .minimal_shared import get_current_character as get_character
    
    # Get the current character
    current_character = get_current_character()
    
    # Look for character-specific prompts first
    character_prompts_path = os.path.join(characters_folder, current_character, 'prompts.json')
    
    # Control output verbosity using the DEBUG flag from enhanced_logic.py
    try:
        # Import DEBUG flag if it exists
        try:
            from .enhanced_logic import DEBUG
        except ImportError:
            DEBUG = False  # Default to False if not available
            
        # Try to load character-specific prompts
        if os.path.exists(character_prompts_path):
            with open(character_prompts_path, 'r', encoding='utf-8') as f:
                mood_prompts = json.load(f)
                if DEBUG:
                    print(f"Loaded mood prompts for character: {current_character}")
        else:
            # Fall back to global prompts
            prompts_path = os.path.join(characters_folder, 'prompts.json')
            with open(prompts_path, 'r', encoding='utf-8') as f:
                mood_prompts = json.load(f)
                if DEBUG:
                    print(f"Using global prompts.json - character-specific prompts not found")
    except FileNotFoundError:
        print(f"Error loading prompts: character or global prompts.json not found. Using default prompts.")
        mood_prompts = {
            "happy": "RESPOND WITH JOY AND ENTHUSIASM.",
            "sad": "RESPOND WITH KINDNESS AND COMFORT.",
            "flirty": "RESPOND WITH A TOUCH OF MYSTERY AND CHARM.",
            "angry": "RESPOND CALMLY AND WISELY.",
            "neutral": "KEEP RESPONSES SHORT AND NATURAL.",
            "fearful": "RESPOND WITH REASSURANCE.",
            "surprised": "RESPOND WITH AMAZEMENT.",
            "disgusted": "RESPOND WITH UNDERSTANDING.",
            "joyful": "RESPOND WITH EXUBERANCE."
        }
    except Exception as e:
        print(f"Error loading prompts: {e}")
        mood_prompts = {}

    # Get the mood prompt but don't print it in normal logging
    mood_prompt = mood_prompts.get(mood, "")
    
    # Debug output only if DEBUG is enabled
    if 'DEBUG' in locals() and DEBUG:
        print(f"Selected prompt for {current_character} ({mood}): {mood_prompt[:100]}...")
    
    return mood_prompt

def analyze_mood(user_input):
    #日本語対応していないのでいったん省略
    return "neutral"

def sanitize_response(response):
    # Remove <think>...</think> blocks first
    response = re.sub(r'<think>[\s\S]*?<\/think>', '', response)
    # Remove asterisks and other formatting
    response = re.sub(r'\*.*?\*', '', response)
    response = re.sub(r'[^\w\s,.\'!?]', '', response)
    # Trim any whitespace
    return response.strip()

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    

async def openai_text_to_speech(prompt, output_path, api_key=None):
    file_extension = Path(output_path).suffix.lstrip('.').lower()
    print(f"file_extension={file_extension}")
    print(f"output_path={output_path}")

    voice_speed = float(os.getenv("VOICE_SPEED", "1.0"))

    async with aiohttp.ClientSession() as session:
        if file_extension == 'wav':
            pcm_data = await fetch_pcm_audio(OPENAI_MODEL_TTS, OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL, session, api_key)
            save_pcm_as_wav(pcm_data, output_path)
        else:
            try:
                async with session.post(
                    url=OPENAI_TTS_URL,
                    headers={"Authorization": f"Bearer {api_key or OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={"model": OPENAI_MODEL_TTS, "voice": OPENAI_TTS_VOICE, "input": prompt, "response_format": file_extension, "speed": voice_speed, "language": LANGUAGE},
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

                print("Audio generated successfully with OpenAI.")
            except aiohttp.ClientError as e:
                print(f"Error during OpenAI TTS: {e}")
                
async def fetch_pcm_audio(model: str, voice: str, input_text: str, api_url: str, session: aiohttp.ClientSession, api_key=None) -> bytes:
    pcm_data = io.BytesIO()
    
    try:
        async with session.post(
            url=api_url,
            headers={"Authorization": f"Bearer {api_key or OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "voice": voice, "input": input_text, "response_format": 'pcm', "language": LANGUAGE},
            timeout=30
        ) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(8192):
                pcm_data.write(chunk)
    except aiohttp.ClientError as e:
        print(f"An error occurred while trying to fetch the audio stream: {e}")
        raise

    return pcm_data.getvalue()         

def save_pcm_as_wav(pcm_data: bytes, file_path: str, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

def chatgpt_streamed(user_input, system_message, mood_prompt, conversation_history, api_key=None):
    full_response = ""
    print(f"Debug: streamed started. MODEL_PROVIDER: {MODEL_PROVIDER}")

    # Calculate token limit based on character limit Approximate token conversion, So if MAX_CHAR_LENGTH is 500, then 500 * 4 // 3 = 666 tokens
    token_limit = min(4000, MAX_CHAR_LENGTH * 4 // 3)

    if MODEL_PROVIDER == 'openai':
        messages = [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}]
        headers = {'Authorization': f'Bearer {api_key or OPENAI_API_KEY}', 'Content-Type': 'application/json'}
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": token_limit  # Approximate token conversion
        }
        try:
            print(f"Debug: Sending request to OpenAI: {OPENAI_BASE_URL}")
            response = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, stream=True, timeout=45)
            response.raise_for_status()

            print("Starting OpenAI stream...")
            line_buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line:
                    try:
                        chunk = json.loads(line)
                        delta_content = chunk['choices'][0]['delta'].get('content', '')
                        if delta_content:
                            line_buffer += delta_content
                            if '\n' in line_buffer:
                                lines = line_buffer.split('\n')
                                for line in lines[:-1]:
                                    print(NEON_GREEN + line + RESET_COLOR)
                                    full_response += line + '\n'
                                line_buffer = lines[-1]
                    except json.JSONDecodeError:
                        continue
            if line_buffer:
                print(NEON_GREEN + line_buffer + RESET_COLOR)
                full_response += line_buffer
            print("\nOpenAI stream complete.")

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to OpenAI model: {e}"
            print(f"Debug: OpenAI error - {e}")
            
    print(f"streaming complete. Response length: {PINK}{len(full_response)}{RESET_COLOR}")
    return full_response


async def transcribe_with_openai_api(audio_file, model="whisper-1", api_key: str | None = None):
    """Transcribe audio using OpenAI's API"""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("API key missing. Please set OPENAI_API_KEY in your environment.")
    
    # Make the API call to OpenAI
    api_url = "https://api.openai.com/v1/audio/transcriptions"
    
    async with aiohttp.ClientSession() as session:
        with open(audio_file, "rb") as audio_file_data:
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                                audio_file_data.read(),
                                filename=os.path.basename(audio_file),
                                content_type='audio/wav')
            
            # Use the model directly
            form_data.add_field('model', model)
            form_data.add_field('language', LANGUAGE)
            
            headers = {
                "Authorization": f"Bearer {key}"
            }
            
            async with session.post(api_url, data=form_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    transcription = result.get("text", "")
                    return transcription
                else:
                    error_text = await response.text()
                    print(f"Error from OpenAI API: {error_text}")
                    raise Exception(f"Transcription error: {response.status} - {error_text}")
