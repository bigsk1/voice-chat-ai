import os
import asyncio
import time
import aiohttp
import pyaudio
import wave
import numpy as np
import requests
import json
import base64
from PIL import ImageGrab
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel
import soundfile as sf
from textblob import TextBlob
from pathlib import Path
import anthropic
import re
import io
from datetime import datetime
from pydub import AudioSegment
from .shared import clients, get_current_character

# Load environment variables
load_dotenv()

SparkTTS = None
SPARKTTS_AVAILABLE = False
torch = None

audio_playback_stop_requested = False
audio_playback_pause_requested = False

def request_audio_playback_stop():
    global audio_playback_stop_requested, audio_playback_pause_requested
    audio_playback_stop_requested = True
    audio_playback_pause_requested = False

def reset_audio_playback_stop():
    global audio_playback_stop_requested, audio_playback_pause_requested
    audio_playback_stop_requested = False
    audio_playback_pause_requested = False

def is_audio_playback_stop_requested():
    return audio_playback_stop_requested

def request_audio_playback_pause():
    global audio_playback_pause_requested
    audio_playback_pause_requested = True

def request_audio_playback_resume():
    global audio_playback_pause_requested
    audio_playback_pause_requested = False

def is_audio_playback_pause_requested():
    return audio_playback_pause_requested

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
XAI_MODEL = os.getenv('XAI_MODEL', 'grok-4-1-fast-non-reasoning')
XAI_BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
XAI_TTS_URL = os.getenv('XAI_TTS_URL', 'https://api.x.ai/v1/tts')
XAI_TTS_VOICE = os.getenv('XAI_TTS_VOICE', 'eve')
XAI_TTS_LANGUAGE = os.getenv('XAI_TTS_LANGUAGE', 'en')
XAI_TTS_FORMAT = os.getenv('XAI_TTS_FORMAT', 'mp3')
XAI_TTS_SAMPLE_RATE = os.getenv('XAI_TTS_SAMPLE_RATE')
XAI_TTS_BIT_RATE = os.getenv('XAI_TTS_BIT_RATE')
XAI_TTS_TIMEOUT = int(os.getenv('XAI_TTS_TIMEOUT', '180'))
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-7-sonnet-20250219')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_TTS_VOICE = os.getenv('ELEVENLABS_TTS_VOICE')
ELEVENLABS_TTS_MODEL = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_multilingual_v2')
KOKORO_BASE_URL = os.getenv('KOKORO_BASE_URL', 'http://localhost:8880/v1')
KOKORO_TTS_VOICE = os.getenv('KOKORO_TTS_VOICE', 'af_bella')
TYPECAST_API_KEY = os.getenv('TYPECAST_API_KEY')
TYPECAST_TTS_VOICE = os.getenv('TYPECAST_TTS_VOICE')
TYPECAST_TTS_MODEL = os.getenv('TYPECAST_TTS_MODEL', 'ssfm-v30')
TYPECAST_EMOTION_PRESET = os.getenv('TYPECAST_EMOTION_PRESET', 'normal')
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))
VOICE_SPEED = os.getenv('VOICE_SPEED', '1.0')
SPARKTTS_MODEL_DIR = os.getenv('SPARKTTS_MODEL_DIR', 'pretrained_models/Spark-TTS-0.5B')
SPARKTTS_MAX_CHARS = int(os.getenv('SPARKTTS_MAX_CHARS', 1000))

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
BLUE = '\033[94m'
RESET_COLOR = '\033[0m'

# Initialize OpenAI API key if available
if OPENAI_API_KEY:
    OpenAI.api_key = OPENAI_API_KEY
else:
    print(f"{YELLOW}OPENAI_API_KEY not set in .env file. OpenAI services disabled.{RESET_COLOR}")
    # Set to None to ensure proper error handling when OpenAI services are attempted
    OpenAI.api_key = None

# Capitalize the first letter of the character name
character_display_name = CHARACTER_NAME.capitalize()

def _cuda_available() -> bool:
    try:
        import torch as torch_module

        return bool(torch_module.cuda.is_available())
    except ImportError:
        return False

def ensure_sparktts_available():
    global SparkTTS, SPARKTTS_AVAILABLE, torch
    if SparkTTS is not None:
        return True

    try:
        import sys
        import logging
        import warnings
        import torch as torch_module

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from cli.SparkTTS import SparkTTS as SparkTTS_cls

        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
        torch = torch_module
        SparkTTS = SparkTTS_cls
        SPARKTTS_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"Spark-TTS import failed: {e}")
        SPARKTTS_AVAILABLE = False
        return False

# Check for CUDA availability only when PyTorch is installed.
device = "cuda" if _cuda_available() else "cpu"

# Check if Faster Whisper should be loaded at startup
FASTER_WHISPER_LOCAL = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

# Initialize whisper model as None to lazy load
whisper_model = None

# Paths for character-specific files
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
characters_folder = os.path.join(project_dir, 'characters', CHARACTER_NAME)
character_prompt_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.txt")
character_audio_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.wav")

# Load Spark-TTS configuration
sparktts_model = None

# Initialize Spark-TTS model
if TTS_PROVIDER == 'sparktts':
    if not ensure_sparktts_available():
        print("Spark-TTS is not available. Please ensure it's properly installed.")
        TTS_PROVIDER = 'openai'
        print("Switched to default TTS provider: openai")
    else:
        print(f"Initializing Spark-TTS model from {SPARKTTS_MODEL_DIR}...")
        try:
            torch_device = torch.device(device)
            print(f"Using device: {torch_device} (CUDA available: {torch.cuda.is_available()})")
            sparktts_model = SparkTTS(model_dir=Path(SPARKTTS_MODEL_DIR), device=torch_device)
            print(f"Spark-TTS model loaded successfully on {torch_device}.")
        except Exception as e:
            print(f"Failed to load Spark-TTS model: {e}")
            TTS_PROVIDER = 'openai'
            print("Switched to default TTS provider: openai")

def init_ollama_model(model_name):
    global OLLAMA_MODEL
    OLLAMA_MODEL = model_name
    print(f"Switched to Ollama model: {model_name}")

def init_openai_model(model_name):
    global OPENAI_MODEL
    OPENAI_MODEL = model_name
    print(f"Switched to OpenAI model: {model_name}")
    
def init_xai_model(model_name):
    global XAI_MODEL
    XAI_MODEL = model_name
    print(f"Switched to XAI model: {model_name}")

def init_anthropic_model(model_name):
    global ANTHROPIC_MODEL
    ANTHROPIC_MODEL = model_name
    print(f"Switched to Anthropic model: {model_name}")

def init_openai_tts_voice(voice_name):
    global OPENAI_TTS_VOICE
    OPENAI_TTS_VOICE = voice_name
    print(f"Switched to OpenAI TTS voice: {voice_name}")

def init_elevenlabs_tts_voice(voice_name):
    global ELEVENLABS_TTS_VOICE
    ELEVENLABS_TTS_VOICE = voice_name
    print(f"Switched to ElevenLabs TTS voice: {voice_name}")

def init_kokoro_tts_voice(voice_name):
    global KOKORO_TTS_VOICE
    KOKORO_TTS_VOICE = voice_name
    print(f"Switched to Kokoro TTS voice: {voice_name}")

def init_xai_tts_voice(voice_name):
    global XAI_TTS_VOICE
    XAI_TTS_VOICE = voice_name
    print(f"Switched to xAI TTS voice: {voice_name}")

def init_typecast_tts_voice(voice_id):
    global TYPECAST_TTS_VOICE
    TYPECAST_TTS_VOICE = voice_id
    print(f"Switched to Typecast TTS voice: {voice_id}")

def init_voice_speed(speed_value):
    global VOICE_SPEED
    VOICE_SPEED = speed_value
    print(f"Switched to global voice speed: {speed_value}")

def init_set_tts(set_tts):
    global TTS_PROVIDER, sparktts_model
    if set_tts == 'sparktts':
        if not ensure_sparktts_available():
            print("Spark-TTS is not available. Please ensure it's properly installed.")
            loop = asyncio.get_running_loop()
            loop.create_task(send_message_to_clients(json.dumps({
                "action": "error",
                "message": "Spark-TTS is not available."
            })))
            return
        
        print(f"Initializing Spark-TTS model from {SPARKTTS_MODEL_DIR}...")
        try:
            torch_device = torch.device(device)
            print(f"Using device: {torch_device} (CUDA available: {torch.cuda.is_available()})")
            sparktts_model = SparkTTS(model_dir=Path(SPARKTTS_MODEL_DIR), device=torch_device)
            print(f"Spark-TTS model loaded successfully on {torch_device}.")
            TTS_PROVIDER = set_tts
        except Exception as e:
            print(f"Failed to load Spark-TTS model: {e}")
            loop = asyncio.get_running_loop()
            loop.create_task(send_message_to_clients(json.dumps({
                "action": "error",
                "message": f"Failed to load Spark-TTS model: {str(e)}"
            })))
    else:
        TTS_PROVIDER = set_tts
        sparktts_model = None
        print(f"Switched to TTS Provider: {set_tts}")

def init_set_provider(set_provider):
    global MODEL_PROVIDER
    MODEL_PROVIDER = set_provider
    print(f"Switched to Model Provider: {set_provider}")
    

# Function to display ElevenLabs quota
def display_elevenlabs_quota():
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=30
        )
        response.raise_for_status()
        user_data = response.json()
        character_count = user_data['subscription']['character_count']
        character_limit = user_data['subscription']['character_limit']
        print(f"{NEON_GREEN}ElevenLabs Character Usage: {character_count} / {character_limit}{RESET_COLOR}")
    except Exception as e:
        print(f"{YELLOW}Could not fetch ElevenLabs quota: {e}{RESET_COLOR}")

if TTS_PROVIDER == "elevenlabs":
    display_elevenlabs_quota()
    
# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Substitute dynamic template variables into a character prompt.
# Supported placeholders: {current_date_time}, {current_date}, {current_time},
# {current_weekday}, {current_iso}. Harmless no-op if none are present.
def render_prompt_template(template):
    if not template:
        return template
    now = datetime.now()
    return (
        template
        .replace("{current_date_time}", now.strftime("%A, %B %d, %Y at %I:%M %p"))
        .replace("{current_date}", now.strftime("%A, %B %d, %Y"))
        .replace("{current_time}", now.strftime("%I:%M %p"))
        .replace("{current_weekday}", now.strftime("%A"))
        .replace("{current_iso}", now.strftime("%Y-%m-%dT%H:%M:%S"))
    )

# Function to play audio using PyAudio
async def play_audio(file_path):
    await asyncio.to_thread(sync_play_audio, file_path)

def sync_play_audio(file_path):
    print("Starting audio playback")
    if is_audio_playback_stop_requested():
        print("Audio playback skipped because stop was requested.")
        return

    file_extension = Path(file_path).suffix.lstrip('.').lower()
    
    temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
    
    if file_extension == 'mp3':
        audio = AudioSegment.from_mp3(file_path)
        audio.export(temp_wav_path, format="wav")
        file_path = temp_wav_path
    
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data and not is_audio_playback_stop_requested():
            while is_audio_playback_pause_requested() and not is_audio_playback_stop_requested():
                time.sleep(0.05)
            if is_audio_playback_stop_requested():
                break
            stream.write(data)
            data = wf.readframes(1024)
            time.sleep(0)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            stream.close()
        wf.close()
        p.terminate()
    print("Finished audio playback")

    pass

output_dir = os.path.join(project_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

if TTS_PROVIDER == 'sparktts':
    print(f"{NEON_GREEN}Using device: {device}{RESET_COLOR}")
print(f"{NEON_GREEN}Model provider: {MODEL_PROVIDER}{RESET_COLOR}")
print(f"{NEON_GREEN}Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else XAI_MODEL if MODEL_PROVIDER == 'xai' else ANTHROPIC_MODEL if MODEL_PROVIDER == 'anthropic' else OLLAMA_MODEL}{RESET_COLOR}")
print(f"{NEON_GREEN}Character: {character_display_name}{RESET_COLOR}")
print(f"{NEON_GREEN}Text-to-Speech provider: {TTS_PROVIDER}{RESET_COLOR}")
print(f"To stop chatting say Quit or Exit. One moment please loading...")

async def process_and_play(prompt, audio_file_pth):
    # Always get the current character name to ensure we have the right audio file
    current_character = get_current_character()
    
    # Update characters_folder path to point to the current character's folder
    current_characters_folder = os.path.join(project_dir, 'characters', current_character)
    
    # Override the provided audio path with the current character's audio file
    # This ensures we always use the correct character voice even after switching
    current_audio_file = os.path.join(current_characters_folder, f"{current_character}.wav")
    
    # Fall back to the provided path if the current character file doesn't exist
    # Could just point to one fallback .wav file for all characters but this works.
    if not os.path.exists(current_audio_file):
        current_audio_file = audio_file_pth
        print(f"Warning: Using fallback audio file as {current_audio_file} not found")
    else:
        # Using current character audio without printing to CLI
        pass
        
    if TTS_PROVIDER == 'openai':
        output_path = os.path.join(output_dir, 'output.wav')
        try:
            await openai_text_to_speech(prompt, output_path)
            # print(f"Generated audio file at: {output_path}")
            if os.path.exists(output_path):
                print("Playing generated audio...")
                await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
                await play_audio(output_path)
                await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
            else:
                print("Error: Audio file not found.")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": "Failed to generate audio with OpenAI TTS"
                }))
        except Exception as e:
            print(f"Error in OpenAI TTS: {str(e)}")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": f"OpenAI TTS error: {str(e)}"
            }))
    elif TTS_PROVIDER == 'elevenlabs':
        output_path = os.path.join(output_dir, 'output.mp3')
        success = await elevenlabs_text_to_speech(prompt, output_path)
        # Only attempt to play if TTS was successful
        if success and os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        elif not success:
            print("Failed to generate ElevenLabs audio.")
            # Error notifications are now handled in elevenlabs_text_to_speech
        else:
            print("Error: ElevenLabs audio file not found after generation.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "ElevenLabs audio file not found after generation"
            }))
    elif TTS_PROVIDER == 'kokoro':
        output_path = os.path.join(output_dir, 'output.wav')
        success = await kokoro_text_to_speech(prompt, output_path)
        if success and os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        elif not success:
            print("Failed to generate Kokoro audio.")
        else:
            print("Error: Kokoro audio file not found after generation.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "Kokoro audio file not found after generation"
            }))
    elif TTS_PROVIDER == 'xai':
        output_path = os.path.join(output_dir, f"output.{xai_tts_file_extension()}")
        success = await xai_text_to_speech(prompt, output_path)
        if success and os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        elif not success:
            print("Failed to generate xAI audio.")
        else:
            print("Error: xAI audio file not found after generation.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "xAI audio file not found after generation"
            }))
    elif TTS_PROVIDER == 'sparktts':
        if sparktts_model is not None:
            try:
                # Spark-TTS inference
                wav_np = await asyncio.to_thread(
                    sparktts_model.inference,
                    text=prompt,
                    prompt_speech_path=Path(current_audio_file),
                    prompt_text=None,  # Will auto-transcribe if needed
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                src_path = os.path.join(output_dir, 'output.wav')
                # Spark-TTS already returns numpy array
                sf.write(src_path, wav_np, sparktts_model.sample_rate)
                print("Audio generated successfully with Spark-TTS.")
                await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
                await play_audio(src_path)
                await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
            except Exception as e:
                print(f"Error during Spark-TTS audio generation: {e}")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": f"Spark-TTS error: {str(e)}"
                }))
        else:
            print("Spark-TTS model is not loaded. Please ensure initialization succeeded.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "Spark-TTS model is not loaded"
            }))
    elif TTS_PROVIDER == 'typecast':
        output_path = os.path.join(output_dir, 'output.wav')
        success = await typecast_text_to_speech(prompt, output_path)
        if success and os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        elif not success:
            print("Failed to generate Typecast audio.")
        else:
            print("Error: Typecast audio file not found after generation.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "Typecast audio file not found after generation"
            }))
    else:
        print(f"Unknown TTS provider: {TTS_PROVIDER}")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": f"Unknown TTS provider: {TTS_PROVIDER}"
        }))


async def send_message_to_clients(message):
    """Send a message to all connected clients
    
    Args:
        message: Either a string or a dictionary to send to clients
    """
    # Convert dictionary to JSON string if needed
    if isinstance(message, dict):
        message_str = json.dumps(message)
    else:
        message_str = message
        
    for client in clients:
        try:
            await client.send_text(message_str)
        except Exception as e:
            print(f"Error sending message to client: {e}")

def save_pcm_as_wav(pcm_data: bytes, file_path: str, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2):
    wav_file = wave.open(file_path, 'wb')
    try:
        if not isinstance(wav_file, wave.Wave_write):
            raise TypeError(f"Expected Wave_write for mode 'wb', got {type(wav_file).__name__}")
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    finally:
        wav_file.close()

async def openai_text_to_speech(prompt, output_path):
    file_extension = Path(output_path).suffix.lstrip('.').lower()

    voice_speed = float(os.getenv("VOICE_SPEED", "1.0"))

    async with aiohttp.ClientSession() as session:
        if file_extension == 'wav':
            pcm_data = await fetch_pcm_audio(OPENAI_MODEL_TTS, OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL, session)
            save_pcm_as_wav(pcm_data, output_path)
        else:
            try:
                async with session.post(
                    url=OPENAI_TTS_URL,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={"model": OPENAI_MODEL_TTS, "voice": OPENAI_TTS_VOICE, "input": prompt, "response_format": file_extension, "speed": voice_speed},
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

                print("Audio generated successfully with OpenAI.")
            except aiohttp.ClientError as e:
                print(f"Error during OpenAI TTS: {e}")

async def fetch_pcm_audio(model: str, voice: str, input_text: str, api_url: str, session: aiohttp.ClientSession) -> bytes:
    pcm_data = io.BytesIO()
    
    try:
        async with session.post(
            url=api_url,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "voice": voice, "input": input_text, "response_format": 'pcm'},
            timeout=30
        ) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(8192):
                pcm_data.write(chunk)
    except aiohttp.ClientError as e:
        print(f"An error occurred while trying to fetch the audio stream: {e}")
        raise

    return pcm_data.getvalue()

def xai_tts_file_extension():
    codec = XAI_TTS_FORMAT.lower()
    return "wav" if codec == "wav" else "mp3"

def build_xai_tts_payload(text):
    payload = {
        "text": text,
        "voice_id": XAI_TTS_VOICE,
        "language": XAI_TTS_LANGUAGE,
    }

    codec = XAI_TTS_FORMAT.lower()
    if codec:
        output_format = {"codec": codec}
        if XAI_TTS_SAMPLE_RATE:
            output_format["sample_rate"] = int(XAI_TTS_SAMPLE_RATE)
        if XAI_TTS_BIT_RATE and codec == "mp3":
            output_format["bit_rate"] = int(XAI_TTS_BIT_RATE)
        payload["output_format"] = output_format

    return payload

async def xai_text_to_speech(text, output_path):
    if not XAI_API_KEY:
        print("XAI_API_KEY is not set. Cannot use xAI TTS.")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": "XAI_API_KEY is not set"
        }))
        return False

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        timeout = aiohttp.ClientTimeout(total=XAI_TTS_TIMEOUT)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                XAI_TTS_URL,
                headers=headers,
                json=build_xai_tts_payload(text),
                timeout=timeout,
            ) as response:
                if response.status == 200:
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    print("Audio generated successfully with xAI TTS.")
                    return True

                error_text = await response.text()
                print(f"Error generating xAI speech (HTTP {response.status}): {error_text}")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": f"xAI TTS error: {response.status}"
                }))
                return False
    except asyncio.TimeoutError:
        print("xAI TTS request timed out. Try a shorter text or check your connection.")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": "xAI TTS request timed out. Text may be too long."
        }))
        return False
    except Exception as e:
        print(f"Error during xAI TTS generation: {str(e)}")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": f"xAI TTS error: {str(e)}"
        }))
        return False

async def elevenlabs_text_to_speech(text, output_path):
    CHUNK_SIZE = 1024
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_TTS_VOICE}/stream"

    # Get global voice speed from environment (default to 1.0)
    voice_speed = os.getenv("VOICE_SPEED", "1.0")

    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": ELEVENLABS_TTS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": voice_speed
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            try:
                # Increase timeout for longer content
                timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout for larger audio files
                async with session.post(tts_url, headers=headers, json=data, timeout=timeout) as response:
                    if response.status == 200:
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                f.write(chunk)
                        print("Audio stream saved successfully.")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"Error generating speech (HTTP {response.status}): {error_text}")
                        # Notify clients about the error
                        await send_message_to_clients(json.dumps({
                            "action": "error",
                            "message": f"ElevenLabs TTS error: {response.status}"
                        }))
                        return False
            except asyncio.TimeoutError:
                print("ElevenLabs TTS request timed out. Try a shorter text or check your connection.")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": "ElevenLabs TTS request timed out. Text may be too long."
                }))
                return False
            except Exception as e:
                print(f"Error during ElevenLabs TTS API call: {str(e)}")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": f"ElevenLabs TTS error: {str(e)}"
                }))
                return False
    except Exception as e:
        print(f"Critical error in ElevenLabs TTS: {str(e)}")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": "Failed to connect to ElevenLabs TTS service"
        }))
        return False

async def typecast_text_to_speech(text, output_path):
    try:
        from typecast import Typecast
        from typecast.models import TTSRequest, TTSModel, Output, PresetPrompt

        client = Typecast(api_key=TYPECAST_API_KEY)

        request = TTSRequest(
            text=text,
            voice_id=TYPECAST_TTS_VOICE,
            model=TTSModel(TYPECAST_TTS_MODEL),
            prompt=PresetPrompt(emotion_preset=TYPECAST_EMOTION_PRESET),
            output=Output(audio_format="wav"),
        )

        response = await asyncio.to_thread(client.text_to_speech, request)

        with open(output_path, 'wb') as f:
            f.write(response.audio_data)

        print("Audio generated successfully with Typecast.")
        return True
    except ImportError:
        print("typecast-python package is not installed. Run: pip install typecast-python")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": "typecast-python package is not installed"
        }))
        return False
    except Exception as e:
        print(f"Error in Typecast TTS: {str(e)}")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": f"Typecast TTS error: {str(e)}"
        }))
        return False

def sanitize_response(response):
    # Remove model planning blocks before sending text to TTS.
    response = remove_response_meta_blocks(response)
    response = normalize_response_punctuation(response, preserve_xai_tags=TTS_PROVIDER == 'xai')
    # Remove asterisks and other formatting
    response = re.sub(r'\*.*?\*', '', response)
    if TTS_PROVIDER == 'xai':
        response = re.sub(r'[^\w\s,.;:\'!?\[\]<>\/%()&-]', '', response)
    else:
        response = re.sub(r'[^\w\s,.\'!?]', '', response)
    # Trim any whitespace
    return response.strip()

MOJIBAKE_PATTERN = re.compile(r'[\u00c3\u00c2\u00e2][\u0080-\u00ff]?|[\u0080-\u009f]')
UNICODE_DASH_PATTERN = re.compile(r'\s*[\u2012\u2013\u2014\u2015]\s*')
RESPONSE_META_BLOCK_PATTERN = re.compile(
    r'<(?P<tag>think|policy|analysis|reasoning|scratchpad|internal|planning|plan|voicefilter)\b[^>]*>[\s\S]*?</(?P=tag)\s*>',
    re.IGNORECASE
)
XAI_ALLOWED_WRAPPING_TAGS = {
    'soft', 'whisper', 'loud', 'build-intensity', 'decrease-intensity',
    'higher-pitch', 'lower-pitch', 'slow', 'fast', 'sing-song', 'singing',
    'laugh-speak', 'emphasis', 'pause', 'long-pause'
}
XML_TAG_PATTERN = re.compile(r'</?([A-Za-z][\w-]*)(?:\s[^>]*)?>')

def repair_mojibake_text(text):
    if not text or not MOJIBAKE_PATTERN.search(text):
        return text

    try:
        repaired = text.encode('latin-1').decode('utf-8')
    except UnicodeError:
        return text

    return repaired if repaired else text

def remove_response_meta_blocks(text):
    if not text:
        return text
    return RESPONSE_META_BLOCK_PATTERN.sub('', text).strip()

def remove_unsupported_xml_tags(text, preserve_xai_tags=False):
    if not text:
        return text

    def replace_tag(match):
        tag_name = match.group(1).lower()
        if preserve_xai_tags and tag_name in XAI_ALLOWED_WRAPPING_TAGS:
            return f"</{tag_name}>" if match.group(0).startswith("</") else f"<{tag_name}>"
        return ""

    return XML_TAG_PATTERN.sub(replace_tag, text)

def normalize_response_punctuation(text, preserve_xai_tags=False):
    text = remove_response_meta_blocks(text)
    text = remove_unsupported_xml_tags(text, preserve_xai_tags=preserve_xai_tags)
    text = repair_mojibake_text(text)
    text = UNICODE_DASH_PATTERN.sub(', ', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r',\s*,+', ', ', text)
    return text.strip()

STORY_DECISION_PROMPT_PATTERN = re.compile(
    r'\s+((?:What(?:[’\']ll| will)|What do|Where do|How do|Which|Choose|Decide|Act|Speak|Continue)\b[^?\n]{0,140}\?)\s*$',
    re.IGNORECASE
)

def format_story_response_text(text):
    text = normalize_response_punctuation(text)
    text = re.sub(r'(?<=[.!?])\s+(?=\S)', '  ', text)
    text = re.sub(r'\s*\*\*STATUS\s*-\s*([^*]+)\*\*\s*', r'\n\nSTATUS - \1\n', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=[^\s])\s+STATUS\s*-\s*', r'\n\nSTATUS - ', text, flags=re.IGNORECASE)
    text = STORY_DECISION_PROMPT_PATTERN.sub(r'\n\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# Per-character TTS filter.
# Each character folder may contain a tts_filter.json of the form:
#   {
#     "strip_line_patterns": ["^SCENE CLOCK\\b", "^ARRIVAL\\b"],
#     "strip_patterns": ["\\[META:.*?\\]"],
#     "replace": [{"pattern": "\\bHVAC\\b", "with": "H V A C"}]
#   }
# strip_line_patterns: any LINE that matches one of these regexes is dropped
# from the TTS-bound text (but NOT from conversation history or UI display).
# strip_patterns: substrings (within or across lines) matching these regexes
# are removed from the TTS-bound text.
# replace: pronunciation normalization pairs applied to the TTS-bound text.
# The filter is a no-op if the file is missing or malformed.
_TTS_FILTER_CACHE = {}

def load_tts_filter(character_name):
    if not character_name:
        return None
    filter_path = os.path.join(project_dir, 'characters', character_name, 'tts_filter.json')
    if not os.path.exists(filter_path):
        return None
    try:
        mtime = os.path.getmtime(filter_path)
    except OSError:
        return None
    cached = _TTS_FILTER_CACHE.get(filter_path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(filter_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load TTS filter for {character_name}: {e}")
        _TTS_FILTER_CACHE[filter_path] = (mtime, None)
        return None
    try:
        compiled = {
            'strip_line_patterns': [
                re.compile(p, re.IGNORECASE)
                for p in raw.get('strip_line_patterns', [])
                if isinstance(p, str) and p
            ],
            'strip_patterns': [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in raw.get('strip_patterns', [])
                if isinstance(p, str) and p
            ],
            'replace': [
                (re.compile(r['pattern'], re.IGNORECASE), r.get('with', ''))
                for r in raw.get('replace', [])
                if isinstance(r, dict) and isinstance(r.get('pattern'), str) and r.get('pattern')
            ],
        }
    except re.error as e:
        print(f"Warning: invalid regex in tts_filter.json for {character_name}: {e}")
        _TTS_FILTER_CACHE[filter_path] = (mtime, None)
        return None
    _TTS_FILTER_CACHE[filter_path] = (mtime, compiled)
    return compiled

def apply_tts_filter(text, character_name):
    if not text or not character_name:
        return text
    flt = load_tts_filter(character_name)
    if not flt:
        return text
    out = text
    if flt['strip_line_patterns']:
        kept_lines = []
        for line in out.splitlines():
            if any(p.search(line) for p in flt['strip_line_patterns']):
                continue
            kept_lines.append(line)
        out = "\n".join(kept_lines)
    for p in flt['strip_patterns']:
        out = p.sub('', out)
    for p, repl in flt['replace']:
        out = p.sub(repl, out)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return out

def is_xai_tts_enabled():
    return TTS_PROVIDER == 'xai'

XAI_INLINE_SPEECH_TAG_PATTERN = re.compile(
    r'\[(?:pause|long-pause|hum-tune|laugh|chuckle|giggle|cry|tsk|tongue-click|lip-smack|breath|inhale|exhale|sigh)\]',
    re.IGNORECASE
)
XAI_WRAPPING_SPEECH_TAG_PATTERN = re.compile(
    r'</?(?:soft|whisper|loud|build-intensity|decrease-intensity|higher-pitch|lower-pitch|slow|fast|sing-song|singing|laugh-speak|emphasis|pause|long-pause)>',
    re.IGNORECASE
)

def strip_xai_speech_tags(text):
    text = remove_response_meta_blocks(text)
    if not is_xai_tts_enabled():
        return remove_unsupported_xml_tags(text).strip()

    text = XAI_INLINE_SPEECH_TAG_PATTERN.sub('', text)
    text = XAI_WRAPPING_SPEECH_TAG_PATTERN.sub('', text)
    text = remove_unsupported_xml_tags(text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def xai_speech_tag_prompt():
    if not is_xai_tts_enabled():
        return ""

    return """
When generating the assistant's response, include 1 to 3 xAI Grok TTS speech tags for natural delivery.
Use tags where they improve emotion, pacing, or character performance, especially in long story/game narration.
Inline tags go at the moment the sound should happen, for example: "That was unexpected. [laugh] I did not see that coming."
Wrapping tags must wrap complete phrases, for example: "<whisper>This part is a secret.</whisper> Then continue normally."
Useful inline tags: [pause], [long-pause], [laugh], [chuckle], [giggle], [sigh], [breath], [inhale], [exhale].
Useful wrapping/control tags: <whisper>, <soft>, <loud>, <slow>, <fast>, <higher-pitch>, <lower-pitch>, <emphasis>, <laugh-speak>, <pause>.
Good story/game usage: "<slow>The lamp gutters once.</slow> [pause] Then the sea goes flat."
Never include internal planning or policy blocks such as <policy>, <analysis>, <think>, <reasoning>, or <scratchpad>.
Do not invent other XML/control tags such as <xai-grok-voice> or <voicefilter>.
Do not stack many tags together. Do not put tags in status blocks or decision prompts. Do not explain the tags. Do not use tags unless the active TTS provider is xAI.
""".strip()

def analyze_mood(user_input):
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity
    print(f"Sentiment polarity: {polarity}")

    flirty_keywords = [
        "flirt", "love", "crush", "charming", "amazing", "attractive", "sexy",
        "cute", "sweet", "darling", "adorable", "alluring", "seductive", "beautiful",
        "handsome", "gorgeous", "hot", "pretty", "romantic", "sensual", "passionate",
        "enchanting", "irresistible", "dreamy", "lovely", "captivating", "enticing",
        "sex", "makeout", "kiss", "hug", "cuddle", "snuggle", "romance", "date",
        "relationship", "flirtatious", "admire", "desire",
        "affectionate", "tender", "intimate", "fond", "smitten", "infatuated",
        "enamored", "yearning", "longing", "attracted", "tempting", "teasing",
        "playful", "coy", "wink", "flatter", "compliment", "woo", "court",
        "seduce", "charm", "beguile", "enthrall", "fascinate", "mesmerize",
        "allure", "tantalize", "tease", "caress", "embrace", "nuzzle", "smooch",
        "adore", "cherish", "treasure", "fancy", "chemistry", "spark", "connection",
        "attraction", "magnetism", "charisma", "appeal", "desirable", "delicious",
        "delightful", "divine", "heavenly", "angelic", "bewitching", "spellbinding",
        "hypnotic", "magical", "enchanted", "soulmate", "sweetheart", "honey",
        "dear", "beloved", "precious", "sugar", "babe", "baby",
        "sweetie", "cutie", "stunning", "ravishing"
    ]
    angry_keywords = [
        "angry", "furious", "mad", "annoyed", "pissed off", "irate", "rage",
        "enraged", "livid", "outraged", "frustrated", "infuriated", "hostile",
        "bitter", "seething", "fuming", "irritated", "agitated", "resentful",
        "indignant", "exasperated", "heated", "antagonized", "provoked", "wrathful",
        "fuckyou", "pissed", "fuckoff", "fuck", "die", "kill", "murder",
        "violent", "hateful", "hate", "despise", "loathe", "detest", "abhor",
        "incensed", "inflamed", "raging", "storming", "explosive", "fierce",
        "vicious", "vindictive", "spiteful", "venomous", "cruel", "savage",
        "ferocious", "threatening", "menacing", "intimidating", "aggressive",
        "combative", "confrontational", "argumentative", "belligerent",
        "antagonistic", "contentious", "quarrelsome", "rebellious", "defiant",
        "obstinate", "stubborn", "uncooperative", "difficult", "impossible",
        "unreasonable", "irrational", "foolish", "stupid", "idiotic", "moronic",
        "dumb", "ignorant", "incompetent", "useless", "worthless", "pathetic"
    ]
    sad_keywords = [
        "sad", "depressed", "down", "unhappy", "crying", "miserable", "grief",
        "heartbroken", "sorrowful", "gloomy", "melancholy", "despondent", "blue",
        "dejected", "hopeless", "desolate", "devastated", "lonely", "anguished",
        "woeful", "forlorn", "tearful", "mourning", "hurt", "pained", "suffering",
        "despair", "distressed", "troubled", "broken", "crushed", "defeated",
        "discouraged", "disheartened", "dispirited", "downcast", "downtrodden",
        "heavy-hearted", "inconsolable", "low", "mournful", "pessimistic",
        "somber", "upset", "weeping", "wretched", "grieving", "lamenting",
        "depressing", "dismal", "dreary", "glum", "joyless", "lost", "tragic",
        "wounded", "yearning", "abandoned", "afflicted", "alone", "bereft",
        "crestfallen", "dark", "destroyed", "empty", "hurting", "isolated"
    ]
    fearful_keywords = [
        "scared", "afraid", "fear", "terrified", "nervous", "anxious", "dread",
        "worried", "frightened", "alarmed", "panicked", "horrified", "petrified",
        "paranoid", "apprehensive", "uneasy", "spooked", "timid",
        "phobic", "jittery", "trembling", "shaken", "intimidated",
        "terror", "panic", "fright", "horror", "dreadful", "scary", "creepy",
        "haunted", "traumatized", "unsettled", "unnerved", "aghast",
        "startled", "jumpy", "skittish", "wary", "suspicious", "insecure", "unsafe",
        "vulnerable", "helpless", "defenseless", "exposed", "trapped", "cornered",
        "paralyzed", "frozen", "quaking", "quivering", "shivering", "shuddering",
        "terrifying", "menacing", "ominous", "sinister", "foreboding", "eerie",
        "spine-chilling", "blood-curdling", "hair-raising", "nightmarish",
        "monstrous", "ghastly", "freaked out", "creeped out", "scared stiff",
        "scared silly", "scared witless", "scared to death", "fear-stricken",
        "panic-stricken", "terror-stricken", "horror-struck", "shell-shocked"
    ]
    surprised_keywords = [
        "surprised", "amazed", "astonished", "shocked", "stunned", "wow",
        "flabbergasted", "astounded", "speechless", "dumbfounded",
        "bewildered", "awestruck", "thunderstruck", "taken aback", "floored",
        "mindblown", "unexpected", "unbelievable", "incredible", "remarkable",
        "extraordinary", "staggering", "overwhelming", "breathtaking",
        "gobsmacked", "dazed", "stupefied", "staggered", "agape", "wonderstruck",
        "spellbound", "transfixed", "mystified", "perplexed",
        "baffled", "confounded", "stumped", "puzzled", "disoriented",
        "disbelieving", "incredulous", "amazement", "astonishment",
        "wonder", "marvel", "miracle", "revelation", "bombshell", "bolt from the blue",
        "eye-opening", "jaw-dropping", "mind-boggling", "out of the blue",
        "shocker", "unpredictable", "unforeseen",
        "unanticipated", "inconceivable", "unimaginable", "unthinkable",
        "beyond belief", "hard to believe", "who would have thought",
        "never saw that coming", "caught off guard", "blindsided"
    ]
    disgusted_keywords = [
        "disgusted", "revolted", "sick", "nauseated", "repulsed", "yuck",
        "grossed out", "appalled", "offended", "detested", "repugnant", "vile",
        "loathsome", "repellent", "abhorrent", "hideous", "nasty", "foul",
        "distasteful", "sickening", "unpleasant", "gross",
        "repulsive", "stomach-turning", "queasy", "nauseous", "disgusting",
        "putrid", "rancid", "fetid", "rank", "rotten", "decaying", "spoiled",
        "contaminated", "tainted", "filthy", "dirty", "unsanitary", "unwholesome",
        "objectionable", "repellant", "revolting", "sordid", "vulgar",
        "crude", "obscene", "disagreeable", "unpalatable", "unsavory",
        "squalid", "mucky", "grotesque", "grungy",
        "icky", "nauseating", "odious", "obnoxious", "repelling", "sickly",
        "stomach-churning", "unappealing", "unappetizing", "unbearable", "vomit-inducing",
        "yucky", "ugh", "eww", "blegh", "blech", "ew"
    ]
    happy_keywords = [
        "happy", "pleased", "content", "satisfied", "great",
        "positive", "upbeat", "bright", "cheery", "merry", "lighthearted",
        "gratified", "blessed", "fortunate", "lucky", "peaceful", "serene", 
        "comfortable", "at ease", "fulfilled", "optimistic", "hopeful", "sunny",
        "cheerful", "pleasant", "contented", "glad", "jolly",
        "carefree", "untroubled", "tranquil", "relaxed", "calm",
        "heartwarming", "uplifting", "encouraging",
        "promising", "favorable", "agreeable", "enjoyable", "satisfying",
        "rewarding", "worthwhile", "meaningful", "enriching", "beneficial"
    ]
    joyful_keywords = [
        "joyful", "elated", "overjoyed", "ecstatic", "jubilant", "blissful",
        "delighted", "radiant", "exuberant", "enthusiastic", "euphoric", "thrilled",
        "gleeful", "giddy", "bouncing", "celebrating", "dancing", "singing",
        "laughing", "beaming", "glowing", "soaring", "floating", "exhilarated",
        "on cloud nine", "in seventh heaven", "over the moon", "walking on air",
        "jumping for joy", "bursting with happiness", "on top of the world",
        "tickled pink", "beside oneself", "in high spirits", "full of beans",
        "bubbling over", "in raptures", "in paradise", "in heaven", "delirious",
        "intoxicated", "flying high", "riding high", "whooping it up", "rejoicing",
        "reveling", "jubilating", "triumphant", "victorious", "festive"
    ]
    neutral_keywords = [
        "okay", "alright", "fine", "neutral", "so-so", "indifferent",
        "meh", "unremarkable", "average", "mediocre", "moderate", "standard",
        "typical", "ordinary", "regular", "common", "plain", "fair", "tolerable",
        "acceptable", "passable", "adequate", "middle-ground", "balanced"
    ]

    mood = "neutral"  # Default value

    if any(keyword in user_input.lower() for keyword in flirty_keywords):
        mood = "flirty"
    elif any(keyword in user_input.lower() for keyword in angry_keywords) or polarity < -0.7:
        mood = "angry"
    elif any(keyword in user_input.lower() for keyword in sad_keywords) or polarity < -0.3:
        mood = "sad"
    elif any(keyword in user_input.lower() for keyword in fearful_keywords):
        mood = "fearful"
    elif any(keyword in user_input.lower() for keyword in surprised_keywords):
        mood = "surprised"
    elif any(keyword in user_input.lower() for keyword in disgusted_keywords):
        mood = "disgusted"
    elif any(keyword in user_input.lower() for keyword in happy_keywords) or polarity > 0.7:
        mood = "happy"
    elif any(keyword in user_input.lower() for keyword in joyful_keywords) or polarity > 0.4:
        mood = "joyful"
    elif any(keyword in user_input.lower() for keyword in neutral_keywords) or (-0.3 <= polarity <= 0.4):
        mood = "neutral"
    
    # Color mapping for different moods
    mood_colors = {
        "flirty": "\033[95m",    # Purple
        "angry": "\033[91m",     # Red
        "sad": "\033[94m",       # Blue
        "fearful": "\033[93m",   # Yellow
        "surprised": "\033[96m", # Cyan
        "disgusted": "\033[90m", # Dark Gray
        "happy": "\033[92m",     # Green
        "joyful": "\033[38;5;208m", # Orange
        "neutral": "\033[92m"    # Green (default)
    }
    
    # Get the appropriate color for the detected mood
    color = mood_colors.get(mood, "\033[92m")
    
    # Print the detected mood with the corresponding color
    print(f"{color}Detected mood: {mood}\033[0m")
    print()  # Add an empty line for spacing in CLI output
        
    return mood

def chatgpt_streamed(user_input, system_message, mood_prompt, conversation_history):
    full_response = ""
    print(f"Debug: streamed started. MODEL_PROVIDER: {MODEL_PROVIDER}")

    # Calculate token limit based on character limit Approximate token conversion, So if MAX_CHAR_LENGTH is 500, then 500 * 4 // 3 = 666 tokens
    token_limit = min(4000, MAX_CHAR_LENGTH * 4 // 3)

    if MODEL_PROVIDER == 'ollama':
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}],
            "stream": True,
            "options": {"num_predict": -2, "temperature": 1.0}
        }
        try:
            print(f"Debug: Sending request to Ollama: {OLLAMA_BASE_URL}/v1/chat/completions")
            response = requests.post(f'{OLLAMA_BASE_URL}/v1/chat/completions', headers=headers, json=payload, stream=True, timeout=45)
            response.raise_for_status()

            line_buffer = ""
            for raw_line in response.iter_lines(decode_unicode=False):
                line = raw_line.decode('utf-8', errors='replace').strip()
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

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to Ollama model: {e}"
            print(f"Debug: Ollama error - {e}")
    
    elif MODEL_PROVIDER == 'xai':
        messages = [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}]
        headers = {
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": XAI_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": token_limit  # Approximate token conversion
        }
        try:
            print(f"Debug: Sending request to XAI: {XAI_BASE_URL}")
            response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()

            print("Starting XAI stream...")
            line_buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line:
                    try:
                        chunk = json.loads(line)
                        delta_content = chunk['choices'][0]['delta'].get('content', '')
                        # print(f"Raw delta_content: {repr(delta_content)}")
                        if delta_content:
                            delta_content = repair_mojibake_text(delta_content)
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
            full_response = repair_mojibake_text(full_response)
            print("\nXAI stream complete.")

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to XAI model: {e}"
            print(f"Debug: XAI error - {e}")

    elif MODEL_PROVIDER == 'openai':
        messages = [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}]
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
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
            
    elif MODEL_PROVIDER == 'anthropic':
        if anthropic is None:
            full_response = "Error: Anthropic library not installed. Please install using: pip install anthropic"
            print(f"Debug: {full_response}")
            return full_response
            
        # Convert the conversation history to Anthropic format
        anthropic_messages = []
        for msg in conversation_history:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        try:
            # Create the client with default settings
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            print(f"Debug: Sending request to Anthropic with model {ANTHROPIC_MODEL}")
            
            # Format system and mood prompt for Anthropic
            system_content = system_message + "\n" + mood_prompt
            
            # Start the streaming message request
            with client.messages.stream(
            max_tokens=token_limit,  # Approximate token conversion
            model=ANTHROPIC_MODEL,
            system=system_content,
            messages=anthropic_messages + [{"role": "user", "content": user_input}],
            temperature=0.8
        ) as stream:
                print("Starting Anthropic stream...")
                line_buffer = ""
                
                # Process the stream events
                for event in stream:
                    if event.type == "content_block_delta":
                        delta_content = event.delta.text
                        if delta_content:
                            line_buffer += delta_content
                            if '\n' in line_buffer:
                                lines = line_buffer.split('\n')
                                for line in lines[:-1]:
                                    print(NEON_GREEN + line + RESET_COLOR)
                                    full_response += line + '\n'
                                line_buffer = lines[-1]
                
                # Print any remaining content in the buffer
                if line_buffer:
                    print(NEON_GREEN + line_buffer + RESET_COLOR)
                    full_response += line_buffer
                    
                print("\nAnthropic stream complete.")
        
        except Exception as e:
            full_response = f"Error connecting to Anthropic model: {e}"
            print(f"Debug: Anthropic error - {e}")

    print(f"streaming complete. Response length: {PINK}{len(full_response)}{RESET_COLOR}")
    return full_response

def save_conversation_history(conversation_history):
    """Save conversation history to a file."""
    try:
        # Import with alias to avoid potential shadowing issues
        from .shared import get_current_character as get_character
        
        current_character = get_character()
        
        # Check if this is a story or game character
        is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
        print(f"Saving history for {current_character} ({is_story_character=})")
        
        if is_story_character:
            # Save to character-specific history file
            character_dir = os.path.join(characters_folder, current_character)
            os.makedirs(character_dir, exist_ok=True)
            history_file = os.path.join(character_dir, "conversation_history.txt")
        else:
            # Save to global history file
            history_file = "conversation_history.txt"
        
        with open(history_file, "w", encoding="utf-8") as file:
            for message in conversation_history:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n\n")  # Add extra newline for readability
    except Exception as e:
        print(f"Error saving conversation history: {e}")
        return {"status": "error", "message": str(e)}
    return {"status": "success"}

def transcribe_with_whisper(audio_file):
    """Transcribe audio using local Faster Whisper model"""
    global whisper_model
    
    # Lazy load the model only when needed
    if whisper_model is None:
        # Check for CUDA availability (use global device or check torch if available)
        whisper_device = "cuda" if _cuda_available() else "cpu"
        
        # Default model size (adjust as needed)
        model_size = "medium.en" if whisper_device == "cuda" else "tiny.en"
        
        try:
            print(f"Lazy-loading Faster-Whisper on {whisper_device}...")
            whisper_model = WhisperModel(model_size, device=whisper_device, compute_type="float16" if whisper_device == "cuda" else "int8")
            print("Faster-Whisper initialized successfully.")
        except Exception as e:
            print(f"Error initializing Faster-Whisper on {whisper_device}: {e}")
            print("Falling back to CPU mode...")
            
            # Force CPU fallback
            model_size = "tiny.en"
            whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("Faster-Whisper initialized on CPU successfully.")
    
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

def detect_silence(data, threshold=1000, chunk_size=1024):   # threshold is More sensitive silence detection, lower to speed up
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.mean(np.abs(audio_data)) < threshold

async def record_audio(file_path, silence_threshold=512, silence_duration=2.5, chunk_size=1024):  # 2.0 seconds of silence adjust as needed, if not picking up your voice increase to 4.0
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
    frames = []
    print("Recording...")
    await send_message_to_clients(json.dumps({"action": "recording_started"}))  # Notify frontend
    silent_chunks = 0
    speaking_chunks = 0
    while True:
        data = stream.read(chunk_size)
        frames.append(data)
        if detect_silence(data, threshold=silence_threshold, chunk_size=chunk_size):
            silent_chunks += 1
            if silent_chunks > silence_duration * (16000 / chunk_size):
                break
        else:
            silent_chunks = 0
            speaking_chunks += 1
        if speaking_chunks > silence_duration * (16000 / chunk_size) * 10:
            break
    print("Recording stopped.")
    await send_message_to_clients(json.dumps({"action": "recording_stopped"}))  # Notify frontend
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

async def execute_once(question_prompt):
    temp_image_path = os.path.join(output_dir, 'temp_img.jpg')
    
    # Determine the audio file format based on the TTS provider this is for the image analysis only see app_logic.py for the user chatbot conversation
    if TTS_PROVIDER == 'elevenlabs':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.mp3')  # Use mp3 for ElevenLabs
        max_char_length = MAX_CHAR_LENGTH  # Set a higher limit for ElevenLabs
    elif TTS_PROVIDER == 'kokoro':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for Kokoro
        max_char_length = MAX_CHAR_LENGTH  # Set a higher limit for Kokoro
    elif TTS_PROVIDER == 'xai':
        temp_audio_path = os.path.join(output_dir, f"temp_audio.{xai_tts_file_extension()}")  # Use configured xAI format
        max_char_length = MAX_CHAR_LENGTH  # REST xAI supports up to 15,000 chars; keep app-level limit
    elif TTS_PROVIDER == 'openai':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for OpenAI
        max_char_length = MAX_CHAR_LENGTH  # Set a higher limit for OpenAI
    else:
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for Spark-TTS
        max_char_length = SPARKTTS_MAX_CHARS  # Spark-TTS character limit

    image_path = await take_screenshot(temp_image_path)
    response = await analyze_image(image_path, question_prompt)
    text_response = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response received.')

    # Truncate response based on the TTS provider's limit
    if len(text_response) > max_char_length:
        text_response = text_response[:max_char_length] + "..."

    print(text_response)

    await generate_speech(text_response, temp_audio_path)

    if TTS_PROVIDER == 'elevenlabs':
        # Convert MP3 to WAV if ElevenLabs is used
        temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
        audio = AudioSegment.from_mp3(temp_audio_path)
        audio.export(temp_wav_path, format="wav")
        await play_audio(temp_wav_path)
    else:
        await play_audio(temp_audio_path)

    os.remove(image_path)
    return text_response

async def execute_screenshot_and_analyze():
    # Import the necessary modules at the beginning of the function
    from .shared import get_current_character, conversation_history
    from .app_logic import save_conversation_history as save_conversation_history_app, save_character_specific_history as save_character_specific_history_app
    
    question_prompt = "What do you see in this image? Keep it short but detailed and answer any follow up questions about it"
    print("Taking screenshot and analyzing...")
    text_response = await execute_once(question_prompt)
    
    # Add the AI's response to the conversation history
    conversation_history.append({"role": "assistant", "content": text_response})
    
    # Save the updated conversation history
    current_character = get_current_character()
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    
    if is_story_character:
        save_character_specific_history_app(conversation_history, current_character)
    else:
        save_conversation_history_app(conversation_history)
    
    # Send the response to any connected websocket clients
    await send_message_to_clients(f"{current_character}: {text_response}")
    
    print("\nReady for the next question....")

async def take_screenshot(temp_image_path):
    await asyncio.sleep(5)
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize((1024, 1024))
    screenshot.save(temp_image_path, 'JPEG')
    return temp_image_path

# Encode Image
async def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Analyze Image
async def analyze_image(image_path, question_prompt):
    encoded_image = await encode_image(image_path)
    
    if MODEL_PROVIDER == 'ollama':
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": "llava",
            "prompt": question_prompt,
            "images": [encoded_image],
            "stream": False
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{OLLAMA_BASE_URL}/api/generate', headers=headers, json=payload, timeout=30) as response:
                    print(f"Response status code: {response.status}")
                    if response.status == 200:
                        print("Using ollama for image analysis")
                        response_json = await response.json()
                        return {"choices": [{"message": {"content": response_json.get('response', 'No response received.')}}]}
                    elif response.status == 404:
                        return {"choices": [{"message": {"content": "The llava model is not available on this server."}}]}
                    else:
                        response.raise_for_status()
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return {"choices": [{"message": {"content": "Failed to process the image with the llava model."}}]}
    
    elif MODEL_PROVIDER == 'xai':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {XAI_API_KEY}"
        }
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": question_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "high"}}
            ]
        }
        payload = {
            "model": "grok-2-vision-1212",
            "temperature": 0.5,
            "messages": [message],
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        print("Using xAI for image analysis")
                        return await response.json()
                    else:
                        # If XAI returns an error,
                        # fall back to OpenAI's image analysis
                        print("XAI image analysis failed or not supported, falling back to OpenAI")
                        return await fallback_to_openai_image_analysis(encoded_image, question_prompt)
        except aiohttp.ClientError as e:
            print(f"XAI image analysis failed: {e}, falling back to OpenAI")
            return await fallback_to_openai_image_analysis(encoded_image, question_prompt)
    
    else:  # OpenAI as default
        return await fallback_to_openai_image_analysis(encoded_image, question_prompt)

async def fallback_to_openai_image_analysis(encoded_image, question_prompt):
    """Helper function for OpenAI image analysis fallback"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": question_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "high"}}
        ]
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 1.0,
        "messages": [message],
        "max_tokens": 1000
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                print("Using OpenAI for image analysis")
                return await response.json()
    except aiohttp.ClientError as e:
        print(f"OpenAI fallback request failed: {e}")
        return {"choices": [{"message": {"content": "Failed to process the image with both XAI and OpenAI models."}}]}


async def generate_speech(text, temp_audio_path):
    if TTS_PROVIDER == 'openai':
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": OPENAI_MODEL_TTS, "voice": OPENAI_TTS_VOICE, "speed": float(VOICE_SPEED), "input": text, "response_format": "wav"}
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENAI_TTS_URL, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    with open(temp_audio_path, "wb") as audio_file:
                        audio_file.write(await response.read())
                else:
                    print(f"Failed to generate speech: {response.status} - {await response.text()}")

    elif TTS_PROVIDER == 'elevenlabs':
        await elevenlabs_text_to_speech(text, temp_audio_path)
    
    elif TTS_PROVIDER == 'kokoro':
        await kokoro_text_to_speech(text, temp_audio_path)

    elif TTS_PROVIDER == 'xai':
        await xai_text_to_speech(text, temp_audio_path)

    elif TTS_PROVIDER == 'typecast':
        await typecast_text_to_speech(text, temp_audio_path)

    else:  # Spark-TTS
        if sparktts_model is not None:
            try:
                wav_np = await asyncio.to_thread(
                    sparktts_model.inference,
                    text=text,
                    prompt_speech_path=Path(character_audio_file),
                    prompt_text=None,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                # Spark-TTS already returns numpy array
                sf.write(temp_audio_path, wav_np, sparktts_model.sample_rate)
                print("Audio generated successfully with Spark-TTS.")
            except Exception as e:
                print(f"Error during Spark-TTS audio generation: {e}")
        else:
            print("Spark-TTS model is not loaded.")

async def kokoro_text_to_speech(text, output_path):
    """Convert text to speech using Kokoro TTS API."""
    try:
        # Using direct aiohttp request
        kokoro_url = f"{KOKORO_BASE_URL}/audio/speech"
        
        # Get voice speed from environment
        voice_speed = float(os.getenv("VOICE_SPEED", "1.0"))
        
        # Prepare payload with the format expected by Kokoro API
        payload = {
            "model": "kokoro",
            "voice": KOKORO_TTS_VOICE,
            "input": text,
            "response_format": "wav",  # Use wav format for more compatibility
            "speed": voice_speed
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add Basic Auth if credentials are provided
        kokoro_username = os.getenv("KOKORO_USERNAME", "")
        kokoro_password = os.getenv("KOKORO_PASSWORD", "")
        
        if kokoro_username and kokoro_password:
            import base64
            auth_str = f"{kokoro_username}:{kokoro_password}"
            auth_bytes = auth_str.encode('ascii')
            base64_auth = base64.b64encode(auth_bytes).decode('ascii')
            headers["Authorization"] = f"Basic {base64_auth}"
        
        # Make the request with SSL verification disabled
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(kokoro_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    # Save the audio data to file
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024):
                            f.write(chunk)
                    
                    print("Audio generated successfully with Kokoro.")
                    return True
                else:
                    error_text = await response.text()
                    print(f"Error from Kokoro API: HTTP {response.status} - {error_text}")
                    await send_message_to_clients(json.dumps({
                        "action": "error",
                        "message": f"Kokoro TTS error: HTTP {response.status}"
                    }))
                    return False
                
    except Exception as e:
        print(f"Error during Kokoro TTS generation: {e}")
        await send_message_to_clients(json.dumps({
            "action": "error",
            "message": f"Kokoro TTS error: {str(e)}"
        }))
        return False

async def user_chatbot_conversation():
    from .app_logic import load_character_specific_history

    def local_adjust_prompt(mood):
        current_character_name = os.getenv("CHARACTER_NAME", "wizard")
        character_prompts_path = os.path.join(project_dir, "characters", current_character_name, "prompts.json")
        global_prompts_path = os.path.join(project_dir, "characters", "prompts.json")

        try:
            prompts_path = character_prompts_path if os.path.exists(character_prompts_path) else global_prompts_path
            with open(prompts_path, "r", encoding="utf-8") as f:
                mood_prompts = json.load(f)
            return mood_prompts.get(mood, mood_prompts.get("neutral", "KEEP RESPONSES SHORT AND NATURAL."))
        except Exception as e:
            print(f"Error loading mood prompt: {e}")
            return "KEEP RESPONSES SHORT AND NATURAL."

    # Track previous character
    previous_character = os.getenv("PREVIOUS_CHARACTER_NAME", "")
    
    # Get current character
    current_character = os.getenv("CHARACTER_NAME", "wizard")
    
    # Check if we're switching characters
    is_character_switch = previous_character != "" and previous_character != current_character
    if is_character_switch:
        print(f"Character switch detected: {previous_character} -> {current_character}")
        
        # Update environment variable for next run
        os.environ["PREVIOUS_CHARACTER_NAME"] = current_character
    
    # Check if this is a story/game character
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    print(f"Starting conversation with character: {current_character}")
    
    # Initialize conversation history based on character type
    if is_story_character:
        # Try to load history from character-specific file
        conversation_history = load_character_specific_history(current_character)
        if conversation_history:
            print(f"Loaded {len(conversation_history)} messages from character-specific history")
        else:
            print(f"No previous history found for {current_character}, starting fresh")
            conversation_history = []
    else:
        # Use global history for standard characters
        conversation_history = []
        # Try to load from global file
        try:
            history_file = "conversation_history.txt"
            if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                temp_history = []
                with open(history_file, "r", encoding="utf-8") as file:
                    current_role = None
                    current_content = ""
                    
                    for line in file:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        if line.startswith("User:"):
                            # Save previous message if exists
                            if current_role:
                                temp_history.append({"role": current_role, "content": current_content.strip()})
                            
                            # Start new user message
                            current_role = "user"
                            current_content = line[5:].strip()
                        elif line.startswith("Assistant:"):
                            # Save previous message if exists
                            if current_role:
                                temp_history.append({"role": current_role, "content": current_content.strip()})
                            
                            # Start new assistant message
                            current_role = "assistant"
                            current_content = line[10:].strip()
                        else:
                            # Continue previous message
                            current_content += "\n" + line
                    
                    # Add the last message
                    if current_role:
                        temp_history.append({"role": current_role, "content": current_content.strip()})
                
                conversation_history = temp_history
                print(f"Loaded {len(conversation_history)} messages from global history")
        except Exception as e:
            print(f"Error loading global history: {e}")
            conversation_history = []
    
    # Debug info about history state
    print(f"Starting conversation with character {current_character}, history size: {len(conversation_history)}")
    
    base_system_message_template = open_file(character_prompt_file)

    quit_phrases = ["quit", "Quit", "Quit.", "Exit.", "exit", "Exit"]
    screenshot_phrases = [
        "what's on my screen", 
        "take a screenshot", 
        "show me my screen", 
        "analyze my screen", 
        "what do you see on my screen", 
        "screen capture", 
        "screenshot"
    ]

    try:
        while True:
            audio_file = "temp_recording.wav"
            record_audio(audio_file)
            user_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)
            print(CYAN + "You:", user_input + RESET_COLOR)
            
            # Check for quit phrases with word boundary check
            words = user_input.lower().split()
            if any(phrase.lower().rstrip('.') == word for phrase in quit_phrases for word in words):
                print("Quitting the conversation...")
                break
                
            conversation_history.append({"role": "user", "content": user_input})
            
            if any(phrase in user_input.lower() for phrase in screenshot_phrases):
                await execute_screenshot_and_analyze()  # Note the 'await' here
                continue
            
            mood = analyze_mood(user_input)
            mood_prompt = local_adjust_prompt(mood)
            tag_prompt = xai_speech_tag_prompt()
            if tag_prompt:
                mood_prompt = f"{mood_prompt}\n\n{tag_prompt}"
            
            print(PINK + f"{character_display_name}:..." + RESET_COLOR)
            base_system_message = render_prompt_template(base_system_message_template)
            chatbot_response = chatgpt_streamed(user_input, base_system_message, mood_prompt, conversation_history)
            display_response = strip_xai_speech_tags(chatbot_response)
            is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
            if is_story_character:
                display_response = format_story_response_text(display_response)
            conversation_history.append({"role": "assistant", "content": display_response})
            # TTS path: apply per-character tts_filter.json BEFORE sanitize_response
            # so its regex patterns can still see colons, brackets, etc.
            tts_source = apply_tts_filter(chatbot_response, current_character)
            sanitized_response = sanitize_response(tts_source)
            if len(sanitized_response) > 400:
                sanitized_response = sanitized_response[:400] + "..."
            prompt2 = sanitized_response
            await process_and_play(prompt2, character_audio_file)  # Note the 'await' here
            if is_story_character:
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]
            else:
                if len(conversation_history) > 30:
                    conversation_history = conversation_history[-30:]

            # Save conversation history after each message exchange
            save_conversation_history(conversation_history)

    except KeyboardInterrupt:
        print("Quitting the conversation...")

if __name__ == "__main__":
    asyncio.run(user_chatbot_conversation())

