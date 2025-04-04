import os
import asyncio
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
from TTS.api import TTS
import soundfile as sf
from textblob import TextBlob
from pathlib import Path
import anthropic
import re
import io
import torch
from pydub import AudioSegment
import shutil
from .shared import clients, get_current_character
from .stream_helpers import stream_openai_response
import time


import logging
logging.getLogger("transformers").setLevel(logging.ERROR)  # transformers 4.48+ warning

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load environment variables
load_dotenv()

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
ELEVENLABS_TTS_SPEED = os.getenv('ELEVENLABS_TTS_SPEED', '1')
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))
XTTS_SPEED = os.getenv('XTTS_SPEED', '1.1') 
XTTS_NUM_CHARS = int(os.getenv('XTTS_NUM_CHARS', 255))
SILENCE_DURATION_SECONDS = float(os.getenv("SILENCE_DURATION_SECONDS", "2.0"))
os.environ["COQUI_TOS_AGREED"] = "1"

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize OpenAI API key if available
if OPENAI_API_KEY:
    OpenAI.api_key = OPENAI_API_KEY
else:
    print(f"{YELLOW}OPENAI_API_KEY not set in .env file. OpenAI services disabled.{RESET_COLOR}")
    # Set to None to ensure proper error handling when OpenAI services are attempted
    OpenAI.api_key = None

# Debug flag for audio levels
DEBUG_AUDIO_LEVELS = os.getenv("DEBUG_AUDIO_LEVELS", "false").lower() == "true"

# Capitalize the first letter of the character name
character_display_name = CHARACTER_NAME.capitalize()

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable CuDNN explicitly - enable this if you get cudnn errors or change in xtts-v2/config.json
# torch.backends.cudnn.enabled = False

# Check if Faster Whisper should be loaded at startup
FASTER_WHISPER_LOCAL = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

# Initialize whisper model as None to lazy load
whisper_model = None

# Default model size (adjust as needed)
model_size = "medium.en"

if FASTER_WHISPER_LOCAL:
    try:
        print(f"Attempting to load Faster-Whisper on {device}...")
        whisper_model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        print("Faster-Whisper initialized successfully.")
    except Exception as e:
        print(f"Error initializing Faster-Whisper on {device}: {e}")
        print("Falling back to CPU mode...")

        # Force CPU fallback
        device = "cpu"
        model_size = "tiny.en"  # Use a smaller model for CPU performance
        whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Faster-Whisper initialized on CPU successfully.")
else:
    print("Faster-Whisper initialization skipped (FASTER_WHISPER_LOCAL=false). Will use OpenAI API for transcription or load on demand.")

# Paths for character-specific files
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
characters_folder = os.path.join(project_dir, 'characters', CHARACTER_NAME)
character_prompt_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.txt")
character_audio_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.wav")

# Load XTTS configuration
tts = None

# Initialize TTS model with automatic downloading
if TTS_PROVIDER == 'xtts':
    print("Initializing XTTS model (may download on first run)...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("Model downloaded, loading into memory...")
        tts = tts.to(device)  # Move to device after download
        
        num_chars = XTTS_NUM_CHARS
        # Set the character limit
        tts.synthesizer.tts_model.args.num_chars = num_chars  # default is 255 we are overriding it 
        
        print("XTTS model loaded successfully.")
    except Exception as e:
        print(f"Failed to load XTTS model: {e}")
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

def init_xtts_speed(speed_value):
    global XTTS_SPEED
    XTTS_SPEED = speed_value
    print(f"Switched to XTTS speed: {speed_value}")

def init_set_tts(set_tts):
    global TTS_PROVIDER, tts
    if set_tts == 'xtts':
        print("Initializing XTTS model (may download on first run)...")
        try:
            os.environ["COQUI_TOS_AGREED"] = "1"  # Auto-agree to terms
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("Model downloaded, loading into memory...")
            tts = tts.to(device)
            num_chars = XTTS_NUM_CHARS
            tts.synthesizer.tts_model.args.num_chars = num_chars # default is 255 we are overriding it warning on cpu will take much longer
            print("XTTS model loaded successfully.")
            TTS_PROVIDER = set_tts
        except Exception as e:
            print(f"Failed to load XTTS model: {e}")
            loop = asyncio.get_running_loop()
            loop.create_task(send_message_to_clients(json.dumps({
                "action": "error",
                "message": "Failed to load XTTS model. Please check your internet connection or model availability."
            })))
    else:
        TTS_PROVIDER = set_tts
        tts = None
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

# Function to play audio using PyAudio
async def play_audio(file_path):
    await asyncio.to_thread(sync_play_audio, file_path)

def sync_play_audio(file_path):
    print("Starting audio playback")
    file_extension = Path(file_path).suffix.lstrip('.').lower()
    
    temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
    
    if file_extension == 'mp3':
        audio = AudioSegment.from_mp3(file_path)
        audio.export(temp_wav_path, format="wav")
        file_path = temp_wav_path
    
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Finished audio playback")

    pass

# Update output paths
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
static_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs')

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(static_output_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model provider: {MODEL_PROVIDER}")
print(f"Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else XAI_MODEL if MODEL_PROVIDER == 'xai' else ANTHROPIC_MODEL if MODEL_PROVIDER == 'anthropic' else OLLAMA_MODEL}")
print(f"Character: {character_display_name}")
print(f"Text-to-Speech provider: {TTS_PROVIDER}")
print("To stop chatting say Quit or Exit. One moment please loading...")

async def process_and_play(prompt, audio_file_pth, remote_playback=False):
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
        
    # Check if audio bridge is enabled
    audio_bridge_enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
    
    # Generate audio file based on TTS provider
    if TTS_PROVIDER == 'openai':
        output_path = os.path.join(output_dir, 'output.wav')
        try:
            await openai_text_to_speech(prompt, output_path)
            # print(f"Generated audio file at: {output_path}")
            if os.path.exists(output_path):
                print("Playing generated audio...")
                await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
                
                # If audio bridge is enabled, send audio to connected clients
                if audio_bridge_enabled:
                    try:
                        from .audio_bridge.audio_bridge_server import audio_bridge
                        if audio_bridge.is_enabled() and audio_bridge.clients_set:
                            print(f"Sending audio to {len(audio_bridge.clients_set)} audio bridge clients")
                            
                            if remote_playback:
                                # Send audio URL for client-side playback
                                audio_url = copy_to_static_output(output_path)
                                print(f"Sending audio URL for client-side playback: {audio_url}")
                                # Send to each connected client
                                for client_id in list(audio_bridge.clients_set):
                                    await audio_bridge.send_audio(client_id, audio_url, is_url=True)
                            else:
                                # Send binary audio data (traditional method)
                                # Read the audio file
                                with open(output_path, "rb") as f:
                                    audio_data = f.read()
                                # Send to each connected client
                                for client_id in list(audio_bridge.clients_set):
                                    await audio_bridge.send_audio(client_id, audio_data)
                    except Exception as e:
                        print(f"Error sending audio via bridge: {e}")
                
                # Play locally only if not remote playback
                if not remote_playback:
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
            
            # If audio bridge is enabled, send audio to connected clients
            if audio_bridge_enabled:
                try:
                    from .audio_bridge.audio_bridge_server import audio_bridge
                    if audio_bridge.is_enabled() and audio_bridge.clients_set:
                        print(f"Sending audio to {len(audio_bridge.clients_set)} audio bridge clients")
                        
                        if remote_playback:
                            # Send audio URL for client-side playback
                            audio_url = copy_to_static_output(output_path)
                            print(f"Sending audio URL for client-side playback: {audio_url}")
                            # Send to each connected client
                            for client_id in list(audio_bridge.clients_set):
                                await audio_bridge.send_audio(client_id, audio_url, is_url=True)
                        else:
                            # Send binary audio data (traditional method)
                            # Convert MP3 to WAV for consistency
                            temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
                            audio = AudioSegment.from_mp3(output_path)
                            audio.export(temp_wav_path, format="wav")
                            # Read the audio file
                            with open(temp_wav_path, "rb") as f:
                                audio_data = f.read()
                            # Send to each connected client
                            for client_id in list(audio_bridge.clients_set):
                                await audio_bridge.send_audio(client_id, audio_data)
                except Exception as e:
                    print(f"Error sending audio via bridge: {e}")
            
            # Play locally only if not remote playback
            if not remote_playback:
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
    elif TTS_PROVIDER == 'xtts':
        if tts is not None:
            try:
                wav = await asyncio.to_thread(
                    tts.tts,
                    text=prompt,
                    speaker_wav=current_audio_file,  # Use the updated current character audio
                    language="en",
                    speed=float(XTTS_SPEED)
                )
                src_path = os.path.join(output_dir, 'output.wav')
                sf.write(src_path, wav, tts.synthesizer.tts_config.audio["sample_rate"])
                print("Audio generated successfully with XTTS.")
                await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
                
                # If audio bridge is enabled, send audio to connected clients
                if audio_bridge_enabled:
                    try:
                        from .audio_bridge.audio_bridge_server import audio_bridge
                        if audio_bridge.is_enabled() and audio_bridge.clients_set:
                            print(f"Sending audio to {len(audio_bridge.clients_set)} audio bridge clients")
                            
                            if remote_playback:
                                # Send audio URL for client-side playback
                                audio_url = copy_to_static_output(src_path)
                                print(f"Sending audio URL for client-side playback: {audio_url}")
                                # Send to each connected client
                                for client_id in list(audio_bridge.clients_set):
                                    await audio_bridge.send_audio(client_id, audio_url, is_url=True)
                            else:
                                # Send binary audio data (traditional method)
                                # Read the audio file
                                with open(src_path, "rb") as f:
                                    audio_data = f.read()
                                # Send to each connected client
                                for client_id in list(audio_bridge.clients_set):
                                    await audio_bridge.send_audio(client_id, audio_data)
                    except Exception as e:
                        print(f"Error sending audio via bridge: {e}")
                
                # Play locally only if not remote playback
                if not remote_playback:
                    await play_audio(src_path)
                    
                await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
            except Exception as e:
                print(f"Error during XTTS audio generation: {e}")
                await send_message_to_clients(json.dumps({
                    "action": "error",
                    "message": f"XTTS error: {str(e)}"
                }))
        else:
            print("XTTS model is not loaded. Please ensure initialization succeeded.")
            await send_message_to_clients(json.dumps({
                "action": "error",
                "message": "XTTS model is not loaded"
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
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

async def openai_text_to_speech(prompt, output_path):
    file_extension = Path(output_path).suffix.lstrip('.').lower()

    async with aiohttp.ClientSession() as session:
        if file_extension == 'wav':
            pcm_data = await fetch_pcm_audio(OPENAI_MODEL_TTS, OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL, session)
            save_pcm_as_wav(pcm_data, output_path)
        else:
            try:
                async with session.post(
                    url=OPENAI_TTS_URL,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={"model": OPENAI_MODEL_TTS, "voice": OPENAI_TTS_VOICE, "input": prompt, "response_format": file_extension},
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

async def elevenlabs_text_to_speech(text, output_path):
    CHUNK_SIZE = 1024
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_TTS_VOICE}/stream"

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
            "speed": ELEVENLABS_TTS_SPEED
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

def sanitize_response(response):
    # Remove <think>...</think> blocks first
    response = re.sub(r'<think>[\s\S]*?<\/think>', '', response)
    # Remove asterisks and other formatting
    response = re.sub(r'\*.*?\*', '', response)
    response = re.sub(r'[^\w\s,.\'!?]', '', response)
    # Trim any whitespace
    return response.strip()

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
            response = requests.post(f'{OLLAMA_BASE_URL}/v1/chat/completions', headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()

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
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Default model size (adjust as needed)
        model_size = "medium.en" if device == "cuda" else "tiny.en"
        
        try:
            print(f"Lazy-loading Faster-Whisper on {device}...")
            whisper_model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
            print("Faster-Whisper initialized successfully.")
        except Exception as e:
            print(f"Error initializing Faster-Whisper on {device}: {e}")
            print("Falling back to CPU mode...")
            
            # Force CPU fallback
            device = "cpu"
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

async def record_audio(file_path, silence_threshold=25, silence_duration=SILENCE_DURATION_SECONDS, chunk_size=1024, no_fallback=False):
    """Record audio from microphone or WebRTC audio bridge"""
    
    # Check if audio bridge is enabled
    audio_bridge_enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
    
    # If audio bridge is enabled, use it
    if audio_bridge_enabled:
        try:
            from .audio_bridge.audio_bridge_server import audio_bridge
            from .audio_bridge.audio_processor import audio_processor
            import traceback
            
            print(f"Using WebRTC audio bridge for recording with silence_threshold: {silence_threshold}")
            
            # Get client IDs from clients_set
            client_ids = list(audio_bridge.clients_set)
            if not client_ids:
                print("No audio bridge clients connected")
                if no_fallback:
                    print("Fallback to local microphone disabled, returning empty audio")
                    # Create an empty WAV file
                    wf = wave.open(file_path, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    # Write a short silence (0.5 seconds)
                    wf.writeframes(b'\x00\x00' * 8000)
                    wf.close()
                    return
                print("Falling back to local microphone")
            else:
                try:
                    print(f"Found {len(client_ids)} audio bridge clients: {client_ids}")
                    
                    # Create a collection for client audio chunks
                    client_audio_chunks = []
                    
                    # Set a timeout (15 seconds - reduced from 20)
                    start_time = asyncio.get_event_loop().time()
                    timeout = 7.0  # seconds
                    
                    # Wait for audio data to arrive - show progress indicators
                    print("Waiting for WebRTC audio data...")
                    
                    # Add a keyboard detection for cancellation
                    print("Press Ctrl+C to cancel waiting and use local microphone")
                    
                    # Initialize tracking variables 
                    detected_streaming = False
                    last_display_time = 0
                    audio_received = False
                    
                    while (asyncio.get_event_loop().time() - start_time) < timeout:
                        try:
                            # Check if any clients are streaming
                            streaming_clients = [cid for cid in client_ids 
                                                if audio_bridge.is_client_streaming.get(cid, False)]
                            
                            # If we detect streaming clients for the first time, log it
                            if streaming_clients and not detected_streaming:
                                detected_streaming = True
                                print(f"Detected {len(streaming_clients)} streaming clients: {streaming_clients}")
                            
                            # If clients are streaming, try to get audio
                            collected_in_iteration = False
                            
                            if streaming_clients:
                                # Check all streaming clients even if we get data from one
                                for client_id in streaming_clients:
                                    # Explicitly check for available audio chunks
                                    pcm_chunks = len(audio_bridge.audio_pcm.get(client_id, []))
                                    if pcm_chunks > 0:
                                        # We have PCM data, get it now
                                        try:
                                            audio_chunk = await audio_bridge.receive_audio(client_id)
                                            if audio_chunk and len(audio_chunk) > 0:
                                                client_audio_chunks.append(audio_chunk)
                                                total_size = sum(len(chunk) for chunk in client_audio_chunks)
                                                collected_in_iteration = True
                                                audio_received = True
                                                
                                                # Log every 5 chunks to reduce console spam
                                                if len(client_audio_chunks) % 5 == 0:
                                                    print(f"Received audio chunks: {len(client_audio_chunks)}, total: {total_size} bytes")
                                            else:
                                                print(f"Warning: Received empty chunk from client {client_id} despite having {pcm_chunks} PCM chunks")
                                        except Exception as e:
                                            print(f"Error receiving audio from client {client_id}: {e}")
                            
                            # If we have plenty of data, break out early
                            total_size = sum(len(chunk) for chunk in client_audio_chunks) if client_audio_chunks else 0
                            if total_size > 24000:  # ~0.75s of audio
                                print(f"Collected sufficient audio data: {total_size} bytes, breaking loop")
                                break
                            
                            # Print status occasionally if we're not collecting data
                            current_time = asyncio.get_event_loop().time()
                            if (not collected_in_iteration) and (current_time - last_display_time) >= 3.0:
                                elapsed = int(current_time - start_time)
                                last_display_time = current_time
                                
                                # Check if we have any data at all
                                if client_audio_chunks:
                                    total_size = sum(len(chunk) for chunk in client_audio_chunks)
                                    print(f"Waiting for more audio... {elapsed}s elapsed, have {len(client_audio_chunks)} chunks ({total_size} bytes)")
                                else:
                                    print(f"Waiting for audio... {elapsed}s elapsed, no data yet")
                                    
                                    # Detailed diagnostics
                                    if elapsed > 10:
                                        # If we're waiting too long, just break out
                                        print("Waited too long for WebRTC audio")
                                        if no_fallback:
                                            print("Fallback to local microphone disabled, returning empty audio")
                                            # Process whatever data we have, even if it's empty
                                            break
                                        print("Falling back to local microphone")
                                        break
                            
                            # Brief pause to avoid tight loop
                            await asyncio.sleep(0.1)
                            
                        except KeyboardInterrupt:
                            print("\nWaiting for WebRTC audio cancelled by user")
                            if no_fallback:
                                raise Exception("Audio recording cancelled")
                            print("Using local microphone")
                            break
                        except asyncio.CancelledError:
                            print("\nOperation cancelled")
                            if no_fallback:
                                raise Exception("Audio recording cancelled")
                            print("Falling back to local microphone")
                            break
                        except Exception as e:
                            print(f"Error in WebRTC audio polling: {e}")
                            await asyncio.sleep(0.1)
                    
                    # Process any audio we collected - even small amounts
                    # For WebRTC, we consider 2000 bytes a viable minimum (lowered from 4000)
                    if client_audio_chunks and sum(len(chunk) for chunk in client_audio_chunks) > 2000:
                        try:
                            total_size = sum(len(chunk) for chunk in client_audio_chunks)
                            print(f"Processing {len(client_audio_chunks)} audio chunks ({total_size} bytes)")
                            
                            # Combine all audio chunks
                            combined_audio = b''.join(client_audio_chunks)
                            
                            # If we have very little data, print a warning but still try
                            if total_size < 8000:
                                print(f"Warning: Limited audio data collected ({total_size} bytes), but attempting conversion")
                            
                            # Convert to WAV - with debug mode enabled for verbose output
                            print("Converting WebRTC audio to WAV format...")
                            wav_data = audio_processor.convert_webrtc_audio_to_wav(combined_audio)
                            
                            # Write to file
                            with open(file_path, 'wb') as f:
                                f.write(wav_data)
                            
                            # Check if file was created successfully
                            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                wav_size = os.path.getsize(file_path)
                                print(f"Successfully saved audio bridge data as WAV ({wav_size} bytes) to {file_path}")
                                return  # Exit the function, we're done
                            else:
                                print(f"Failed to save valid WAV file (file empty or missing)")
                                if no_fallback:
                                    # Create an empty WAV file
                                    wf = wave.open(file_path, 'wb')
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)  # 16-bit
                                    wf.setframerate(16000)
                                    # Write a short silence (0.5 seconds)
                                    wf.writeframes(b'\x00\x00' * 8000)
                                    wf.close()
                                    return
                        except Exception as e:
                            print(f"Error processing WebRTC audio: {e}")
                            print(traceback.format_exc())
                            if no_fallback:
                                # Create an empty WAV file
                                wf = wave.open(file_path, 'wb')
                                wf.setnchannels(1)
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(16000)
                                # Write a short silence (0.5 seconds)
                                wf.writeframes(b'\x00\x00' * 8000)
                                wf.close()
                                return
                    else:
                        if client_audio_chunks:
                            total_size = sum(len(chunk) for chunk in client_audio_chunks)
                            print(f"Insufficient audio collected: {len(client_audio_chunks)} chunks ({total_size} bytes)")
                        else:
                            print("No audio chunks collected from WebRTC clients")
                        
                        if no_fallback:
                            print("Fallback to local microphone disabled, returning empty audio")
                            # Create an empty WAV file
                            wf = wave.open(file_path, 'wb')
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)
                            # Write a short silence (0.5 seconds)
                            wf.writeframes(b'\x00\x00' * 8000)
                            wf.close()
                            return
                        print("No usable audio from bridge, falling back to local microphone")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    if no_fallback:
                        raise Exception("Audio recording cancelled")
                    print("Using local microphone")
                except asyncio.CancelledError:
                    print("\nOperation cancelled")
                    if no_fallback:
                        raise Exception("Audio recording cancelled")
                    print("Falling back to local microphone")
                except Exception as e:
                    print(f"Error in WebRTC audio bridge: {e}")
                    print(traceback.format_exc())
                    if no_fallback:
                        # Create an empty WAV file
                        wf = wave.open(file_path, 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)
                        # Write a short silence (0.5 seconds)
                        wf.writeframes(b'\x00\x00' * 8000)
                        wf.close()
                        return
        except Exception as e:
            print(f"Error using audio bridge: {e}")
            print(traceback.format_exc())
            if no_fallback:
                # Create an empty WAV file
                wf = wave.open(file_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                # Write a short silence (0.5 seconds)
                wf.writeframes(b'\x00\x00' * 8000)
                wf.close()
                return
            print("Falling back to local microphone")
    
    # If no_fallback is enabled and we've reached this point with an enabled audio bridge, we should not continue
    if audio_bridge_enabled and no_fallback:
        print("Error: Audio bridge enabled but failed, and fallback is disabled. This should not happen.")
        # Create an empty WAV file
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        # Write a short silence (0.5 seconds)
        wf.writeframes(b'\x00\x00' * 8000)
        wf.close()
        return
    
    # If we reach here, use local microphone
    print("Recording from local microphone...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
    frames = []
    print("Recording... (Talk now)")
    await send_message_to_clients(json.dumps({"action": "recording_started"}))  # Notify frontend
    silent_chunks = 0
    speaking_chunks = 0
    max_duration = 30  # Maximum recording duration in seconds
    max_chunks = int(max_duration * 16000 / chunk_size)  # Calculate max chunks
    chunk_count = 0
    
    # Calculate the number of silent chunks needed before stopping
    silence_chunks_threshold = int(silence_duration * (16000 / chunk_size))
    
    # Enforce a minimum silence threshold
    minimum_threshold = 100
    if silence_threshold < minimum_threshold:
        print(f"Warning: Silence threshold {silence_threshold} is too low, using minimum of {minimum_threshold}")
        silence_threshold = minimum_threshold
        
    # Print silence threshold information
    print(f"Using silence_threshold: {silence_threshold}, silence_duration: {silence_duration}s ({silence_chunks_threshold} chunks)")
    
    try:
        while chunk_count < max_chunks:  # Add a maximum duration cap
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            chunk_count += 1
            
            # Calculate audio level directly
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            
            # Get audio level for debugging
            if DEBUG_AUDIO_LEVELS:
                if chunk_count % 10 == 0:  # Print every 10th chunk to reduce spam
                    print(f"Audio level: {level:.2f}, threshold: {silence_threshold}, silent chunks: {silent_chunks}/{silence_chunks_threshold}")
            
            # Detect silence using direct level comparison
            if level < silence_threshold:
                silent_chunks += 1
                if silent_chunks > silence_chunks_threshold and speaking_chunks > 0:
                    # End recording if we've detected sufficient silence after speech
                    print(f"Silence detected for {silence_duration}s after speech, stopping recording")
                    break
            else:
                silent_chunks = 0
                speaking_chunks += 1
                # If this is the first speech detected, notify
                if speaking_chunks == 1:
                    print("Speech detected!")
            
            # Maximum recording duration reached
            if speaking_chunks > silence_duration * (16000 / chunk_size) * 10:
                print(f"Maximum recording duration reached ({max_duration}s)")
                break
            
            # Status update every ~5 seconds
            if chunk_count % int(5 * 16000 / chunk_size) == 0:
                seconds = chunk_count * chunk_size / 16000
                print(f"Still recording... ({seconds:.1f}s)")
        
        # If we never detected speech, give a message
        if speaking_chunks == 0:
            print("No speech detected during recording")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
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
        
        audio_size = os.path.getsize(file_path)
        print(f"Audio bridge created file: {file_path} ({audio_size} bytes)")
        
        # Check if we actually recorded anything meaningful
        if speaking_chunks == 0:
            print("Warning: No speech detected in the recording")

async def execute_once(question_prompt):
    temp_image_path = os.path.join(output_dir, 'temp_img.jpg')
    
    # Determine the audio file format based on the TTS provider this is for the image analysis only see app_logic.py for the user chatbot conversation
    if TTS_PROVIDER == 'elevenlabs':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.mp3')  # Use mp3 for ElevenLabs
        max_char_length = MAX_CHAR_LENGTH  # Set a higher limit for ElevenLabs
    elif TTS_PROVIDER == 'openai':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for OpenAI
        max_char_length = MAX_CHAR_LENGTH  # Set a higher limit for OpenAI
    else:
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for XTTS
        max_char_length = XTTS_NUM_CHARS  # Set a lower limit for XTTS , default is 255 testing 1000+, on 4090 taking 90 secs for 2000 chars quality is bad

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

async def execute_screenshot_and_analyze():
    question_prompt = "What do you see in this image? Keep it short but detailed and answer any follow up questions about it"
    print("Taking screenshot and analyzing...")
    await execute_once(question_prompt)
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
        # First, try XAI's image analysis if it's supported
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {XAI_API_KEY}"
        }
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": question_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "low"}}
            ]
        }
        payload = {
            "model": XAI_MODEL,
            "temperature": 0.5,
            "messages": [message],
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        # If XAI doesn't support image analysis or returns an error,
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
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}", "detail": "low"}}
        ]
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.5,
        "messages": [message],
        "max_tokens": 1000
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        print(f"OpenAI fallback request failed: {e}")
        return {"choices": [{"message": {"content": "Failed to process the image with both XAI and OpenAI models."}}]}


async def generate_speech(text, temp_audio_path):
    if TTS_PROVIDER == 'openai':
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": OPENAI_MODEL_TTS, "voice": OPENAI_TTS_VOICE, "input": text, "response_format": "wav"}
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENAI_TTS_URL, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    with open(temp_audio_path, "wb") as audio_file:
                        audio_file.write(await response.read())
                else:
                    print(f"Failed to generate speech: {response.status} - {await response.text()}")

    elif TTS_PROVIDER == 'elevenlabs':
        await elevenlabs_text_to_speech(text, temp_audio_path)

    else:  # XTTS
        if tts is not None:
            try:
                wav = await asyncio.to_thread(
                    tts.tts,
                    text=text,
                    speaker_wav=character_audio_file,
                    language="en",
                    speed=float(os.getenv('XTTS_SPEED', '1.1'))
                )
                sf.write(temp_audio_path, wav, tts.synthesizer.tts_config.audio["sample_rate"])
                print("Audio generated successfully with XTTS.")
            except Exception as e:
                print(f"Error during XTTS audio generation: {e}")
        else:
            print("XTTS model is not loaded.")

async def user_chatbot_conversation():
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
    
    base_system_message = open_file(character_prompt_file)
    
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
            
            print(PINK + f"{character_display_name}:..." + RESET_COLOR)
            chatbot_response = chatgpt_streamed(user_input, base_system_message, mood, conversation_history)
            conversation_history.append({"role": "assistant", "content": chatbot_response})
            sanitized_response = sanitize_response(chatbot_response)
            if len(sanitized_response) > 400:
                sanitized_response = sanitized_response[:400] + "..."
            prompt2 = sanitized_response
            await process_and_play(prompt2, character_audio_file)  # Note the 'await' here
            if current_character.startswith("story_") or current_character.startswith("game_"):
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]
            else:
                if len(conversation_history) > 30:
                    conversation_history = conversation_history[-30:]

            # Save conversation history after each message exchange
            save_conversation_history(conversation_history)

    except KeyboardInterrupt:
        print("Quitting the conversation...")

def load_character_specific_history(character_name):
    """
    Load conversation history from a character-specific file for story/game characters.
    
    Args:
        character_name: The name of the character
        
    Returns:
        list: The conversation history or an empty list if not found
    """
    try:
        # Only process for story/game characters
        if not character_name.startswith("story_") and not character_name.startswith("game_"):
            print(f"Not a story/game character: {character_name}")
            return []
        
        # Create character-specific history file path
        character_dir = os.path.join(characters_folder, character_name)
        history_file = os.path.join(character_dir, "conversation_history.txt")
        
        # Check if file exists
        if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
            print(f"No character-specific history found for {character_name}")
            return []
        
        print(f"Loading character-specific history for {character_name}")
        
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
        
        print(f"Loaded {len(temp_history)} messages from character-specific history file")
        return temp_history
    except Exception as e:
        print(f"Error loading character-specific history: {e}")
        return []

async def process_message(self, message, client_id=None, remote_playback=False):
    """Process a message from the user and return the response"""
    # Always use remote playback regardless of the parameter value
    remote_playback = True
    
    if not message or message.strip() == "":
        return "I didn't catch that. Could you please repeat?"

    try:
        # Get character context
        character = get_current_character()
            
        # Log the message
        logger.info(f"Processing message from user to character {character}: {message}")
        
        # Add sentiment analysis
        sentiment = TextBlob(message).sentiment
        if sentiment:
            logger.info(f"Sentiment polarity: {sentiment.polarity}")
            mood = "positive" if sentiment.polarity > 0.1 else "negative" if sentiment.polarity < -0.1 else "neutral"
            logger.info(f"Detected mood: {mood}")
        
        # Get response streaming
        model_provider = MODEL_PROVIDER
        response = ""
        
        # Debug info
        logger.info(f"Debug: streamed started. MODEL_PROVIDER: {model_provider}")
                        
        # Common parameters for all streaming
        kwargs = {
            "character": character,
            "prompt": message,
        }
        
        if model_provider == "openai":
            logger.info(f"Debug: Sending request to OpenAI: {OPENAI_BASE_URL}")
            logger.info("Starting OpenAI stream...")
            
            # Get the model from the session or use the default
            model = OPENAI_MODEL
                
            # Stream the response
            try:
                async for chunk in stream_openai_response(character=character, prompt=message, model=model):
                    print(chunk, end="")
                    response += chunk
                
                print("\n") 
                logger.info("OpenAI stream complete.")
            except Exception as e:
                logger.error(f"Error streaming OpenAI response: {e}")
                response = f"I'm sorry, I encountered an error while generating a response: {str(e)}"
        
        # Add other model providers here as needed
        
        logger.info(f"streaming complete. Response length: {len(response)}")
        
        # Generate audio for the response
        logger.info("Playing generated audio...")
        
        # ALWAYS use remote playback to ensure audio plays on the client side
        logger.info("Forcing client-side audio playback for all responses")
        
        # If we're using audio bridge and have a client_id, ensure client gets both message types
        if client_id:
            try:
                # First send a message to the client with play_on_client set to true
                from app.audio_bridge.audio_bridge_server import audio_bridge_server
                await audio_bridge_server.send_message_to_client(client_id, {
                    "type": "transcription",
                    "text": message,
                    "response": response,
                    "play_on_client": True  # Force client-side playback
                })
                
                # Also generate audio URL if possible and send that too
                try:
                    # Create a temporary WAV file path in the outputs directory
                    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
                    output_path = os.path.join(outputs_dir, f"response_{int(time.time())}.wav")
                    
                    # Generate audio file using the chosen TTS provider
                    if TTS_PROVIDER == "openai":
                        from app.app import openai_text_to_speech
                        await openai_text_to_speech(response, output_path)
                    elif TTS_PROVIDER == "elevenlabs":
                        from app.app import elevenlabs_text_to_speech
                        await elevenlabs_text_to_speech(response, output_path)
                        
                    # Check if file was generated
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        # Copy the file to the static directory for URL access
                        from app.app import copy_to_static_output
                        audio_url = copy_to_static_output(output_path)
                        
                        # Send the response with audio URL for client-side playback
                        await audio_bridge_server.send_message_to_client(client_id, {
                            "type": "transcription",
                            "text": message,
                            "response": response,
                            "play_on_client": True,
                            "audio_url": audio_url
                        })
                except Exception as e:
                    logger.error(f"Error generating audio URL: {e}")
                    
            except Exception as e:
                logger.error(f"Error sending client message: {e}")
        
        return response

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Sorry, there was an error processing your message: {str(e)}"

def copy_to_static_output(source_file):
    """
    Copy a file from the outputs directory to the static/outputs directory
    so it can be accessed via HTTP.
    
    Args:
        source_file: Path to the source file in the outputs directory
        
    Returns:
        The URL path to access the file via HTTP
    """
    try:
        # Get the filename
        filename = os.path.basename(source_file)
        
        # Create the destination path
        dest_path = os.path.join(static_output_dir, filename)
        
        # Copy the file
        shutil.copy2(source_file, dest_path)
        
        # Return the absolute URL path (always start with /)
        url_path = f"/static/outputs/{filename}"
        
        # Debug output
        print(f"Copied audio file from {source_file} to {dest_path}")
        print(f"Generated URL path: {url_path}")
        print(f"File exists at destination: {os.path.exists(dest_path)}")
        print(f"File size at destination: {os.path.getsize(dest_path)}")
        
        return url_path
    except Exception as e:
        print(f"Error copying file to static outputs: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(user_chatbot_conversation())

