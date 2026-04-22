# use: uv run python cli.py   (or: python cli.py with your venv activated)
# PyTorch is optional unless TTS_PROVIDER=sparktts (install torch per INSTALL.md).

import os
import time
import pyaudio
import numpy as np
import wave
import requests
import json
import base64
from PIL import ImageGrab
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import anthropic
from faster_whisper import WhisperModel
import soundfile as sf
from textblob import TextBlob
from pathlib import Path
import re
import io
from pydub import AudioSegment
import warnings

# Load environment variables
load_dotenv()

MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'openai')
CHARACTER_NAME = os.getenv('CHARACTER_NAME', 'wizard')
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
OPENAI_TTS_URL = os.getenv('OPENAI_TTS_URL', 'https://api.openai.com/v1/audio/speech')
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', 'alloy')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions')
OPENAI_TRANSCRIPTION_MODEL = os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'gpt-4o-mini-transcribe')
OPENAI_MODEL_TTS = os.getenv('OPENAI_MODEL_TTS', 'gpt-4o-mini-tts')
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
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))
VOICE_SPEED = os.getenv('VOICE_SPEED', '1.0')
SPARKTTS_MODEL_DIR = os.getenv('SPARKTTS_MODEL_DIR', 'pretrained_models/Spark-TTS-0.5B')
SPARKTTS_MAX_CHARS = int(os.getenv('SPARKTTS_MAX_CHARS', 1000))
TYPECAST_API_KEY = os.getenv('TYPECAST_API_KEY')
TYPECAST_TTS_VOICE = os.getenv('TYPECAST_TTS_VOICE')
TYPECAST_TTS_MODEL = os.getenv('TYPECAST_TTS_MODEL', 'ssfm-v30')
TYPECAST_EMOTION_PRESET = os.getenv('TYPECAST_EMOTION_PRESET', 'normal')

audio_playback_stop_requested = False

def request_audio_playback_stop():
    global audio_playback_stop_requested
    audio_playback_stop_requested = True

def reset_audio_playback_stop():
    global audio_playback_stop_requested
    audio_playback_stop_requested = False

def is_audio_playback_stop_requested():
    return audio_playback_stop_requested

def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


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

# Capitalize the first letter of the character name
character_display_name = CHARACTER_NAME.capitalize()

# Check for CUDA availability (uses PyTorch if installed; else CPU)
device = "cuda" if _cuda_available() else "cpu"

# Check if Faster Whisper should be loaded at startup
FASTER_WHISPER_LOCAL = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

# Initialize whisper model as None to lazy load
whisper_model = None

# Paths for character-specific files
project_dir = os.path.dirname(os.path.abspath(__file__))
characters_folder = os.path.join(project_dir, 'characters', CHARACTER_NAME)
character_prompt_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.txt")
character_audio_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.wav")

# Initialize Spark-TTS only when selected (pulls in PyTorch / cli.SparkTTS)
sparktts_model = None
if TTS_PROVIDER == 'sparktts':
    SparkTTS_cls = None
    try:
        import sys

        _root = os.path.dirname(os.path.abspath(__file__))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from cli.SparkTTS import SparkTTS as SparkTTS_cls
    except ImportError as e:
        print(f"Spark-TTS import failed: {e}")
    if SparkTTS_cls is None:
        print("Spark-TTS is not available. Install torch and Spark-TTS extras (see INSTALL.md).")
        TTS_PROVIDER = 'openai'
        print("Switched to default TTS provider: openai")
    else:
        print(f"Initializing Spark-TTS model from {SPARKTTS_MODEL_DIR}...")
        try:
            import torch

            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module="torch.nn.utils.weight_norm",
            )
            torch_device = torch.device(device)
            print(f"Using device: {torch_device} (CUDA available: {torch.cuda.is_available()})")
            sparktts_model = SparkTTS_cls(model_dir=Path(SPARKTTS_MODEL_DIR), device=torch_device)
            print(f"Spark-TTS model loaded successfully on {torch_device}.")
        except Exception as e:
            print(f"Failed to load Spark-TTS model: {e}")
            TTS_PROVIDER = 'openai'
            print("Switched to default TTS provider: openai")

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
def play_audio(file_path):
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

# Model and device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = os.path.join(project_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

if TTS_PROVIDER == 'sparktts':
    print(f"Using device: {device}")
print(f"Model provider: {MODEL_PROVIDER}")
print(f"Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else XAI_MODEL if MODEL_PROVIDER == 'xai' else ANTHROPIC_MODEL if MODEL_PROVIDER == 'anthropic' else OLLAMA_MODEL}")
print(f"Character: {character_display_name}")
print(f"Text-to-Speech provider: {TTS_PROVIDER}")
print(f"Text-to-Speech model: {OPENAI_MODEL_TTS if TTS_PROVIDER == 'openai' else 'xai-grok-tts' if TTS_PROVIDER == 'xai' else ELEVENLABS_TTS_MODEL if TTS_PROVIDER == 'elevenlabs' else 'kokoro-tts' if TTS_PROVIDER == 'kokoro' else 'Spark-TTS-0.5B' if TTS_PROVIDER == 'sparktts' else TYPECAST_TTS_MODEL if TTS_PROVIDER == 'typecast' else 'Unknown'}")
print("To stop chatting say Quit or Exit. One moment please loading...")

# Function to synthesize speech
def process_and_play(prompt, audio_file_pth):
    if TTS_PROVIDER == 'openai':
        output_path = os.path.join(output_dir, 'output.wav')
        openai_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            play_audio(output_path)
        else:
            print("Error: Audio file not found.")
    elif TTS_PROVIDER == 'elevenlabs':
        output_path = os.path.join(output_dir, 'output.mp3')
        elevenlabs_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            # Convert MP3 to WAV if ElevenLabs is used
            temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
            audio = AudioSegment.from_mp3(output_path)
            audio.export(temp_wav_path, format="wav")
            play_audio(temp_wav_path)
        else:
            print("Error: Audio file not found.")
    elif TTS_PROVIDER == 'kokoro':
        output_path = os.path.join(output_dir, 'output.wav')
        kokoro_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            play_audio(output_path)
        else:
            print("Error: Audio file not found.")
    elif TTS_PROVIDER == 'xai':
        output_path = os.path.join(output_dir, f"output.{xai_tts_file_extension()}")
        success = xai_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if success and os.path.exists(output_path):
            print("Playing generated audio...")
            play_audio(output_path)
        elif not success:
            print("Failed to generate xAI audio.")
        else:
            print("Error: Audio file not found.")
    elif TTS_PROVIDER == 'sparktts':
        if sparktts_model is not None:
            try:
                wav_np = sparktts_model.inference(
                    text=prompt,
                    prompt_speech_path=Path(audio_file_pth),
                    prompt_text=None,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                src_path = os.path.join(output_dir, 'output.wav')
                sf.write(src_path, wav_np, sparktts_model.sample_rate)
                print("Audio generated successfully with Spark-TTS.")
                play_audio(src_path)
            except Exception as e:
                print(f"Error during Spark-TTS audio generation: {e}")
        else:
            print("Spark-TTS model is not loaded. Please ensure initialization succeeded.")
    elif TTS_PROVIDER == 'typecast':
        output_path = os.path.join(output_dir, 'output.wav')
        typecast_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            play_audio(output_path)
        else:
            print("Error: Audio file not found.")

def save_pcm_as_wav(pcm_data: bytes, file_path: str, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2):
    """ Saves PCM data as a WAV file. """
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

def fetch_pcm_audio(model: str, voice: str, input_text: str, api_url: str) -> bytes:
    """ Fetches PCM audio data from the OpenAI API. """
    client = OpenAI()
    pcm_data = io.BytesIO()
    
    try:
        response = requests.post(
            url=api_url,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "voice": voice,
                "input": input_text,
                "response_format": 'pcm'
            },
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=8192):
            pcm_data.write(chunk)

    except OpenAIError as e:
        print(f"An error occurred while trying to fetch the audio stream: {e}")
        raise

    return pcm_data.getvalue()

def openai_text_to_speech(prompt, output_path):
    file_extension = Path(output_path).suffix.lstrip('.').lower()

    if file_extension == 'wav':
        pcm_data = fetch_pcm_audio(OPENAI_MODEL_TTS, OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL)
        save_pcm_as_wav(pcm_data, output_path)
    else:
        try:
            response = requests.post(
                url=OPENAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL_TTS,
                    "voice": OPENAI_TTS_VOICE,
                    "speed": float(VOICE_SPEED),
                    "input": prompt,
                    "response_format": file_extension
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("Audio generated successfully with OpenAI.")
            play_audio(output_path)
        except requests.HTTPError as e:
            print(f"Error during OpenAI TTS: {e}")

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

def xai_text_to_speech(text, output_path):
    """Convert text to speech using xAI Grok TTS."""
    if not XAI_API_KEY:
        print("XAI_API_KEY is not set. Cannot use xAI TTS.")
        return False

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            XAI_TTS_URL,
            headers=headers,
            json=build_xai_tts_payload(text),
            stream=True,
            timeout=XAI_TTS_TIMEOUT,
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Audio generated successfully with xAI TTS.")
        return True
    except requests.HTTPError as e:
        error_text = e.response.text if e.response is not None else str(e)
        print(f"Error during xAI TTS: {error_text}")
        return False
    except requests.RequestException as e:
        print(f"Error during xAI TTS: {e}")
        return False

def elevenlabs_text_to_speech(text, output_path):
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
            "speed": float(VOICE_SPEED)
        }
    }

    response = requests.post(tts_url, headers=headers, json=data, stream=True, timeout=30)

    if response.ok:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        print("Audio stream saved successfully.")
    else:
        print("Error generating speech:", response.text)

def kokoro_text_to_speech(text, output_path):
    """Convert text to speech using Kokoro TTS API."""
    try:
        # Define the API endpoint
        kokoro_url = f"{KOKORO_BASE_URL}/audio/speech"
        
        # Prepare payload with the format expected by Kokoro API
        payload = {
            "model": "kokoro",
            "voice": KOKORO_TTS_VOICE,
            "input": text,
            "response_format": "wav",  # Use wav format for more compatibility
            "speed": float(VOICE_SPEED)  # Use the global VOICE_SPEED parameter
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
        response = requests.post(kokoro_url, json=payload, headers=headers, timeout=30, verify=False)
        
        if response.status_code == 200:
            # Save the audio data to file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print("Audio generated successfully with Kokoro.")
            return True
        else:
            print(f"Error from Kokoro API: HTTP {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error during Kokoro TTS generation: {e}")
        return False

def typecast_text_to_speech(text, output_path):
    """Convert text to speech using Typecast API."""
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

        response = client.text_to_speech(request)

        with open(output_path, 'wb') as f:
            f.write(response.audio_data)

        print("Audio generated successfully with Typecast.")
        return True
    except ImportError:
        print("typecast-python package is not installed. Run: pip install typecast-python")
        return False
    except Exception as e:
        print(f"Error during Typecast TTS generation: {e}")
        return False

def sanitize_response(response):
    # Remove model planning blocks before sending text to TTS.
    response = remove_response_meta_blocks(response)
    response = normalize_response_punctuation(response, preserve_xai_tags=TTS_PROVIDER == 'xai')
    # Remove asterisks and emojis
    response = re.sub(r'\*.*?\*', '', response)
    if TTS_PROVIDER == 'xai':
        response = re.sub(r'[^\w\s,.;:\'!?\[\]<>\/%()&-]', '', response)
    else:
        response = re.sub(r'[^\w\s,.\'!?]', '', response)
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
        "happy", "pleased", "content", "satisfied", "good", "great",
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
    elif any(keyword in user_input.lower() for keyword in angry_keywords):
        mood = "angry"
    elif any(keyword in user_input.lower() for keyword in sad_keywords):
        mood = "sad"
    elif any(keyword in user_input.lower() for keyword in fearful_keywords):
        mood = "fearful"
    elif any(keyword in user_input.lower() for keyword in surprised_keywords):
        mood = "surprised"
    elif any(keyword in user_input.lower() for keyword in disgusted_keywords):
        mood = "disgusted"
    elif any(keyword in user_input.lower() for keyword in happy_keywords):
        mood = "happy"
    elif any(keyword in user_input.lower() for keyword in joyful_keywords) or polarity > 0.3:
        mood = "joyful"
    elif any(keyword in user_input.lower() for keyword in neutral_keywords):
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
        
    return mood

def adjust_prompt(mood):
    prompts_path = os.path.join(characters_folder, 'prompts.json')
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            mood_prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error loading prompts: {prompts_path} not found. Using default prompts.")
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

    mood_prompt = mood_prompts.get(mood, "")
    return mood_prompt

def chatgpt_streamed(user_input, system_message, mood_prompt, conversation_history):
    """
    Function to send a query to either the Ollama model or OpenAI model
    """
    # Calculate token limit based on character limit
    token_limit = min(4000, MAX_CHAR_LENGTH * 4 // 3)
    
    if MODEL_PROVIDER == 'ollama':
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}],
            "stream": True,
            "options": {
                "num_predict": -2,
                "temperature": 1.0
            }
        }
        response = requests.post(f'{OLLAMA_BASE_URL}/v1/chat/completions', headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        full_response = ""
        line_buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                line = line[5:].strip()  # Remove the "data:" prefix
            if line:
                try:
                    chunk = json.loads(line)
                    delta_content = chunk['choices'][0]['delta'].get('content', '')
                    if delta_content:
                        line_buffer += delta_content
                        if '\n' in line_buffer:
                            lines = line_buffer.split('\n')
                            for line in lines[:-1]:
                                if not is_xai_tts_enabled():
                                    print(NEON_GREEN + line + RESET_COLOR)
                                full_response += line + '\n'
                            line_buffer = lines[-1]
                except json.JSONDecodeError:
                    continue
        if line_buffer:
            if not is_xai_tts_enabled():
                print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer
        return full_response
    
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
            "temperature": 0.8,
            "max_tokens": token_limit  # Using our calculated token limit for xAI
        }
        response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        full_response = ""
        print("Starting XAI stream...")
        for raw_line in response.iter_lines(decode_unicode=False):
            line = raw_line.decode('utf-8', errors='replace').strip()
            if line.startswith("data:"):
                line = line[5:].strip() 
            if line:
                try:
                    chunk = json.loads(line)
                    delta_content = chunk['choices'][0]['delta'].get('content', '')
                    if delta_content:
                        delta_content = repair_mojibake_text(delta_content)
                        if not is_xai_tts_enabled():
                            print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                        full_response += delta_content
                except json.JSONDecodeError:
                    continue
        full_response = repair_mojibake_text(full_response)
        print("\nXAI stream complete.")
        return full_response

    elif MODEL_PROVIDER == 'anthropic':
        if anthropic is None:
            print("Error: Anthropic library not installed. Please install using: pip install anthropic")
            return "I apologize, but the Anthropic API is not available. Please install the required library or choose a different model provider."
            
        # Format the conversation history for Anthropic
        anthropic_messages = []
        for msg in conversation_history:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        try:
            # Create the client with default settings
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            print(f"Starting Anthropic stream...")
            
            # Format system and mood prompt for Anthropic
            system_content = system_message + "\n" + mood_prompt
            
            # Variables to store the full response and line buffer
            full_response = ""
            line_buffer = ""
            
            # Start the streaming message request
            with client.messages.stream(
                max_tokens=token_limit,  # Using our calculated token limit for Anthropic
                model=ANTHROPIC_MODEL,
                system=system_content,
                messages=anthropic_messages + [{"role": "user", "content": user_input}],
                temperature=0.8
            ) as stream:
                # Process the stream events
                for event in stream:
                    if event.type == "content_block_delta":
                        delta_content = event.delta.text
                        if delta_content:
                            if not is_xai_tts_enabled():
                                print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                            full_response += delta_content
            
            print("\nAnthropic stream complete.")
            return full_response
        
        except Exception as e:
            error_message = f"Error connecting to Anthropic model: {e}"
            print(f"Error: {error_message}")
            return error_message

    elif MODEL_PROVIDER == 'openai':
        messages = [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}]
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": token_limit  # Using the new parameter name for OpenAI
        }
        response = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        full_response = ""
        print("Starting OpenAI stream...")
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                line = line[5:].strip() 
            if line:
                try:
                    chunk = json.loads(line)
                    delta_content = chunk['choices'][0]['delta'].get('content', '')
                    if delta_content:
                        if not is_xai_tts_enabled():
                            print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                        full_response += delta_content
                except json.JSONDecodeError:
                    continue
        print("\nOpenAI stream complete.")
        return full_response

# Function to transcribe the recorded audio using faster-whisper
def transcribe_with_whisper(audio_file):
    global whisper_model

    if whisper_model is None:
        whisper_device = "cuda" if _cuda_available() else "cpu"
        model_size = "medium.en" if whisper_device == "cuda" else "tiny.en"

        try:
            print(f"Lazy-loading Faster-Whisper on {whisper_device}...")
            whisper_model = WhisperModel(
                model_size,
                device=whisper_device,
                compute_type="float16" if whisper_device == "cuda" else "int8",
            )
            print("Faster-Whisper initialized successfully.")
        except Exception as e:
            print(f"Error initializing Faster-Whisper on {whisper_device}: {e}")
            print("Falling back to CPU mode...")
            whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            print("Faster-Whisper initialized on CPU successfully.")

    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

def detect_silence(data, threshold=1000, chunk_size=1024):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.mean(np.abs(audio_data)) < threshold

# Function to record audio from the microphone and save to a file
def record_audio(file_path, silence_threshold=512, silence_duration=4.0, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
    frames = []
    print(f"{PINK}Recording...{RESET_COLOR}")
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
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def execute_once(question_prompt):
    temp_image_path = os.path.join(output_dir, 'temp_img.jpg')
    
    # Determine the audio file format based on the TTS provider
    if TTS_PROVIDER == 'elevenlabs':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.mp3')
    elif TTS_PROVIDER == 'xai':
        temp_audio_path = os.path.join(output_dir, f"temp_audio.{xai_tts_file_extension()}")
    else:
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')

    image_path = take_screenshot(temp_image_path)
    response = analyze_image(image_path, question_prompt)
    text_response = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response received.')

    max_char_length = MAX_CHAR_LENGTH
    if len(text_response) > max_char_length:
        text_response = text_response[:max_char_length] + "..."

    print(text_response)

    generate_speech(text_response, temp_audio_path)

    if TTS_PROVIDER == 'elevenlabs':
        # Convert MP3 to WAV if ElevenLabs is used
        temp_wav_path = os.path.join(output_dir, 'temp_output.wav')
        audio = AudioSegment.from_mp3(temp_audio_path)
        audio.export(temp_wav_path, format="wav")
        play_audio(temp_wav_path)
    else:
        play_audio(temp_audio_path)

    os.remove(image_path)

def execute_screenshot_and_analyze():
    question_prompt = "What do you see in this image? Keep it short but detailed and answer any follow up questions about it"
    print("Taking screenshot and analyzing...")
    execute_once(question_prompt)
    print("\nReady for the next question....")

def take_screenshot(temp_image_path):
    time.sleep(5)  # Wait for 5 seconds before taking a screenshot
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize((512, 512))
    screenshot.save(temp_image_path, 'JPEG')
    return temp_image_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, question_prompt):
    encoded_image = encode_image(image_path)

    if MODEL_PROVIDER == 'ollama':
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": "llava",
            "prompt": question_prompt,
            "images": [encoded_image],
            "stream": False
        }
        try:
            response = requests.post(f'{OLLAMA_BASE_URL}/api/generate', headers=headers, json=payload, timeout=20)
            print(f"Response status code: {response.status_code}")  # Debugging statement
            if response.status_code == 200:
                print("Using ollama for image analysis")
                return {"choices": [{"message": {"content": response.json().get('response', 'No response received.')}}]}
            elif response.status_code == 404:
                return {"choices": [{"message": {"content": "The llava model is not available on this server."}}]}
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")  # Debugging statement
            return {"choices": [{"message": {"content": "Failed to process the image with the llava model."}}]}
    
    elif MODEL_PROVIDER == 'xai':
        # First try XAI image analysis
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
            response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                print("Using xAI for image analysis")
                return response.json()
            else:
                print("XAI image analysis failed or not supported, falling back to OpenAI")
                # Fall back to OpenAI image analysis
                return analyze_image_with_openai(encoded_image, question_prompt)
        except requests.exceptions.RequestException as e:
            print(f"XAI image analysis failed: {e}, falling back to OpenAI")
            return analyze_image_with_openai(encoded_image, question_prompt)
    
    else:
        return analyze_image_with_openai(encoded_image, question_prompt)

# Add helper function for OpenAI image analysis fallback
def analyze_image_with_openai(encoded_image, question_prompt):
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
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        print("Using OpenAI for image analysis")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"OpenAI request failed: {e}")
        return {"choices": [{"message": {"content": "Failed to process the image with both XAI and OpenAI models."}}]}


def generate_speech(text, temp_audio_path):
    if TTS_PROVIDER == 'openai':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": OPENAI_MODEL_TTS,
            "voice": OPENAI_TTS_VOICE,
            "speed": float(VOICE_SPEED),
            "input": text,
            "response_format": "wav"
        }
        response = requests.post(OPENAI_TTS_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            with open(temp_audio_path, "wb") as audio_file:
                audio_file.write(response.content)
        else:
            print(f"Failed to generate speech: {response.status_code} - {response.text}")
    elif TTS_PROVIDER == 'elevenlabs':
        elevenlabs_text_to_speech(text, temp_audio_path)
    elif TTS_PROVIDER == 'kokoro':
        kokoro_text_to_speech(text, temp_audio_path)
    elif TTS_PROVIDER == 'xai':
        xai_text_to_speech(text, temp_audio_path)
    elif TTS_PROVIDER == 'typecast':
        typecast_text_to_speech(text, temp_audio_path)
    else:  # Spark-TTS
        if sparktts_model is not None:
            try:
                wav_np = sparktts_model.inference(
                    text=text,
                    prompt_speech_path=Path(character_audio_file),
                    prompt_text=None,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                sf.write(temp_audio_path, wav_np, sparktts_model.sample_rate)
                print("Audio generated successfully with Spark-TTS.")
            except Exception as e:
                print(f"Error during Spark-TTS audio generation: {e}")
        else:
            print("Spark-TTS model is not loaded.")

def transcribe_with_openai_api(audio_file, model="gpt-4o-mini-transcribe"):
    """Transcribe audio using OpenAI's API"""
    if not OPENAI_API_KEY:
        raise ValueError("API key missing. Please set OPENAI_API_KEY in your environment.")
    
    # Make the API call to OpenAI
    api_url = "https://api.openai.com/v1/audio/transcriptions"
    
    with open(audio_file, "rb") as audio_file_data:
        files = {
            'file': (os.path.basename(audio_file), audio_file_data, 'audio/wav')
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data = {
            'model': model
        }
        
        response = requests.post(api_url, headers=headers, files=files, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            transcription = result.get("text", "")
            return transcription
        else:
            error_text = response.text
            print(f"Error from OpenAI API: {error_text}")
            raise Exception(f"Transcription error: {response.status_code} - {error_text}")

def transcribe_audio(audio_file):
    """Transcribe audio using either local Faster Whisper or OpenAI API"""
    if FASTER_WHISPER_LOCAL:
        print(f"Using Faster Whisper for transcription")
        return transcribe_with_whisper(audio_file)
    else:
        print(f"Transcription (model: {OPENAI_TRANSCRIPTION_MODEL})")
        return transcribe_with_openai_api(audio_file, OPENAI_TRANSCRIPTION_MODEL)

def load_character_specific_history(character_name):
    """
    Load conversation history from a character-specific file for story/game characters.
    Only to be used for characters with names starting with story_ or game_
    
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
        character_dir = os.path.join('characters', character_name)
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
            print(f"Not a story/game character: {character_name}")
            return {"status": "error", "message": "Not a story/game character"}
            
        # Create character-specific history file path
        character_dir = os.path.join('characters', character_name)
        os.makedirs(character_dir, exist_ok=True)
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
        print(f"Error saving character-specific history: {e}")
        return {"status": "error", "message": str(e)}

def save_global_conversation_history(history):
    """
    Save conversation history to the global history file.
    To be used for regular characters (not story_/game_ prefixed).
    
    Args:
        history: The conversation history to save
        
    Returns:
        dict: Status of the operation
    """
    try:
        # Save to global history file
        history_file = "conversation_history.txt"
        
        print(f"Saving global conversation history")
        
        with open(history_file, "w", encoding="utf-8") as file:
            for message in history:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n\n")  # Extra newline for readability
                
        print(f"Saved {len(history)} messages to global history file")
        return {"status": "success"}
    except Exception as e:
        print(f"Error saving global conversation history: {e}")
        return {"status": "error", "message": str(e)}

def user_chatbot_conversation():
    reset_audio_playback_stop()

    # Get current character
    current_character = os.getenv('CHARACTER_NAME', 'wizard')
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    
    # Initialize conversation history based on character type
    if is_story_character:
        # Try to load history from character-specific file
        loaded_history = load_character_specific_history(current_character)
        if loaded_history:
            conversation_history = loaded_history
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
            user_input = transcribe_audio(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file 
            print(CYAN + "You:", user_input + RESET_COLOR)
            
            # Check for quit phrases - simplified and more robust check
            if any(user_input.lower().strip().rstrip('.') == phrase.lower().rstrip('.') for phrase in quit_phrases):
                print("Quitting the conversation...")
                break
                
            conversation_history.append({"role": "user", "content": user_input})
            
            if any(phrase in user_input.lower() for phrase in screenshot_phrases):
                execute_screenshot_and_analyze()
                continue
            
            mood = analyze_mood(user_input)
            mood_prompt = adjust_prompt(mood)
            tag_prompt = xai_speech_tag_prompt()
            if tag_prompt:
                mood_prompt = f"{mood_prompt}\n\n{tag_prompt}"
            
            print(PINK + f"{character_display_name}:..." + RESET_COLOR)
            chatbot_response = chatgpt_streamed(user_input, base_system_message, mood_prompt, conversation_history)
            sanitized_response = sanitize_response(chatbot_response)
            display_response = strip_xai_speech_tags(chatbot_response)
            current_character = os.getenv('CHARACTER_NAME', 'wizard')
            is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
            if is_story_character:
                display_response = format_story_response_text(display_response)
            if is_xai_tts_enabled():
                print(NEON_GREEN + display_response + RESET_COLOR)
            conversation_history.append({"role": "assistant", "content": display_response})
            if is_story_character:
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]
                save_character_specific_history(conversation_history, current_character)
            else:
                if len(conversation_history) > 30:
                    conversation_history = conversation_history[-30:]
                save_global_conversation_history(conversation_history)
            if len(sanitized_response) > MAX_CHAR_LENGTH:  # Limit response length for audio generation
                sanitized_response = sanitized_response[:MAX_CHAR_LENGTH] + "..."
            prompt2 = sanitized_response
            process_and_play(prompt2, character_audio_file)
    except KeyboardInterrupt:
        request_audio_playback_stop()
        print("Quitting the conversation...")

if __name__ == "__main__":
    user_chatbot_conversation()
