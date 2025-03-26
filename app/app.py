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
import re
import io
import torch
from pydub import AudioSegment
from .shared import clients, get_current_character

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)  # transformers 4.48+ warning

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
XAI_API_KEY = os.getenv('XAI_API_KEY')
XAI_MODEL = os.getenv('XAI_MODEL', 'grok-2-1212')
XAI_BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_TTS_VOICE = os.getenv('ELEVENLABS_TTS_VOICE')
ELEVENLABS_TTS_MODEL = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_multilingual_v2')
ELEVENLABS_TTS_SPEED = os.getenv('ELEVENLABS_TTS_SPEED', '1')
XTTS_SPEED = os.getenv('XTTS_SPEED', '1.1') 
os.environ["COQUI_TOS_AGREED"] = "1"

# Initialize OpenAI API key
OpenAI.api_key = OPENAI_API_KEY

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Capitalize the first letter of the character name
character_display_name = CHARACTER_NAME.capitalize()

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable CuDNN explicitly - enable this if you get cudnn errors or change in xtts-v2/config.json
# torch.backends.cudnn.enabled = False

# Default model size (adjust as needed)
model_size = "medium.en"

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

output_dir = os.path.join(project_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model provider: {MODEL_PROVIDER}")
print(f"Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else XAI_MODEL if MODEL_PROVIDER == 'xai' else OLLAMA_MODEL}")
print(f"Character: {character_display_name}")
print(f"Text-to-Speech provider: {TTS_PROVIDER}")
print("To stop chatting say Quit, Leave or Exit. Say, what's on my screen, to have AI view screen. One moment please loading...")

async def process_and_play(prompt, audio_file_pth):
    # Always get the current character name to ensure we have the right audio file
    current_character = get_current_character()
    
    # Update characters_folder path to point to the current character's folder
    current_characters_folder = os.path.join(project_dir, 'characters', current_character)
    
    # Override the provided audio path with the current character's audio file
    # This ensures we always use the correct character voice even after switching
    current_audio_file = os.path.join(current_characters_folder, f"{current_character}.wav")
    
    # Fall back to the provided path if the current character file doesn't exist
    if not os.path.exists(current_audio_file):
        current_audio_file = audio_file_pth
        print(f"Warning: Using fallback audio file as {current_audio_file} not found")
    else:
        print(f"Using current character audio: {current_character}")
        
    if TTS_PROVIDER == 'openai':
        output_path = os.path.join(output_dir, 'output.wav')
        await openai_text_to_speech(prompt, output_path)
        # print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        else:
            print("Error: Audio file not found.")
    elif TTS_PROVIDER == 'elevenlabs':
        output_path = os.path.join(output_dir, 'output.mp3')
        await elevenlabs_text_to_speech(prompt, output_path)
        # print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            await send_message_to_clients(json.dumps({"action": "ai_start_speaking"}))
            await play_audio(output_path)
            await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
        else:
            print("Error: Audio file not found.")
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
                await play_audio(src_path)
                await send_message_to_clients(json.dumps({"action": "ai_stop_speaking"}))
            except Exception as e:
                print(f"Error during XTTS audio generation: {e}")
        else:
            print("XTTS model is not loaded. Please ensure initialization succeeded.")
    else:
        print(f"Unknown TTS provider: {TTS_PROVIDER}")


async def send_message_to_clients(message):
    for client in clients:
        await client.send_text(message)

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
            pcm_data = await fetch_pcm_audio("tts-1", OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL, session)
            save_pcm_as_wav(pcm_data, output_path)
        else:
            try:
                async with session.post(
                    url=OPENAI_TTS_URL,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "tts-1", "voice": OPENAI_TTS_VOICE, "input": prompt, "response_format": file_extension},
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

    async with aiohttp.ClientSession() as session:
        async with session.post(tts_url, headers=headers, json=data, timeout=30) as response:
            if response.status == 200:
                with open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                print("Audio stream saved successfully.")
            else:
                print("Error generating speech:", await response.text())

def sanitize_response(response):
    response = re.sub(r'\*.*?\*', '', response)
    response = re.sub(r'[^\w\s,.\'!?]', '', response)
    return response.strip()

def analyze_mood(user_input):
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity
    print(f"Sentiment polarity: {polarity}")

    flirty_keywords = [
        "flirt", "love", "crush", "charming", "amazing", "attractive",
        "cute", "sweet", "darling", "adorable", "alluring", "seductive"
    ]
    angry_keywords = [
        "angry", "furious", "mad", "annoyed", "pissed off", "irate",
        "enraged", "livid", "outraged", "frustrated", "infuriated"
    ]
    sad_keywords = [
        "sad", "depressed", "down", "unhappy", "crying", "miserable",
        "heartbroken", "sorrowful", "gloomy", "melancholy", "despondent"
    ]
    fearful_keywords = [
        "scared", "afraid", "fear", "terrified", "nervous", "anxious",
        "worried", "frightened", "alarmed", "panicked", "horrified"
    ]
    surprised_keywords = [
        "surprised", "amazed", "astonished", "shocked", "stunned",
        "flabbergasted", "astounded", "speechless", "startled"
    ]
    disgusted_keywords = [
        "disgusted", "revolted", "sick", "nauseated", "repulsed",
        "grossed out", "appalled", "offended", "detested"
    ]
    joyful_keywords = [
        "joyful", "happy", "elated", "glad", "delighted", "pleased",
        "cheerful", "content", "satisfied", "thrilled", "ecstatic"
    ]
    neutral_keywords = [
        "okay", "alright", "fine", "neutral", "so-so", "indifferent",
        "meh", "unremarkable", "average", "mediocre"
    ]

    if any(keyword in user_input.lower() for keyword in flirty_keywords):
        return "flirty"
    elif any(keyword in user_input.lower() for keyword in angry_keywords):
        return "angry"
    elif any(keyword in user_input.lower() for keyword in sad_keywords):
        return "sad"
    elif any(keyword in user_input.lower() for keyword in fearful_keywords):
        return "fearful"
    elif any(keyword in user_input.lower() for keyword in surprised_keywords):
        return "surprised"
    elif any(keyword in user_input.lower() for keyword in disgusted_keywords):
        return "disgusted"
    elif any(keyword in user_input.lower() for keyword in joyful_keywords) or polarity > 0.3:
        return "joyful"
    elif any(keyword in user_input.lower() for keyword in neutral_keywords):
        return "neutral"
    else:
        return "neutral"

def chatgpt_streamed(user_input, system_message, mood_prompt, conversation_history):
    full_response = ""
    print(f"Debug: chatgpt_streamed started. MODEL_PROVIDER: {MODEL_PROVIDER}")

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
            "stream": True
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
        payload = {"model": OPENAI_MODEL, "messages": messages, "stream": True}
        try:
            print(f"Debug: Sending request to OpenAI: {OPENAI_BASE_URL}")
            response = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, stream=True, timeout=30)
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

    print(f"Debug: chatgpt_streamed completed. Response length: {len(full_response)}")
    return full_response

def save_conversation_history(conversation_history):
    with open("conversation_history.txt", "w", encoding="utf-8") as file:
        for message in conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            file.write(f"{role}: {content}\n")

def transcribe_with_whisper(audio_file):
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
    
    # Determine the audio file format based on the TTS provider
    if TTS_PROVIDER == 'elevenlabs':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.mp3')  # Use mp3 for ElevenLabs
        max_char_length = 500  # Set a higher limit for ElevenLabs
    elif TTS_PROVIDER == 'openai':
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for OpenAI
        max_char_length = 500  # Set a higher limit for OpenAI
    else:
        temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')  # Use wav for XTTS
        max_char_length = 250  # Set a lower limit for XTTS

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
        payload = {"model": "tts-1", "voice": OPENAI_TTS_VOICE, "input": text, "response_format": "wav"}
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
    conversation_history = []
    base_system_message = open_file(character_prompt_file)
    
    quit_phrases = ["quit", "Quit", "Quit.", "Exit.", "exit", "Exit", "leave", "Leave."]
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
            if any(phrase.lower().rstrip('.') in words for phrase in quit_phrases):
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
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

            # Save conversation history after each message exchange
            save_conversation_history(conversation_history)

    except KeyboardInterrupt:
        print("Quitting the conversation...")

if __name__ == "__main__":
    asyncio.run(user_chatbot_conversation())

