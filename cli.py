# use python cli.py to run CLI version

import os
import torch
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
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
from textblob import TextBlob
from pathlib import Path
import re
import io
from pydub import AudioSegment

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
XAI_MODEL = os.getenv('XAI_MODEL', 'grok-2-1212')
XAI_BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_TTS_VOICE = os.getenv('ELEVENLABS_TTS_VOICE')
ELEVENLABS_TTS_MODEL = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_multilingual_v2')
ELEVENLABS_TTS_SPEED = os.getenv('ELEVENLABS_TTS_SPEED', '1')
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))
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
    print("Faster-Whisper initialization skipped (FASTER_WHISPER_LOCAL=false). Will use OpenAI API for transcription.")

# Paths for character-specific files
project_dir = os.path.dirname(os.path.abspath(__file__))
characters_folder = os.path.join(project_dir, 'characters', CHARACTER_NAME)
character_prompt_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.txt")
character_audio_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.wav")

# Initialize TTS model
tts = None
if TTS_PROVIDER == 'xtts':
    print("Initializing XTTS model (may download on first run)...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("XTTS model loaded successfully.")
    except Exception as e:
        print(f"Failed to load XTTS model: {e}")
        TTS_PROVIDER = 'openai'  # Fallback to OpenAI
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

# Model and device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = os.path.join(project_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model provider: {MODEL_PROVIDER}")
print(f"Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else XAI_MODEL if MODEL_PROVIDER == 'xai' else OLLAMA_MODEL}")
print(f"Character: {character_display_name}")
print(f"Text-to-Speech provider: {TTS_PROVIDER}")
print(f"Text-to-Speech model: {OPENAI_MODEL_TTS if TTS_PROVIDER == 'openai' else ELEVENLABS_TTS_MODEL if TTS_PROVIDER == 'elevenlabs' else 'local' if TTS_PROVIDER == 'xtts' else 'Unknown'}")
print("To stop chatting say Quit, Leave or Exit. Say, what's on my screen, to have AI view screen. One moment please loading...")

# Function to synthesize speech using XTTS
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
    elif TTS_PROVIDER == 'xtts':
        if tts is not None:
            try:
                wav = tts.tts(
                    text=prompt,
                    speaker_wav=audio_file_pth,  # For voice cloning
                    language="en",
                    speed=float(XTTS_SPEED)
                )
                src_path = os.path.join(output_dir, 'output.wav')
                sf.write(src_path, wav, tts.synthesizer.tts_config.audio["sample_rate"])
                print("Audio generated successfully with XTTS.")
                play_audio(src_path)
            except Exception as e:
                print(f"Error during XTTS audio generation: {e}")
        else:
            print("XTTS model is not loaded. Please ensure initialization succeeded.")

def save_pcm_as_wav(pcm_data: bytes, file_path: str, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2):
    """ Saves PCM data as a WAV file. """
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

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
            "speed": ELEVENLABS_TTS_SPEED
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

def sanitize_response(response):
    # Remove asterisks and emojis
    response = re.sub(r'\*.*?\*', '', response)
    response = re.sub(r'[^\w\s,.\'!?]', '', response)
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
                                print(NEON_GREEN + line + RESET_COLOR)
                                full_response += line + '\n'
                            line_buffer = lines[-1]
                except json.JSONDecodeError:
                    continue
        if line_buffer:
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
            "stream": True
        }
        response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        full_response = ""
        print("Starting XAI stream...")
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                line = line[5:].strip() 
            if line:
                try:
                    chunk = json.loads(line)
                    delta_content = chunk['choices'][0]['delta'].get('content', '')
                    if delta_content:
                        print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                        full_response += delta_content
                except json.JSONDecodeError:
                    continue
        print("\nXAI stream complete.")
        return full_response

    elif MODEL_PROVIDER == 'openai':
        messages = [{"role": "system", "content": system_message + "\n" + mood_prompt}] + conversation_history + [{"role": "user", "content": user_input}]
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "stream": True
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
                        print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                        full_response += delta_content
                except json.JSONDecodeError:
                    continue
        print("\nOpenAI stream complete.")
        return full_response

# Function to transcribe the recorded audio using faster-whisper
def transcribe_with_whisper(audio_file):
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
    print("Recording...")
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
            response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
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
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
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
    else:  # XTTS
        if tts is not None:
            try:
                wav = tts.tts(
                    text=text,
                    speaker_wav=character_audio_file,
                    language="en",
                    speed=float(XTTS_SPEED)
                )
                sf.write(temp_audio_path, wav, tts.synthesizer.tts_config.audio["sample_rate"])
                print("Audio generated successfully with XTTS.")
            except Exception as e:
                print(f"Error during XTTS audio generation: {e}")
        else:
            print("XTTS model is not loaded.")

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
    quit_phrases = ["quit", "Quit", "Quit.", "Exit.", "exit", "Exit", "leave", "Leave", "Leave."]
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
            
            print(PINK + f"{character_display_name}:..." + RESET_COLOR)
            chatbot_response = chatgpt_streamed(user_input, base_system_message, mood_prompt, conversation_history)
            conversation_history.append({"role": "assistant", "content": chatbot_response})
            sanitized_response = sanitize_response(chatbot_response)
            if len(sanitized_response) > MAX_CHAR_LENGTH:  # Limit response length for audio generation
                sanitized_response = sanitized_response[:MAX_CHAR_LENGTH] + "..."
            prompt2 = sanitized_response
            process_and_play(prompt2, character_audio_file)
            current_character = os.getenv('CHARACTER_NAME', 'wizard')
            if current_character.startswith("story_") or current_character.startswith("game_"):
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]
                # Save to character-specific history
                save_character_specific_history(conversation_history, current_character)
            else:
                if len(conversation_history) > 30:
                    conversation_history = conversation_history[-30:]
                # Save to global history file
                save_global_conversation_history(conversation_history)
    except KeyboardInterrupt:
        print("Quitting the conversation...")

if __name__ == "__main__":
    user_chatbot_conversation()
