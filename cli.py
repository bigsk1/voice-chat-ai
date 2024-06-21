# use python app.py to run cli version 

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
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
from textblob import TextBlob
from pathlib import Path
import re
import io

# Load environment variables
load_dotenv()

MODEL_PROVIDER = os.getenv('MODEL_PROVIDER')

CHARACTER_NAME = os.getenv('CHARACTER_NAME')

TTS_PROVIDER = os.getenv('TTS_PROVIDER')
OPENAI_TTS_URL = os.getenv('OPENAI_TTS_URL')
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', 'alloy')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')

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

# Set up the faster-whisper model
model_size = "medium.en"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Paths for character-specific files
project_dir = os.path.dirname(os.path.abspath(__file__))
characters_folder = os.path.join(project_dir, 'characters', CHARACTER_NAME)
character_prompt_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.txt")
character_audio_file = os.path.join(characters_folder, f"{CHARACTER_NAME}.wav")

# Load XTTS configuration
xtts_config_path = os.path.join(project_dir, "XTTS-v2", "config.json")
xtts_checkpoint_dir = os.path.join(project_dir, "XTTS-v2")

xtts_config = XttsConfig()
xtts_config.load_json(xtts_config_path)

# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir=xtts_checkpoint_dir, eval=True)
xtts_model.cuda()  # Move the model to GPU if available

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')
    # Create a PyAudio instance
    p = pyaudio.PyAudio()
    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Model and device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = os.path.join(project_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model provider: {MODEL_PROVIDER}")
print(f"Model: {OPENAI_MODEL if MODEL_PROVIDER == 'openai' else OLLAMA_MODEL}")
print(f"Character: {character_display_name}")
print(f"Text-to-Speech provider: {TTS_PROVIDER}")
print("To stop chatting say Quit, Leave or Exit. Say, what's on my screen, to have AI view screen. One moment please loading...")

# Function to synthesize speech using XTTS
def process_and_play(prompt, audio_file_pth):
    if TTS_PROVIDER == 'openai':
        output_path = os.path.join(output_dir, 'output.wav')  # or 'output.mp3' if you want MP3
        openai_text_to_speech(prompt, output_path)
        print(f"Generated audio file at: {output_path}")
        if os.path.exists(output_path):
            print("Playing generated audio...")
            play_audio(output_path)
        else:
            print("Error: Audio file not found.")
    else:
        tts_model = xtts_model
        try:
            # Use XTTS to synthesize speech
            outputs = tts_model.synthesize(
                prompt,  # Pass the prompt as a string directly
                xtts_config,
                speaker_wav=audio_file_pth,  # Pass the file path directly
                gpt_cond_len=24,
                temperature=0.2,
                language='en',
                speed=1.1  # Use the speed from .env
            )

            # Get the synthesized audio tensor from the dictionary
            synthesized_audio = outputs['wav']

            # Save the synthesized audio to the output path
            src_path = os.path.join(output_dir, 'output.wav')
            sample_rate = xtts_config.audio.sample_rate
            sf.write(src_path, synthesized_audio, sample_rate)

            print("Audio generated successfully with XTTS.")
            play_audio(src_path)
        except Exception as e:
            print(f"Error during XTTS audio generation: {e}")

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
        pcm_data = fetch_pcm_audio("tts-1", OPENAI_TTS_VOICE, prompt, OPENAI_TTS_URL)
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
                    "model": "tts-1",
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

def sanitize_response(response):
    # Remove asterisks and emojis
    response = re.sub(r'\*.*?\*', '', response)
    response = re.sub(r'[^\w\s,.\'!?]', '', response)
    return response.strip()

def analyze_mood(user_input):
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity
    print(f"Sentiment polarity: {polarity}")  # Debugging statement

    # Custom keyword lists for detecting specific emotions
    flirty_keywords = ["flirt", "love", "crush", "charming", "amazing", "attractive"]
    angry_keywords = ["angry", "furious", "mad", "annoyed", "pissed off"]
    sad_keywords = ["sad", "depressed", "down", "unhappy", "crying"]
    fearful_keywords = ["scared", "afraid", "fear", "terrified", "nervous"]
    surprised_keywords = ["surprised", "amazed", "astonished", "shocked"]
    disgusted_keywords = ["disgusted", "revolted", "sick", "nauseated"]
    joyful_keywords = ["joyful", "happy", "elated", "glad", "delighted"]
    neutral_keywords = ["okay", "alright", "fine", "neutral"]

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

    print(f"Detected mood: {mood}")
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
    temp_audio_path = os.path.join(output_dir, 'temp_audio.wav')

    image_path = take_screenshot(temp_image_path)
    response = analyze_image(image_path, question_prompt)
    text_response = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response received.')

    # Truncate response if too long for XTTS
    max_char_length = 350
    if len(text_response) > max_char_length:
        text_response = text_response[:max_char_length] + "..."

    print(text_response)

    generate_speech(text_response, temp_audio_path)
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
            # print(f"Response content: {response.content}")  # Debugging statement
            if response.status_code == 200:
                return {"choices": [{"message": {"content": response.json().get('response', 'No response received.')}}]}
            elif response.status_code == 404:
                return {"choices": [{"message": {"content": "The llava model is not available on this server."}}]}
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")  # Debugging statement
            return {"choices": [{"message": {"content": "Failed to process the image with the llava model."}}]}
    else:
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
            print(f"Request failed: {e}")  # Debugging statement
            return {"choices": [{"message": {"content": "Failed to process the image with the OpenAI model."}}]}


def generate_speech(text, temp_audio_path):
    if TTS_PROVIDER == 'openai':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": "tts-1",
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
    else:
        tts_model = xtts_model
        try:
            # Use XTTS to synthesize speech
            outputs = tts_model.synthesize(
                text,  # Pass the prompt as a string directly
                xtts_config,
                speaker_wav=character_audio_file,  # Pass the file path directly
                gpt_cond_len=24,
                temperature=0.2,
                language='en',
                speed=float(os.getenv('XTTS_SPEED', '1.1'))  # Use the speed from .env
            )
            # Get the synthesized audio tensor from the dictionary
            synthesized_audio = outputs['wav']
            sample_rate = xtts_config.audio.sample_rate
            sf.write(temp_audio_path, synthesized_audio, sample_rate)
            print("Audio generated successfully with XTTS.")
        except Exception as e:
            print(f"Error during XTTS audio generation: {e}")

def user_chatbot_conversation():
    conversation_history = []
    base_system_message = open_file(character_prompt_file)
    quit_phrases = ["quit", "Quit", "Quit.", "Exit.", "exit", "Leave."]
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
            os.remove(audio_file)  # Clean up the temporary audio file 
            print(CYAN + "You:", user_input + RESET_COLOR)
            if user_input.strip() in quit_phrases:
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
            if len(sanitized_response) > 400:  # Limit response length for audio generation
                sanitized_response = sanitized_response[:400] + "..."
            prompt2 = sanitized_response
            process_and_play(prompt2, character_audio_file)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
    except KeyboardInterrupt:
        print("Quitting the conversation...")

if __name__ == "__main__":
    user_chatbot_conversation()
