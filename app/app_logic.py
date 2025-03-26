import os
import asyncio
from threading import Thread
from fastapi import APIRouter
from .shared import clients, continue_conversation, conversation_history, get_current_character
from .app import (
    analyze_mood,
    chatgpt_streamed,
    sanitize_response,
    process_and_play,
    execute_screenshot_and_analyze,
    open_file,
    init_ollama_model,
    init_openai_model,
    init_xai_model,
    init_openai_tts_voice,
    init_elevenlabs_tts_voice,
    init_xtts_speed,
    init_set_tts,
    init_set_provider,
    save_conversation_history,
    send_message_to_clients,
)
from .transcription import transcribe_audio
import json
import logging
import requests

router = APIRouter()
characters_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "characters")

# Global variable to store the current transcription model
FASTER_WHISPER_LOCAL = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"
current_transcription_model = "gpt-4o-mini-transcribe"
use_local_whisper = FASTER_WHISPER_LOCAL  # Initialize based on environment

# Function to update the transcription model
def set_transcription_model(model_name):
    global current_transcription_model, use_local_whisper
    if model_name == "local_whisper":
        use_local_whisper = True
    else:
        current_transcription_model = model_name
        use_local_whisper = False
    print(f"Transcription set to: {'Local Whisper' if use_local_whisper else current_transcription_model}")
    return {"status": "success", "message": f"Transcription model set to: {'Local Whisper' if use_local_whisper else current_transcription_model}"}

async def record_audio_and_transcribe():
    """Record audio and transcribe it using the selected method"""
    
    # Create a custom callback that works with our clients set
    async def status_callback(status_data):
        message = json.dumps(status_data) if isinstance(status_data, dict) else status_data
        # Use the existing send_message_to_clients function from shared
        await send_message_to_clients(message)
        
    # Use our new unified transcription module
    user_input = await transcribe_audio(
        transcription_model=current_transcription_model,
        use_local=use_local_whisper,
        send_status_callback=status_callback
    )
    
    return user_input

# We can keep this as a utility function but it's not used directly with transcription
async def send_message_to_all_clients(message):
    for client_websocket in clients:
        try:
            await client_websocket.send_text(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")

async def process_text(user_input):
    current_character = get_current_character()
    character_folder = os.path.join('characters', current_character)
    character_prompt_file = os.path.join(character_folder, f"{current_character}.txt")
    character_audio_file = os.path.join(character_folder, f"{current_character}.wav")

    base_system_message = open_file(character_prompt_file)
    mood = analyze_mood(user_input)
    mood_prompt = adjust_prompt(mood)

    chatbot_response = chatgpt_streamed(user_input, base_system_message, mood_prompt, conversation_history)
    sanitized_response = sanitize_response(chatbot_response)
    if len(sanitized_response) > 400:  # Limit response length for audio generation
        sanitized_response = sanitized_response[:500] + "..."
    prompt2 = sanitized_response
    await process_and_play(prompt2, character_audio_file)

    conversation_history.append({"role": "assistant", "content": chatbot_response})
    save_conversation_history(conversation_history)
    return chatbot_response

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

@router.post("/start_conversation")
async def start_conversation():
    global continue_conversation
    continue_conversation = True
    conversation_thread = Thread(target=asyncio.run, args=(conversation_loop(),))
    conversation_thread.start()
    return {"message": "Conversation started"}

@router.post("/stop_conversation")
async def stop_conversation():
    global continue_conversation
    continue_conversation = False
    return {"message": "Conversation stopped"}

async def conversation_loop():
    global continue_conversation
    while continue_conversation:
        user_input = await record_audio_and_transcribe() 
        conversation_history.append({"role": "user", "content": user_input})
        save_conversation_history(conversation_history)
        await send_message_to_clients(f"You: {user_input}")
        print(f"You: {user_input}")

        # Check for quit phrases with word boundary check
        words = user_input.lower().split()
        if any(phrase.lower() in words for phrase in quit_phrases):
            print("Quitting the conversation...")
            await stop_conversation()
            break

        # Check for screenshot phrases - match only if the full phrase exists in input
        if any(phrase in user_input.lower() for phrase in screenshot_phrases):
            await execute_screenshot_and_analyze()
            continue

        try:
            chatbot_response = await process_text(user_input)
        except Exception as e:
            chatbot_response = f"An error occurred: {e}"
            print(chatbot_response)

        current_character = get_current_character()
        await send_message_to_clients(f"{current_character.capitalize()}: {chatbot_response}")
        # print(f"{current_character.capitalize()}: {chatbot_response}")

def set_env_variable(key: str, value: str):
    os.environ[key] = value
    if key == "OLLAMA_MODEL":
        init_ollama_model(value)  # Reinitialize Ollama model
    if key == "OPENAI_MODEL":
        init_openai_model(value)  # Reinitialize OpenAI model
    if key == "XAI_MODEL":
        init_xai_model(value)  # Reinitialize XAI model
    if key == "OPENAI_TTS_VOICE":
        init_openai_tts_voice(value)  # Reinitialize OpenAI TTS voice
    if key == "ELEVENLABS_TTS_VOICE":
        init_elevenlabs_tts_voice(value)  # Reinitialize Elevenlabs TTS voice
    if key == "XTTS_SPEED":
        init_xtts_speed(value)  # Reinitialize XTTS speed
    if key == "TTS_PROVIDER":
        init_set_tts(value)      # Reinitialize TTS Providers
    if key == "MODEL_PROVIDER":
        init_set_provider(value)  # Reinitialize Model Providers

def adjust_prompt(mood):
    """Load mood-specific prompts from the character's prompts.json file."""
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

    # The key issue: we only want to log the mood, not the entire prompt
    # Just print the mood name, not the full prompt text
    print(f"Detected mood: {mood}")
    
    # Get the mood prompt but don't print it in normal logging
    mood_prompt = mood_prompts.get(mood, "")
    
    # Debug output only if DEBUG is enabled
    if 'DEBUG' in locals() and DEBUG:
        print(f"Selected prompt for {current_character} ({mood}): {mood_prompt[:50]}...")
    
    return mood_prompt

async def fetch_ollama_models():
    """Fetch available models from Ollama API"""
    try:
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            # Extract just the model names from the response
            models = [model['name'] for model in models_data.get('models', [])]
            
            # If models list is empty (older Ollama versions format), try alternate path
            if not models and 'models' not in models_data:
                models = [model['name'] for model in models_data]
                
            return {"models": models}
        else:
            logging.warning(f"Failed to fetch Ollama models: {response.status_code}")
            return {"models": ["llama3.2"], "error": f"Failed to fetch models: {response.status_code}"}
    except Exception as e:
        logging.error(f"Error fetching Ollama models: {e}")
        return {"models": ["llama3.2"], "error": f"Error connecting to Ollama: {str(e)}"}

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