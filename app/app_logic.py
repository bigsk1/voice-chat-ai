import os
import asyncio
from threading import Thread
from fastapi import APIRouter
from pydantic import BaseModel
from .shared import clients, continue_conversation, conversation_history
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

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Define the CharacterModel
class CharacterModel(BaseModel):
    character: str

router = APIRouter()
characters_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "characters")

# Maximum character length for audio generation
MAX_CHAR_LENGTH = int(os.getenv('MAX_CHAR_LENGTH', 500))

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
    # Import with alias to avoid potential shadowing issues
    from .shared import get_current_character as get_character
    
    current_character = get_character()
    character_folder = os.path.join('characters', current_character)
    character_prompt_file = os.path.join(character_folder, f"{current_character}.txt")
    character_audio_file = os.path.join(character_folder, f"{current_character}.wav")

    base_system_message = open_file(character_prompt_file)
    mood = analyze_mood(user_input)
    mood_prompt = adjust_prompt(mood)

    chatbot_response = chatgpt_streamed(user_input, base_system_message, mood_prompt, conversation_history)
    sanitized_response = sanitize_response(chatbot_response)
    # Limit the response length to the MAX_CHAR_LENGTH for audio generation
    if len(sanitized_response) > MAX_CHAR_LENGTH:
        sanitized_response = sanitized_response[:MAX_CHAR_LENGTH] + "..."
    prompt2 = sanitized_response
    await process_and_play(prompt2, character_audio_file)

    conversation_history.append({"role": "assistant", "content": chatbot_response})
    
    # Check if this is a story or game character
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    
    if is_story_character:
        # Save to character-specific history file
        save_character_specific_history(conversation_history, current_character)
        print(f"Saved character-specific history for {current_character}")
    else:
        # Save to global history file
        save_conversation_history(conversation_history)
        print(f"Saved global history for {current_character}")
        
    return chatbot_response

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

@router.post("/start_conversation")
async def start_conversation():
    global continue_conversation # noqa: F824
    
    # Set flag to continue conversation
    continue_conversation = True
    
    # Import with alias to avoid potential shadowing issues
    from .shared import conversation_history, get_current_character as get_character, set_conversation_active
    
    # Get the current character
    current_character = get_character()
    print(f"Starting conversation with character: {current_character}")
    
    # Set conversation_active to True
    set_conversation_active(True)
    
    # Determine if character is story/game character
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    
    # Handle history based on character type
    if is_story_character:
        # For story/game characters: preserve existing history or load from character-specific file
        print(f"Using character-specific history for {current_character}")
        loaded_history = load_character_specific_history(current_character)
        if loaded_history:
            # Clear existing history and load from file
            conversation_history.clear()
            conversation_history.extend(loaded_history)
            print(f"Loaded {len(loaded_history)} messages from character-specific history")
        else:
            # If no history exists, make sure in-memory history is cleared too
            conversation_history.clear()
            print("No previous character-specific history found, starting fresh")
    else:
        # For standard characters: make sure we're starting with an empty history
        print(f"Clearing conversation history for standard character: {current_character}")
        conversation_history.clear()
        
        # Load history from file - only for existing global history
        history_file = "conversation_history.txt"
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            print("Loading history from global file")
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
            
            # Add loaded messages to conversation history
            conversation_history.extend(temp_history)
            print(f"Loaded {len(temp_history)} messages from global history")
    
    print(f"Starting conversation with character {current_character}, history size: {len(conversation_history)}")
    
    # Wait for speech
    # print("Waiting for speech...")
    await send_message_to_clients({"type": "waiting"})
    
    # Start conversation thread
    Thread(target=asyncio.run, args=(conversation_loop(),)).start()
    
    return {"status": "started"}

@router.post("/stop_conversation")
async def stop_conversation():
    global continue_conversation # noqa: F824
    continue_conversation = False
    return {"message": "Conversation stopped"}

async def conversation_loop():
    global continue_conversation # noqa: F824
    
    # Import with alias to avoid potential shadowing issues
    from .shared import get_current_character as get_character
    
    while continue_conversation:
        user_input = await record_audio_and_transcribe() 
        
        # Check if user_input is None and handle it
        if user_input is None:
            print("Warning: Received None input from transcription")
            continue
            
        conversation_history.append({"role": "user", "content": user_input})
        
        # Get current character to check if it's a story/game character
        current_character = get_character()
        is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
        
        # Save history based on character type
        if is_story_character:
            save_character_specific_history(conversation_history, current_character)
            print(f"Saved user input to character-specific history for {current_character}")
        else:
            save_conversation_history(conversation_history)
            # print(f"Saved user input to global history for {current_character}")
            
        await send_message_to_clients(f"You: {user_input}")
        print(CYAN + f"You: {user_input}" + RESET_COLOR)

        # Check for quit phrases with word boundary check
        words = user_input.lower().split()
        if any(phrase.lower().rstrip('.') == word for phrase in quit_phrases for word in words):
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

        current_character = get_character()
        await send_message_to_clients(chatbot_response)
        # await send_message_to_clients(f"{current_character.capitalize()}: {chatbot_response}") # to use for character names
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
    # Import with alias to avoid potential shadowing issues
    from .shared import get_current_character as get_character
    
    # Get the current character
    current_character = get_character()
    
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

def is_client_active(client):
    """Check if a client is still active"""
    # This is a placeholder - implement connection checking as needed
    return True

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
            print(f"Character file not found: {character_file_path}")
            # Return a default prompt for the assistant character
            return "You are a helpful AI assistant."
            
        with open(character_file_path, 'r', encoding='utf-8') as file:
            character_prompt = file.read()
            
        print(f"Loaded character prompt for {character_name}: {len(character_prompt)} chars")
        return character_prompt
    except Exception as e:
        print(f"Error loading character prompt: {e}")
        # Return a default prompt for the assistant character
        return "You are a helpful AI assistant."

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
            print(f"Not a story/game character: {character_name}, using global history instead")
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
        logging.error(f"Error loading character-specific history: {e}")
        return []

@router.post("/set_character")
async def set_api_character(character: CharacterModel):
    """Set the current character."""
    # Import with alias to avoid potential shadowing issues
    from .shared import set_current_character, get_current_character, conversation_history
    
    previous_character = get_current_character()
    new_character = character.character
    
    print(f"Switching character: {previous_character} -> {new_character}")
    
    # Always save the previous character's history if needed
    is_previous_story_character = previous_character and (
        previous_character.startswith("story_") or previous_character.startswith("game_")
    )
    
    if is_previous_story_character and conversation_history:
        # Save the current history to character-specific file before switching
        save_character_specific_history(conversation_history, previous_character)
        print(f"Saved history for previous character: {previous_character}")
    
    # Set the new character
    set_current_character(new_character)
    
    # Always clear the global history when switching characters
    conversation_history.clear()
    print(f"Cleared in-memory conversation history")
    
    # Delete the global history file and create a new empty one
    history_file = "conversation_history.txt"
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"Deleted global history file")
    
    # Create empty history file
    with open(history_file, "w", encoding="utf-8") as f:
        pass
    print(f"Created empty global history file")
    
    return {"status": "success", "message": f"Character set to {new_character}"}