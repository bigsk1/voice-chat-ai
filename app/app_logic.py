import os
import asyncio
from threading import Thread
from fastapi import APIRouter
from .shared import clients, continue_conversation, conversation_history, get_current_character, is_client_active, set_client_inactive
from .app import (
    transcribe_with_whisper,
    analyze_mood,
    chatgpt_streamed,
    sanitize_response,
    process_and_play,
    record_audio,
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
)
import json

router = APIRouter()
characters_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "characters")

async def record_audio_and_transcribe():
    audio_file = "temp_recording.wav"
    await record_audio(audio_file)
    user_input = transcribe_with_whisper(audio_file)
    os.remove(audio_file)  # Clean up the temporary audio file
    return user_input


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

async def send_message_to_clients(message: str):
    """Send a message to all connected clients, handling closed connections gracefully."""
    closed_clients = set()
    for client in clients:
        if is_client_active(client):
            try:
                await client.send_json({"message": message})
            except RuntimeError as e:
                if "websocket.send" in str(e) and "websocket.close" in str(e):
                    # WebSocket is already closed
                    print(f"Client connection already closed: {e}")
                    closed_clients.add(client)
                    set_client_inactive(client)
                else:
                    # Some other RuntimeError
                    print(f"Error sending message to client: {e}")
                    closed_clients.add(client)
                    set_client_inactive(client)
            except Exception as e:
                # Any other exception
                print(f"Error sending message to client: {e}")
                closed_clients.add(client)
                set_client_inactive(client)
    
    # No need to remove from clients set here - let the WebSocketDisconnect handler do it

async def conversation_loop():
    global continue_conversation
    while continue_conversation:
        user_input = await record_audio_and_transcribe() 
        conversation_history.append({"role": "user", "content": user_input})
        save_conversation_history(conversation_history)
        await send_message_to_clients(f"You: {user_input}")
        print(f"You: {user_input}")

        if any(phrase in user_input.strip() for phrase in quit_phrases):
            print("Quitting the conversation...")
            await stop_conversation()
            break

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