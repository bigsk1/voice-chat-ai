import os
import asyncio
import json
from threading import Thread
from fastapi import APIRouter
from .shared import clients, conversation_history, get_current_character, is_client_active, set_client_inactive
from .app_logic import (
    save_conversation_history,
    open_file,
    analyze_mood,
    adjust_prompt,
    sanitize_response,
    characters_folder
)
from .transcription import transcribe_audio

router = APIRouter()

# Enhanced-specific variables
enhanced_conversation_active = False
enhanced_conversation_thread = None
enhanced_speed = "1.0"
enhanced_voice = os.getenv("OPENAI_TTS_VOICE", "coral")
enhanced_model = os.getenv("OPENAI_MODEL", "gpt-4o")
enhanced_tts_model = os.getenv("OPENAI_MODEL_TTS", "gpt-4o-mini-tts")
enhanced_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")

# Debug flags to control verbose output
DEBUG_AUDIO_LEVELS = False  # Set to True to see audio level output in the console
DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # Control detailed debug output

# Quit phrases that will stop the conversation when detected
QUIT_PHRASES = ["quit", "exit", "leave", "end", "bye", "goodbye"]

async def send_message_to_enhanced_clients(message):
    """Send message to clients using the enhanced websocket."""
    for client in clients:
        if not is_client_active(client):
            continue
            
        try:
            # Check if we're dealing with a WebSocket object
            if hasattr(client, 'send_json') and hasattr(client, 'send_text'):
                if isinstance(message, str):
                    # If it's already a string, send as text
                    await client.send_text(message)
                else:
                    # Otherwise send as JSON
                    await client.send_json(message)
            # If the client doesn't have the expected methods, log an error
            else:
                print(f"Client {client} doesn't support expected WebSocket methods")
                set_client_inactive(client)
        except RuntimeError as e:
            if "websocket.send" in str(e) and "websocket.close" in str(e):
                # WebSocket is already closed
                print(f"Client connection already closed: {e}")
                set_client_inactive(client)
            else:
                # Some other RuntimeError
                print(f"RuntimeError sending message to client: {e}")
                set_client_inactive(client)
        except Exception as e:
            print(f"Error sending message to client: {e}")
            set_client_inactive(client)

async def record_enhanced_audio_and_transcribe():
    """Record audio and transcribe it using the enhanced transcription model."""
    return await transcribe_audio(
        transcription_model=enhanced_transcription_model,
        use_local=False,  # Enhanced always uses OpenAI API
        send_status_callback=send_message_to_enhanced_clients
    )

async def enhanced_text_to_speech(text, character_audio_file, detected_mood=None):
    """Convert text to speech using the enhanced TTS model with emotional voice instructions."""
    try:
        # Import required libraries
        import json
        import aiohttp
        import os
        import tempfile
        from pathlib import Path
        import asyncio
        import wave
        import pyaudio
        
        # Create a temporary file to store the speech audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("API key missing. Please set OPENAI_API_KEY in your environment.")
            return
        
        # Prepare parameters for the enhanced TTS
        voice = enhanced_voice
        speed = float(enhanced_speed)
        model = enhanced_tts_model
        
        # Get current character and load its prompt
        character_name = get_current_character()
        character_prompt_file = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        
        # Extract voice instructions from the character prompt file
        base_voice_instructions = {}
        try:
            if os.path.exists(character_prompt_file):
                with open(character_prompt_file, 'r', encoding='utf-8') as f:
                    character_content = f.read()
                    
                    # Look for VOICE INSTRUCTIONS section
                    if "VOICE INSTRUCTIONS:" in character_content:
                        voice_section = character_content.split("VOICE INSTRUCTIONS:")[1].strip()
                        # Extract up to the next major section (usually marked by ALL CAPS)
                        next_section = None
                        for line in voice_section.split('\n\n'):
                            if line and line == line.upper() and len(line) > 10:
                                next_section = line
                                break
                                
                        if next_section:
                            voice_section = voice_section.split(next_section)[0].strip()
                        
                        # Parse the structured instructions
                        current_key = None
                        for line in voice_section.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                                
                            if ":" in line and line.split(":")[0].strip() in ["Voice", "Pacing", "Pronunciation", 
                                                                            "Emotion", "Inflection", "Word Choice"]:
                                current_key = line.split(":")[0].strip()
                                base_voice_instructions[current_key] = line.split(":", 1)[1].strip()
                            elif current_key:
                                # Continuation of previous key
                                base_voice_instructions[current_key] += " " + line
        except Exception as e:
            print(f"Error parsing voice instructions: {e}")
        
        # Get character-specific mood prompt from the character's prompts.json
        mood_voice_instructions = {}
        try:
            if detected_mood:
                character_prompts_path = os.path.join(characters_folder, character_name, 'prompts.json')
                if os.path.exists(character_prompts_path):
                    with open(character_prompts_path, 'r', encoding='utf-8') as f:
                        mood_prompts = json.load(f)
                        mood_prompt = mood_prompts.get(detected_mood, "")
                        
                        # Parse the structured mood-specific instructions
                        for segment in mood_prompt.split(". "):
                            if "Voice:" in segment:
                                mood_voice_instructions["Voice"] = segment.split("Voice:")[1].strip()
                            elif "Pacing:" in segment:
                                mood_voice_instructions["Pacing"] = segment.split("Pacing:")[1].strip()
                            elif "Emotion:" in segment:
                                mood_voice_instructions["Emotion"] = segment.split("Emotion:")[1].strip()
                            elif "Inflection:" in segment:
                                mood_voice_instructions["Inflection"] = segment.split("Inflection:")[1].strip()
                            elif "Pronunciation:" in segment:
                                mood_voice_instructions["Pronunciation"] = segment.split("Pronunciation:")[1].strip()
        except Exception as e:
            print(f"Error parsing mood instructions: {e}")
        
        # Build the final structured voice instructions for the TTS model
        structured_instructions = []
        
        # Add character identity
        character_display_name = character_name.replace('_', ' ')
        structured_instructions.append(f"Character: {character_display_name}")
        
        # Priority: mood-specific instructions override base instructions
        for key in ["Voice", "Pacing", "Pronunciation", "Emotion", "Inflection"]:
            if key in mood_voice_instructions:
                structured_instructions.append(f"{key}: {mood_voice_instructions[key]}")
            elif key in base_voice_instructions:
                structured_instructions.append(f"{key}: {base_voice_instructions[key]}")
        
        # Add mood context
        if detected_mood:
            structured_instructions.append(f"Current Mood: {detected_mood}")
            
        # Format the final instructions in the OpenAI-preferred format
        voice_instructions = "\n\n".join(structured_instructions)
            
        # Log TTS parameters - only if DEBUG is enabled
        if DEBUG:
            print(f"TTS model: {model} | Voice: {voice} | Speed: {speed}")
        
        # Notify that AI is about to speak
        await send_message_to_enhanced_clients({"action": "ai_start_speaking"})
        
        # Make the actual API call to OpenAI for TTS
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav"  # Changed from mp3 to wav for lower latency
        }
        
        # Add voice instructions for gpt-4o-mini-tts
        if model == "gpt-4o-mini-tts" and voice_instructions:
            payload["voice_instructions"] = voice_instructions
            
            # Debug - print instructions only if DEBUG is enabled
            if DEBUG and detected_mood:
                print(f"Voice instructions for {detected_mood} mood:")
                print(voice_instructions)
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # Save the audio file
                    with open(temp_filename, "wb") as f:
                        f.write(await response.read())
                    
                    # Signal that audio is about to play (for animation synchronization)
                    await send_message_to_enhanced_clients({"action": "audio_actually_playing"})
                    
                    # Play audio using PyAudio - similar to the main page implementation
                    try:
                        print("Starting audio playback")
                        wf = wave.open(temp_filename, 'rb')
                        p = pyaudio.PyAudio()
                        
                        # Set up a buffered stream for lower latency
                        # Use a smaller buffer size for quicker start (512 instead of 1024)
                        buffer_size = 512
                        
                        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                        channels=wf.getnchannels(),
                                        rate=wf.getframerate(),
                                        output=True,
                                        frames_per_buffer=buffer_size)
                        
                        # Read initial data to start quickly
                        data = wf.readframes(buffer_size)
                        
                        # Stream the audio data
                        while data and len(data) > 0:
                            stream.write(data)
                            data = wf.readframes(buffer_size)
                            
                        # Clean up resources
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        wf.close()  # Close the wave file properly
                        print("Finished audio playback")
                        
                    except Exception as e:
                        print(f"Error playing audio: {e}")
                        # Just wait an estimated amount of time if playback fails
                        estimated_duration = len(text) * 0.08
                        await asyncio.sleep(estimated_duration)
                    
                else:
                    error_text = await response.text()
                    print(f"Error from OpenAI TTS API: {error_text}")
                    raise Exception(f"TTS API error: {response.status} - {error_text}")
        
        # Clean up the temporary file
        try:
            # Add a small delay to ensure file is released before deletion (Windows issue)
            await asyncio.sleep(0.2)
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except Exception as e:
            print(f"Error removing temporary TTS file: {e}")
            # Non-critical error, we can continue even if file cleanup fails
        
        # Signal that AI has stopped speaking
        await send_message_to_enhanced_clients({"action": "ai_stop_speaking"})
        
    except Exception as e:
        print(f"Error in enhanced text-to-speech: {e}")
        await send_message_to_enhanced_clients({
            "action": "error",
            "message": f"TTS Error: {str(e)}"
        })
        # Make sure to stop speaking in case of error
        await send_message_to_enhanced_clients({"action": "ai_stop_speaking"})

async def enhanced_conversation_loop():
    """Main conversation loop for the enhanced interface."""
    global enhanced_conversation_active, conversation_history
    try:
        # Keep context of the conversation
        local_conversation_history = []
        
        # Check if there's any history to load - only load if there's actual content
        if conversation_history and len(conversation_history) > 0:
            print(f"Loading {len(conversation_history)} previous messages from conversation history")
            local_conversation_history = conversation_history.copy()
            
            # Display the conversation history in the UI
            for msg in local_conversation_history:
                if msg["role"] == "user":
                    await send_message_to_enhanced_clients({
                        "message": f"You: {msg['content']}"
                    })
                elif msg["role"] == "assistant":
                    await send_message_to_enhanced_clients({
                        "message": msg['content']
                    })
        else:
            print("Starting with empty conversation history")
        
        character_name = get_current_character()
        character_prompt_file = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        character_audio_file = os.path.join(characters_folder, character_name, f"{character_name}.wav")
        
        base_system_message = open_file(character_prompt_file)
        
        # Don't automatically show a greeting message
        # Instead, show character selection confirmation similar to main page
        await send_message_to_enhanced_clients({"message": f"Character: {character_name.replace('_', ' ')}", "type": "system-message"})
        
        while enhanced_conversation_active:
            # Listen for user input
            try:
                # Record audio and convert to text
                user_input = await record_enhanced_audio_and_transcribe()
                
                # Check if recording was successful
                if not user_input or user_input.strip() == "":
                    await send_message_to_enhanced_clients({
                        "action": "error", 
                        "message": "Sorry, I couldn't hear what you said. Please try again."
                    })
                    continue
                    
                # Check for quit phrases with word boundary check
                words = user_input.lower().split()
                if any(phrase.lower() in words for phrase in QUIT_PHRASES):
                    await send_message_to_enhanced_clients({
                        "message": "Goodbye! Ending our conversation now."
                    })
                    break
                    
                # Process user input - analyze mood, etc.
                sentiment_score = analyze_mood(user_input)
                detected_mood = adjust_prompt(sentiment_score)
                                
                # Send user message to UI
                await send_message_to_enhanced_clients({
                    "message": f"You: {user_input}"
                })
                
                # Add to local history
                local_conversation_history.append({"role": "user", "content": user_input})
                
                # Always print basic turn info but keep it minimal
                print(f"User: \"{user_input[:40]}{'...' if len(user_input) > 40 else ''}\"")
                
                # Get mood-specific prompt from character's prompts.json - but without logging
                mood_prompt = ""
                character_prompts_path = os.path.join(characters_folder, character_name, 'prompts.json')
                
                try:
                    if os.path.exists(character_prompts_path):
                        with open(character_prompts_path, 'r', encoding='utf-8') as f:
                            mood_prompts = json.load(f)
                            mood_prompt = mood_prompts.get(detected_mood, "")
                            
                            # Only print detailed prompt info if DEBUG is enabled
                            if DEBUG:
                                print(f"Loaded mood prompts for character: {character_name}")
                                print(f"Selected prompt for {character_name} ({detected_mood}): {mood_prompt[:50]}...")
                except Exception as e:
                    if DEBUG:
                        print(f"Error loading character prompts: {str(e)}")
                    
                # Get response from LLM
                ai_response = await enhanced_chat_completion(
                    user_input, 
                    base_system_message, 
                    mood_prompt,
                    local_conversation_history[:-1] if len(local_conversation_history) > 1 else None,
                    detected_mood
                )
                
                # Clean up the response
                ai_response = sanitize_response(ai_response)
                
                # Add to conversation history
                local_conversation_history.append({"role": "assistant", "content": ai_response})
                
                # Manage conversation history size - keep last 20 messages
                if len(local_conversation_history) > 20:
                    local_conversation_history = local_conversation_history[-20:]
                
                # Update global conversation history
                conversation_history = local_conversation_history.copy()
                
                # Optionally, save the conversation history to a file
                save_conversation_history(conversation_history)
                
                # Print AI response summary
                print(f"AI: \"{ai_response[:40]}{'...' if len(ai_response) > 40 else ''}\"")
                
                # Convert to speech - this will play the audio first
                await enhanced_text_to_speech(ai_response, character_audio_file, detected_mood)
                
                # Send AI response to client after audio finishes playing
                await send_message_to_enhanced_clients({"message": ai_response})
                
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                # Attempt to continue anyway
                await send_message_to_enhanced_clients({
                    "action": "error", 
                    "message": f"Error: {str(e)}. Please try again."
                })
                
                # Optional: take a short break to avoid rapid errors
                await asyncio.sleep(1)
                
    except Exception as e:
        print(f"Fatal error in enhanced conversation loop: {e}")
    finally:
        # Make sure to clean up
        enhanced_conversation_active = False
        print("Enhanced conversation thread stopping...")

async def enhanced_chat_completion(prompt, system_message, mood_prompt, conversation_history=None, detected_mood=None):
    """Get chat completion from OpenAI using the specified model."""
    try:
        # Import required libraries
        import json
        import aiohttp
        import os
        
        # Get API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "API key missing. Please set OPENAI_API_KEY in your environment."
        
        # Use the selected model
        model = enhanced_model
        
        # Combine the system message with mood prompt from sentiment analysis
        combined_system_message = f"{system_message}\n{mood_prompt}"
        
        # Print streamlined debug information only if DEBUG is enabled
        if DEBUG:
            print(f"Chat model: {model} | System prompt length: {len(combined_system_message)} chars")
        
        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": combined_system_message}
        ]
        
        # If conversation history is provided, add it to the messages
        if conversation_history:
            # Find name-related information in conversation history
            name_info = ""
            for msg in conversation_history:
                if msg["role"] == "user" and ("my name is" in msg["content"].lower() or "i am" in msg["content"].lower() or "i'm" in msg["content"].lower()):
                    name_info = msg["content"]
                    break
            
            # If name information was found, add it at the beginning for context
            if name_info:
                # Add a summary note about user identity 
                identity_note = {"role": "system", "content": f"Note: The user previously shared: \"{name_info}\". Remember this context."}
                messages.insert(1, identity_note)
                
                # Debug log about user identity if DEBUG is enabled
                if DEBUG:
                    print(f"Added user identity context: {name_info[:40]}...")
            
            # Now add all the conversation history
            messages.extend(conversation_history)
            
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Set up the API request
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 2000
        }
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    try:
                        response_text = data['choices'][0]['message']['content']
                        return response_text
                    except (KeyError, IndexError) as e:
                        print(f"Unexpected API response structure: {e}")
                        return f"Error parsing API response: {str(e)}"
                else:
                    error_text = await response.text()
                    print(f"Error from OpenAI Chat API: {error_text}")
                    return f"Chat API error: {response.status} - {error_text}"
        
    except Exception as e:
        print(f"Error in enhanced chat completion: {e}")
        return f"Error: {str(e)}"

async def start_enhanced_conversation(character=None, speed=None, model=None, voice=None, ttsModel=None, transcriptionModel=None):
    """Start a new enhanced conversation."""
    global enhanced_conversation_active, enhanced_conversation_thread, enhanced_speed, enhanced_voice, enhanced_model, enhanced_tts_model, enhanced_transcription_model, conversation_history
    
    # Set character if provided
    if character:
        from .shared import set_current_character
        set_current_character(character)
    
    # Set speed if provided
    if speed:
        enhanced_speed = speed
        
    # Set model if provided
    if model:
        enhanced_model = model
        
    # Set voice if provided
    if voice:
        enhanced_voice = voice
        
    # Set TTS model if provided
    if ttsModel:
        enhanced_tts_model = ttsModel
        
    # Set transcription model if provided
    if transcriptionModel:
        enhanced_transcription_model = transcriptionModel
    
    # Check if already running
    if enhanced_conversation_active:
        print("Enhanced conversation already running")
        return {"status": "already_running"}
    
    # IMPORTANT FIX: Ensure history is properly reset
    # Clear the in-memory history completely
    conversation_history.clear()
    
    # Reload conversation history from file to ensure it's in sync
    try:
        history_file = "conversation_history.txt"
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            # File exists and has content, load it
            temp_history = []
            with open(history_file, "r", encoding="utf-8") as file:
                current_role = None
                current_content = ""
                
                for line in file:
                    line = line.strip()
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
                    
            # Set the global conversation history
            conversation_history = temp_history
            print(f"Loaded {len(conversation_history)} messages from history file")
        else:
            # File doesn't exist or is empty
            print("No history found or empty history file, starting with empty conversation")
            conversation_history = []
    except Exception as e:
        print(f"Error loading conversation history from file: {e}")
        # Safety fallback - ensure history is empty if loading fails
        conversation_history = []
    
    # Set active flag
    enhanced_conversation_active = True
    
    # Start the conversation in a separate thread
    enhanced_conversation_thread = Thread(target=asyncio.run, args=(enhanced_conversation_loop(),))
    enhanced_conversation_thread.start()
    
    return {"status": "started"}

async def stop_enhanced_conversation():
    """Stop the currently running enhanced conversation."""
    global enhanced_conversation_active, enhanced_conversation_thread
    
    # Set the flag to stop the conversation loop
    enhanced_conversation_active = False
    
    # Send a message to clients that conversation is stopping
    try:
        await send_message_to_enhanced_clients({
            "message": "Stopping conversation...",
            "action": "conversation_stopped"
        })
    except Exception as e:
        print(f"Error sending stop message: {e}")
    
    if enhanced_conversation_thread:
        try:
            # Give it a moment to clean up
            await asyncio.sleep(0.5)
            # Thread can't be cancelled like a Task, it will stop on its own when enhanced_conversation_active is set to False
            print("Enhanced conversation thread stopping...")
        except Exception as e:
            print(f"Error stopping enhanced conversation: {e}")
    
    # Clear the thread reference
    enhanced_conversation_thread = None
    return {"status": "stopped"} 