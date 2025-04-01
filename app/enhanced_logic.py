import os
import asyncio
import json
from threading import Thread
from fastapi import APIRouter
from .shared import clients, conversation_history, is_client_active, set_client_inactive
from .app_logic import (
    save_conversation_history,
    open_file,
    analyze_mood,
    sanitize_response,
    characters_folder,
    load_character_specific_history,
    save_character_specific_history
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
QUIT_PHRASES = ["quit", "exit"]

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
            if DEBUG:
                print(f"Character file not found: {character_file_path}")
            return None
            
        with open(character_file_path, 'r', encoding='utf-8') as file:
            character_prompt = file.read()
            
        if DEBUG:
            print(f"Loaded character prompt for {character_name}: {len(character_prompt)} chars")
            
        return character_prompt
    except Exception as e:
        print(f"Error loading character prompt: {e}")
        return None

async def send_message_to_enhanced_clients(message):
    """Send message to clients using the enhanced websocket."""
    client_count = 0
    active_count = 0
    
    for client in clients:
        client_count += 1
        if not is_client_active(client):
            continue
            
        active_count += 1
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
    
    # Only log client counts when there's a problem
    if active_count == 0 and client_count > 0:
        print(f"Warning: No active clients to receive message ({client_count} total clients)")

async def record_enhanced_audio_and_transcribe():
    """Record audio and transcribe it using the enhanced transcription model."""
    return await transcribe_audio(
        transcription_model=enhanced_transcription_model,
        use_local=False,  # Enhanced always uses OpenAI API
        send_status_callback=send_message_to_enhanced_clients
    )

async def enhanced_text_to_speech(text, detected_mood=None):
    """Convert text to speech using the enhanced TTS model with emotional voice instructions."""
    try:
        # Import required libraries
        import json
        import aiohttp
        import os
        import asyncio
        import wave
        import pyaudio
        
        # Import get_current_character at the top to avoid shadowing
        from .shared import get_current_character as get_character
        
        # Use a persistent file in the outputs directory instead of a temporary file
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)  # Ensure the outputs directory exists
        
        # Create a persistent file path for enhanced audio
        enhanced_audio_filename = os.path.join(outputs_dir, "output_enhanced.wav")
        
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
        character_name = get_character()
        character_prompt_file = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        
        # Extract voice instructions from the character prompt file
        base_voice_instructions = {}
        try:
            if os.path.exists(character_prompt_file):
                with open(character_prompt_file, 'r', encoding='utf-8') as f:
                    character_content = f.read()
                    
                    # Debug - print the content being parsed
                    if DEBUG:
                        print("\nParsing character file:")
                        print(f"File: {character_prompt_file}")
                        print(f"Content length: {len(character_content)} chars")
                    
                    # Look for VOICE INSTRUCTIONS section
                    if "VOICE INSTRUCTIONS:" in character_content:
                        voice_section = character_content.split("VOICE INSTRUCTIONS:")[1].strip()
                        if DEBUG:
                            print("\nFound VOICE INSTRUCTIONS section:")
                            print("-------------------")
                            print(voice_section[:200] + "..." if len(voice_section) > 200 else voice_section)
                            print("-------------------\n")
                            
                        # Extract up to the next major section (usually marked by ALL CAPS)
                        next_section = None
                        for line in voice_section.split('\n\n'):
                            if line and line == line.upper() and len(line) > 10:
                                next_section = line
                                if DEBUG:
                                    print(f"Found next section marker: {next_section}")
                                break
                                
                        if next_section:
                            voice_section = voice_section.split(next_section)[0].strip()
                            
                        # Parse the structured instructions
                        current_key = None
                        unrecognized_instructions = []
                        
                        for line in voice_section.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Remove leading hyphen if present
                            if line.startswith('- '):
                                line = line[2:]
                                
                            # Map "Voice Quality" to "Voice"
                            if line.startswith("Voice Quality:"):
                                line = "Voice:" + line.split(":", 1)[1]
                                
                            if ":" in line:
                                # Extract the key (category) before the colon
                                potential_key = line.split(":")[0].strip()
                                # Map common variations to standard categories
                                category_mapping = {
                                    "Voice Quality": "Voice",
                                    "Voice": "Voice",
                                    "Voice Type": "Voice",
                                    "Voice Style": "Voice",
                                    "Pacing": "Pacing",
                                    "Speed": "Pacing",
                                    "Tempo": "Pacing",
                                    "Pronunciation": "Pronunciation",
                                    "Articulation": "Pronunciation",
                                    "Delivery": "Delivery",
                                    "Speaking Style": "Delivery",
                                    "Tone": "Tone",
                                    "Tone Quality": "Tone",
                                    "Mood Tone": "Tone",
                                    "Inflection": "Inflection",
                                    "Pitch Pattern": "Inflection",
                                    "Word Choice": "Word Choice",
                                    "Vocabulary": "Word Choice",
                                    "Language": "Word Choice",
                                    "Emphasis": "Emphasis",
                                    "Stress": "Emphasis",
                                    "Focus": "Emphasis",
                                    "Pauses": "Pauses",
                                    "Breaks": "Pauses",
                                    "Timing": "Pauses",
                                    "Emotion": "Emotion",
                                    "Emotional Tone": "Emotion",
                                    "Feeling": "Emotion"
                                }
                                
                                if potential_key in category_mapping:
                                    current_key = category_mapping[potential_key]
                                    base_voice_instructions[current_key] = line.split(":", 1)[1].strip()
                                    if DEBUG:
                                        print(f"Found base instruction - {current_key}: {base_voice_instructions[current_key][:50]}...")
                                else:
                                    # Store unrecognized instruction
                                    instruction_content = line.split(":", 1)[1].strip()
                                    if instruction_content:  # Only add if there's actual content
                                        unrecognized_instructions.append(f"{potential_key}: {instruction_content}")
                                        if DEBUG:
                                            print(f"Found unrecognized instruction - {potential_key}: {instruction_content[:50]}...")
                            elif current_key:
                                # Continuation of previous key
                                base_voice_instructions[current_key] += " " + line
                                if DEBUG:
                                    print(f"Added continuation to {current_key}")
                    else:
                        if DEBUG:
                            print("No VOICE INSTRUCTIONS section found in character file")
        except Exception as e:
            print(f"Error parsing voice instructions: {e}")
            if DEBUG:
                print(f"Character content that failed to parse:")
                print(character_content[:200] + "..." if len(character_content) > 200 else character_content)
        
        # Get character-specific mood prompt from the character's prompts.json
        mood_voice_instructions = {}
        try:
            if detected_mood:
                character_prompts_path = os.path.join(characters_folder, character_name, 'prompts.json')
                if os.path.exists(character_prompts_path):
                    with open(character_prompts_path, 'r', encoding='utf-8') as f:
                        mood_prompts = json.load(f)
                        mood_prompt = mood_prompts.get(detected_mood, "")
                        
                        if DEBUG:
                            print(f"\nParsing mood prompt for {detected_mood}:")
                            print("-------------------")
                            print(mood_prompt[:200] + "..." if len(mood_prompt) > 200 else mood_prompt)
                            print("-------------------\n")
                        
                        # First, split by periods that are followed by a voice instruction category
                        instruction_parts = []
                        current_part = ""
                        
                        # Split the mood prompt into parts while preserving the structure
                        for char in mood_prompt:
                            current_part += char
                            if char == '.' and len(current_part) > 2:
                                # Look ahead for voice instruction categories
                                next_words = mood_prompt[len(current_part):].strip()
                                if any(next_words.startswith(category + ":") for category in [
                                    "Voice", "Pacing", "Pronunciation", "Delivery",
                                    "Tone", "Inflection", "Word Choice", "Emphasis",
                                    "Pauses", "Emotion"
                                ]):
                                    if DEBUG:
                                        print(f"Found instruction part: {current_part.strip()}")
                                    instruction_parts.append(current_part.strip())
                                    current_part = ""
                        
                        # Add the last part if there is one
                        if current_part.strip():
                            if DEBUG:
                                print(f"Adding final part: {current_part.strip()}")
                            instruction_parts.append(current_part.strip())
                        
                        # Now parse each instruction part
                        for part in instruction_parts:
                            for category in [
                                "Voice", "Pacing", "Pronunciation", "Delivery",
                                "Tone", "Inflection", "Word Choice", "Emphasis",
                                "Pauses", "Emotion"
                            ]:
                                if f"{category}:" in part:
                                    value = part.split(f"{category}:")[1].strip()
                                    # Remove any trailing periods or other categories
                                    value = value.split(".")[0].strip()
                                    mood_voice_instructions[category] = value
                                    if DEBUG:
                                        print(f"Found mood instruction - {category}: {value[:50]}...")
                else:
                    if DEBUG:
                        print(f"No prompts.json found at {character_prompts_path}")
        except Exception as e:
            print(f"Error parsing mood instructions: {e}")
            if DEBUG:
                print(f"Mood prompt that failed to parse: {mood_prompt}")
        
        # Build the final structured voice instructions for the TTS model
        structured_instructions = []
        
        # Add character identity
        character_display_name = character_name.replace('_', ' ')
        structured_instructions.append(f"Character: {character_display_name}")
        
        if DEBUG:
            print("\nCompiling final voice instructions:")
            print(f"Base instructions found: {list(base_voice_instructions.keys())}")
            print(f"Mood instructions found: {list(mood_voice_instructions.keys())}")
        
        # Define all possible voice instruction categories
        voice_categories = [
            "Voice", "Pacing", "Pronunciation", "Delivery",
            "Tone", "Inflection", "Word Choice", "Emphasis",
            "Pauses", "Emotion"
        ]
        
        # Priority: mood-specific instructions override base instructions
        for key in voice_categories:
            if key in mood_voice_instructions:
                structured_instructions.append(f"{key}: {mood_voice_instructions[key]}")
                if DEBUG:
                    print(f"Using mood-specific {key} instruction")
            elif key in base_voice_instructions:
                structured_instructions.append(f"{key}: {base_voice_instructions[key]}")
                if DEBUG:
                    print(f"Using base {key} instruction")
            elif DEBUG:
                print(f"No instruction found for {key}")
                
        # Add any unrecognized instructions at the end
        if "Additional" in base_voice_instructions:
            structured_instructions.append("Additional Instructions:")
            structured_instructions.append(base_voice_instructions["Additional"])
            if DEBUG:
                print("Added unrecognized instructions to final output")
        
        # Add mood context
        if detected_mood and mood_prompt:
            structured_instructions.append(f"Current Mood: {detected_mood}")
            structured_instructions.append(f"Mood Context: {mood_prompt}")
            
        # Format the final instructions in the OpenAI-preferred format
        voice_instructions = "\n\n".join(structured_instructions)
            
        # Log TTS parameters and instructions if DEBUG is enabled
        if DEBUG:
            print(f"\nTTS Configuration:")
            print(f"Model: {model} | Voice: {voice} | Speed: {speed}")
            print("\nVoice Instructions:")
            print("-------------------")
            print(voice_instructions)
            print("-------------------\n")
        
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
            if DEBUG:
                print(f"Voice instructions being sent to TTS model:")
                print(voice_instructions)
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # Save the audio file
                    with open(enhanced_audio_filename, "wb") as f:
                        f.write(await response.read())
                    
                    # Signal that audio is about to play (for animation synchronization)
                    await send_message_to_enhanced_clients({"action": "audio_actually_playing"})
                    
                    # Play audio using PyAudio - similar to the main page implementation
                    try:
                        wf = wave.open(enhanced_audio_filename, 'rb')
                        p = pyaudio.PyAudio()
                        
                        # Set up a buffered stream for lower latency
                        # Use a smaller buffer size for quicker start (512 instead of 1024)
                        buffer_size = 1024
                        
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
                        
                    except Exception as e:
                        print(f"Error playing audio: {e}")
                        # Just wait an estimated amount of time if playback fails
                        estimated_duration = len(text) * 0.08
                        await asyncio.sleep(estimated_duration)
                    
                else:
                    error_text = await response.text()
                    print(f"Error from OpenAI TTS API: {error_text}")
                    raise Exception(f"TTS API error: {response.status} - {error_text}")
        
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

    # Import get_current_character at the top level to avoid shadowing
    from .shared import get_current_character as get_character

    try:
        # Keep context of the conversation
        local_conversation_history = []
        
        # Check if there's any history to load - only load if there's actual content
        if conversation_history and len(conversation_history) > 0:
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
        
        character_name = get_character()
        character_prompt_file = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        # character_audio_file = os.path.join(characters_folder, character_name, f"{character_name}.wav")
        
        base_system_message = open_file(character_prompt_file)
        
        # Don't automatically show a greeting message
        # Instead, show character selection confirmation similar to main page
        await send_message_to_enhanced_clients({"message": f"Character: {character_name.replace('_', ' ')}", "type": "system-message"})
        
        # Check if this is a story/game character for special handling
        is_story_character = character_name.startswith("story_") or character_name.startswith("game_")
        
        # Function to save history based on character type
        async def save_history():
            if is_story_character:
                # Save to character-specific history file for story/game characters
                # Ensure the character directory exists
                character_dir = os.path.join(characters_folder, character_name)
                os.makedirs(character_dir, exist_ok=True)
                
                save_character_specific_history(local_conversation_history, character_name)
            else:
                # Save to global history file for standard characters
                save_conversation_history(local_conversation_history)
                
        # Main conversation loop
        while enhanced_conversation_active:
            try:
                # Update client status
                await send_message_to_enhanced_clients({"action": "status", "status": "ready"})
                
                # Show listening animation
                await send_message_to_enhanced_clients({"action": "mic", "status": "listening"})
                
                # Wait for user to speak
                user_input = await record_enhanced_audio_and_transcribe()
                
                # Check if there was an error in transcription
                if not user_input or user_input == "ERROR":
                    print("Error transcribing audio or no speech detected")
                    await send_message_to_enhanced_clients({"action": "mic", "status": "off"})
                    await send_message_to_enhanced_clients({"message": "No speech detected or there was an error. Please try again.", "type": "system-message"})
                    continue
                
                # Print user input to CLI with "You:" prefix
                print("\033[96mYou:", user_input + "\033[0m")  # Cyan text for user messages
                
                # Display user message
                await send_message_to_enhanced_clients({"message": f"You: {user_input}"})
                await send_message_to_enhanced_clients({"action": "mic", "status": "processing"})
                
                # Check for quit commands
                words = user_input.lower().split()
                if any(phrase.lower().rstrip('.') == word for phrase in QUIT_PHRASES for word in words):
                    await send_message_to_enhanced_clients({"message": "Conversation ended.", "type": "system-message"})
                    break
                
                # Add user input to conversation history
                local_conversation_history.append({"role": "user", "content": user_input})
                
                # Detect user mood from input (if enabled)
                detected_mood = analyze_mood(user_input)
                
                # If mood is detected, try to load a custom prompt for that mood
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
                    local_conversation_history[:-1] if len(local_conversation_history) > 1 else None
                )
                
                # Clean up the response
                ai_response = sanitize_response(ai_response)
                
                # Log the AI response in green text to CLI
                print("\033[92m" + ai_response + "\033[0m")  # Green text
                
                # Don't display the AI response in the UI yet - will be displayed after audio finishes
                
                # Add to conversation history
                local_conversation_history.append({"role": "assistant", "content": ai_response})
                
                # Manage conversation history size - keep last 30 messages for global history and 100 for stories and games
                if character_name.startswith("story_") or character_name.startswith("game_"):
                    if len(local_conversation_history) > 100:
                        local_conversation_history = local_conversation_history[-100:]
                else:
                    if len(local_conversation_history) > 30:
                        local_conversation_history = local_conversation_history[-30:]
                
                # Update conversation history
                conversation_history = local_conversation_history.copy()
                
                # Save history based on character type
                await save_history()
                
                # Convert text to speech and play audio
                await enhanced_text_to_speech(ai_response, detected_mood)
                
                # Now that audio is finished, display the message in the UI
                await send_message_to_enhanced_clients({
                    "message": ai_response
                })
                
                # Remove the "processing" animation once response is complete
                await send_message_to_enhanced_clients({"action": "mic", "status": "off"})
                
            except asyncio.CancelledError:
                print("Task was cancelled")
                break
            except Exception as e:
                print(f"Error in enhanced conversation loop: {e}")
                await send_message_to_enhanced_clients({"message": f"Error processing request: {str(e)}", "type": "error-message"})
                await send_message_to_enhanced_clients({"action": "mic", "status": "off"})
                
        # Final save when conversation ends
        await save_history()
                
    except Exception as e:
        print(f"Error in enhanced conversation loop: {e}")
    finally:
        # Always make sure to turn off the mic status indicator
        try:
            await send_message_to_enhanced_clients({"action": "mic", "status": "off"})
        except:
            pass
        
        # Set active flag to False
        enhanced_conversation_active = False

async def enhanced_chat_completion(prompt, system_message, mood_prompt, conversation_history=None):
    """Get chat completion from OpenAI using the specified model."""
    try:
        # Import required libraries
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
            "temperature": 0.8,
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
    
    # Import get_current_character at the top level to avoid shadowing
    from .shared import get_current_character as get_character
    
    # Track previous character for history management
    previous_character = get_character()
    
    # Set character if provided
    if character:
        from .shared import set_current_character
        set_current_character(character)
    
    # Get current character after setting
    current_character = get_character()
    
    # Check if we're switching characters
    is_character_switch = previous_character != current_character
    if is_character_switch:
        # Save previous character's history if it was a story/game character
        is_previous_story_character = previous_character.startswith("story_") or previous_character.startswith("game_")
        if is_previous_story_character and conversation_history:
            # Save the previous character's history before we clear it
            save_character_specific_history(conversation_history, previous_character)
        
        # Always clear history when switching characters to prevent mixing
        conversation_history.clear()
        
        # Also delete the global history file
        history_file = "conversation_history.txt"
        if os.path.exists(history_file):
            os.remove(history_file)
            
        # Create empty history file
        with open(history_file, "w", encoding="utf-8") as f:
            pass
        
        # Send a message to clear the UI
        try:
            await send_message_to_enhanced_clients({
                "action": "clear_character_switch",
                "message": f"Switching to character: {current_character.replace('_', ' ')}",
                "type": "system-message"
            })
        except Exception as e:
            print(f"Error sending character switch message: {e}")
    
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
        return {"status": "already_running"}
    
    # Check if this is a story or game character
    is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
    
    # Handle history based on character type
    if is_story_character:
        # For story/game characters: load from character-specific file
        loaded_history = load_character_specific_history(current_character)
        # Clear existing history to ensure we only have the loaded history
        conversation_history.clear()
        if loaded_history:
            conversation_history.extend(loaded_history)
        else:
            print("No previous character-specific history found, starting fresh")
    else:
        # For standard characters: ensure history is cleared
        # Only reload from global file if it exists and has content
        history_file = "conversation_history.txt"
        # Make sure in-memory history is clear
        conversation_history.clear()
        
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            # File exists and has content, load it
            try:
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
            except Exception as e:
                print(f"Error loading global history: {e}")
        else:
            # File doesn't exist or is empty
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
        except Exception as e:
            print(f"Error stopping enhanced conversation: {e}")
    
    # Clear the thread reference
    enhanced_conversation_thread = None
    return {"status": "stopped"} 