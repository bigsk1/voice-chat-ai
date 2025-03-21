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

router = APIRouter()

# Enhanced-specific variables
enhanced_conversation_active = False
enhanced_conversation_thread = None
enhanced_tone = "neutral"
enhanced_speed = "1.0"
enhanced_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
enhanced_model = os.getenv("OPENAI_MODEL", "gpt-4o")
enhanced_tts_model = os.getenv("OPENAI_MODEL_TTS", "gpt-4o-mini-tts")
enhanced_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")

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
    try:
        # Import required libraries
        import json
        import aiohttp
        import tempfile
        import pyaudio
        import wave
        import numpy as np
        import os
        import time
        
        # First, record the audio using similar logic to the existing app
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Recording parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        # Use a lower threshold to be more sensitive to input
        SILENCE_THRESHOLD = 300  # Lower threshold to detect quieter speech
        SILENCE_DURATION = 2.0  # in seconds
        
        # Recording logic
        p = pyaudio.PyAudio()
        
        # Debug info about audio devices
        print("\nAudio input devices:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:  # Only input devices
                print(f"Device {i}: {dev_info['name']}")
        print("Using default input device\n")
        
        # Open the stream with input_device_index=None to use default device
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        # Detect silence function
        def detect_silence(data, threshold=300):
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.mean(np.abs(audio_data))
            print(f"Audio level: {level}")  # Debug to see audio levels
            return level < threshold
        
        # Wait for user to start speaking (break initial silence)
        print("Waiting for speech...")
        await send_message_to_enhanced_clients({"action": "waiting_for_speech", "message": "Listening... Please speak now."})
        
        # Flush initial buffer
        for _ in range(5):
            stream.read(CHUNK)
            
        initial_silent_chunks = 0
        silence_broken = False
        
        # Wait for user to start speaking
        while not silence_broken and enhanced_conversation_active:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not detect_silence(data, threshold=SILENCE_THRESHOLD):
                silence_broken = True
                print("Speech detected, recording started...")
                break
            
            initial_silent_chunks += 1
            # If waiting too long (15 seconds), abort
            if initial_silent_chunks > 15 * (RATE / CHUNK):
                print("No speech detected after timeout. Aborting.")
                stream.stop_stream()
                stream.close()
                p.terminate()
                await send_message_to_enhanced_clients({
                    "action": "error", 
                    "message": "No speech detected. Please check your microphone and try again."
                })
                return "No speech detected. Please try again."
                
            # Every 2 seconds, check if the conversation is still active
            if initial_silent_chunks % (2 * int(RATE / CHUNK)) == 0:
                if not enhanced_conversation_active:
                    print("Conversation stopped while waiting for speech.")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return None
                
                # Provide feedback every few seconds
                if initial_silent_chunks > 0 and initial_silent_chunks % (4 * int(RATE / CHUNK)) == 0:
                    await send_message_to_enhanced_clients({
                        "action": "waiting_for_speech", 
                        "message": "Still listening... Please speak or check your microphone."
                    })
                    
        # Now begin actual recording
        frames = []
        print("Enhanced recording...")
        await send_message_to_enhanced_clients({"action": "recording_started"})
        
        # Add the initial speech chunk that broke the silence
        if silence_broken:
            frames.append(data)
        
        silent_chunks = 0
        speaking_chunks = 0
        
        # Continue recording until silence is detected
        while enhanced_conversation_active:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if detect_silence(data, threshold=SILENCE_THRESHOLD):
                silent_chunks += 1
                if silent_chunks > SILENCE_DURATION * (RATE / CHUNK):
                    break
            else:
                silent_chunks = 0
                speaking_chunks += 1
            if speaking_chunks > SILENCE_DURATION * (RATE / CHUNK) * 15:  # Allow longer recordings
                break
                
        print("Enhanced recording stopped.")
        await send_message_to_enhanced_clients({"action": "recording_stopped"})
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # If no substantial recording was made, return None
        if len(frames) < 10:
            try:
                os.unlink(temp_filename)
            except:
                pass
            await send_message_to_enhanced_clients({
                "action": "error", 
                "message": "Recording too short. Please try again."
            })
            return "Recording too short. Please try again."
        
        # Save the recorded audio
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Get API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "API key missing. Please set OPENAI_API_KEY in your environment."
        
        # Use the selected transcription model
        model = enhanced_transcription_model
        print(f"Using transcription model: {model}")
        
        # Make the actual API call to OpenAI
        api_url = "https://api.openai.com/v1/audio/transcriptions"
        
        async with aiohttp.ClientSession() as session:
            with open(temp_filename, "rb") as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                    audio_file.read(),
                                    filename=os.path.basename(temp_filename),
                                    content_type='audio/wav')
                form_data.add_field('model', model if model != "gpt-4o-transcribe" and model != "gpt-4o-mini-transcribe" else "whisper-1")
                
                headers = {
                    "Authorization": f"Bearer {openai_api_key}"
                }
                
                async with session.post(api_url, data=form_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcription = result.get("text", "")
                    else:
                        error_text = await response.text()
                        print(f"Error from OpenAI API: {error_text}")
                        return f"Transcription error: {response.status} - {error_text}"
        
        # Clean up temp file
        try:
            os.unlink(temp_filename)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
            
        return transcription
        
    except Exception as e:
        print(f"Error in enhanced audio recording/transcription: {e}")
        return f"Error: {str(e)}"

async def enhanced_text_to_speech(text, character_audio_file):
    """Convert text to speech using the enhanced TTS model."""
    try:
        # Import required libraries
        import json
        import aiohttp
        import os
        import tempfile
        from pathlib import Path
        import asyncio
        from pydub import AudioSegment
        from pydub.playback import play
        
        # Create a temporary file to store the speech audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
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
        model = enhanced_tts_model if enhanced_tts_model else "tts-1"
        
        # Additional instructions to adjust tone based on the selected setting
        voice_instructions = ""
        if enhanced_tone == "friendly":
            voice_instructions = "Speak in a warm, friendly tone that makes the listener feel comfortable."
        elif enhanced_tone == "professional":
            voice_instructions = "Speak in a clear, professional manner suitable for a business context."
        elif enhanced_tone == "excited":
            voice_instructions = "Speak with enthusiasm and excitement in your voice."
        elif enhanced_tone == "empathetic":
            voice_instructions = "Speak with empathy and understanding in your voice."
        elif enhanced_tone == "serious":
            voice_instructions = "Speak in a serious, no-nonsense tone."
        
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
            "response_format": "mp3"
        }
        
        # If we have voice instructions, include them
        if voice_instructions:
            payload["voice_instructions"] = voice_instructions
            
        print(f"Using TTS model: {model}, voice: {voice}, speed: {speed}")
        
        # Notify that AI is about to speak
        await send_message_to_enhanced_clients({"action": "ai_start_speaking"})
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # Save the audio file
                    with open(temp_filename, "wb") as f:
                        f.write(await response.read())
                    
                    # Play audio using pydub - avoid system audio players like VLC
                    try:
                        # Use pydub's play function which doesn't launch external players
                        sound = AudioSegment.from_mp3(temp_filename)
                        play(sound)
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
            os.unlink(temp_filename)
        except Exception as e:
            print(f"Error removing temporary TTS file: {e}")
        
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

async def enhanced_chat_completion(prompt, system_message, mood_prompt, conversation_history=None):
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
        
        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": system_message + "\n" + mood_prompt}
        ]
        
        # If conversation history is provided, add it to the messages
        if conversation_history:
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
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        print(f"Using chat model: {model}")
        
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

async def enhanced_conversation_loop():
    """Main conversation loop for the enhanced interface."""
    global enhanced_conversation_active
    try:
        # Keep context of the conversation
        local_conversation_history = []
        
        character_name = get_current_character()
        character_prompt_file = os.path.join(characters_folder, character_name, f"{character_name}.txt")
        character_audio_file = os.path.join(characters_folder, character_name, f"{character_name}.wav")
        
        base_system_message = open_file(character_prompt_file)
        
        # Greeting message (just send to UI, don't play audio yet)
        greeting = f"Hello! I'm {character_name.replace('_', ' ')}. How can I help you today?"
        await send_message_to_enhanced_clients({"message": greeting})
        
        # Don't play greeting audio automatically - wait for user interaction
        # await enhanced_text_to_speech(greeting, character_audio_file)
        
        while enhanced_conversation_active:
            # Record and transcribe
            user_input = await record_enhanced_audio_and_transcribe()
            
            if not user_input or user_input.lower() in ["quit", "exit", "leave"]:
                await send_message_to_enhanced_clients({
                    "message": "Conversation ended."
                })
                break
                
            # Send user message to frontend
            await send_message_to_enhanced_clients({
                "message": f"You: {user_input}"
            })
            
            # Add to history
            local_conversation_history.append({"role": "user", "content": user_input})
            
            # Analyze mood
            mood = analyze_mood(user_input)
            mood_prompt = adjust_prompt(mood)
            
            # Get AI response - pass conversation history for context
            chatbot_response = await enhanced_chat_completion(
                user_input, 
                base_system_message, 
                mood_prompt, 
                local_conversation_history if len(local_conversation_history) > 1 else None
            )
            
            # Add to history
            local_conversation_history.append({"role": "assistant", "content": chatbot_response})
            
            # Limit conversation history to last 10 exchanges to prevent context overflow
            if len(local_conversation_history) > 20:
                # Keep system message and last 10 exchanges
                local_conversation_history = local_conversation_history[-20:]
            
            # Send to frontend
            await send_message_to_enhanced_clients({
                "message": chatbot_response
            })
            
            # Convert to speech
            sanitized_response = sanitize_response(chatbot_response)
            if len(sanitized_response) > 500:  # Limit length for TTS
                sanitized_response = sanitized_response[:500] + "..."
                
            await enhanced_text_to_speech(sanitized_response, character_audio_file)
            
            # Update global conversation history
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": chatbot_response}
            ])
            save_conversation_history(conversation_history)
            
    except Exception as e:
        print(f"Error in enhanced conversation loop: {e}")
        await send_message_to_enhanced_clients({
            "action": "error",
            "message": f"Error: {str(e)}"
        })
    finally:
        enhanced_conversation_active = False

async def start_enhanced_conversation(character=None, speed=None, tone=None, model=None, voice=None, ttsModel=None, transcriptionModel=None):
    """Start the enhanced conversation with the specified settings."""
    global enhanced_conversation_active, enhanced_conversation_thread
    global enhanced_tone, enhanced_speed, enhanced_voice, enhanced_model, enhanced_tts_model, enhanced_transcription_model
    
    if enhanced_conversation_active:
        await stop_enhanced_conversation()
    
    # Update settings if provided
    if character:
        from .shared import set_current_character
        set_current_character(character)
    if speed:
        enhanced_speed = speed
    if tone:
        enhanced_tone = tone
    if model:
        enhanced_model = model
    if voice:
        enhanced_voice = voice
    if ttsModel:
        enhanced_tts_model = ttsModel
    if transcriptionModel:
        enhanced_transcription_model = transcriptionModel
    
    enhanced_conversation_active = True
    enhanced_conversation_thread = Thread(target=asyncio.run, args=(enhanced_conversation_loop(),))
    enhanced_conversation_thread.daemon = True
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