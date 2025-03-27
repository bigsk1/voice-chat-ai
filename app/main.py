import json
import os
import signal
import uvicorn
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from .shared import clients, set_current_character, conversation_history, add_client, remove_client
from .app_logic import start_conversation, stop_conversation, set_env_variable, save_conversation_history, characters_folder, set_transcription_model, fetch_ollama_models
from .enhanced_logic import start_enhanced_conversation, stop_enhanced_conversation
# Import the OpenAI Realtime API functions
from .realtime_sessions import (
    RealtimeSession,
    create_realtime_session,
    active_sessions,
    broadcast_to_session_clients
)
import logging
from threading import Thread
import uuid
import base64
from typing import Dict, List, Any, Optional, Callable
import websockets
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define available characters
CHARACTERS = ["assistant", "wizard", "pirate", "alien", "robot", "detective", "cowboy"]

app = FastAPI()

# Mount static files and templates
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    model_provider = os.getenv("MODEL_PROVIDER")
    character_name = os.getenv("CHARACTER_NAME", "wizard") 
    tts_provider = os.getenv("TTS_PROVIDER")
    openai_tts_voice = os.getenv("OPENAI_TTS_VOICE")
    openai_model = os.getenv("OPENAI_MODEL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    xtts_speed = os.getenv("XTTS_SPEED")
    elevenlabs_voice = os.getenv("ELEVENLABS_TTS_VOICE")
    faster_whisper_local = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_provider": model_provider,
        "character_name": character_name,
        "tts_provider": tts_provider,
        "openai_tts_voice": openai_tts_voice,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "xtts_speed": xtts_speed,
        "elevenlabs_voice": elevenlabs_voice,
        "faster_whisper_local": faster_whisper_local,
    })

@app.get("/characters")
async def get_characters():
    if not os.path.exists(characters_folder):
        logger.warning(f"Characters folder not found: {characters_folder}")
        return {"characters": ["Assistant"]}  # fallback
    
    try:
        character_dirs = [d for d in os.listdir(characters_folder) 
                        if os.path.isdir(os.path.join(characters_folder, d))]
        if not character_dirs:
            logger.warning("No character folders found")
            return {"characters": ["Assistant"]}  # fallback
        return {"characters": character_dirs}
    except Exception as e:
        logger.error(f"Error listing characters: {e}")
        return {"characters": ["Assistant"]}  # fallback in case of error

@app.get("/elevenlabs_voices")
async def get_elevenlabs_voices():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voices_file = os.path.join(project_dir, 'elevenlabs_voices.json')
    example_file = os.path.join(project_dir, 'elevenlabs_voices.json.example')
    
    # If the elevenlabs_voices.json file doesn't exist but the example does, create from example
    if not os.path.exists(voices_file) and os.path.exists(example_file):
        try:
            logger.info("elevenlabs_voices.json not found. Creating from example file.")
            with open(example_file, 'r', encoding='utf-8') as src:
                with open(voices_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            logger.info("Created elevenlabs_voices.json from example file.")
        except Exception as e:
            logger.error(f"Error creating elevenlabs_voices.json: {e}")
            
    # If file still doesn't exist, create a minimal version
    if not os.path.exists(voices_file):
        try:
            logger.info("Creating minimal elevenlabs_voices.json.")
            default_content = {
                "voices": {},
                "_comment": "This is a placeholder file. Replace with your own voice IDs from ElevenLabs."
            }
            with open(voices_file, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, indent=2)
            logger.info("Created minimal elevenlabs_voices.json file.")
        except Exception as e:
            logger.error(f"Error creating minimal elevenlabs_voices.json: {e}")
            return {"voices": []}
    
    try:
        with open(voices_file, 'r', encoding='utf-8') as f:
            voices = json.load(f)
        return voices
    except Exception as e:
        logger.error(f"Error reading elevenlabs_voices.json: {e}")
        return {"voices": []}

@app.get("/enhanced", response_class=HTMLResponse)
async def get_enhanced(request: Request):
    return templates.TemplateResponse("enhanced.html", {"request": request})

@app.get("/enhanced_defaults")
async def get_enhanced_defaults():
    from .enhanced_logic import enhanced_voice, enhanced_model, enhanced_tts_model, enhanced_transcription_model
    from .shared import get_current_character
    
    return {
        "character": get_current_character(),
        "voice": enhanced_voice,
        "model": enhanced_model,
        "tts_model": enhanced_tts_model,
        "transcription_model": enhanced_transcription_model
    }

@app.post("/set_character")
async def set_character(request: Request):
    data = await request.json()
    character = data.get("character")
    if not character:
        return {"status": "error", "message": "Character name is required"}

@app.post("/start_conversation")
async def start_conversation_route():
    Thread(target=lambda: asyncio.run(start_conversation())).start()
    return {"status": "started"}

@app.post("/stop_conversation")
async def stop_conversation_route():
    await stop_conversation()
    return {"status": "stopped"}

@app.post("/start_enhanced_conversation")
async def start_enhanced_conversation_route(request: Request):
    data = await request.json()
    character = data.get("character")
    speed = data.get("speed")
    model = data.get("model")
    voice = data.get("voice")
    tts_model = data.get("ttsModel")
    transcription_model = data.get("transcriptionModel")
    
    asyncio.create_task(start_enhanced_conversation(
        character=character,
        speed=speed,
        model=model,
        voice=voice,
        ttsModel=tts_model,
        transcriptionModel=transcription_model
    ))
    
    return {"status": "started"}

@app.post("/stop_enhanced_conversation")
async def stop_enhanced_conversation_route():
    await stop_enhanced_conversation()
    return {"status": "stopped"}

@app.post("/clear_history")
async def clear_history():
    global conversation_history
    conversation_history.clear()  # Clear the in-memory conversation history
    save_conversation_history(conversation_history)  # Save the cleared state to the file
    return {"status": "cleared"}

@app.get("/download_history")
async def download_history():
    # Create a temporary file with a unique name different from the main history file
    temp_file = f"temp_download_{uuid.uuid4().hex}.txt"
    
    # Format it the same way as the save_conversation_history function in app.py
    with open(temp_file, "w", encoding="utf-8") as file:
        for message in conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            file.write(f"{role}: {content}\n")
    
    # Return the file and ensure it will be cleaned up after sending
    return FileResponse(
        temp_file,
        media_type="text/plain",
        filename="conversation_history.txt",
        background=BackgroundTask(lambda: os.remove(temp_file) if os.path.exists(temp_file) else None)
    )

@app.get("/download_enhanced_history")
async def download_enhanced_history():
    # First load history from file to ensure we have the latest content
    try:
        history_file = "conversation_history.txt"
        temp_history = []
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            # File exists and has content, load it
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
        
        # Create a temporary file with a unique name different from the main history file
        temp_file = f"temp_download_{uuid.uuid4().hex}.txt"
        
        # Format it the same way as the save_conversation_history function in app.py
        with open(temp_file, "w", encoding="utf-8") as file:
            for message in temp_history:
                role = message["role"].capitalize()
                content = message["content"]
                file.write(f"{role}: {content}\n")
        
        # Return the file and ensure it will be cleaned up after sending
        return FileResponse(
            temp_file,
            media_type="text/plain",
            filename="conversation_history.txt",
            background=BackgroundTask(lambda: os.remove(temp_file) if os.path.exists(temp_file) else None)
        )
    except Exception as e:
        print(f"Error creating download file: {e}")
        return HTTPException(status_code=500, detail="Failed to create download file")

@app.post("/set_transcription_model")
async def update_transcription_model(request: Request):
    data = await request.json()
    model_name = data.get("model")
    if not model_name:
        return {"status": "error", "message": "Model name is required"}
    
    return set_transcription_model(model_name)

@app.get("/ollama_models")
async def get_ollama_models():
    """
    Fetch available models from Ollama
    """
    return await fetch_ollama_models()

@app.get("/openai_ephemeral_key")
async def get_openai_ephemeral_key():
    """
    Generate an ephemeral key for OpenAI API access from the browser
    
    In a production environment, you would use a service like Supabase or a proper server-side
    authentication system. For simplicity in this demo, we're just returning the API key directly.
    """
    try:
        # Get the API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("OPENAI_API_KEY not set in environment")
            return {"error": "API key not configured"}
        
        # In a real application, you might want to create a temporary token or session
        # For this demo, we'll just return the key directly
        # WARNING: This exposes your API key in production!
        
        # Add logging to help debug
        logger.info(f"Returning ephemeral key (first 5 chars): {api_key[:5]}...")
        
        # Return in the exact format expected by the WebRTC client
        return {
            "client_secret": {
                "value": api_key
            }
        }
    except Exception as e:
        logger.error(f"Error generating ephemeral key: {e}")
        return {"error": str(e)}

@app.post("/openai_realtime_proxy")
async def proxy_openai_realtime(request: Request):
    """
    Proxy endpoint to relay WebRTC connection to OpenAI API.
    This avoids CORS issues when connecting directly from the browser.
    """
    try:
        # Get the API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Get the SDP from the request body
        body = await request.body()
        sdp = body.decode('utf-8')
        
        # Get the model parameter from query params
        model = request.query_params.get('model', 'gpt-4o-realtime-preview-2024-12-17')
        
        # Log the request (without the full SDP for privacy)
        logger.info(f"Proxying WebRTC connection to OpenAI Realtime API for model: {model}")
        
        # Forward to OpenAI
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.openai.com/v1/realtime?model={model}",
                content=sdp,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/sdp",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            
            # Return the same status code and content
            from fastapi.responses import Response
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type="application/sdp"
            )
    
    except Exception as e:
        logger.error(f"Error proxying to OpenAI: {e}")
        return HTTPException(status_code=500, detail=f"Error proxying to OpenAI: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    add_client(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["action"] == "stop":
                await stop_conversation()
            elif message["action"] == "start":
                selected_character = message["character"]
                await stop_conversation()  # Ensure any running conversation stops
                set_current_character(selected_character)
                await start_conversation()
            elif message["action"] == "set_character":
                set_current_character(message["character"])
                await websocket.send_json({"message": f"Character: {message['character']}"})
            elif message["action"] == "set_provider":
                set_env_variable("MODEL_PROVIDER", message["provider"])
            elif message["action"] == "set_tts":
                set_env_variable("TTS_PROVIDER", message["tts"])
            elif message["action"] == "set_openai_voice":
                set_env_variable("OPENAI_TTS_VOICE", message["voice"])
            elif message["action"] == "set_openai_model":
                set_env_variable("OPENAI_MODEL", message["model"])
            elif message["action"] == "set_ollama_model":
                set_env_variable("OLLAMA_MODEL", message["model"])
            elif message["action"] == "set_xtts_speed":
                set_env_variable("XTTS_SPEED", message["speed"])
            elif message["action"] == "set_elevenlabs_voice":
                set_env_variable("ELEVENLABS_TTS_VOICE", message["voice"])
            elif message["action"] == "clear":
                conversation_history.clear()
                await websocket.send_json({"message": "Conversation history cleared."})
    except WebSocketDisconnect:
        remove_client(websocket)
        logger.info(f"Client disconnected from standard websocket")
    except Exception as e:
        logger.error(f"Error in standard websocket: {e}")
        # Still remove the client to prevent resource leaks
        remove_client(websocket)

@app.websocket("/ws_enhanced")
async def enhanced_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    add_client(websocket)
    try:
        # Just keep the connection alive without actively reading
        # Simply notify the client that the connection is established
        await websocket.send_json({"action": "connected", "message": "WebSocket connection established"})
        
        # Wait for the client to disconnect rather than actively reading
        while True:
            # This will raise WebSocketDisconnect when client disconnects
            # Process only control messages and heartbeats
            data = await websocket.receive()
            logging.info(f"Received message from client: {data}")
            # Don't try to parse or process normal text messages
            if data["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(1000)
    except WebSocketDisconnect:
        remove_client(websocket)
        logger.info(f"Client disconnected from enhanced websocket")
    except Exception as e:
        logger.error(f"Error in enhanced websocket: {e}")
        # Still remove the client to prevent resource leaks
        remove_client(websocket)

# OpenAI Realtime route (real-time voice chat)
@app.get("/openai_realtime")
async def get_openai_realtime(request: Request):
    """
    Serves the OpenAI Realtime API page for real-time voice conversations.
    """
    try:
        # Get characters from characters folder
        characters = []
        if os.path.exists(characters_folder):
            characters = [d for d in os.listdir(characters_folder) 
                        if os.path.isdir(os.path.join(characters_folder, d))]
        
        # Provide a fallback if no characters found
        if not characters:
            characters = ["assistant"]
            logger.warning("No character folders found, using fallback assistant")
            
        return templates.TemplateResponse(
            "openai_realtime.html", 
            {
                "request": request,
                "characters": characters,
            }
        )
    except Exception as e:
        logger.error(f"Error rendering OpenAI Realtime page: {e}")
        # Fallback with minimal context
        return templates.TemplateResponse(
            "openai_realtime.html", 
            {
                "request": request,
                "characters": ["assistant"],
            }
        )

# WebRTC OpenAI Realtime route (direct WebRTC implementation)
@app.get("/webrtc_realtime")
async def get_webrtc_realtime(request: Request):
    """
    Serves the WebRTC implementation of OpenAI Realtime API page.
    """
    try:
        # Get characters from characters folder
        characters = []
        if os.path.exists(characters_folder):
            characters = [d for d in os.listdir(characters_folder) 
                        if os.path.isdir(os.path.join(characters_folder, d))]
        
        # Provide a fallback if no characters found
        if not characters:
            characters = ["assistant"]
            logger.warning("No character folders found, using fallback assistant")
            
        return templates.TemplateResponse(
            "webrtc_realtime.html", 
            {
                "request": request,
                "characters": characters,
            }
        )
    except Exception as e:
        logger.error(f"Error rendering WebRTC Realtime page: {e}")
        # Fallback with minimal context
        return templates.TemplateResponse(
            "webrtc_realtime.html", 
            {
                "request": request,
                "characters": ["assistant"],
            }
        )

@app.websocket("/ws_openai_realtime")
async def openai_realtime_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for OpenAI Realtime API communication.
    Handles real-time audio streaming and direct WebSocket connections to OpenAI.
    """
    await websocket.accept()
    
    # Generate a unique client ID
    client_id = str(uuid.uuid4())
    
    # Create message queues for this client
    send_queue = asyncio.Queue()
    
    # Reference to the active session for this client
    active_session = None
    
    async def handle_client_messages(websocket, send_queue, client_id):
        """Handle messages from the client."""
        nonlocal active_session
        
        while True:
            try:
                # Receive message from client
                message_data = await websocket.receive()
                logging.info(f"Received message from client: {message_data}")
                
                # Parse the message
                if isinstance(message_data, dict) and "text" in message_data:
                    # It's a text message
                    message = json.loads(message_data["text"])
                elif isinstance(message_data, dict) and "bytes" in message_data:
                    # It's binary data
                    if active_session:
                        await active_session.process_audio_chunk(message_data["bytes"])
                    continue
                elif isinstance(message_data, str):
                    # It's a string message
                    message = json.loads(message_data)
                else:
                    # Unknown message format
                    continue
                    
                # Handle different message types
                if message.get("type") == "start_session" or message.get("action") == "create_session":
                    # Close existing session if there is one
                    if active_session:
                        active_session.close()
                    
                    # Create a new session
                    character = message.get("character", "assistant")
                    voice = message.get("voice", "alloy")
                    
                    try:
                        active_session = RealtimeSession(character=character, voice=voice)
                        await active_session.start()
                        
                        # Send session created message
                        await send_queue.put({
                            "type": "session_created",
                            "session_id": active_session.session_id
                        })
                        
                        # Register message handler
                        active_session.set_message_callback(
                            lambda msg: send_queue.put_nowait(msg)
                        )
                        
                    except Exception as e:
                        logging.error(f"Error starting session: {str(e)}")
                        await send_queue.put({
                            "type": "error",
                            "message": f"Failed to start session: {str(e)}"
                        })
                
                elif message.get("type") == "stop_session" or message.get("action") == "close_session":
                    # Close the session
                    if active_session:
                        active_session.close()
                        active_session = None
                        await send_queue.put({"type": "session_closed"})
                
                elif message.get("type") == "text" or message.get("action") == "send_text":
                    # Send text message to the session
                    if active_session:
                        text = message.get("text", "")
                        if text:
                            await active_session.send_text(text)
                    else:
                        await send_queue.put({
                            "type": "error",
                            "message": "No active session"
                        })

                # Debug commands
                elif message.get("type") == "debug" or message.get("action") == "debug":
                    cmd = message.get("command", "").lower()
                    
                    if cmd == "status":
                        # Return status of current session
                        status_info = {
                            "type": "debug_info",
                            "active_session": bool(active_session),
                            "timestamp": time.time()
                        }
                        
                        if active_session:
                            status_info.update({
                                "session_id": active_session.session_id,
                                "session_active": active_session.active,
                                "session_created_at": active_session.created_at,
                                "session_last_activity": active_session.last_activity,
                                "user_speaking": active_session.user_is_speaking,
                                "ai_speaking": active_session.ai_is_speaking,
                                "character": active_session.character,
                                "model": active_session.model,
                                "voice": active_session.voice,
                                "websocket_connected": bool(active_session.ws and active_session.ws.sock and active_session.ws.sock.connected)
                            })
                        
                        await send_queue.put(status_info)
                    
                    elif cmd == "toggle_audio_debug":
                        # Toggle audio debugging
                        import os
                        current = os.environ.get("DEBUG_AUDIO", "false").lower() == "true"
                        os.environ["DEBUG_AUDIO"] = str(not current).lower()
                        
                        await send_queue.put({
                            "type": "debug_info",
                            "message": f"Audio debugging {'disabled' if current else 'enabled'}"
                        })
                    
                    elif cmd == "toggle_websocket_debug":
                        # Toggle websocket debugging
                        import os
                        current = os.environ.get("DEBUG_WEBSOCKET", "false").lower() == "true"
                        os.environ["DEBUG_WEBSOCKET"] = str(not current).lower()
                        
                        await send_queue.put({
                            "type": "debug_info",
                            "message": f"WebSocket debugging {'disabled' if current else 'enabled'}"
                        })
                    
                    elif cmd == "help":
                        # Show available debug commands
                        await send_queue.put({
                            "type": "debug_info",
                            "message": "Available debug commands:\nstatus - Show session status\ntoggle_audio_debug - Toggle audio debugging\ntoggle_websocket_debug - Toggle WebSocket debugging\nhelp - Show this message"
                        })
                
                elif message.get("type") == "audio" or message.get("action") == "send_audio":
                    # Process audio data
                    if active_session:
                        audio_data = message.get("audio_data")
                        if audio_data:
                            try:
                                # Convert base64 to binary
                                binary_audio = base64.b64decode(audio_data)
                                await active_session.process_audio_chunk(binary_audio)
                            except Exception as e:
                                logging.error(f"Error processing audio: {str(e)}")
                    else:
                        await send_queue.put({
                            "type": "error",
                            "message": "No active session"
                        })
                
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON: {str(e)}")
                await send_queue.put({
                    "type": "error",
                    "message": "Invalid message format"
                })
            except websockets.exceptions.ConnectionClosedOK:
                # Connection closed normally
                break
            except websockets.exceptions.ConnectionClosedError:
                # Connection closed with error
                break
            except Exception as e:
                logging.error(f"Error handling client message: {str(e)}")
                await send_queue.put({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                break

    async def handle_server_messages(websocket, send_queue):
        """Send messages from the queue to the client."""
        while True:
            message = await send_queue.get()
            try:
                await websocket.send_json(message)
            except Exception as e:
                logging.error(f"Error sending message to client: {str(e)}")
                break
    
    try:
        # Start the tasks for handling messages
        client_task = asyncio.create_task(handle_client_messages(websocket, send_queue, client_id))
        server_task = asyncio.create_task(handle_server_messages(websocket, send_queue))
        
        # Wait for either task to complete (or raise an exception)
        done, pending = await asyncio.wait(
            [client_task, server_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            
        # Check for exceptions in completed tasks
        for task in done:
            try:
                await task
            except Exception as e:
                logging.error(f"Task error: {str(e)}")
                
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    
    finally:
        # Close the active session if it exists
        if active_session:
            try:
                active_session.close()  # Use synchronous close
                await send_queue.put({"type": "session_closed"})
            except Exception as e:
                logging.error(f"Error closing session: {str(e)}")
        
        # Close the WebSocket connection
        try:
            await websocket.close()
        except Exception as e:
            logging.error(f"Error closing WebSocket: {str(e)}")

def signal_handler(sig, frame):
    print('\nShutting down gracefully... Press Ctrl+C again to force exit')
    
    try:
        # First, close all active realtime sessions
        if active_sessions:
            print(f"Closing {len(active_sessions)} active realtime sessions...")
            for session_id, session in list(active_sessions.items()):
                print(f"Closing realtime session: {session_id}")
                try:
                    session.close()  # Synchronous close
                except Exception as e:
                    print(f"Error closing session {session_id}: {e}")
        
        # Stop any active enhanced conversation
        try:
            # For async shutdown in sync context, create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # First stop any active conversations
            from .enhanced_logic import enhanced_conversation_active, stop_enhanced_conversation
            if enhanced_conversation_active:
                print("Stopping active enhanced conversation...")
                loop.run_until_complete(stop_enhanced_conversation())
                
            # Then close all WebSocket connections
            for client in list(clients):  # Create a copy of the clients set to avoid modification during iteration
                try:
                    if hasattr(client, 'close'):
                        # Use the same loop for consistency
                        loop.run_until_complete(client.close())
                except Exception as e:
                    print(f"Error closing client: {e}")
                    
            loop.close()
        except Exception as e:
            print(f"Error in graceful shutdown: {e}")
        
        print("Shutdown procedures completed. Exiting...")
        import os
        os._exit(0)  # Force exit as sys.exit() might not work if asyncio is running
        
    except Exception as e:
        print(f"Error during shutdown: {e}")
        import os
        os._exit(1)  # Error exit code

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting server. Press Ctrl+C to exit.")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nServer stopped by keyboard interrupt.")
    finally:
        print("Shutdown complete.")

# Define missing functions
async def broadcast_to_session_clients(session_id, message):
    """Broadcast a message to all clients connected to a session"""
    from .realtime_sessions import active_sessions
    
    if session_id in active_sessions:
        session = active_sessions[session_id]
        for client_queue in session.client_connections:
            try:
                await client_queue.put(message)
            except Exception as e:
                logger.error(f"Error broadcasting message to client: {e}")

async def create_realtime_session(character, model=None, voice=None, client_id=None):
    """Create a new realtime session - wrapper for RealtimeSession"""
    from .realtime_sessions import RealtimeSession, active_sessions
    from .app_logic import load_character_prompt
    
    try:
        # Use defaults if not provided
        model = model or os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
        voice = voice or "alloy"
        
        # Load character instructions
        character_instructions = load_character_prompt(character)
        
        # Create session ID
        session_id = f"realtime_{uuid.uuid4().hex}"
        
        # Initialize session
        session = RealtimeSession(
            session_id=session_id,
            character=character,
            model=model, 
            voice=voice
        )
        
        # Start session
        if await session.start():
            # Register session
            active_sessions[session_id] = session
            
            # Add client if specified
            if client_id and client_id in client_queues:
                session.client_connections.add(client_queues[client_id])
            
            logger.info(f"Created realtime session: {session_id}")
            return session_id
        else:
            logger.error("Failed to start realtime session")
            return None
    except Exception as e:
        logger.error(f"Error creating realtime session: {e}")
        return None

# Define a client queue dictionary
client_queues = {}