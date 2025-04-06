import json
import os
import signal
import uvicorn
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from starlette.background import BackgroundTask
from .shared import clients, set_current_character, conversation_history, add_client, remove_client
from .app_logic import start_conversation, stop_conversation, set_env_variable, save_conversation_history, characters_folder, set_transcription_model, fetch_ollama_models, load_character_prompt, save_character_specific_history
from .enhanced_logic import start_enhanced_conversation, stop_enhanced_conversation
import logging
from threading import Thread
import uuid
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    voice_speed = os.getenv("VOICE_SPEED")
    elevenlabs_voice = os.getenv("ELEVENLABS_TTS_VOICE")
    kokoro_voice = os.getenv("KOKORO_TTS_VOICE")
    faster_whisper_local = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_provider": model_provider,
        "character_name": character_name,
        "tts_provider": tts_provider,
        "openai_tts_voice": openai_tts_voice,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "voice_speed": voice_speed,
        "elevenlabs_voice": elevenlabs_voice,
        "kokoro_voice": kokoro_voice,
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
    try:
        data = await request.json()
        character = data.get("character")
        if not character:
            return {"status": "error", "message": "Character name is required"}
        
        # Import the set_character function from app_logic
        from .app_logic import set_api_character
        from pydantic import BaseModel
        
        # Create a model for the function
        class CharacterModel(BaseModel):
            character: str
        
        # Call the function with the character model
        result = await set_api_character(CharacterModel(character=character))
        return result
    except Exception as e:
        print(f"Error setting character: {e}")
        return {"status": "error", "message": str(e)}

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
    """Clear the conversation history."""
    try:
        # Import with alias to avoid potential shadowing issues
        from .shared import conversation_history, get_current_character as get_character
        
        current_character = get_character()
        
        # Check if this is a story or game character
        is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
        print(f"Clearing history for {current_character} ({is_story_character=})")
        
        # Clear the in-memory history
        conversation_history.clear()
        
        if is_story_character:
            # Clear character-specific history file
            character_dir = os.path.join(characters_folder, current_character)
            history_file = os.path.join(character_dir, "conversation_history.txt")
            
            if os.path.exists(history_file):
                os.remove(history_file)
                print(f"Deleted character-specific history file for {current_character}")
            
            # Write empty history to character-specific file
            save_character_specific_history(conversation_history, current_character)
        else:
            # Clear global history file
            history_file = "conversation_history.txt"
            if os.path.exists(history_file):
                os.remove(history_file)
                print(f"Deleted global history file")
            
            # Write empty history to global file
            save_conversation_history(conversation_history)
        
        return {"status": "cleared"}
    except Exception as e:
        print(f"Error clearing history: {e}")
        return {"status": "error", "message": str(e)}

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
    """Download the conversation history."""
    try:
        # Import with alias to avoid potential shadowing issues
        from .shared import get_current_character as get_character
        
        current_character = get_character()
        
        # Check if this is a story or game character
        is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
        print(f"Downloading history for {current_character} ({is_story_character=})")
        
        if is_story_character:
            # Get from character-specific history file
            character_dir = os.path.join(characters_folder, current_character)
            history_file = os.path.join(character_dir, "conversation_history.txt")
            
            if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
                # Create an empty history file if it doesn't exist
                with open(history_file, "w", encoding="utf-8") as f:
                    f.write(f"No conversation history found for {current_character}.\n")
                
            # Generate download filename based on character
            download_filename = f"{current_character}_history.txt"
            
            return FileResponse(
                history_file,
                media_type="text/plain",
                filename=download_filename
            )
        else:
            # Get from global history file
            history_file = "conversation_history.txt"
            
            if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
                # Create an empty history file if it doesn't exist
                with open(history_file, "w", encoding="utf-8") as f:
                    f.write("No conversation history found.\n")
            
            return FileResponse(
                history_file,
                media_type="text/plain",
                filename="conversation_history.txt"
            )
    except Exception as e:
        print(f"Error downloading history: {e}")
        return PlainTextResponse(f"Error downloading history: {str(e)}", status_code=500)

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
        
        # Get the model parameter from query params or default from environment
        default_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
        model = request.query_params.get('model', default_model)
        
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
            elif message["action"] == "set_xai_model":
                set_env_variable("XAI_MODEL", message["model"])
            elif message["action"] == "set_anthropic_model":
                set_env_variable("ANTHROPIC_MODEL", message["model"])
            elif message["action"] == "set_voice_speed":
                set_env_variable("VOICE_SPEED", message["speed"])
            elif message["action"] == "set_elevenlabs_voice":
                set_env_variable("ELEVENLABS_TTS_VOICE", message["voice"])
            elif message["action"] == "set_kokoro_voice":
                set_env_variable("KOKORO_TTS_VOICE", message["voice"])
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
async def websocket_enhanced_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Add client to the list
    add_client(websocket)
    print(f"Enhanced WebSocket client {id(websocket)} connected")
    logging.info("connection open")
    
    # Notify client they are connected successfully
    try:
        await websocket.send_json({"action": "connected"})
    except:
        pass
    
    try:
        # Process messages from the client
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "ping":
                    # Respond to heartbeats
                    await websocket.send_json({"action": "pong"})
            except json.JSONDecodeError:
                # Not a JSON message
                pass
                
    except WebSocketDisconnect:
        logging.info("Client disconnected from enhanced websocket")
    except Exception as e:
        logging.error(f"Error in enhanced websocket: {e}")
    finally:
        # Remove client from the list on any error or disconnect
        remove_client(websocket)
        print(f"Enhanced WebSocket client {id(websocket)} disconnected")

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
        
        # Get realtime model from environment variable or use default
        realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
            
        return templates.TemplateResponse(
            "webrtc_realtime.html", 
            {
                "request": request,
                "characters": characters,
                "realtime_model": realtime_model,
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
                "realtime_model": "gpt-4o-realtime-preview-2024-12-17",  # Default fallback
            }
        )

@app.get("/api/character/{character_name}")
async def get_character_prompt(character_name: str):
    """
    Get the prompt for a specific character
    """
    try:
        prompt = load_character_prompt(character_name)
        return {"prompt": prompt}
    except Exception as e:
        logger.error(f"Error loading character prompt: {e}")
        return {"error": str(e)}

@app.get("/get_character_history")
async def get_character_history():
    """Get conversation history for currently selected character."""
    try:
        # Import with alias to avoid potential shadowing issues
        from .shared import get_current_character as get_character
        
        current_character = get_character()
        
        # Check if this is a story or game character
        is_story_character = current_character.startswith("story_") or current_character.startswith("game_")
        print(f"Getting history for {current_character} ({is_story_character=})")
        
        if is_story_character:
            # Get from character-specific history file
            character_dir = os.path.join(characters_folder, current_character)
            history_file = os.path.join(character_dir, "conversation_history.txt")
            
            if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_text = f.read()
                return {"status": "success", "history": history_text, "character": current_character}
            else:
                return {"status": "empty", "history": "", "character": current_character}
        else:
            # For non-story characters, return empty history
            return {"status": "not_story_character", "history": "", "character": current_character}
    except Exception as e:
        print(f"Error getting character history: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/kokoro_voices")
async def get_kokoro_voices():
    try:
        # Get the base URL from environment or use default
        kokoro_base_url = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
        
        try:
            # Use the correct API endpoint for voices
            voices_url = f"{kokoro_base_url}/audio/voices"
            
            # Make HTTP request directly
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(voices_url, timeout=3) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Process the voices from the response
                            voices = []
                            for voice_id in data.get("voices", []):
                                # Create a readable name from the ID
                                # Format: language code (af/am) + name
                                parts = voice_id.split('_')
                                if len(parts) >= 2:
                                    lang_code = parts[0]
                                    name = parts[1].capitalize()
                                    gender = "Female" if lang_code == "af" else "Male"
                                    voices.append({
                                        "id": voice_id,
                                        "name": f"{name} ({gender})"
                                    })
                                else:
                                    # Fallback for voices without standard format
                                    voices.append({
                                        "id": voice_id,
                                        "name": voice_id
                                    })
                            
                            return {"voices": voices}
                        else:
                            # Log the error and return empty voices
                            error_text = await response.text()
                            logger.error(f"Error fetching Kokoro voices: HTTP {response.status} - {error_text}")
                            return {"voices": [], "error": f"HTTP Error: {response.status}"}
                except aiohttp.ClientConnectorError as e:
                    # Handle connection errors specifically (server not available)
                    logger.info(f"Kokoro server not available at {kokoro_base_url} - This is normal if you don't have Kokoro running")
                    return {"voices": [], "error": "Kokoro server not available"}
                except asyncio.TimeoutError:
                    # Handle timeout errors
                    # logger.info(f"Timeout connecting to Kokoro server at {kokoro_base_url}")
                    return {"voices": [], "error": "Connection timeout"}
            
        except Exception as e:
            # Log the error and return empty voices with error message
            logger.error(f"Error fetching Kokoro voices: {str(e)}")
            return {"voices": [], "error": str(e)}
            
    except Exception as e:
        logger.error(f"Critical error in get_kokoro_voices: {str(e)}")
        return {"voices": [], "error": str(e)}

def signal_handler(sig, frame):
    print('\nShutting down gracefully... Press Ctrl+C again to force exit')
    
    try:
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