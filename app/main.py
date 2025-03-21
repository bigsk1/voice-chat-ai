import json
import os
import signal
import sys
import uvicorn
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from .shared import clients, get_current_character, set_current_character, conversation_history, add_client, remove_client, clear_conversation_history
from .app_logic import start_conversation, stop_conversation, set_env_variable, save_conversation_history, characters_folder
from .enhanced_logic import start_enhanced_conversation, stop_enhanced_conversation
import logging
from threading import Thread

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
    character_name = os.getenv("CHARACTER_NAME", "grok_xai")  # Default to grok_xai if not in .env
    tts_provider = os.getenv("TTS_PROVIDER")
    openai_tts_voice = os.getenv("OPENAI_TTS_VOICE")
    openai_model = os.getenv("OPENAI_MODEL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    xtts_speed = os.getenv("XTTS_SPEED")
    elevenlabs_voice = os.getenv("ELEVENLABS_TTS_VOICE")

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
    try:
        with open(voices_file, 'r', encoding='utf-8') as f:
            voices = json.load(f)
        return voices
    except FileNotFoundError:
        return {"voices": []}
    except Exception as e:
        return {"error": str(e)}

@app.get("/enhanced", response_class=HTMLResponse)
async def get_enhanced(request: Request):
    return templates.TemplateResponse("enhanced.html", {"request": request})

@app.post("/set_character/{character_name}")
async def set_character(character_name: str):
    set_current_character(character_name)
    return {"status": "success", "character": character_name}

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
    # Create a temporary file with the same format used in app.py save_conversation_history
    temp_file = "conversation_history.txt"
    
    # Format it the same way as the save_conversation_history function in app.py
    with open(temp_file, "w", encoding="utf-8") as file:
        for message in conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            file.write(f"{role}: {content}\n")
    
    # Return the file
    return FileResponse(
        temp_file,
        media_type="text/plain",
        filename="conversation_history.txt"
    )

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

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    
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
        # Safely close all WebSocket connections
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
    
    # Force exit after a short timeout if still running
    print("Shutdown procedures completed. Exiting...")
    import os
    os._exit(0)  # Force exit as sys.exit() might not work if asyncio is running

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