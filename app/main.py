import json
import os
import signal
import sys
import uvicorn
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from .app_logic import (
    start_conversation,
    stop_conversation,
    set_character,
    set_env_variable,
    clients,
)

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

@app.get("/")
async def get(request: Request):
    model_provider = os.getenv("MODEL_PROVIDER")
    character_name = os.getenv("CHARACTER_NAME")
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
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    characters_folder = os.path.join(project_dir, 'characters')
    characters = [name for name in os.listdir(characters_folder) if os.path.isdir(os.path.join(characters_folder, name))]
    return {"characters": characters}

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["action"] == "stop":
                await stop_conversation()
            elif message["action"] == "start":
                selected_character = message["character"]
                await stop_conversation()  # Ensure any running conversation stops
                set_character(selected_character)
                await start_conversation()
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
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
    finally:
        if websocket in clients:
            clients.remove(websocket)

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    for client in clients:
        asyncio.create_task(client.close())
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server stopped by user.")