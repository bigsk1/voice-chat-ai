import os
import logging
from fastapi import FastAPI, UploadFile, File, Depends, Form, HTTPException, Request, WebSocket, Header, Query
from fastapi.responses import Response, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# core.py 経由ではなく services/audio.py のラッパーを使用
#from app.services.audio import transcribe_audio_bytes, generate_response_text, synthesize_text
from .minimal_shared import transcribe_audio_bytes, generate_response_text, synthesize_text,get_current_character, set_current_character,conversation_history,characters_folder,  load_character_prompt

enhanced_voice = os.getenv("OPENAI_TTS_VOICE", "coral")
enhanced_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
enhanced_tts_model = os.getenv("OPENAI_MODEL_TTS", "gpt-4o-mini-tts")
enhanced_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")


# ロガー設定: Uvicorn のエラーログを使用
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# ─── Static Files ─────────────────────────────────
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

# ─── Templates ───────────────────────────────────
templates = Jinja2Templates(directory="app/templates")
ja_templates = Jinja2Templates(directory="app/templates/ja")

# ─── API Key Dependency ───────────────────────────
def get_api_key(authorization: str = Header(None), api_key: str = Header(None)):
    if api_key:
        return api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    return os.getenv("OPENAI_API_KEY")
     
# ─── Request Models ───────────────────────────────
class TextInput(BaseModel):
    text: str

class CharacterInput(BaseModel):
    character: str

# ─── API Endpoints ───────────────────────────────
@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key),
):
    audio_bytes = await file.read()
    logger.info(f"[api_transcribe] Received {len(audio_bytes)} bytes, content_type={file.content_type}, model={model}")
    try:
        text = await transcribe_audio_bytes(audio_bytes, api_key, model=model)
        logger.info(f"[api_transcribe] Transcription result: {text[:50]}...")
    except Exception as e:
        logger.error(f"[api_transcribe] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text}

@app.post("/api/chat")
async def api_chat(
    payload: TextInput,
    api_key: str = Depends(get_api_key),
):
    logger.info(f"[api_chat] User text: {payload.text[:50]}...")
    try:
        response_text = await generate_response_text(payload.text, api_key)
        logger.info(f"[api_chat] Response text: {response_text[:50]}...")
    except Exception as e:
        logger.error(f"[api_chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": response_text}

@app.post("/api/synthesize")
async def api_synthesize(
    payload: TextInput,
    api_key: str = Depends(get_api_key),
):
    logger.info(f"[api_synthesize] Synthesizing text: {payload.text[:50]}...")
    try:
        audio_bytes = await synthesize_text(payload.text, api_key)
        logger.info(f"[api_synthesize] Audio bytes length: {len(audio_bytes)}")
    except Exception as e:
        logger.error(f"[api_synthesize] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=audio_bytes, media_type="audio/wav")

# ─── Characters / Voices / Defaults ───────────────
@app.get("/characters")
async def get_characters():
    dirs = []
    if os.path.isdir(characters_folder):
        dirs = [d for d in os.listdir(characters_folder)
                if os.path.isdir(os.path.join(characters_folder, d))]
    lang = os.getenv("LANGUAGE", "en")
    if lang == "ja":
        filtered = [d for d in dirs if d.startswith("ja")]
    else:
        filtered = [d for d in dirs if not d.startswith("ja")]
    return {"characters": filtered or ["assistant"]}

@app.post("/set_character")
async def set_character(payload: CharacterInput):
    logger.info(f"[set_character] Setting character: {payload.character}")
    try:
        set_current_character(payload.character)
    except Exception as e:
        logger.error(f"[set_character] Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return {"character": get_current_character()}

@app.get("/kokoro_voices")
async def get_kokoro_voices():
    return {"voices": []}

@app.get("/enhanced_defaults")
async def get_enhanced_defaults():
    return {
        "character": get_current_character(),
        "voice": enhanced_voice,
        "model": enhanced_model,
        "tts_model": enhanced_tts_model,
        "transcription_model": enhanced_transcription_model,
    }

# ─── WebSocket Stub ──────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except:
        pass

# ─── Template Chooser ─────────────────────────────
async def _choose_template(request: Request, template_name: str):
    lang = os.getenv("LANGUAGE", "en")
    tpl = ja_templates if lang == "ja" else templates
    return tpl.TemplateResponse(template_name, {"request": request})

# ─── HTML Page Endpoints ─────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index(request: Request):
    return await _choose_template(request, "index.html")

@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_index_html(request: Request):
    return RedirectResponse(url="/")

@app.get("/enhanced", response_class=HTMLResponse, include_in_schema=False)
async def serve_enhanced(request: Request):
    return await _choose_template(request, "enhanced.html")

@app.get("/enhanced.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_enhanced_html(request: Request):
    return RedirectResponse(url="/enhanced")

@app.get("/speech_test", response_class=HTMLResponse, include_in_schema=False)
async def serve_speech_test(request: Request):
    return await _choose_template(request, "speech_test.html")

@app.get("/speech_test.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_speech_test_html(request: Request):
    return RedirectResponse(url="/speech_test")

@app.get("/webrtc_realtime", response_class=HTMLResponse, include_in_schema=False)
async def serve_webrtc(request: Request):
    return await _choose_template(request, "webrtc_realtime.html")

@app.get("/webrtc_realtime.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_webrtc_html(request: Request):
    return RedirectResponse(url="/webrtc_realtime")

@app.post("/clear_history")
async def clear_history(api_key: str = Depends(get_api_key)):
    conversation_history.clear()
    return {"status": "cleared"}

@app.get("/openai_ephemeral_key")
async def get_openai_ephemeral_key(api_key: str = Depends(get_api_key)):
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    logger.info("[openai_ephemeral_key] Returning ephemeral key")
    return {"client_secret": {"value": api_key}}


@app.post("/openai_realtime_proxy")
async def proxy_openai_realtime(request: Request, api_key: str = Depends(get_api_key)):
    """
    Proxy endpoint to relay WebRTC connection to OpenAI API.
    This avoids CORS issues when connecting directly from the browser.
    """
    try:
        # Get the API key
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
    

