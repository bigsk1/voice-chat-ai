from fastapi import FastAPI, UploadFile, File, Depends, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
from typing import Optional
import os

from core import transcribe_audio_bytes, generate_response_text, synthesize_text

app = FastAPI()


def get_api_key(authorization: str = None, api_key: str = None):
    if api_key:
        return api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    return os.getenv("OPENAI_API_KEY")


class TextInput(BaseModel):
    text: str


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key),
):
    audio_bytes = await file.read()
    try:
        text = await transcribe_audio_bytes(audio_bytes, api_key, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text}


@app.post("/api/chat")
async def api_chat(payload: TextInput, api_key: str = Depends(get_api_key)):
    try:
        response_text = await generate_response_text(payload.text, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": response_text}


@app.post("/api/synthesize")
async def api_synthesize(payload: TextInput, api_key: str = Depends(get_api_key)):
    try:
        audio_bytes = await synthesize_text(payload.text, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=audio_bytes, media_type="audio/wav")
