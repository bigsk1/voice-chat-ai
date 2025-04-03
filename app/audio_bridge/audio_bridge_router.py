"""
Audio Bridge Router
Provides FastAPI routes for WebRTC signaling and audio exchange
"""

import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Response
from typing import Dict

from .audio_bridge_server import audio_bridge
from .audio_processor import audio_processor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/audio-bridge", tags=["audio-bridge"])

# Store WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}

@router.get("/status")
async def get_status(request: Request, response: Response):
    """Get the status of the audio bridge"""
    # Skip logging for status check endpoints
    request.state.skip_log = True
    request.state.access_log = False
    
    # Tell FastAPI not to log this request
    response.headers["x-no-log"] = "true"
    
    # Get status from the bridge server
    status = audio_bridge.get_status()
    
    return status

@router.post("/register")
async def register_client(request: Request):
    """Register a new client with the audio bridge"""
    if not audio_bridge.is_enabled():
        return {"status": "error", "message": "Audio bridge is disabled"}
    
    try:
        body = await request.json()
        client_id = body.get("client_id")
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
        
        success = await audio_bridge.register_client(client_id)
        
        if success:
            return {"status": "success", "client_id": client_id}
        else:
            return {"status": "error", "message": "Failed to register client"}
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/unregister")
async def unregister_client(request: Request):
    """Unregister a client from the audio bridge"""
    if not audio_bridge.is_enabled():
        return {"status": "error", "message": "Audio bridge is disabled"}
    
    try:
        body = await request.json()
        client_id = body.get("client_id")
        
        if not client_id:
            return {"status": "error", "message": "Client ID is required"}
        
        success = await audio_bridge.unregister_client(client_id)
        
        if success:
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Failed to unregister client"}
    except Exception as e:
        logger.error(f"Error unregistering client: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/signaling")
async def handle_signaling(request: Request):
    """Handle WebRTC signaling messages"""
    if not audio_bridge.is_enabled():
        return {"status": "error", "message": "Audio bridge is disabled"}
    
    try:
        body = await request.json()
        client_id = body.get("client_id")
        message = body.get("message")
        
        if not client_id or not message:
            return {"status": "error", "message": "Client ID and message are required"}
        
        response = await audio_bridge.handle_signaling(client_id, message)
        
        if response:
            return {"status": "success", "response": response}
        else:
            return {"status": "error", "message": "Failed to handle signaling message"}
    except Exception as e:
        logger.error(f"Error handling signaling: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/receive-audio")
async def receive_audio(request: Request, response: Response):
    """Receive audio from a client"""
    if not audio_bridge.is_enabled():
        return {"status": "error", "message": "Audio bridge is disabled"}
    
    try:
        body = await request.json()
        client_id = body.get("client_id")
        
        if not client_id:
            return {"status": "error", "message": "Client ID is required"}
        
        # Receive audio from client
        webrtc_audio = await audio_bridge.receive_audio(client_id)
        
        if webrtc_audio:
            # Convert WebRTC audio to WAV
            wav_audio = audio_processor.convert_webrtc_audio_to_wav(webrtc_audio)
            
            # Set response headers for WAV audio
            response.headers["Content-Type"] = "audio/wav"
            response.headers["Content-Disposition"] = f"attachment; filename=audio_{client_id}.wav"
            
            return Response(content=wav_audio, media_type="audio/wav")
        else:
            return {"status": "error", "message": "No audio available"}
    except Exception as e:
        logger.error(f"Error receiving audio: {e}")
        return {"status": "error", "message": str(e)}

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio communication"""
    if not audio_bridge.is_enabled():
        await websocket.close(code=1000, reason="Audio bridge is disabled")
        return
    
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Register client if not already registered
        if client_id not in audio_bridge.clients:
            success = await audio_bridge.register_client(client_id)
            if not success:
                await websocket.close(code=1000, reason="Failed to register client")
                return
        
        # Store the WebSocket connection
        websocket_connections[client_id] = websocket
        
        # Main WebSocket loop
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle signaling messages
                if "type" in message:
                    response = await audio_bridge.handle_signaling(client_id, message)
                    
                    if response:
                        await websocket.send_text(json.dumps(response))
                
                # Handle binary audio data
                # In a real implementation, we would handle binary frames as well
        except WebSocketDisconnect:
            # Remove client from active connections
            if client_id in websocket_connections:
                del websocket_connections[client_id]
            
            # Unregister client
            await audio_bridge.unregister_client(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if client_id in websocket_connections:
            del websocket_connections[client_id]
        
        await audio_bridge.unregister_client(client_id) 