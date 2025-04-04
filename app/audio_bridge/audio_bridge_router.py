"""
Audio Bridge Router
Provides FastAPI routes for WebRTC signaling and audio exchange
"""

import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Response
from typing import Dict
import time

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

@router.post("/offer")
async def handle_offer(request: Request):
    """Handle WebRTC offer from client"""
    if not audio_bridge.is_enabled():
        return {"type": "error", "message": "Audio bridge is disabled"}
    
    try:
        # Get the request data
        data = await request.json()
        sdp = data.get("sdp")
        client_id = data.get("client_id")
        
        # Check if required fields are present
        if not sdp or not client_id:
            logger.warning("Missing SDP or client ID in offer request")
            return {"type": "error", "message": "Client ID and SDP are required"}
        
        # Register client if not already registered
        if client_id not in audio_bridge.clients_set:
            await audio_bridge.register_client(client_id)
        
        # Handle the WebRTC offer
        result = await audio_bridge.handle_signaling({
            "type": "offer",
            "sdp": sdp,
            "client_id": client_id
        })
        
        return result
    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"type": "error", "message": f"Error handling offer: {str(e)}"}

@router.websocket("/ws/{client_id}")
async def websocket_handler(websocket: WebSocket, client_id: str):
    """WebSocket handler for audio bridge signaling"""
    if not audio_bridge.is_enabled():
        await websocket.close(code=1000, reason="Audio bridge is disabled")
        return
        
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established with client {client_id}")
        
        # Register the client
        await audio_bridge.register_client(client_id)
        
        # Send a welcome message
        await websocket.send_json({
            "type": "welcome",
            "client_id": client_id,
            "message": "Connected to audio bridge"
        })
        
        # Keep connection alive until closed
        try:
            while True:
                # Wait for messages from the client
                message = await websocket.receive_text()
                
                try:
                    # Parse the message
                    data = json.loads(message)
                    
                    # Add client_id to message if not present
                    if "client_id" not in data:
                        data["client_id"] = client_id
                    
                    # Handle the message
                    response = await audio_bridge.handle_signaling(data)
                    
                    # Send the response
                    if response:
                        await websocket.send_json(response)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from client {client_id}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for client {client_id}")
        finally:
            # Unregister the client
            await audio_bridge.unregister_client(client_id)
            
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        try:
            await websocket.close(code=1011, reason=f"Error: {str(e)}")
        except:
            pass

@router.post("/test")
async def handle_test(request: Request):
    """Test endpoint for the audio bridge"""
    if not audio_bridge.is_enabled():
        return {"status": "error", "message": "Audio bridge is disabled"}
    
    try:
        body = await request.json()
        client_id = body.get("client_id")
        
        if not client_id:
            return {"status": "error", "message": "Client ID is required"}
        
        # Get the current status
        status = audio_bridge.get_status()
        
        # Check if this client is registered
        client_registered = client_id in audio_bridge.clients_set
        
        # Check if we have any audio data from this client
        audio_data_available = False
        if client_id in audio_bridge.client_audio and audio_bridge.client_audio[client_id]:
            audio_data_available = True
        
        # Is this client streaming?
        is_streaming = audio_bridge.is_client_streaming.get(client_id, False)
        
        # Return test response
        return {
            "status": "success",
            "bridge_status": status,
            "client_registered": client_registered,
            "audio_data_available": audio_data_available,
            "is_streaming": is_streaming,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        return {"status": "error", "message": str(e)} 