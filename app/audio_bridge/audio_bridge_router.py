"""
Audio Bridge Router
Provides FastAPI routes for WebRTC signaling and audio exchange
"""

import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Response, File, UploadFile, Form, HTTPException
from typing import Dict
import os
import aiofiles

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
    """Register a client with the audio bridge"""
    try:
        data = await request.json()
        client_id = data.get("client_id")
        
        if not client_id:
            return {"status": "error", "message": "Client ID is required"}
        
        # Register the client
        result = await audio_bridge.register_client(client_id)
        
        if result:
            return {"status": "success", "message": f"Client {client_id} registered"}
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
    if not audio_bridge.enabled:
        return {"type": "error", "message": "Audio bridge is disabled"}
    
    try:
        # Get request data
        data = await request.json()
        
        # Extract the required fields
        sdp = data.get("sdp")
        client_id = data.get("client_id")
        
        # Validate required fields
        if not sdp or not client_id:
            return {"type": "error", "message": "SDP and client ID are required"}
        
        # Handle the WebRTC offer using the signaling handler
        message = {
            "type": "offer",
            "sdp": sdp,
            "client_id": client_id
        }
        
        # Process the offer through the audio bridge server
        result = await audio_bridge.handle_signaling(message)
        return result
    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"type": "error", "message": f"Error: {str(e)}"}

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
async def test_audio_bridge(request: Request):
    """Test the audio bridge connection"""
    try:
        # Get the request data
        data = await request.json()
        client_id = data.get("client_id", "test_client")
        
        # Get detailed status from the audio bridge
        test_response = await audio_bridge.get_test_response(data)
        
        # Return the test response
        return test_response
    except Exception as e:
        logger.error(f"Error testing audio bridge: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error testing audio bridge: {str(e)}",
            "bridge_enabled": audio_bridge.enabled
        }

@router.post("/upload-audio")
async def upload_audio(request: Request, audio: UploadFile = File(...), client_id: str = Form(...)):
    """Handle direct audio uploads from clients"""
    
    logger.info(f"Received audio upload from client {client_id}, size: {request.headers.get('content-length', 'unknown')} bytes")
    
    if not audio.filename.endswith(('.webm', '.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Create a temporary file to store the uploaded audio
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, f"direct_upload_{client_id}_{uuid.uuid4()}.webm")
    
    try:
        # Save the uploaded file
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)
        
        logger.info(f"Saved uploaded audio to {temp_file_path}")
        
        # Process the audio file
        result = await audio_bridge.process_fallback_audio(client_id, temp_file_path)
        
        return {"status": "success", "message": "Audio uploaded and processed successfully", "result": result}
    except Exception as e:
        logger.error(f"Error processing uploaded audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Still return a 200 to avoid client-side errors, but indicate failure
        return {"status": "error", "message": f"Error processing audio: {str(e)}"}

@router.get("/status")
async def get_status():
    """Get the status of the audio bridge server"""
    return {
        "enabled": audio_bridge.is_enabled(),
        "num_clients": len(audio_bridge.clients_set) if audio_bridge.is_enabled() else 0,
    } 