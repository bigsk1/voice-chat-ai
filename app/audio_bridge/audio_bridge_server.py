"""
Audio Bridge Server
Handles WebRTC connections for remote audio access
Allows remote clients to use their microphones and speakers with the server
"""

import asyncio
import os
import logging
import json
import threading
from typing import Dict, Set, Optional, Any, Deque, Union
from collections import deque, defaultdict
import numpy as np
import subprocess
import aiohttp_cors
from aiohttp import web
import time
import uuid
from queue import Queue
import pathlib
from ..app import output_dir, TTS_PROVIDER
import websockets

# Try to import WebRTC components
AIORTC_AVAILABLE = False
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay, MediaBlackhole, MediaRecorder
    from aiortc.mediastreams import AudioFrame, MediaStreamError
    AIORTC_AVAILABLE = True
except ImportError:
    logging.warning("aiortc not available, WebRTC audio bridge disabled")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug mode - set to True for more verbose logging
DEBUG_MODE = os.getenv("DEBUG_AUDIO_BRIDGE", "false").lower() == "true"

# Setup outputs directory for temporary files
def get_outputs_dir():
    """Get the path to the outputs directory"""
    # Get the root directory of the application
    root_dir = pathlib.Path(__file__).parent.parent.parent
    outputs_dir = root_dir / "outputs"
    
    # Create the outputs directory if it doesn't exist
    if not outputs_dir.exists():
        outputs_dir.mkdir(exist_ok=True)
        logger.info(f"Created outputs directory at {outputs_dir}")
    
    return outputs_dir

# WebRTC connections store
class AudioBridgeServer:
    """
    Audio Bridge Server
    Handles WebRTC audio bridge between web clients and server
    """
    
    def __init__(self):
        """Initialize the audio bridge"""
        # Initialize websocket connections dictionary
        self.ws_connections = {}
        
        # Client tracking
        self.clients_set = set()
        self.client_audio = defaultdict(list)
        self.audio_pcm = defaultdict(deque)
        self.connections = {}
        self.data_channels = {}
        self.track_processors = {}
        self.last_audio_time = {}
        self.is_client_streaming = defaultdict(bool)
        self.message_queue = Queue()
        self.enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
        self.port = int(os.getenv("AUDIO_BRIDGE_PORT", "8081"))  # Use 8081 as default
        self.ssl_context = None
        self.client_types = {}     # Track client types
        
        # Audio buffer for each client
        self.audio_buffer = {}
        
        # Peer connection tracking
        self.pcs = set()
        
        # For signaling
        self.waiting_ice_candidates = defaultdict(list)
        
        # Setup outputs directory for temporary files
        self.outputs_dir = get_outputs_dir()
        logger.info(f"Using outputs directory for audio bridge temporary files: {self.outputs_dir}")
        
        # Create certificate if using HTTPS
        if os.getenv("ENABLE_HTTPS", "false").lower() == "true":
            self.ssl_context = self._create_ssl_context()
            
        # Check for FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            logger.info("FFmpeg is available for audio bridge")
            self.ffmpeg_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available, some audio processing features may be limited")
            self.ffmpeg_available = False
        
        # Initialize WebRTC media relay if available
        if AIORTC_AVAILABLE:
            try:
                logger.info("Initializing WebRTC MediaRelay for audio bridge")
                self.relay = MediaRelay()
                logger.info("MediaRelay initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing MediaRelay: {e}")
                self.relay = None
        else:
            self.relay = None
        
        # Debug mode
        self.debug_mode = DEBUG_MODE
        
        # Initialize event loop
        self.loop = None
        
        self._init_audio_dirs()
    
    def _init_audio_dirs(self):
        """Initialize any additional directories or setup needed"""
        # This method is empty as the existing _init_audio_dirs method is called in __init__
        pass
    
    async def register_client(self, client_id: str) -> bool:
        """
        Register a new client with the server.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Registering client: {client_id}")
        self.clients_set.add(client_id)
        self.client_audio[client_id] = deque(maxlen=100)  # Store up to 100 audio chunks
        self.audio_pcm[client_id] = deque(maxlen=100)     # Store up to 100 PCM chunks
        self.last_audio_time[client_id] = asyncio.get_event_loop().time()
        self.is_client_streaming[client_id] = False
        return True
        
    async def unregister_client(self, client_id: str) -> bool:
        """Unregister a client from the server"""
        if client_id in self.clients_set:
            logger.info(f"Unregistering client: {client_id}")
            self.clients_set.remove(client_id)
            
            # Close peer connection if exists
            if client_id in self.connections:
                conn = self.connections[client_id]
                await conn.close()
                del self.connections[client_id]
            
            # Clean up audio data
            if client_id in self.client_audio:
                del self.client_audio[client_id]
                
            if client_id in self.audio_pcm:
                del self.audio_pcm[client_id]
                
            if client_id in self.last_audio_time:
                del self.last_audio_time[client_id]
            
            if client_id in self.is_client_streaming:
                del self.is_client_streaming[client_id]
                
            return True
        return False
        
    async def handle_signaling(self, message):
        """Handle WebRTC signaling message"""
        if not self.enabled:
            logger.warning("Audio bridge is disabled, but received signaling message")
            return {"type": "error", "message": "Audio bridge is disabled"}
            
        if isinstance(message, dict):
            client_id = message.get("client_id")
            if not client_id:
                logger.warning("Received signaling message without client ID")
                return {"type": "error", "message": "Client ID is required"}
            
            # Register client if not already registered
            if client_id not in self.clients_set:
                await self.register_client(client_id)
                
            message_type = message.get("type")
            
            if message_type == "offer":
                logger.info(f"Processing offer from {client_id}")
                sdp = message.get("sdp")
                
                if not sdp:
                    return {"type": "error", "message": "SDP is required for offer"}
                
                # Create peer connection if it doesn't exist
                peer_connection = self.connections.get(client_id)
                if not peer_connection:
                    peer_connection = RTCPeerConnection()
                    self.connections[client_id] = peer_connection
                    
                    @peer_connection.on("connectionstatechange")
                    async def on_connectionstatechange():
                        logger.info(f"Connection state for {client_id}: {peer_connection.connectionState}")
                        
                    @peer_connection.on("datachannel")
                    def on_datachannel(channel):
                        logger.info(f"Data channel established with {client_id}")
                        self.data_channels[client_id] = channel
                        
                        @channel.on("message")
                        def on_message(message):
                            logger.info(f"Received message from {client_id}: {message}")
                            
                    @peer_connection.on("track")
                    def on_track(track):
                        logger.info(f"Received track from {client_id}: {track.kind}")
                        
                        if track.kind == "audio":
                            try:
                                # Create a relay for the track
                                relayed_track = self.relay.subscribe(track)
                                
                                # Create a processor for this track
                                processor = AudioTrackProcessor(client_id, self)
                                processor.track = relayed_track
                                
                                # Store processor
                                self.track_processors[client_id] = processor
                                
                                # Mark client as streaming
                                self.is_client_streaming[client_id] = True
                                self.last_audio_time[client_id] = asyncio.get_event_loop().time()
                                
                                logger.info(f"Successfully set up audio track processor for client {client_id}")
                                
                            except Exception as e:
                                logger.error(f"Error setting up audio track: {e}")
                
                # Set remote description (the offer)
                try:
                    offer = RTCSessionDescription(sdp=sdp, type="offer")
                    await peer_connection.setRemoteDescription(offer)
                    
                    # Create answer
                    answer = await peer_connection.createAnswer()
                    await peer_connection.setLocalDescription(answer)
                    
                    # Return the answer
                    return {
                        "type": "answer",
                        "sdp": peer_connection.localDescription.sdp
                    }
                except Exception as e:
                    logger.error(f"Error creating answer for {client_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"type": "error", "message": f"Error creating answer: {str(e)}"}
                    
            elif message_type == "ice-candidate":
                logger.info(f"Processing ICE candidate from {client_id}")
                candidate = message.get("candidate")
                
                if not candidate:
                    return {"type": "error", "message": "Candidate data is required"}
                
                # Get the peer connection
                peer_connection = self.connections.get(client_id)
                if not peer_connection:
                    return {"type": "error", "message": "No connection found for this client"}
                
                try:
                    if isinstance(candidate, dict):
                        # Get the candidate string
                        candidate_str = candidate.get("candidate", "")
                        
                        if not candidate_str:
                            # This is an "end of candidates" signal, which is normal
                            logger.info(f"Received empty ICE candidate (end of candidates) for {client_id}")
                            return {"type": "success", "message": "Empty ICE candidate acknowledged"}
                        
                        # Extract required parameters from the candidate string
                        # Parse the SDP candidate string to extract the required fields
                        # Format: candidate:foundation component protocol priority ip port type ...
                        parts = candidate_str.split(" ")
                        if len(parts) < 8:
                            return {"type": "error", "message": f"Invalid candidate format: {candidate_str}"}
                        
                        # The first part contains "candidate:foundation"
                        foundation = parts[0].split(":")[1] if ":" in parts[0] else parts[0]
                        component = int(parts[1])
                        protocol = parts[2]
                        priority = int(parts[3])
                        ip = parts[4]
                        port = int(parts[5])
                        candidate_type = parts[7]
                        
                        # Optional related address and port
                        related_address = None
                        related_port = None
                        
                        # Look for related address and port in the remaining parts
                        for i in range(8, len(parts) - 1, 2):
                            if parts[i] == "raddr":
                                related_address = parts[i + 1]
                            elif parts[i] == "rport":
                                related_port = int(parts[i + 1])
                        
                        # Get sdpMid and sdpMLineIndex
                        sdp_mid = candidate.get("sdpMid")
                        sdp_mline_index = candidate.get("sdpMLineIndex")
                        
                        # Create the RTCIceCandidate with all required parameters
                        ice_candidate = RTCIceCandidate(
                            component=component,
                            foundation=foundation,
                            ip=ip,
                            port=port,
                            priority=priority,
                            protocol=protocol,
                            type=candidate_type,
                            relatedAddress=related_address,
                            relatedPort=related_port,
                            sdpMid=sdp_mid,
                            sdpMLineIndex=sdp_mline_index
                        )
                        
                        # Add the candidate to the peer connection
                        await peer_connection.addIceCandidate(ice_candidate)
                        logger.info(f"Added ICE candidate for {client_id}")
                        return {"type": "success", "message": "ICE candidate added"}
                    else:
                        return {"type": "error", "message": "Invalid ICE candidate format - expected a dictionary"}
                    
                except TypeError as e:
                    # Log details about the candidate data to debug the TypeError
                    logger.error(f"TypeError adding ICE candidate for {client_id}: {e}")
                    logger.error(f"Candidate data: {candidate}")
                    return {"type": "error", "message": f"Invalid ICE candidate format: {str(e)}"}
                except Exception as e:
                    logger.error(f"Error adding ICE candidate for {client_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"type": "error", "message": f"Error adding ICE candidate: {str(e)}"}
                    
            else:
                logger.warning(f"Unhandled message type '{message_type}' from {client_id}")
                return {"type": "error", "message": f"Unhandled message type: {message_type}"}
        else:
            logger.warning(f"Received non-dict message: {message}")
            return {"type": "error", "message": "Invalid message format"}
        
    async def send_audio(self, client_id: str, audio_data: Union[bytes, str], is_url: bool = False) -> bool:
        """
        Send audio data to a client
        
        Args:
            client_id: The client ID to send audio to
            audio_data: Either binary audio data or a URL string
            is_url: If True, audio_data is a URL string
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or client_id not in self.clients_set:
            logger.warning(f"Attempted to send audio to unregistered client {client_id}")
            return False
        
        try:
            # If it's a URL, send it as a message with audio_url parameter
            if is_url:
                logger.info(f"Sending audio URL to client {client_id}: {audio_data}")
                await self.send_message_to_client(client_id, json.dumps({
                    "action": "audio",
                    "audio_url": audio_data,
                    "play_on_client": True
                }))
                return True
            
            # If it's binary data, send as WebSocket binary message
            logger.info(f"Sending {len(audio_data)} bytes of audio to client {client_id}")
            
            # Try sending via data channel if available
            data_channel = self.data_channels.get(client_id)
            if data_channel and data_channel.readyState == "open":
                try:
                    data_channel.send(audio_data)
                    logger.info(f"Sent audio via data channel to {client_id}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not send via data channel, falling back to WebSocket: {e}")
            
            # Fallback to WebSocket
            client_ws = self.ws_connections.get(client_id)
            if client_ws:
                await client_ws.send_bytes(audio_data)
                logger.info(f"Sent audio via WebSocket to {client_id}")
                return True
                
            logger.warning(f"No available connection to send audio to client {client_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error sending audio to client {client_id}: {e}")
            return False
        
    async def receive_audio(self, client_id: str) -> Optional[bytes]:
        """Receive audio data from a client"""
        if not self.enabled or client_id not in self.clients_set:
            return None
        
        # First check if we have any PCM data - this is preferred
        if client_id in self.audio_pcm and self.audio_pcm[client_id]:
            try:
                # Get the audio data
                return self.audio_pcm[client_id].popleft()
            except IndexError:
                # No data available
                pass
        
        # Then check if we have any audio data at all
        if client_id in self.client_audio and self.client_audio[client_id]:
            try:
                # Get the audio data
                return self.client_audio[client_id].popleft()
            except IndexError:
                # No data available
                pass
        
        # Special case: if the client is connected but we're not getting audio data,
        # create a small amount of silent audio to keep the pipeline moving
        # This helps with low-volume microphones or connections with issues
        if client_id in self.is_client_streaming and self.is_client_streaming[client_id]:
            # Check when we last received audio
            now = asyncio.get_event_loop().time()
            last_time = self.last_audio_time.get(client_id, 0)
            
            # If it's been more than 2 seconds since last audio
            if now - last_time > 2:
                # Create a silent audio chunk (16-bit PCM, 1 channel, 16kHz)
                # This is a sample of silence (1000 samples at 16kHz = ~60ms)
                if DEBUG_MODE:
                    logger.info(f"Creating synthetic silence for client {client_id} - no audio received for {now - last_time:.1f}s")
                
                # Create 1000 samples of very quiet noise (not complete silence)
                # This helps keep the audio pipeline going
                import numpy as np
                samples = 1000
                # Generate very low volume random noise (-120 to 120 amplitude)
                audio_array = np.random.randint(-120, 120, samples, dtype=np.int16)
                synthetic_audio = audio_array.tobytes()
                
                # Update last audio time
                self.last_audio_time[client_id] = now
                
                # Save synthetic audio to a .wav file in the outputs directory
                self._save_debug_audio(synthetic_audio, f"synthetic_silence_{client_id}")
                
                return synthetic_audio
        
        # No audio data available
        return None
        
    def _save_debug_audio(self, audio_data: bytes, prefix: str):
        """Save audio data to a .wav file in the outputs directory for debugging"""
        if not audio_data:
            return None
            
        try:
            # Generate filename
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}.wav"
            filepath = self.outputs_dir / filename
            
            # Convert raw PCM to WAV
            import wave
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(audio_data)
                
            logger.info(f"Saved debug audio to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")
            return None
        
    async def process_fallback_audio(self, client_id: str, audio_data_or_path: Union[bytes, str]) -> bool:
        """
        Process audio data received from a client
        Used by both WebRTC and WebSocket connections
        
        Args:
            client_id: The client ID that sent the audio
            audio_data_or_path: The raw audio data or a path to an audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or client_id not in self.clients_set:
            logger.warning(f"Attempted to process audio for unregistered client {client_id}")
            return False
        
        try:
            # Check if we have a file path or binary data
            if isinstance(audio_data_or_path, str):
                # We have a file path
                logger.info(f"Processing audio file from path: {audio_data_or_path}")
                
                # Process with AudioProcessor
                from .audio_processor import audio_processor
                audio_filename = await audio_processor.process_audio_file(
                    audio_data_or_path,
                    client_id,
                    silence_threshold=50,
                    silence_duration=3.0
                )
                
                if audio_filename:
                    logger.info(f"Audio processing complete, transcribing file: {audio_filename}")
                    
                    # Get a reference to the app for transcription services
                    from .. import app
                    try:
                        # Use the transcribe_audio function from app.py
                        transcription = await app.transcribe_with_model(
                            audio_filename, 
                            use_local=False  # Use API for better accuracy with remote audio
                        )
                        
                        if transcription:
                            logger.info(f"Transcription: {transcription}")
                            
                            # Send the transcription to the client
                            await self.send_message_to_client(client_id, json.dumps({
                                "action": "transcription",
                                "text": transcription
                            }))
                            
                            # Process the message with AI
                            from ..app_logic import process_text
                            ai_response = await process_text(transcription)
                            
                            # Send the AI response to the client
                            await self.send_message_to_client(client_id, json.dumps({
                                "action": "ai_response",
                                "text": ai_response
                            }))
                            
                            # Log success
                            logger.info(f"Successfully processed audio from client {client_id}")
                            return True
                        else:
                            logger.warning(f"Transcription failed for {audio_filename}")
                            return False
                    except Exception as e:
                        logger.error(f"Error in transcription process: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return False
                else:
                    logger.warning(f"Audio processing did not produce a valid file for {audio_data_or_path}")
                    return False
            else:
                # We have binary data
                logger.info(f"Received binary audio from client {client_id}, {len(audio_data_or_path)} bytes")
                
                # Store the audio data for later retrieval
                if client_id in self.client_audio:
                    self.client_audio[client_id].append(audio_data_or_path)
                    self.last_audio_time[client_id] = asyncio.get_event_loop().time()
                    self.is_client_streaming[client_id] = True
                    logger.info(f"Stored audio from client {client_id}")
                    
                    # Save a copy to the outputs directory for debugging
                    self._save_debug_audio(audio_data_or_path, f"audio_from_client_{client_id}")
                
                return True
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def is_enabled(self):
        """Return whether the audio bridge is enabled"""
        return self.enabled
        
    def get_status(self):
        """Get the status of the audio bridge"""
        active_clients = 0
        
        # Count active clients (those that are streaming audio)
        for client_id in self.clients_set:
            if client_id in self.is_client_streaming and self.is_client_streaming[client_id]:
                active_clients += 1
        
        return {
            "status": "active" if self.enabled else "disabled",
            "enabled": self.enabled,
            "total_clients": len(self.clients_set),
            "active_clients": active_clients,
            "connected": active_clients > 0 or len(self.clients_set) > 0,
            "clients": list(self.clients_set)
        }

    async def handle_status(self, request):
        """Handle status request via HTTP"""
        status = self.get_status()
        return web.json_response(status)

    async def handle_offer(self, request):
        """Handle WebRTC offer from client via HTTP"""
        if not AIORTC_AVAILABLE:
            return web.json_response({"type": "error", "message": "WebRTC is not supported"})
        
        try:
            # Get request data
            data = await request.json()
            sdp = data.get("sdp")
            client_id = data.get("client_id")
            
            # Validate required fields
            if not sdp or not client_id:
                return web.json_response({"type": "error", "message": "SDP and client ID are required"})
            
            # Register client if not already registered
            if client_id not in self.clients_set:
                await self.register_client(client_id)
            
            # Handle the WebRTC offer using our signaling handler
            message = {
                "type": "offer",
                "sdp": sdp,
                "client_id": client_id
            }
            
            result = await self.handle_signaling(message)
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error handling offer via HTTP: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.json_response({"type": "error", "message": f"Error: {str(e)}"})

    async def handle_test(self, request):
        """Handle test request via HTTP"""
        try:
            # Get request data
            data = await request.json()
            client_id = data.get("client_id", "unknown")
            
            # Call the internal method that generates the response
            response = await self.get_test_response(data)
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Error handling test request: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error: {str(e)}",
                "clients": list(self.clients_set)
            })

    async def get_test_response(self, request_data: dict) -> Dict[str, Any]:
        """Generate test response data"""
        client_id = request_data.get("client_id", "unknown")
        
        # Prepare response data
        response = {
            "status": "active" if self.enabled else "disabled",
            "message": "Audio bridge test successful" if self.enabled else "Audio bridge is disabled",
            "client_info": {}
        }
        
        # Add client-specific info if client ID is provided
        if client_id in self.clients_set:
            streaming = self.is_client_streaming.get(client_id, False)
            last_time = self.last_audio_time.get(client_id, 0)
            now = asyncio.get_event_loop().time()
            time_diff = now - last_time if last_time > 0 else 0
            
            # Get audio info
            audio_chunks = len(self.client_audio.get(client_id, []))
            pcm_chunks = len(self.audio_pcm.get(client_id, []))
            
            response["client_info"] = {
                "client_id": client_id,
                "registered": True,
                "streaming": streaming,
                "last_audio_received": f"{time_diff:.1f} seconds ago" if last_time > 0 else "never",
                "audio_chunks": audio_chunks,
                "pcm_chunks": pcm_chunks
            }
        else:
            response["client_info"] = {
                "client_id": client_id,
                "registered": False,
                "message": "Client not registered with audio bridge"
            }
            
            # Include client set info
            response["registered_clients"] = list(self.clients_set)
            
        return response

    async def websocket_handler(self, request):
        """WebSocket handler for audio bridge signaling"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Get client ID from URL
        client_id = request.match_info.get('client_id', str(uuid.uuid4()))
        
        # Register client
        self.add_client(client_id, ws)
        self.ws_connections[client_id] = ws
        logger.info(f"WebSocket connection established with client {client_id}")
        
        # Send welcome message
        await ws.send_json({
            "type": "welcome",
            "client_id": client_id,
            "message": "Connected to audio bridge"
        })
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug(f"Received JSON message from client {client_id}")
                        
                        # Handle client identification
                        if data.get("type") == "identify":
                            client_type = data.get("client_type", "unknown")
                            logger.info(f"Client {client_id} identified as type: {client_type}")
                            
                            # Store client type for later use
                            self.client_types[client_id] = client_type
                            
                            # Send acknowledgement
                            await ws.send_json({
                                "type": "ack",
                                "message": f"Identified as {client_type}"
                            })
                        else:
                            # Handle other message types using our signaling handler
                            result = await self.handle_signaling(data)
                            await ws.send_json(result)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON message: {e}")
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                        
                elif msg.type == web.WSMsgType.BINARY:
                    # Handle binary message (audio data)
                    logger.debug(f"Received binary data from client {client_id}, length: {len(msg.data)}")
                    # Process the audio data
                    from .. import app as app_module
                    await self.handle_data(client_id, msg.data, app=app_module)
                
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with error: {ws.exception()}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in WebSocket handler for client {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Clean up the client connection
            self.remove_client(client_id)
            logger.info(f"Removed client {client_id} from active connections")
        
        return ws

    async def run_server(self):
        """Run the WebRTC audio bridge server"""
        if not AIORTC_AVAILABLE:
            logger.error("Cannot run WebRTC audio bridge - aiortc is not available")
            return
        
        self.loop = asyncio.get_event_loop()
        # Initialize runner to None at the start to ensure it's defined
        self.runner = None
        
        app = web.Application()
        
        # Define routes with explicit handler methods
        app.router.add_get("/audio-bridge/status", self.handle_status)
        app.router.add_get("/audio-bridge/ws/{client_id}", self.websocket_handler)
        app.router.add_post("/audio-bridge/offer", self.handle_offer)
        
        # Add direct HTTP fallback for audio playback
        app.router.add_post("/audio-bridge/direct-play", self.handle_direct_play)
        app.router.add_post("/audio-bridge/check-client", self.handle_check_client)
        
        # Add test endpoint
        app.router.add_post("/audio-bridge/test", self.handle_test)
        
        # Apply CORS to all routes
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            )
        })
        
        # Apply CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        logger.info(f"Starting WebRTC audio bridge server on port {self.port}")
        
        try:
            # Check if we have SSL context for HTTPS
            if self.ssl_context:
                logger.info("Using SSL for WebRTC audio bridge")
                self.runner = web.AppRunner(app)
                await self.runner.setup()
                site = web.TCPSite(self.runner, "0.0.0.0", self.port, ssl_context=self.ssl_context)
                await site.start()
                logger.info(f"WebRTC audio bridge server started on port {self.port} with SSL")
            else:
                # No SSL
                self.runner = web.AppRunner(app)
                await self.runner.setup()
                site = web.TCPSite(self.runner, "0.0.0.0", self.port)
                await site.start()
                logger.info(f"WebRTC audio bridge server started on port {self.port} (no SSL)")
            
            # Keep the server running
            while True:
                await asyncio.sleep(3600)  # Check every hour
        except asyncio.CancelledError:
            logger.info("WebRTC audio bridge server shutting down due to cancellation")
        except Exception as e:
            logger.error(f"Error in WebRTC audio bridge server: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Ensure self.runner is defined and not None before calling cleanup()
            if hasattr(self, 'runner') and self.runner is not None:
                try:
                    await self.runner.cleanup()
                    logger.info("Application runner cleaned up during shutdown")
                except Exception as e:
                    logger.error(f"Error during runner cleanup: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.info("No runner to clean up during shutdown")
            logger.info("WebRTC audio bridge server stopped")
            
    async def handle_direct_play(self, request):
        """Handle direct playback request via HTTP"""
        try:
            data = await request.json()
            client_id = data.get("client_id")
            text = data.get("text")
            
            if not client_id or not text:
                return web.json_response({
                    "success": False, 
                    "message": "Missing client_id or text parameter"
                })
            
            # Log the request
            logger.info(f"Direct playback request from client {client_id}, text length: {len(text)}")
            
            try:
                # Create a temporary WAV file path in the outputs directory
                outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
                output_path = os.path.join(outputs_dir, f"direct_play_{int(time.time())}.wav")
                
                # Get TTS provider from environment
                tts_provider = os.getenv("TTS_PROVIDER", "openai")
                
                # Generate audio file using the chosen TTS provider
                if tts_provider == "openai":
                    from app.app import openai_text_to_speech
                    await openai_text_to_speech(text, output_path)
                elif tts_provider == "elevenlabs":
                    from app.app import elevenlabs_text_to_speech
                    await elevenlabs_text_to_speech(text, output_path)
                else:
                    # Can't generate audio with this provider via direct API
                    return web.json_response({
                        "success": False,
                        "message": f"TTS provider {tts_provider} not supported for direct playback",
                        "use_browser_tts": True,
                        "text": text
                    })
                
                # Check if file was generated
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Copy the file to the static directory for URL access
                    from app.app import copy_to_static_output
                    audio_url = copy_to_static_output(output_path)
                    
                    # Return the audio URL for client-side playback
                    return web.json_response({
                        "success": True,
                        "audio_url": audio_url,
                        "message": "Audio generated successfully"
                    })
            except Exception as e:
                logger.error(f"Error generating audio: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Fallback to browser TTS
            return web.json_response({
                "success": False,
                "message": "Audio generation failed, use browser TTS",
                "use_browser_tts": True,
                "text": text
            })
            
        except Exception as e:
            logger.error(f"Error handling direct playback request: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return web.json_response({
                "success": False,
                "message": f"Server error: {str(e)}"
            })
            
    async def handle_check_client(self, request):
        """Check if a client is registered and can receive audio"""
        try:
            data = await request.json()
            client_id = data.get("client_id")
            
            if not client_id:
                return web.json_response({
                    "success": False, 
                    "message": "Missing client_id parameter"
                })
                
            # Check if client is registered
            is_registered = client_id in self.clients_set
            has_ws_connection = client_id in self.ws_connections
            has_data_channel = client_id in self.data_channels
            
            return web.json_response({
                "success": True,
                "client_id": client_id,
                "registered": is_registered,
                "has_ws_connection": has_ws_connection,
                "has_data_channel": has_data_channel,
                "can_receive_audio": has_ws_connection or has_data_channel
            })
            
        except Exception as e:
            logger.error(f"Error checking client status: {str(e)}")
            return web.json_response({
                "success": False,
                "message": f"Server error: {str(e)}"
            })

    async def stop_server(self):
        """Stop the WebRTC audio bridge server"""
        logger.info("Stopping WebRTC audio bridge server...")
        
        # Close all peer connections
        for client_id, conn in list(self.connections.items()):
            try:
                await conn.close()
                logger.info(f"Closed peer connection for client {client_id}")
            except Exception as e:
                logger.error(f"Error closing peer connection for client {client_id}: {e}")
        
        # Clear all collections
        self.connections.clear()
        self.data_channels.clear()
        self.ws_connections.clear()
        self.track_processors.clear()
        self.clients_set.clear()
        self.client_audio.clear()
        self.audio_pcm.clear()
        self.last_audio_time.clear()
        self.is_client_streaming.clear()
        
        # Clean up runner if it exists
        cleanup_success = False
        if hasattr(self, 'runner') and self.runner is not None:
            try:
                await self.runner.cleanup()
                logger.info("Application runner cleaned up")
                cleanup_success = True
            except Exception as e:
                logger.error(f"Error cleaning up application runner: {e}")
        else:
            logger.info("No runner to clean up - server may not have started fully")
            
        # Explicitly set runner to None to prevent future cleanup attempts
        self.runner = None
        
        logger.info("WebRTC audio bridge server stopped")
        return True

    async def _handle_test_request(self, request):
        """Private helper to properly handle test requests"""
        try:
            data = await request.json()
        except:
            data = {}
            
        client_id = data.get("client_id", "unknown")
        
        # Prepare response data
        response = {
            "status": "active" if self.enabled else "disabled",
            "message": "Audio bridge test successful" if self.enabled else "Audio bridge is disabled",
            "client_info": {}
        }
        
        # Add client-specific info if client ID is provided
        if client_id in self.clients_set:
            streaming = self.is_client_streaming.get(client_id, False)
            last_time = self.last_audio_time.get(client_id, 0)
            now = asyncio.get_event_loop().time()
            time_diff = now - last_time if last_time > 0 else 0
            
            # Get audio info
            audio_chunks = len(self.client_audio.get(client_id, []))
            pcm_chunks = len(self.audio_pcm.get(client_id, []))
            
            response["client_info"] = {
                "client_id": client_id,
                "registered": True,
                "streaming": streaming,
                "last_audio_received": f"{time_diff:.1f} seconds ago" if last_time > 0 else "never",
                "audio_chunks": audio_chunks,
                "pcm_chunks": pcm_chunks
            }
        else:
            response["client_info"] = {
                "client_id": client_id,
                "registered": False,
                "message": "Client not registered with audio bridge"
            }
            
            # Include client set info
            response["registered_clients"] = list(self.clients_set)
            
        return web.json_response(response)

    async def handle_track(self, client_id, track):
        """Handle incoming track from client"""
        if track.kind == "audio":
            logger.info(f"Received audio track from client {client_id}")
            
            # Create processor for this track
            processor = AudioTrackProcessor(client_id, self)
            processor.add_listener(track)
            
            # Store the processor
            self.track_processors[client_id] = processor
            
            # Return the processor as the incoming track
            return processor
        else:
            logger.warning(f"Received non-audio track from client {client_id}, kind: {track.kind}")
            return track

    async def process_audio_frames(self, client_id, processor):
        """Actively process audio frames from the track processor"""
        if not processor or not processor.track:
            logger.error(f"Cannot process audio frames - missing processor or track for {client_id}")
            return
            
        logger.info(f"Starting audio frame processing for client {client_id}")
        last_save_time = time.time()
        
        try:
            # Process frames until stopped
            while True:
                try:
                    # Get the next frame
                    frame = await processor.track.recv()
                    
                    # Convert frame to PCM data
                    pcm_data = frame.to_ndarray().tobytes()
                    
                    # Log audio level occasionally
                    if DEBUG_MODE:
                        try:
                            # Calculate audio level from PCM data
                            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                            level = np.mean(np.abs(audio_array))
                            if processor.frame_count % 100 == 0:
                                logger.info(f"Audio frame {processor.frame_count} from client {client_id}, level: {level:.2f}")
                            
                            # If the audio level is very low, log a warning occasionally
                            if level < 25 and processor.frame_count % 100 == 0:
                                logger.warning(f"Very low audio level detected from client {client_id}: {level:.2f}")
                        except Exception as e:
                            if processor.frame_count % 100 == 0:
                                logger.error(f"Error calculating audio level: {e}")
                    
                    # Store the PCM data directly
                    if pcm_data and len(pcm_data) > 0:
                        self.audio_pcm[client_id].append(pcm_data)
                        self.client_audio[client_id].append(pcm_data)
                        
                        # Save audio frames to file periodically
                        now = time.time()
                        if now - last_save_time >= 5.0:  # Save every 5 seconds
                            try:
                                # Save the audio to a WAV file in the outputs directory
                                self._save_debug_audio(pcm_data, f"processed_audio_{client_id}")
                                last_save_time = now
                            except Exception as e:
                                logger.error(f"Error saving processed audio: {e}")
                        
                        # Update tracking
                        processor.frame_count += 1
                        self.last_audio_time[client_id] = asyncio.get_event_loop().time()
                        self.is_client_streaming[client_id] = True
                        
                        # Log occasionally
                        if DEBUG_MODE and processor.frame_count % 100 == 0:
                            logger.info(f"Processed {processor.frame_count} audio frames from client {client_id}")
                    
                except MediaStreamError:
                    # Track ended
                    logger.info(f"Media stream ended for client {client_id}")
                    break
                except asyncio.CancelledError:
                    # Task cancelled
                    logger.info(f"Audio processing cancelled for client {client_id}")
                    break
                except Exception as e:
                    # Other error
                    logger.error(f"Error processing audio frame for client {client_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue processing
                    await asyncio.sleep(0.1)
                
                # Brief pause to avoid tight loop
                await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in audio frame processing loop for client {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Stopped audio frame processing for client {client_id}")
            self.is_client_streaming[client_id] = False

    async def handle_data(self, client_id, data, app=None):
        """Handle incoming audio data from a client"""
        
        logger.info(f"Received data from client {client_id}: {len(data)} bytes")
        
        if len(data) == 0:
            logger.warning(f"Empty data received from client {client_id}")
            return
        
        # Store audio data from client
        self.client_audio[client_id] = data
        
        # Check if this looks like audio data
        try:
            # Check if debug mode is requested by client
            debug_mode = False
            
            # If the data is small enough, it might be a control message
            if len(data) < 1000 and isinstance(data, bytes):
                try:
                    # Try to decode as JSON
                    json_data = json.loads(data.decode('utf-8'))
                    
                    # Check for debug mode flag
                    if json_data.get('debug_mode') is True:
                        debug_mode = True
                        logger.info(f"Debug mode requested by client {client_id}")
                    
                    # Check for other control messages
                    if json_data.get('type') == 'simulate_audio':
                        logger.info(f"Client {client_id} requested simulated audio processing (debug_mode={debug_mode})")
                        # Process empty audio with debug mode
                        debug_mode = True
                    
                    # This was a control message, not audio data
                    if 'type' in json_data:
                        return
                except:
                    # Not JSON, assume it's binary audio data
                    pass
            
            # Check if this is audio data
            is_present = self.audio_processor.process_audio(
                data, 
                client_id=client_id,
                debug_mode=debug_mode   # Pass the debug mode flag
            )
            
            if is_present:
                logger.info(f"Audio detected from client {client_id}, processing for transcription")
                
                # Get the transcription
                audio_filepath = self.audio_processor.get_latest_audio_file(client_id)
                
                if not audio_filepath or not os.path.exists(audio_filepath):
                    logger.warning(f"No audio file available for client {client_id}")
                    return
                
                # Transcribe the audio
                transcription = self.transcriber.transcribe(audio_filepath)
                
                if not transcription:
                    logger.warning(f"Failed to transcribe audio from client {client_id}")
                    return
                
                logger.info(f"Transcription from client {client_id}: {transcription}")
                
                # Send transcription to client
                await self.send_message_to_client(client_id, {
                    'type': 'transcription',
                    'text': transcription
                })
                
                # Generate a response
                try:
                    # Get the response from the AI
                    response = app.process_message(transcription, client_id=client_id, remote_playback=True)
                    logger.info(f"AI response for client {client_id}: {response[:100]}...")
                    
                    # Response will be sent by the process_message function
                    # through the audio_bridge_server's send_message_to_client
                    
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    
                    # Send error message to client
                    await self.send_message_to_client(client_id, {
                        'type': 'error',
                        'message': f"Error generating response: {str(e)}"
                    })
            else:
                if debug_mode:
                    logger.info(f"No audio detected from client {client_id}, but debug mode is enabled. Sending empty response.")
                    
                    # In debug mode, still send a message to acknowledge receipt
                    await self.send_message_to_client(client_id, {
                        'type': 'status',
                        'message': 'Audio received in debug mode',
                        'status': 'ok'
                    })
                else:
                    logger.warning(f"No audio detected from client {client_id}")
        
        except Exception as e:
            logger.error(f"Error processing audio data from client {client_id}: {str(e)}")
            traceback.print_exc()

    async def send_message_to_client(self, client_id: str, message: str) -> bool:
        """
        Send a message to a specific client
        
        Args:
            client_id: The client ID to send the message to
            message: The message to send (as a string or JSON string)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or client_id not in self.clients_set:
            logger.warning(f"Attempted to send message to unregistered client {client_id}")
            return False
        
        try:
            logger.info(f"Sending message to client {client_id}: {message[:100]}...")
            
            # Try sending via data channel if available
            data_channel = self.data_channels.get(client_id)
            if data_channel and data_channel.readyState == "open":
                try:
                    data_channel.send(message)
                    logger.info(f"Sent message via data channel to {client_id}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not send via data channel, falling back to WebSocket: {e}")
            
            # Fallback to WebSocket
            client_ws = self.ws_connections.get(client_id)
            if client_ws:
                await client_ws.send_str(message)
                logger.info(f"Sent message via WebSocket to {client_id}")
                return True
            
            # No connection available to send message, log error
            logger.warning(f"No available connection to send message to client {client_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            return False

    def add_client(self, client_id, websocket):
        """Add a new client to the connection pool"""
        self.clients_set.add(client_id)
        self.client_audio[client_id] = deque(maxlen=100)
        self.last_audio_time[client_id] = time.time()
        self.client_types[client_id] = "unknown"
        logger.info(f"Added client {client_id} to connection pool")
        
    def remove_client(self, client_id):
        """Remove a client from the connection pool"""
        if client_id in self.clients_set:
            self.clients_set.remove(client_id)
        if client_id in self.client_audio:
            del self.client_audio[client_id]
        if client_id in self.last_audio_time:
            del self.last_audio_time[client_id]
        if client_id in self.client_types:
            del self.client_types[client_id]
        logger.info(f"Removed client {client_id} from connection pool")
    
    async def send_message(self, client_id, message):
        """Send a message to a client"""
        if client_id in self.clients_set:
            try:
                if isinstance(message, dict):
                    message = json.dumps(message)
                # Use the original send_message_to_client implementation
                if client_id in self.ws_connections:
                    await self.ws_connections[client_id].send_str(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                return False
        else:
            logger.warning(f"Cannot send message to unknown client {client_id}")
            return False
            
    # Backward compatibility method
    async def send_message_to_client(self, client_id, message):
        """Alias for send_message for backward compatibility"""
        return await self.send_message(client_id, message)

class AudioTrackProcessor(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, client_id, server):
        super().__init__()  # Initialize the parent class
        self.client_id = client_id
        self.server = server
        self.track = None
        self.pcm_buffer = bytearray()
        self.frame_count = 0
        self.chunk_count = 0
        self.running = True
        self.last_log_time = time.time()
        self.last_save_time = time.time()
        
        # Create deques for this client if they don't exist
        if client_id not in server.audio_pcm:
            server.audio_pcm[client_id] = deque(maxlen=100)
        if client_id not in server.client_audio:
            server.client_audio[client_id] = deque(maxlen=100)
        
        # Mark client as streaming
        server.is_client_streaming[client_id] = True
        
        # Log creation
        logger.info(f"Created AudioTrackProcessor for client {client_id}")
    
    async def recv(self):
        if self.track is None:
            raise MediaStreamError("Track not set")
            
        frame = await self.track.recv()
        
        # Process the frame
        try:
            self.frame_count += 1
            
            # Convert frame to PCM data
            pcm_data = frame.to_ndarray().tobytes()
            
            # Log audio level more frequently for debugging
            now = time.time()
            if now - self.last_log_time >= 1.0:  # Log every second
                try:
                    # Calculate audio level from PCM data
                    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                    level = np.mean(np.abs(audio_array))
                    logger.info(f"Audio level from client {self.client_id}: {level:.2f}")
                    
                    # If the audio level is very low, log a warning
                    if level < 25:
                        logger.warning(f"Very low audio level detected from client {self.client_id}: {level:.2f}")
                    
                    self.last_log_time = now
                except Exception as e:
                    logger.error(f"Error calculating audio level: {e}")
            
            # Store PCM data more frequently (store smaller chunks for better responsiveness)
            # Buffer PCM data
            self.pcm_buffer.extend(pcm_data)
            
            # Store chunks more frequently (on every frame instead of every 2nd)
            # if self.frame_count % 2 == 0:
            # Store on every frame for better responsiveness
            # Store the chunk
            chunk = bytes(self.pcm_buffer)
            
            # Add to both PCM and regular audio buffers
            if len(chunk) > 0:
                self.server.audio_pcm[self.client_id].append(chunk)
                self.server.client_audio[self.client_id].append(chunk)
                
                # Save debug audio to file every few seconds to avoid too many files
                if now - self.last_save_time >= 5.0:  # Save every 5 seconds
                    try:
                        # Save the audio to a WAV file in the outputs directory
                        self.server._save_debug_audio(chunk, f"audio_frame_{self.client_id}")
                        self.last_save_time = now
                    except Exception as e:
                        logger.error(f"Error saving audio frame: {e}")
                
                # Reset buffer
                self.pcm_buffer = bytearray()
                
                # Update counter and log occasionally
                self.chunk_count += 1
                if DEBUG_MODE and self.chunk_count % 5 == 0:
                    logger.info(f"Stored {self.chunk_count} audio chunks from client {self.client_id}")
            
            # Update last activity timestamp
            self.server.last_audio_time[self.client_id] = asyncio.get_event_loop().time()
            
            # Ensure client is marked as streaming
            if not self.server.is_client_streaming[self.client_id]:
                self.server.is_client_streaming[self.client_id] = True
                logger.info(f"Client {self.client_id} is now streaming audio")
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        # Return the frame unchanged
        return frame
        
    def add_listener(self, track):
        self.track = track
        # Log the track details
        logger.info(f"Adding audio track for client {self.client_id}: {track.kind}")
        # Update status
        self.server.is_client_streaming[self.client_id] = True

# Create a singleton instance
audio_bridge = AudioBridgeServer() 