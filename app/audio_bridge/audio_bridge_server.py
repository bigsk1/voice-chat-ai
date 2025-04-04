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
from typing import Dict, Set, Optional, Any, Deque
from collections import deque, defaultdict
import numpy as np
import subprocess
import aiohttp_cors
from aiohttp import web
import time
import uuid
from queue import Queue

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

# WebRTC connections store
class AudioBridgeServer:
    """
    Audio Bridge Server
    Handles WebRTC audio bridge between web clients and server
    """
    
    def __init__(self):
        """Initialize the audio bridge"""
        self.clients_set = set()
        self.client_audio = defaultdict(deque)
        self.audio_pcm = defaultdict(deque)
        self.connections = {}
        self.data_channels = {}
        self.ws_connections = {}
        self.track_processors = {}
        self.last_audio_time = {}
        self.is_client_streaming = {}
        self.message_queue = Queue()
        self.enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
        self.port = int(os.getenv("AUDIO_BRIDGE_PORT", "8081"))  # Use 8081 as default
        self.ssl_context = None
        
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
    
    async def register_client(self, client_id: str) -> bool:
        """Register a new client with the server"""
        if not self.enabled:
            logger.warning("Attempted to register client while bridge is disabled")
            return False
            
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
                
                # Handle different candidate formats
                try:
                    # Check for sdpCandidate format - this is our custom format
                    if isinstance(candidate, dict) and "sdpCandidate" in candidate:
                        ice_candidate = RTCIceCandidate(
                            sdpMid=candidate.get("sdpMid"),
                            sdpMLineIndex=candidate.get("sdpMLineIndex"),
                            candidate=candidate.get("sdpCandidate")
                        )
                    # Check for standard format
                    elif isinstance(candidate, dict) and "candidate" in candidate:
                        ice_candidate = RTCIceCandidate(
                            sdpMid=candidate.get("sdpMid"),
                            sdpMLineIndex=candidate.get("sdpMLineIndex"),
                            candidate=candidate.get("candidate")
                        )
                    # Direct use of the candidate object
                    else:
                        ice_candidate = RTCIceCandidate(**candidate)
                    
                    await peer_connection.addIceCandidate(ice_candidate)
                    logger.info(f"Added ICE candidate for {client_id}")
                    return {"type": "success", "message": "ICE candidate added"}
                except TypeError as e:
                    # Log details about the candidate data to debug the TypeError
                    logger.error(f"TypeError adding ICE candidate for {client_id}: {e}")
                    logger.error(f"Candidate data: {candidate}")
                    return {"type": "error", "message": f"Invalid ICE candidate format: {str(e)}"}
                except Exception as e:
                    logger.error(f"Error adding ICE candidate for {client_id}: {e}")
                    return {"type": "error", "message": f"Error adding ICE candidate: {str(e)}"}
                    
            else:
                logger.warning(f"Unhandled message type '{message_type}' from {client_id}")
                return {"type": "error", "message": f"Unhandled message type: {message_type}"}
        else:
            logger.warning(f"Received non-dict message: {message}")
            return {"type": "error", "message": "Invalid message format"}
        
    async def send_audio(self, client_id: str, audio_data: bytes) -> bool:
        """Send audio data to a client"""
        if not self.enabled or client_id not in self.clients_set:
            return False
            
        # In a real implementation, this would send audio over WebRTC data channel
        logger.info(f"Would send {len(audio_data)} bytes of audio to {client_id}")
        return True
        
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
                
                return synthetic_audio
        
        # No audio data available
        return None
        
    async def process_fallback_audio(self, client_id: str, audio_data: bytes) -> bool:
        """
        Process audio data received from a client
        Used by both WebRTC and WebSocket connections
        
        Args:
            client_id: The client ID that sent the audio
            audio_data: The raw audio data 
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or client_id not in self.clients_set:
            logger.warning(f"Attempted to process audio for unregistered client {client_id}")
            return False
        
        try:
            # Log the received audio
            logger.info(f"Received audio from client {client_id}, {len(audio_data)} bytes")
            
            # Store the audio data for later retrieval
            if client_id in self.client_audio:
                self.client_audio[client_id].append(audio_data)
                self.last_audio_time[client_id] = asyncio.get_event_loop().time()
                self.is_client_streaming[client_id] = True
                logger.info(f"Stored audio from client {client_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
        
    def is_enabled(self):
        """Return whether the audio bridge is enabled"""
        return self.enabled
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the audio bridge"""
        active_clients = 0
        streaming_clients = 0
        
        # Count active clients based on recent audio reception
        now = asyncio.get_event_loop().time()
        for client_id, last_time in self.last_audio_time.items():
            # Client is active if we've received audio in the last 30 seconds
            if now - last_time < 30:
                active_clients += 1
                
        # Count streaming clients
        for client_id, is_streaming in self.is_client_streaming.items():
            if is_streaming:
                streaming_clients += 1
                
        return {
            "status": "active" if self.enabled else "disabled",
            "total_clients": len(self.clients_set),
            "active_clients": active_clients,
            "streaming_clients": streaming_clients,
            "webrtc_available": AIORTC_AVAILABLE
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
        await self.register_client(client_id)
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
                        data["client_id"] = client_id  # Ensure client ID is set
                        
                        # Handle the message with our signaling handler
                        result = await self.handle_signaling(data)
                        await ws.send_json(result)
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON from client {client_id}")
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        await ws.send_json({"type": "error", "message": str(e)})
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with error: {ws.exception()}")
                    break
        finally:
            # Unregister client on disconnect
            await self.unregister_client(client_id)
            logger.info(f"WebSocket connection closed for client {client_id}")
        
        return ws

    async def run_server(self):
        """Run the WebRTC audio bridge server"""
        if not AIORTC_AVAILABLE:
            logger.error("Cannot run WebRTC audio bridge - aiortc is not available")
            return
        
        self.loop = asyncio.get_event_loop()
        
        app = web.Application()
        
        # Define routes with explicit handler methods
        app.router.add_get("/audio-bridge/status", self.handle_status)
        app.router.add_get("/audio-bridge/ws/{client_id}", self.websocket_handler)
        app.router.add_post("/audio-bridge/offer", self.handle_offer)
        
        # Add test endpoint
        app.router.add_post("/audio-bridge/test", self.handle_test)
        
        # Add CORS middleware
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
            logger.info("WebRTC audio bridge server shutting down")
        except Exception as e:
            logger.error(f"Error in WebRTC audio bridge server: {e}")
        finally:
            if hasattr(self, 'runner'):
                await self.runner.cleanup()
            logger.info("WebRTC audio bridge server stopped")
            
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
        if hasattr(self, 'runner') and self.runner is not None:
            try:
                await self.runner.cleanup()
                logger.info("Application runner cleaned up")
                self.runner = None
            except Exception as e:
                logger.error(f"Error cleaning up application runner: {e}")
        
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