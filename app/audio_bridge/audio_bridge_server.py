"""
Audio Bridge Server
Handles WebRTC connections for remote audio access
Allows remote clients to use their microphones and speakers with the server
"""

import asyncio
import os
import logging
from typing import Dict, Set, Optional, Any, Deque
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebRTC connections store
class AudioBridgeServer:
    def __init__(self):
        # Store active peer connections 
        self.connections: Dict[str, Any] = {}
        # Set of client IDs
        self.clients: Set[str] = set()
        # Flag to check if the bridge is enabled
        self.enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
        # Store audio data from clients
        self.client_audio: Dict[str, Deque[bytes]] = {}
        # Store the last time we received audio from a client
        self.last_audio_time: Dict[str, float] = {}
        
        if self.enabled:
            logger.info("Audio Bridge Server initialized and ENABLED")
        else:
            logger.info("Audio Bridge Server initialized but DISABLED")
            
    async def register_client(self, client_id: str) -> bool:
        """Register a new client connection"""
        if not self.enabled:
            logger.warning("Attempted to register client but bridge is disabled")
            return False
            
        if client_id in self.clients:
            # Client already registered - this is likely a reconnect or ping
            # Update the last audio time to keep the client active
            self.last_audio_time[client_id] = asyncio.get_event_loop().time()
            logger.info(f"Client {client_id} re-registered (ping)")
            return True
            
        self.clients.add(client_id)
        self.client_audio[client_id] = deque(maxlen=10)  # Store up to 10 audio chunks
        self.last_audio_time[client_id] = asyncio.get_event_loop().time()
        logger.info(f"Client {client_id} registered")
        return True
        
    async def unregister_client(self, client_id: str) -> bool:
        """Unregister a client connection"""
        if not self.enabled:
            return False
            
        if client_id not in self.clients:
            return False
            
        self.clients.remove(client_id)
        if client_id in self.client_audio:
            del self.client_audio[client_id]
        if client_id in self.last_audio_time:
            del self.last_audio_time[client_id]
        logger.info(f"Client {client_id} unregistered")
        return True
        
    async def handle_signaling(self, client_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle signaling message from client"""
        if not self.enabled:
            return {"type": "error", "message": "Audio bridge is disabled"}
            
        if client_id not in self.clients:
            return {"type": "error", "message": "Client not registered"}
            
        # Basic message validation
        if "type" not in message:
            return {"type": "error", "message": "Invalid signaling message"}
            
        message_type = message["type"]
        
        if message_type == "offer":
            # Store the offer and return mock answer
            # In a real implementation, this would create proper WebRTC answers
            logger.info(f"Received offer from {client_id}")
            return {
                "type": "answer",
                "sdp": "mockSDP"  # In real implementation, generate proper SDP
            }
            
        elif message_type == "ice-candidate":
            # Handle ICE candidate
            logger.info(f"Received ICE candidate from {client_id}")
            return {"type": "ack"}
            
        elif message_type == "disconnect":
            # Handle disconnect request
            await self.unregister_client(client_id)
            return {"type": "disconnected"}
            
        return {"type": "error", "message": f"Unsupported message type: {message_type}"}
        
    async def send_audio(self, client_id: str, audio_data: bytes) -> bool:
        """Send audio data to a client"""
        if not self.enabled or client_id not in self.clients:
            return False
            
        # In a real implementation, this would send audio over WebRTC data channel
        logger.info(f"Would send {len(audio_data)} bytes of audio to {client_id}")
        return True
        
    async def receive_audio(self, client_id: str) -> Optional[bytes]:
        """Receive audio data from a client"""
        if not self.enabled or client_id not in self.clients:
            return None
            
        # Check if we have any audio from this client
        if client_id in self.client_audio and self.client_audio[client_id]:
            # Get the oldest audio chunk from the queue
            audio_data = self.client_audio[client_id].popleft()
            logger.info(f"Returning {len(audio_data)} bytes of audio from client {client_id}")
            return audio_data
        
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
        if not self.enabled or client_id not in self.clients:
            logger.warning(f"Attempted to process audio for unregistered client {client_id}")
            return False
        
        try:
            # Log the received audio
            logger.info(f"Received audio from client {client_id}, {len(audio_data)} bytes")
            
            # Store the audio data for later retrieval
            if client_id in self.client_audio:
                self.client_audio[client_id].append(audio_data)
                self.last_audio_time[client_id] = asyncio.get_event_loop().time()
                logger.info(f"Stored audio from client {client_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
        
    def is_enabled(self) -> bool:
        """Check if the audio bridge is enabled"""
        return self.enabled
    
    def get_active_clients_count(self) -> int:
        """Get the number of active clients"""
        # A client is considered active if we've received audio from them in the last 30 seconds
        current_time = asyncio.get_event_loop().time()
        active_clients = [client_id for client_id in self.clients 
                         if client_id in self.last_audio_time and 
                         current_time - self.last_audio_time[client_id] < 30]
        return len(active_clients)
    
    def get_status(self) -> dict:
        """Get the full status of the audio bridge"""
        active_clients = self.get_active_clients_count() if self.enabled else 0
        
        return {
            "enabled": self.enabled,
            "active_clients": active_clients,
            "total_clients": len(self.clients) if self.enabled else 0,
            "status": "active" if self.enabled else "disabled"
        }

# Create a singleton instance
audio_bridge = AudioBridgeServer() 