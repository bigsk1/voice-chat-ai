"""
OpenAI Realtime API integration using direct WebSocket connection

This module handles real-time voice conversations with OpenAI using their
direct WebSocket API as described at:
https://platform.openai.com/docs/guides/realtime
"""

import os
import json
import time
import base64
import logging
import uuid
import websocket
import threading
import queue
import asyncio
import random
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable more detailed debug output when needed
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "false").lower() == "true"
DEBUG_WEBSOCKET = os.getenv("DEBUG_WEBSOCKET", "false").lower() == "true"

# Limit audio logging to prevent terminal flooding
def log_audio_size(data_type, data, log_data=False):
    """Log audio data size without printing raw content"""
    if DEBUG_AUDIO:
        if log_data:
            # Only log first 10 bytes as hex for debugging
            preview = data[:10].hex() if data else ""
            preview_str = f", preview: {preview}..." if preview else ""
        else:
            preview_str = ""
            
        logger.info(f"{data_type}: {len(data)} bytes{preview_str}")
    else:
        # Just log size occasionally to avoid flooding
        if random.random() < 0.05:  # Only log ~5% of messages
            logger.info(f"{data_type}: {len(data)} bytes")

# Log helpful information about debug settings
if DEBUG_AUDIO or DEBUG_WEBSOCKET:
    logger.info(f"DEBUG MODE ENABLED - Audio: {DEBUG_AUDIO}, WebSocket: {DEBUG_WEBSOCKET}")
    logger.info("To disable debug output, set environment variables DEBUG_AUDIO=false and DEBUG_WEBSOCKET=false")

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # The latest model

# Global conversation history
conversation_history = []

def save_conversation_history(history):
    """Save conversation history to a file"""
    # This is a placeholder - implement as needed
    pass

class RealtimeSession:
    """
    Manages a direct WebSocket connection to OpenAI's Realtime API for voice conversations
    """
    def __init__(self, session_id=None, character="assistant", model=OPENAI_REALTIME_MODEL, voice="alloy"):
        """Initialize the session"""
        self.session_id = session_id or f"sess_{uuid.uuid4().hex}"
        self.character = character
        self.model = model
        self.voice = voice
        self.active = False
        self.created_at = time.time()
        self.last_activity = time.time()
        self.ai_is_speaking = False
        self.user_is_speaking = False
        self.transcript_history = []
        self.event_counter = 0
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.audio_queue = queue.Queue()
        
        # Connected clients tracking
        self.client_connections = set()
        
        # Message callback
        self.message_callback = None
    
    async def start(self) -> bool:
        """Start a new session with the specified character"""
        try:
            # Load character instructions
            from app.app_logic import load_character_prompt
            character_instructions = load_character_prompt(self.character)
            
            # Start the realtime session
            return await self.start_session(character_instructions)
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return False
    
    def set_message_callback(self, callback: Callable) -> None:
        """Set a callback function to handle messages from the session"""
        self.message_callback = callback
        
    def _broadcast_message(self, message: dict) -> None:
        """Broadcast a message using the callback if set"""
        if self.message_callback:
            try:
                self.message_callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
                
    async def _async_broadcast_message(self, message: dict) -> None:
        """Asynchronously broadcast a message"""
        if self.message_callback:
            try:
                # Create a future that can be awaited
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                
                def callback_done(task):
                    if not future.done():
                        if task.exception():
                            future.set_exception(task.exception())
                        else:
                            future.set_result(task.result())
                
                # Run the callback in the event loop
                task = asyncio.run_coroutine_threadsafe(
                    self.message_callback(message), 
                    loop
                )
                task.add_done_callback(callback_done)
                
                # Wait for the callback to complete
                return await asyncio.wait_for(future, timeout=2.0)
            except Exception as e:
                logger.error(f"Error in async message callback: {e}")
    
    async def start_session(self, system_instructions: str) -> bool:
        """Start a new Realtime session with OpenAI API"""
        try:
            self.character_instructions = system_instructions
            
            # Direct WebSocket connection
            return self.connect(system_instructions)
            
        except Exception as e:
            logger.error(f"Error starting realtime session: {e}")
            return False
    
    def connect(self, system_instructions: str = None) -> bool:
        """Connect to the OpenAI Realtime API"""
        if self.active:
            logger.warning("Session already active")
            return True
        
        # WebSocket URL with model as query parameter
        ws_url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        
        # Headers as specified in the documentation
        headers = [
            f"Authorization: Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta: realtime=v1"
        ]
        
        # Create WebSocket connection
        try:
            # Enable trace for debugging (set to False in production)
            websocket.enableTrace(False)
            
            # Create WebSocket
            self.ws = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to establish
            start_time = time.time()
            while not self.active and time.time() - start_time < 5:
                time.sleep(0.1)
            
            # Start audio sender thread
            if self.active:
                threading.Thread(target=self._audio_sender_thread).start()
                
                # Set instructions and voice if provided
                if system_instructions:
                    self.set_character_instructions(system_instructions)
                
                if self.voice != "alloy":
                    self.set_session_voice(self.voice)
            
            return self.active
            
        except Exception as e:
            logger.error(f"Error connecting to Realtime API: {e}")
            return False
    
    def get_event_id(self):
        """Generate a unique event ID"""
        self.event_counter += 1
        return f"event_{int(time.time())}_{self.event_counter}"
    
    async def process_audio_chunk(self, audio_data: bytes) -> bool:
        """Process an audio chunk from the user and send it to the realtime session"""
        if not self.ws or not self.active:
            logger.error("No active WebSocket connection for audio processing")
            return False
        
        try:
            # Set flag indicating user is speaking
            self.user_is_speaking = True
            self.last_activity = time.time()
            
            # Log audio data size and sample analysis
            log_audio_size("User audio", audio_data, True)
            
            # Broadcast user activity to all clients
            await broadcast_to_session_clients(self.session_id, {
                "action": "user_speaking",
            })
            
            # Queue the audio data for sending
            self.audio_queue.put(audio_data)
            return True
            
        except Exception as e:
            logger.error(f"Error queueing audio data: {e}")
            self.user_is_speaking = False
            return False
    
    async def commit_audio(self) -> bool:
        """Commit the audio buffer after user stops speaking"""
        if not self.ws or not self.active:
            logger.error("No active WebSocket connection for audio processing")
            return False
        
        try:
            # Send commit message with correct format
            message = {
                "event_id": self.get_event_id(),
                "type": "input_audio_buffer.commit"
                # No additional parameters needed for commit
            }
            
            # Log the commit message
            logger.info("Committing audio buffer")
            
            # Send using the thread-safe method
            self.ws.send(json.dumps(message))
            
            # Update status flags
            self.user_is_speaking = False
            self.last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error committing audio buffer: {e}")
            return False
    
    async def send_text(self, text: str) -> bool:
        """Send a text message to the realtime session"""
        if not self.ws or not self.active:
            logger.error("No active WebSocket connection for sending text")
            return False
        
        try:
            # Log the message being sent
            logger.info(f"Sending text message: {text[:50]}..." if len(text) > 50 else f"Sending text message: {text}")
            
            # Format message according to OpenAI documentation (fixed format)
            # https://platform.openai.com/docs/guides/realtime
            message = {
                "event_id": self.get_event_id(),
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            }
            
            # Send message
            logger.info(f"Sending formatted message to OpenAI: {json.dumps(message)[:100]}...")
            self.ws.send(json.dumps(message))
            
            # Also update transcript with user message for history
            self.transcript_history.append({
                "role": "user",
                "content": text
            })
            
            # Flag that user is not speaking after sending text
            self.user_is_speaking = False
            
            # Update last activity timestamp
            self.last_activity = time.time()
            
            # Broadcast to clients that message was sent
            self._broadcast_message({
                "type": "transcript",
                "role": "user",
                "text": text
            })
            
            return True
        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            return False
    
    def set_session_voice(self, voice: str) -> bool:
        """Update the session voice"""
        if not self.ws or not self.active:
            logger.error("Cannot update voice: Not connected")
            return False
        
        try:
            # Update session with new voice
            message = {
                "event_id": self.get_event_id(),
                "type": "session.update",
                "session": {
                    "voice": voice
                }
            }
            
            self.ws.send(json.dumps(message))
            self.voice = voice
            return True
            
        except Exception as e:
            logger.error(f"Error updating voice: {e}")
            return False
    
    def set_character_instructions(self, instructions: str) -> bool:
        """Update the character instructions"""
        if not self.ws or not self.active:
            logger.error("Cannot update instructions: Not connected")
            return False
        
        try:
            # Update session with new instructions
            message = {
                "event_id": self.get_event_id(),
                "type": "session.update",
                "session": {
                    "instructions": instructions
                }
            }
            
            self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error updating instructions: {e}")
            return False
    
    def close(self) -> bool:
        """Close the Realtime session (synchronous version)"""
        success = True
        
        # Set session as inactive
        self.active = False
        
        try:
            # Close WebSocket
            if self.ws:
                self.ws.close()
            
            # Wait for thread to terminate
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
            
            # Update global conversation history
            global conversation_history
            conversation_history.extend(self.transcript_history)
            save_conversation_history(conversation_history)
            
            # Notify clients of session closure
            self._broadcast_message({
                "type": "session_closed"
            })
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing realtime session: {e}")
            return False
            
    async def async_close(self) -> bool:
        """Asynchronous version of the close method"""
        success = self.close()  # Call the synchronous version
        
        # Add additional async notification
        try:
            await broadcast_to_session_clients(self.session_id, {
                "action": "session_closed"
            })
        except Exception as e:
            logger.error(f"Error in async notification during close: {e}")
            
        return success
    
    def _on_open(self, ws):
        """Called when WebSocket connection is established"""
        logger.info("Connected to OpenAI Realtime API")
        self.active = True
        self.last_activity = time.time()
    
    def _on_message(self, ws, message):
        """Called when a message is received from the WebSocket"""
        try:
            # Log the message if debug enabled, but avoid logging large audio content
            if DEBUG_WEBSOCKET:
                # Check if this is likely a binary audio message (could be base64 encoded)
                if len(message) > 1000 and ("audio" in message.lower() or "content" in message.lower()):
                    logger.debug(f"Received large WebSocket message: {len(message)} bytes (audio content)")
                else:
                    # Log the first 200 chars of the message to avoid flooding logs
                    preview = message[:200] + ("..." if len(message) > 200 else "")
                    logger.debug(f"Received WebSocket message: {preview}")
            
            # Parse the message
            data = json.loads(message)
            message_type = data.get("type", "")
            
            # Always log message type for easier debugging (important to see what's happening)
            logger.info(f"OpenAI message type: {message_type}")
            
            # Process based on message type
            if message_type == "session.created":
                self.active = True
                logger.info("WebSocket session created successfully")
                
                # Broadcast session creation to clients
                self._broadcast_message({
                    "type": "session_created",
                    "session_id": self.session_id
                })
            
            elif message_type == "conversation.item.message.completed":
                logger.info(f"AI message completed")
                self.ai_is_speaking = False
            
            elif message_type == "conversation.item.speech.started":
                logger.info(f"AI started speaking")
                self.ai_is_speaking = True
            
            elif message_type == "conversation.item.speech.completed":
                logger.info(f"AI speech completed")
                self.ai_is_speaking = False
            
            elif message_type == "conversation.item.speech.data":
                # This contains audio data - don't log the data itself
                log_audio_size("Received AI audio", data.get("content", ""))
                
                try:
                    # Extract the audio content (base64 encoded)
                    audio_content = data.get("content", "")
                    
                    # Broadcast to clients
                    self._broadcast_message({
                        "type": "audio_data",
                        "format": "mp3",  # OpenAI sends mp3 data
                        "audio_data": audio_content  # Keep base64 format for client
                    })
                    
                    # Debug audio playback issues
                    if DEBUG_AUDIO:
                        try:
                            decoded_size = len(base64.b64decode(audio_content)) if audio_content else 0
                            logger.debug(f"Decoded audio size: {decoded_size} bytes")
                            
                            # Check if we're getting extremely small audio chunks
                            if decoded_size > 0 and decoded_size < 100:
                                logger.warning("Very small audio chunk received - might not be playable")
                        except Exception as e:
                            logger.error(f"Error analyzing audio data: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing speech data: {e}")
            
            elif message_type == "conversation.item.text.created":
                # AI is generating text response
                content = data.get("content", {})
                text = content.get("text", "")
                
                logger.info(f"AI response text: {text[:50]}..." if len(text) > 50 else f"AI response text: {text}")
                
                # Add to transcript history
                if text:
                    self.transcript_history.append({
                        "role": "assistant",
                        "content": text
                    })
                    
                    # Broadcast text to clients
                    self._broadcast_message({
                        "type": "transcript",
                        "role": "assistant",
                        "text": text
                    })
            
            # Handle error messages
            elif message_type == "error":
                error = data.get("error", {})
                error_message = error.get("message", "Unknown error")
                logger.error(f"Error from OpenAI: {error_message}")
                
                # Broadcast error to clients
                self._broadcast_message({
                    "type": "error",
                    "message": f"OpenAI error: {error_message}"
                })
                
            else:
                # Log any unexpected message types
                logger.info(f"Unhandled message type: {message_type}")
                if DEBUG_WEBSOCKET:
                    logger.debug(f"Full message content: {message[:500]}...")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Called when a WebSocket error occurs"""
        logger.error(f"WebSocket error: {error}")
        
        # Add more detailed debug info if enabled
        if DEBUG_WEBSOCKET and hasattr(error, "__dict__"):
            logger.error(f"WebSocket error details: {error.__dict__}")
        
        # Broadcast error to clients
        self._broadcast_message({
            "type": "error",
            "message": f"WebSocket error: {str(error)}"
        })
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when the WebSocket connection is closed"""
        logger.info(f"Connection closed: {close_status_code} - {close_msg}")
        self.active = False
        
        # Broadcast closure to clients
        self._broadcast_message({
            "type": "session_closed"
        })
    
    def _audio_sender_thread(self):
        """Thread to send audio data from the queue"""
        audio_chunks_sent = 0
        total_bytes_sent = 0
        
        while self.active:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Convert to base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Create audio append message with correct format
                audio_message = {
                    "event_id": self.get_event_id(),
                    "type": "input_audio_buffer.append",
                    "chunk": {
                        "encoding": "base64",
                        "content": audio_base64
                    }
                }
                
                # Send audio data
                if self.ws and self.active:
                    self.ws.send(json.dumps(audio_message))
                    
                    # Log periodically
                    audio_chunks_sent += 1
                    total_bytes_sent += len(audio_data)
                    if audio_chunks_sent % 20 == 0:  # Log every 20 chunks
                        logger.info(f"Sent {audio_chunks_sent} audio chunks ({total_bytes_sent} bytes) to OpenAI")
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audio sender thread: {e}")
                time.sleep(0.1)  # Avoid tight loop in case of persistent errors
    
    def _broadcast_to_clients_sync(self, message):
        """Synchronous helper to broadcast messages to all clients"""
        # Use the message callback if set
        if self.message_callback:
            self.message_callback(message)
        
        # Also broadcast to any directly connected clients (legacy support)
        for client_queue in self.client_connections:
            try:
                client_queue.put_nowait(message)
            except Exception as e:
                logger.error(f"Error queuing message for client: {e}")

# Global session registry
active_sessions = {}
client_queues = {}

async def broadcast_to_session_clients(session_id, message):
    """Broadcast a message to all clients connected to a session"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        for client_queue in session.client_connections:
            await client_queue.put(message)

async def create_realtime_session(character, model, voice, client_id=None):
    """Create a new realtime session"""
    try:
        from app.app_logic import load_character_prompt
        
        # Load character instructions
        character_instructions = load_character_prompt(character)
        
        # Create session ID
        session_id = f"realtime_{uuid.uuid4().hex}"
        
        # Initialize session
        session = RealtimeSession(
            session_id=session_id,
            model=model, 
            voice=voice
        )
        
        # Start session
        if await session.start_session(character_instructions):
            # Register session
            active_sessions[session_id] = session
            
            # Add client if specified
            if client_id and client_id in client_queues:
                session.client_connections.add(client_queues[client_id])
            
            logger.info(f"Created realtime session: {session_id}")
            return session_id
        else:
            logger.error("Failed to start realtime session")
            return None
    except Exception as e:
        logger.error(f"Error creating realtime session: {e}")
        return None

async def process_websocket_message(session_id, message, client_id):
    """Process a message from a WebSocket client"""
    if session_id not in active_sessions:
        return {"action": "error", "message": "Invalid session ID"}
    
    session = active_sessions[session_id]
    
    try:
        # Parse message
        data = json.loads(message)
        action = data.get("action")
        
        # Process based on action
        if action == "audio":
            # Convert base64 audio to bytes
            audio_data = base64.b64decode(data.get("data", ""))
            await session.process_audio_chunk(audio_data)
            return {"action": "audio_received"}
        
        elif action == "commit_audio":
            # User stopped speaking, commit the audio buffer
            await session.commit_audio()
            return {"action": "audio_committed"}
        
        elif action == "text":
            # Send text message
            await session.send_text(data.get("content", ""))
            return {"action": "text_sent"}
        
        elif action == "close_session":
            # Close the session
            await session.async_close()
            # Remove from active sessions
            if session_id in active_sessions:
                del active_sessions[session_id]
            return {"action": "session_closed"}
        
        else:
            return {"action": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        return {"action": "error", "message": str(e)} 