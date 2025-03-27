#!/usr/bin/env python3
"""
Test client for OpenAI's Realtime API using WebRTC

This script implements a test client using the WebRTC approach as documented by OpenAI
https://platform.openai.com/docs/guides/realtime
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
import base64
import time
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import MediaStreamTrack, AudioStreamTrack

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is required")
    sys.exit(1)

MODEL = "gpt-4o-realtime-preview-2024-12-17"
BASE_URL = "https://api.openai.com/v1/realtime"

# Audio file for testing (optional)
AUDIO_FILE = None  # "test_audio.wav"

class MicrophoneStreamTrack(AudioStreamTrack):
    """
    An audio track that captures from microphone
    """
    def __init__(self):
        super().__init__()
        self.audio_player = None
        if AUDIO_FILE and os.path.exists(AUDIO_FILE):
            # Use audio file for testing
            self.audio_player = MediaPlayer(AUDIO_FILE)
            logger.info(f"Using audio file: {AUDIO_FILE}")
        else:
            # Use microphone (requires additional setup)
            try:
                self.audio_player = MediaPlayer("default:none", format="pulse")
                logger.info("Using default microphone")
            except Exception as e:
                logger.error(f"Failed to initialize microphone: {e}")
                logger.info("Will generate silence instead")
                self.audio_player = None
    
    async def recv(self):
        # If we have an audio player, use it, otherwise generate silence
        if self.audio_player and hasattr(self.audio_player, "audio"):
            frame = await self.audio_player.audio.recv()
            return frame
        else:
            # Generate silent audio frames
            import fractions
            from av import AudioFrame
            from aiortc.mediastreams import AUDIO_PTIME
            
            sample_rate = 48000
            samples = int(AUDIO_PTIME * sample_rate)
            frame = AudioFrame(format="s16", layout="mono", samples=samples)
            frame.sample_rate = sample_rate
            frame.pts = int(time.time() * sample_rate)
            frame.time_base = fractions.Fraction(1, sample_rate)
            return frame

async def create_session_and_connect():
    """
    Create a realtime session and connect to it using WebRTC
    """
    # Create a peer connection
    pc = RTCPeerConnection()
    
    # Set up to play remote audio from the model
    recorder = MediaRecorder("ai_response.wav")
    
    # Add local audio track for microphone input
    audio_track = MicrophoneStreamTrack()
    pc.addTrack(audio_track)
    
    # Set up data channel for sending and receiving events
    data_channel = pc.createDataChannel("oai-events")
    
    @data_channel.on("open")
    def on_open():
        logger.info("Data channel is open")
        
        # Example: Send a message to the AI
        message = {
            "event_id": f"event_{int(time.time())}",
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, can you tell me about the WebRTC API?"
                    }
                ]
            }
        }
        data_channel.send(json.dumps(message))
        
    @data_channel.on("message")
    def on_message(message):
        # Handle messages from OpenAI
        try:
            data = json.loads(message)
            logger.info(f"Received message type: {data.get('type')}")
            
            if data.get("type") == "conversation.item.text.created":
                content = data.get("content", {})
                text = content.get("text", "")
                logger.info(f"AI response: {text}")
                
            elif data.get("type") == "conversation.item.message.completed":
                logger.info("AI message completed")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    # Create an offer
    await pc.setLocalDescription(await pc.createOffer())
    
    logger.info("Creating session...")
    
    # Construct the request URL with model parameter
    request_url = f"{BASE_URL}?model={MODEL}"
    
    # Extract SDP from local description
    sdp = pc.localDescription.sdp
    
    logger.info(f"Sending SDP to OpenAI: {sdp[:100]}...")
    
    try:
        # Send the SDP to OpenAI
        async with aiohttp.ClientSession() as session:
            async with session.post(
                request_url,
                data=sdp,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/sdp",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to create session: {response.status} - {error_text}")
                    return
                
                # Get the answer SDP
                answer_sdp = await response.text()
                logger.info(f"Received answer SDP: {answer_sdp[:100]}...")
                
                # Create an answer description and set it
                answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                await pc.setRemoteDescription(answer)
                
                logger.info("WebRTC connection established!")
                
                # Set up recorder for incoming audio
                @pc.on("track")
                def on_track(track):
                    logger.info(f"Received {track.kind} track from OpenAI")
                    if track.kind == "audio":
                        logger.info("Adding audio track to recorder")
                        recorder.addTrack(track)
                        recorder.start()
                
                # Keep the connection alive for 60 seconds
                logger.info("Session active - will run for 60 seconds")
                await asyncio.sleep(60)
                
                # Close the recorder
                if recorder:
                    recorder.stop()
                
                # Close the connection
                await pc.close()
                logger.info("Session closed")
                
    except Exception as e:
        logger.error(f"Error during session: {e}")
        if pc:
            await pc.close()

async def main():
    """Main function to run the WebRTC client"""
    logger.info("Starting OpenAI Realtime WebRTC test client")
    await create_session_and_connect()

if __name__ == "__main__":
    asyncio.run(main()) 