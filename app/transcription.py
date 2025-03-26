import os
import pyaudio
import wave
import numpy as np
import json
import aiohttp
import tempfile
import torch
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug flag for audio levels
DEBUG_AUDIO_LEVELS = os.getenv("DEBUG_AUDIO_LEVELS", "false").lower() == "true"

# Check for local Faster Whisper setting
FASTER_WHISPER_LOCAL = os.getenv("FASTER_WHISPER_LOCAL", "true").lower() == "true"

# Initialize whisper model as None to lazy load
whisper_model = None

def initialize_whisper_model():
    """Initialize the Faster Whisper model - only called when needed"""
    global whisper_model
    
    if whisper_model is not None:
        return whisper_model
        
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default model size (adjust as needed)
    model_size = "medium.en"
    
    try:
        print(f"Attempting to load Faster-Whisper on {device}...")
        whisper_model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        print("Faster-Whisper initialized successfully.")
    except Exception as e:
        print(f"Error initializing Faster-Whisper on {device}: {e}")
        print("Falling back to CPU mode...")

        # Force CPU fallback
        device = "cpu"
        model_size = "tiny.en"  # Use a smaller model for CPU performance
        whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Faster-Whisper initialized on CPU successfully.")
        
    return whisper_model

def transcribe_with_whisper(audio_file):
    """Transcribe audio using local Faster Whisper model"""
    # Lazy load the model only when needed
    model = initialize_whisper_model()
    
    segments, info = model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

async def transcribe_with_openai_api(audio_file, model="gpt-4o-mini-transcribe"):
    """Transcribe audio using OpenAI's API"""
    if not OPENAI_API_KEY:
        raise ValueError("API key missing. Please set OPENAI_API_KEY in your environment.")
    
    # Make the API call to OpenAI
    api_url = "https://api.openai.com/v1/audio/transcriptions"
    
    async with aiohttp.ClientSession() as session:
        with open(audio_file, "rb") as audio_file_data:
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                                audio_file_data.read(),
                                filename=os.path.basename(audio_file),
                                content_type='audio/wav')
            
            # Use the model directly
            form_data.add_field('model', model)
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            async with session.post(api_url, data=form_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    transcription = result.get("text", "")
                    return transcription
                else:
                    error_text = await response.text()
                    print(f"Error from OpenAI API: {error_text}")
                    raise Exception(f"Transcription error: {response.status} - {error_text}")

def detect_silence(data, threshold=512, chunk_size=1024):
    """Detect silence in audio data"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    level = np.mean(np.abs(audio_data))
    # Only print audio levels if debug is enabled
    if DEBUG_AUDIO_LEVELS:
        print(f"Audio level: {level}")
    return level < threshold

async def record_audio(file_path, silence_threshold=512, silence_duration=2.5, chunk_size=1024, send_status_callback=None):
    """Record audio to a file path
    
    Args:
        file_path: Path to save the recorded audio
        silence_threshold: Threshold for silence detection
        silence_duration: Duration of silence to stop recording
        chunk_size: Size of audio chunks
        send_status_callback: Callback to send status updates (optional)
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
    frames = []
    print("Recording...")
    
    # Notify frontend if callback provided
    if send_status_callback:
        await send_status_callback({"action": "recording_started"})
        
    silent_chunks = 0
    speaking_chunks = 0
    
    while True:
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
        
        if detect_silence(data, threshold=silence_threshold, chunk_size=chunk_size):
            silent_chunks += 1
            if silent_chunks > silence_duration * (16000 / chunk_size):
                break
        else:
            silent_chunks = 0
            speaking_chunks += 1
            
        if speaking_chunks > silence_duration * (16000 / chunk_size) * 10:
            break
            
    print("Recording stopped.")
    
    # Notify frontend if callback provided
    if send_status_callback:
        await send_status_callback({"action": "recording_stopped"})
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

async def record_audio_enhanced(send_status_callback=None, silence_threshold=300, silence_duration=2.0):
    """Enhanced audio recording with waiting for speech detection
    
    Args:
        send_status_callback: Callback to send status messages
        silence_threshold: Threshold for silence detection
        silence_duration: Duration of silence to stop recording
    
    Returns:
        Path to the recorded audio file
    """
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    # Recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    # Recording logic
    p = pyaudio.PyAudio()
    
    # Debug info about audio devices - only show once
    if DEBUG_AUDIO_LEVELS:
        print("\nAudio input devices:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:  # Only input devices
                print(f"Device {i}: {dev_info['name']}")
        print("Using default input device\n")
    
    # Open the stream with input_device_index=None to use default device
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    # Wait for user to start speaking
    print("Waiting for speech...")
    if send_status_callback:
        await send_status_callback({"action": "waiting_for_speech", "message": "Listening... Please speak now."})
    
    # Flush initial buffer
    for _ in range(5):
        stream.read(CHUNK)
        
    initial_silent_chunks = 0
    silence_broken = False
    
    # Wait for user to start speaking
    while not silence_broken:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if not detect_silence(data, threshold=silence_threshold):
            silence_broken = True
            print("Speech detected, recording started...")
            break
        
        initial_silent_chunks += 1
        # If waiting too long (15 seconds), abort
        if initial_silent_chunks > 15 * (RATE / CHUNK):
            print("No speech detected after timeout. Aborting.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            try:
                os.unlink(temp_filename)
            except:
                pass
            if send_status_callback:
                await send_status_callback({
                    "action": "error", 
                    "message": "No speech detected. Please check your microphone and try again."
                })
            return None
            
        # Every 2 seconds, provide feedback
        if initial_silent_chunks % (2 * int(RATE / CHUNK)) == 0 and initial_silent_chunks > 0 and initial_silent_chunks % (4 * int(RATE / CHUNK)) == 0:
            if send_status_callback:
                await send_status_callback({
                    "action": "waiting_for_speech", 
                    "message": "Still listening... Please speak or check your microphone."
                })
                
    # Now begin actual recording
    frames = []
    print("Enhanced recording...")
    if send_status_callback:
        await send_status_callback({"action": "recording_started"})
    
    # Add the initial speech chunk that broke the silence
    if silence_broken:
        frames.append(data)
    
    silent_chunks = 0
    speaking_chunks = 0
    
    # Continue recording until silence is detected
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        if detect_silence(data, threshold=silence_threshold):
            silent_chunks += 1
            if silent_chunks > silence_duration * (RATE / CHUNK):
                break
        else:
            silent_chunks = 0
            speaking_chunks += 1
        if speaking_chunks > silence_duration * (RATE / CHUNK) * 15:  # Allow longer recordings
            break
            
    print("Enhanced recording stopped.")
    if send_status_callback:
        await send_status_callback({"action": "recording_stopped"})
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # If no substantial recording was made, return None
    if len(frames) < 10:
        try:
            os.unlink(temp_filename)
        except:
            pass
        if send_status_callback:
            await send_status_callback({
                "action": "error", 
                "message": "Recording too short. Please try again."
            })
        return None
    
    # Save the recorded audio
    wf = wave.open(temp_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return temp_filename

async def send_status_message(callback, message):
    """Helper function to send status messages through the callback
    
    This handles both dictionary and string message formats.
    
    Args:
        callback: The callback function to send the message through
        message: Either a dictionary or string message
    """
    if callback:
        await callback(message)

async def transcribe_audio(transcription_model="gpt-4o-mini-transcribe", use_local=False, send_status_callback=None):
    """Main function to record audio and transcribe it
    
    Args:
        transcription_model: Model to use for OpenAI transcription
        use_local: Whether to use local Faster Whisper
        send_status_callback: Callback to send status messages
    
    Returns:
        Transcribed text
    """
    try:
        # Create an async wrapper for the callback
        async def callback_wrapper(msg):
            if send_status_callback:
                await send_status_message(send_status_callback, msg)
                
        # Record audio with enhanced mode
        temp_filename = await record_audio_enhanced(
            send_status_callback=callback_wrapper
        )
        
        if not temp_filename:
            return None
            
        # Transcribe based on method
        if use_local:
            # Lazy initialize Faster Whisper if needed
            if whisper_model is None:
                initialize_whisper_model()
                
            transcription = transcribe_with_whisper(temp_filename)
        else:
            # Use OpenAI API
            transcription = await transcribe_with_openai_api(temp_filename, transcription_model)
            
        # Clean up temp file
        try:
            os.unlink(temp_filename)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
            
        return transcription
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        if send_status_callback:
            await send_status_message(send_status_callback, {
                "action": "error", 
                "message": f"Error: {str(e)}"
            })
        return f"Error: {str(e)}" 