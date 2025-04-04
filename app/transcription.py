import os
import pyaudio
import wave
import numpy as np
import aiohttp
import torch
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import time

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug flag for audio levels
DEBUG_AUDIO_LEVELS = os.getenv("DEBUG_AUDIO_LEVELS", "false").lower() == "true"

# Silence duration
SILENCE_DURATION_SECONDS = float(os.getenv("SILENCE_DURATION_SECONDS", "2.0"))

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

def detect_silence(data, threshold=300, chunk_size=1024):
    """Detect if the given audio data is silence
    
    Args:
        data: Audio data bytes
        threshold: Threshold for silence detection (lower values make it more sensitive)
        chunk_size: Size of audio chunks
        
    Returns:
        Boolean indicating if the audio is silence
    """
    try:
        # Use numpy for faster processing
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate RMS level (more accurate than simple mean)
        # Ensure we don't hit division by zero
        if len(audio_data) == 0:
            return True
            
        # Calculate RMS value
        level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        
        # Print low level audio warnings
        if DEBUG_AUDIO_LEVELS and level > 0 and level < 50:
            print(f"Very low audio level detected: {level:.2f}")
        
        # Only print audio levels if debug is enabled
        if DEBUG_AUDIO_LEVELS:
            print(f"Audio level: {level:.2f}, threshold: {threshold}")
            
        # Return True if silent, False if sound
        return level < threshold
    except Exception as e:
        print(f"Error in detect_silence: {e}")
        # Return default value on error
        return True
    

async def record_audio(file_path, silence_threshold=512, silence_duration=SILENCE_DURATION_SECONDS, chunk_size=1024, send_status_callback=None, no_fallback=False):
    """Record audio to a file path
    
    Args:
        file_path: Path to save the recorded audio
        silence_threshold: Threshold for silence detection
        silence_duration: Duration of silence to stop recording
        chunk_size: Size of audio chunks
        send_status_callback: Callback to send status updates (optional)
        no_fallback: If True, don't fall back to local microphone on error
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
    
    # Set a shorter silence duration - makes it stop faster after speech ends
    silence_chunks_threshold = int(silence_duration * (16000 / chunk_size)) 
    
    while True:
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
        
        # Get audio level for debug
        if DEBUG_AUDIO_LEVELS:
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            print(f"Current audio level: {level:.2f}, threshold: {silence_threshold}, silent chunks: {silent_chunks}/{silence_chunks_threshold}")
        
        if detect_silence(data, threshold=silence_threshold, chunk_size=chunk_size):
            silent_chunks += 1
            if silent_chunks >= silence_chunks_threshold:
                print(f"Silence detected for {silence_duration} seconds, stopping recording")
                break
        else:
            # Reset the silent chunk counter if any sound is detected
            silent_chunks = 0
            speaking_chunks += 1
            
        # Maximum recording time (10 times the silence duration)
        if speaking_chunks > silence_duration * (16000 / chunk_size) * 10:
            print("Maximum recording time reached")
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

async def record_audio_enhanced(send_status_callback=None, silence_threshold=200, silence_duration=SILENCE_DURATION_SECONDS):
    """Enhanced audio recording with waiting for speech detection
    
    Args:
        send_status_callback: Callback to send status messages
        silence_threshold: Threshold for silence detection (lower is more sensitive)
        silence_duration: Duration of silence to stop recording (reduced to SILENCE_DURATION_SECONDS seconds)
    
    Returns:
        Path to the recorded audio file
    """
    # Enforce a minimum threshold to ensure silence detection works
    minimum_threshold = 100
    if silence_threshold < minimum_threshold:
        print(f"Warning: Silence threshold {silence_threshold} is too low, using minimum of {minimum_threshold}")
        silence_threshold = minimum_threshold
    
    # Check if audio bridge is enabled
    audio_bridge_enabled = os.getenv("ENABLE_AUDIO_BRIDGE", "false").lower() == "true"
    
    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Created outputs directory at {outputs_dir}")
    
    # Generate a unique filename with timestamp
    timestamp = int(time.time())
    output_filename = os.path.join(outputs_dir, f"recording_{timestamp}.wav")
    
    # If audio bridge is enabled, check if we have any clients
    if audio_bridge_enabled:
        try:
            from .audio_bridge.audio_bridge_server import audio_bridge
            if audio_bridge.clients_set:
                print(f"Audio bridge enabled with {len(audio_bridge.clients_set)} clients")
                
                # Call record_audio with the correct parameters and shortened silence duration
                await record_audio(
                    file_path=output_filename,
                    silence_threshold=max(25, minimum_threshold),  # Ensure minimum threshold
                    silence_duration=silence_duration,  # Use the reduced silence duration
                    send_status_callback=send_status_callback,
                    no_fallback=True
                )
                
                # Print a success message
                print(f"Audio bridge created file: {output_filename}")
                return output_filename
        except Exception as e:
            print(f"Error using audio bridge in enhanced mode: {e}")
            import traceback
            print(traceback.format_exc())
    
    # Recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    # Recording logic
    p = pyaudio.PyAudio()
    
    # Always show audio devices when recording starts to help with debugging
    print("\nAudio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only input devices
            print(f"Device {i}: {dev_info['name']}")
    print("Using default input device\n")
    
    # Open the stream with input_device_index=None to use default device
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    # Wait for user to start speaking
    print(YELLOW + "Waiting for speech... (threshold: " + str(silence_threshold) + ")" + RESET_COLOR)
    if send_status_callback:
        await send_status_callback({"action": "waiting_for_speech"})
    
    # Flush initial buffer
    for _ in range(5):
        stream.read(CHUNK)
        
    initial_silent_chunks = 0
    silence_broken = False
    
    # Wait for user to start speaking
    while not silence_broken:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Detect speech using our threshold
        audio_data = np.frombuffer(data, dtype=np.int16)
        level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        
        if level > silence_threshold:
            # Using direct level comparison for more reliable speech detection
            silence_broken = True
            print(f"Speech detected, level: {level:.2f}, recording started...")
            break
        
        initial_silent_chunks += 1
        # If waiting too long (15 seconds), abort
        if initial_silent_chunks > 15 * (RATE / CHUNK):
            print("No speech detected after timeout. Aborting.")
            # Try reading audio levels directly to confirm microphone is working
            print("Testing microphone with direct read:")
            test_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(test_data, dtype=np.int16)
            level = np.mean(np.abs(audio_data))
            print(f"Direct audio level test: {level} (threshold: {silence_threshold})")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            try:
                os.unlink(output_filename)
            except:
                pass
            if send_status_callback:
                await send_status_callback({
                    "action": "error", 
                    "message": "No speech detected. Please check your microphone and try again. Audio level: " + str(level)
                })
            return None
            
        # Provide more frequent feedback
        if initial_silent_chunks % int(RATE / CHUNK) == 0 and initial_silent_chunks > 0:
            print(f"Still waiting for speech... ({initial_silent_chunks} chunks), level: {level:.2f}")
            if send_status_callback:
                await send_status_callback({
                    "action": "waiting_for_speech"
                })

    # Now begin actual recording
    frames = []
    print("Enhanced recording...")
    if send_status_callback:
        await send_status_callback({"action": "recording_started"})
    
    # Add the initial speech chunk that broke the silence
    if silence_broken:
        frames.append(data)
    
    # Continue recording until silence is detected
    silent_chunks = 0
    speaking_chunks = 0
    silence_chunks_threshold = int(silence_duration * (RATE / CHUNK))
    
    # Continue recording until silence is detected
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate audio level directly for more reliable detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        
        # Get audio level for debug
        if DEBUG_AUDIO_LEVELS:
            print(f"Current audio level: {level:.2f}, threshold: {silence_threshold}, silent chunks: {silent_chunks}/{silence_chunks_threshold}")
        
        # Check for silence by direct level comparison
        if level < silence_threshold:
            silent_chunks += 1
            if silent_chunks >= silence_chunks_threshold:
                print(f"Silence detected for {SILENCE_DURATION_SECONDS} seconds, stopping recording")
                break
        else:
            # Reset silent chunks counter completely when sound is detected
            silent_chunks = 0
            speaking_chunks += 1
            
        # Maximum recording duration (15 times the silence duration)
        if speaking_chunks > silence_duration * (RATE / CHUNK) * 15:
            print("Maximum recording time reached")
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
            os.unlink(output_filename)
        except:
            pass
        if send_status_callback:
            await send_status_callback({
                "action": "error", 
                "message": "Recording too short. Please try again."
            })
        return None
    
    # Save the recorded audio
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return output_filename

async def send_status_message(callback, message):
    """Helper function to send status messages through the callback
    
    This handles both dictionary and string message formats.
    
    Args:
        callback: The callback function to send the message through
        message: Either a dictionary or string message
    """
    if callback:
        await callback(message)

async def transcribe_audio(transcription_model="gpt-4o-mini-transcribe", use_local=False, send_status_callback=None, silence_threshold=100):
    """
    Record audio, transcribe it, and return the transcription.
    
    Args:
        transcription_model: Model to use for transcription
        use_local: Whether to use local or OpenAI transcription
        send_status_callback: Function to call to send status messages to client
        silence_threshold: Threshold for silence detection (minimum is 100)
        
    Returns:
        The transcribed text
    """
    try:
        # Create an async wrapper for the callback
        async def callback_wrapper(msg):
            if send_status_callback:
                await send_status_message(send_status_callback, msg)
        
        # Enforce minimum threshold to ensure silence detection works
        minimum_threshold = 100
        if silence_threshold < minimum_threshold:
            print(f"Warning: Silence threshold {silence_threshold} is too low, using minimum of {minimum_threshold}")
            silence_threshold = minimum_threshold
                
        # Record audio with enhanced mode and shorter silence duration
        temp_filename = await record_audio_enhanced(
            send_status_callback=callback_wrapper,
            silence_threshold=silence_threshold,
            silence_duration=SILENCE_DURATION_SECONDS  # Use shorter silence duration for faster response
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