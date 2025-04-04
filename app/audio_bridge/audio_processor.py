"""
Audio Processor
Handles conversion and processing of audio between WebRTC bridge and application
Provides an abstraction layer for audio input/output operations
"""

import os
import wave
import logging
import subprocess
import tempfile
import io
import struct
import math
from typing import Optional
import time
from pydub import AudioSegment
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        # Audio format configuration
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16-bit audio
        
    def convert_webrtc_audio_to_wav(self, webrtc_audio: bytes) -> bytes:
        """Convert WebRTC audio data to WAV format
        
        Args:
            webrtc_audio: Raw audio data from WebRTC
            
        Returns:
            WAV audio data as bytes
        """
        if len(webrtc_audio) == 0:
            logger.error("Cannot convert empty audio data")
            raise ValueError("Audio data is empty")
        
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
                logger.info(f"Created outputs directory at {outputs_dir}")
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            
            # Save incoming audio to a file in the outputs directory
            raw_input_path = os.path.join(outputs_dir, f"webrtc_raw_{timestamp}.raw")
            with open(raw_input_path, 'wb') as raw_file:
                raw_file.write(webrtc_audio)
            
            # Create output file path in the outputs directory
            temp_output_path = os.path.join(outputs_dir, f"webrtc_wav_{timestamp}.wav")

            try:
                # For small chunks, handle them differently - as raw PCM
                if len(webrtc_audio) < 10000:
                    logger.info(f"Small audio chunk detected ({len(webrtc_audio)} bytes), using specialized handling")
                    # Try as raw PCM at 16kHz, mono, 16-bit
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file if it exists
                        '-f', 'f32le',  # Try as 32-bit float PCM first
                        '-ar', str(self.sample_rate),  # Sampling rate
                        '-ac', str(self.channels),  # Channels 
                        '-i', raw_input_path,  # Input file
                        '-acodec', 'pcm_s16le',  # Force 16-bit PCM
                        temp_output_path  # Output file
                    ]
                else:
                    # Try to guess format for larger chunks
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file if it exists
                        '-f', 'auto',  # Try to auto-detect format
                        '-i', raw_input_path,  # Input file
                        '-ar', str(self.sample_rate),  # Sampling rate
                        '-ac', str(self.channels),  # Channels
                        '-acodec', 'pcm_s16le',  # Force 16-bit PCM
                        temp_output_path  # Output file
                    ]
                
                logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                # Check if conversion was successful
                if process.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {stderr.decode('utf-8')}")
                    # Try an alternative method if first conversion failed
                    fallback_cmd = [
                        'ffmpeg',
                        '-y',
                        '-f', 's16le',  # Try as 16-bit PCM
                        '-ar', str(self.sample_rate),
                        '-ac', str(self.channels),
                        '-i', raw_input_path,
                        '-acodec', 'pcm_s16le',
                        temp_output_path
                    ]
                    
                    logger.info(f"Trying fallback FFmpeg command: {' '.join(fallback_cmd)}")
                    process = subprocess.Popen(
                        fallback_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"Fallback FFmpeg conversion also failed: {stderr.decode('utf-8')}")
                        raise ValueError(f"Could not convert audio data: {stderr.decode('utf-8')}")
                
                # Read the converted WAV file
                with open(temp_output_path, 'rb') as wav_file:
                    wav_data = wav_file.read()
                
                logger.info(f"Successfully converted WebRTC audio to WAV format ({len(wav_data)} bytes)")
                
                # Keep the files in the outputs directory for debugging
                logger.info(f"Raw audio saved at: {raw_input_path}")
                logger.info(f"Converted WAV saved at: {temp_output_path}")
                
                return wav_data
                
            except Exception as e:
                logger.error(f"Error converting WebRTC audio: {e}")
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                raise
        except Exception as e:
            logger.error(f"Error in convert_webrtc_audio_to_wav: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_wav_from_pcm(self, pcm_data: bytes) -> bytes:
        """Convert PCM audio data to WAV format
        
        Args:
            pcm_data: Raw PCM audio data
            
        Returns:
            WAV audio data as bytes
        """
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            wav_path = os.path.join(outputs_dir, f"pcm_to_wav_{timestamp}.wav")
            
            # Create WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_data)
            
            # Read the WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            logger.info(f"Successfully converted PCM data to WAV format ({len(wav_data)} bytes), saved at {wav_path}")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error converting PCM data to WAV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_wav_from_raw_audio(self, raw_data: bytes) -> bytes:
        """Convert raw audio data to WAV format
        
        Args:
            raw_data: Raw audio data
            
        Returns:
            WAV audio data as bytes
        """
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            raw_path = os.path.join(outputs_dir, f"raw_audio_{timestamp}.raw")
            wav_path = os.path.join(outputs_dir, f"raw_to_wav_{timestamp}.wav")
            
            # Save raw data to file
            with open(raw_path, 'wb') as f:
                f.write(raw_data)
            
            # Try to convert using ffmpeg
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-f', 's16le',  # 16-bit PCM
                '-ar', str(self.sample_rate),  # Sampling rate
                '-ac', str(self.channels),  # Channels
                '-i', raw_path,  # Input file
                wav_path  # Output file
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            logger.info(f"Successfully converted raw audio to WAV format ({len(wav_data)} bytes), saved at {wav_path}")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error converting raw audio to WAV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_synthetic_wav(self) -> bytes:
        """Create a synthetic WAV file with a beep
        
        Returns:
            WAV audio data as bytes
        """
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            wav_path = os.path.join(outputs_dir, f"synthetic_beep_{timestamp}.wav")
            
            # Generate beep sound (1 second, 440Hz)
            duration = 1.0  # seconds
            frequency = 440.0  # Hz
            sample_rate = 16000  # samples per second
            
            # Generate a sine wave
            samples = int(duration * sample_rate)
            audio_data = []
            for i in range(samples):
                sample = int(32767.0 * math.sin(2 * math.pi * frequency * i / sample_rate))
                audio_data.append(sample)
            
            # Convert to bytes
            audio_bytes = b''
            for sample in audio_data:
                audio_bytes += struct.pack('<h', sample)
            
            # Create WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            
            # Read the WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            logger.info(f"Created synthetic beep WAV file ({len(wav_data)} bytes), saved at {wav_path}")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error creating synthetic WAV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Create an empty WAV
            return self._create_empty_wav()
    
    def _create_empty_wav(self) -> bytes:
        """Create an empty WAV file (silent)
        
        Returns:
            WAV audio data as bytes
        """
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            
            # Generate timestamp for unique filenames
            timestamp = int(time.time())
            wav_path = os.path.join(outputs_dir, f"empty_silent_{timestamp}.wav")
            
            # Create silent audio (0.5 seconds)
            duration = 0.5  # seconds
            sample_rate = 16000  # samples per second
            samples = int(duration * sample_rate)
            silence = b'\x00\x00' * samples  # 16-bit silence
            
            # Create WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(silence)
            
            # Read the WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            logger.info(f"Created empty WAV file ({len(wav_data)} bytes), saved at {wav_path}")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error creating empty WAV: {e}")
            
            # Create minimal valid WAV data as a last resort
            # WAV header + minimal data
            wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            logger.info("Created minimal empty WAV data as last resort")
            return wav_header

    def convert_wav_to_webrtc_audio(self, wav_audio: bytes) -> bytes:
        """
        Convert WAV audio to format suitable for WebRTC
        
        Args:
            wav_audio: WAV format audio data
            
        Returns:
            Audio data in WebRTC-compatible format
        """
        try:
            # In a real implementation, this would convert WAV to WebRTC's format
            # This is a simplified version that assumes WebRTC can handle raw PCM
            
            # Read the WAV header and extract the PCM data
            with io.BytesIO(wav_audio) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # Check if mono and 16-bit
                    if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                        # Would need to convert, but that's beyond the scope here
                        pass
                    
                    # Extract raw PCM data
                    pcm_data = wav_file.readframes(wav_file.getnframes())
            
            return pcm_data
        except Exception as e:
            logger.error(f"Error converting WAV to WebRTC audio: {e}")
            return bytes()

    async def process_audio_file(self, file_path, client_id=None, silence_threshold=200, silence_duration=1.5):
        """
        Process an audio file directly for transcription
        
        Args:
            file_path: Path to the audio file
            client_id: Optional client ID for tracking
            silence_threshold: Threshold for silence detection
            silence_duration: Duration of silence to stop recording
            
        Returns:
            Path to the processed file or None if processing failed
        """
        try:
            import os
            from pydub import AudioSegment
            
            logger.info(f"Processing audio file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return None
                
            # Check file size
            file_size = os.path.getsize(file_path)
            logger.info(f"Audio file size: {file_size} bytes")
            
            if file_size < 100:  # Arbitrary small size to catch empty files
                logger.warning(f"Audio file too small: {file_size} bytes")
                return None
                
            # Create a base filename for output
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Just use the input file if it's already a wav file in the right format
            # Otherwise convert it to wav
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.wav':
                # Already a WAV file, use it directly
                output_path = file_path
            else:
                # Convert to WAV
                timestamp = int(time.time())
                output_path = os.path.join(output_dir, f"recording_{timestamp}.wav")
                
                try:
                    if file_ext == '.webm':
                        # Use ffmpeg for WebM conversion
                        import subprocess
                        cmd = f'ffmpeg -i "{file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_path}"'
                        subprocess.run(cmd, shell=True, check=True)
                    else:
                        # Use pydub for other formats
                        audio = AudioSegment.from_file(file_path)
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(output_path, format="wav")
                    
                    logger.info(f"Converted audio to WAV: {output_path}")
                except Exception as e:
                    logger.error(f"Error converting audio file: {e}")
                    return None
            
            # Log success and return the path
            logger.info(f"Audio file ready for transcription: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def process_audio(self, audio_data, client_id=None, silence_threshold=500, silence_duration=2.0, debug_mode=False):
        """
        Process audio data and save as WAV file if valid speech is detected
        
        Args:
            audio_data (bytes): Raw audio data to process
            client_id (str): Client ID for tracking audio files
            silence_threshold (int): Threshold for silence detection (lower means more sensitive)
            silence_duration (float): Duration of silence in seconds to consider end of speech
            debug_mode (bool): If True, process audio regardless of levels for debugging

        Returns:
            bool: True if audio was processed and contains speech, False otherwise
        """
        try:
            # Ensure minimum silence threshold to avoid false triggers
            if silence_threshold < 100:
                logger.warning(f"Silence threshold {silence_threshold} is too low, using minimum of 100")
                silence_threshold = 100
            
            # Parse data into numpy array
            if not audio_data or len(audio_data) == 0:
                logger.debug("No audio data to process")
                return False
            
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check if we have enough data to process
            if len(audio_array) < 100:
                if debug_mode:
                    logger.info(f"Short audio data ({len(audio_array)} samples), but processing in debug mode")
                else:
                    logger.debug(f"Audio data too short ({len(audio_array)} samples), skipping")
                    return False
            
            # Calculate audio levels
            max_level = np.max(np.abs(audio_array))
            rms_level = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            
            # Log audio level every 5th call to avoid flooding logs
            self.call_count += 1
            if debug_mode or self.call_count % 5 == 0 or max_level > silence_threshold:
                logger.info(f"Audio levels - Max: {max_level}, RMS: {rms_level:.2f}, Length: {len(audio_array)} samples")
            
            # Detect if this contains speech (or we're in debug mode)
            speech_detected = max_level > silence_threshold
            force_processing = debug_mode or len(audio_array) > 32000  # Also process if chunk is large enough
            
            if speech_detected or force_processing:
                # Create temporary WAV file
                client_suffix = f"_{client_id}" if client_id else ""
                output_filename = f"recording_{int(time.time())}{client_suffix}.wav"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save WAV file with audio data
                self._save_audio_as_wav(audio_data, output_path)
                
                # Store file reference for this client
                if client_id:
                    self.client_audio_files[client_id] = output_path
                
                if speech_detected:
                    logger.info(f"Speech detected in audio data, saved to {output_path}")
                elif debug_mode:
                    logger.info(f"Processing in debug mode, saved to {output_path}")
                else:
                    logger.info(f"Forced processing of non-speech audio, saved to {output_path}")
                
                # Return True to indicate audio was processed
                return True
            else:
                logger.debug(f"No speech detected in audio data, max level {max_level} < threshold {silence_threshold}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# Create a singleton instance
audio_processor = AudioProcessor() 