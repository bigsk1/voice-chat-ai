"""
Audio Processor
Handles conversion and processing of audio between WebRTC bridge and application
Provides an abstraction layer for audio input/output operations
"""

import io
import logging
import wave
import os
import subprocess
import tempfile
import struct
from typing import Optional
import math

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
        """
        Convert audio received from WebRTC to WAV format
        
        Args:
            webrtc_audio: Raw audio data from WebRTC
            
        Returns:
            WAV format audio data
        """
        try:
            # Check if input is empty
            if not webrtc_audio or len(webrtc_audio) == 0:
                logger.warning("Received empty WebRTC audio data")
                return self._create_empty_wav()
            
            logger.info(f"Converting {len(webrtc_audio)} bytes of WebRTC audio to WAV")
            
            # Analyze the first 20 bytes to help with debugging
            if len(webrtc_audio) >= 20:
                sample_bytes = webrtc_audio[:20]
                hex_values = ' '.join(f'{b:02x}' for b in sample_bytes)
                logger.info(f"First 20 bytes of audio data: {hex_values}")
            
            # Check if the data might be Opus - look for common Opus header patterns
            if len(webrtc_audio) > 8 and webrtc_audio[:4] in [b'OggS', b'Opus']:
                logger.info("Audio data appears to be in Opus format")
            
            # First try to treat as PCM data directly - most reliable approach
            try:
                logger.info("Attempting direct PCM to WAV conversion")
                wav_data = self._create_wav_from_pcm(webrtc_audio)
                logger.info(f"PCM conversion successful, created {len(wav_data)} bytes of WAV data")
                return wav_data
            except Exception as e:
                logger.warning(f"Failed to convert as PCM, trying with raw audio approach: {e}")
                
                try:
                    logger.info("Attempting raw audio to WAV conversion")
                    wav_data = self._create_wav_from_raw_audio(webrtc_audio)
                    logger.info(f"Raw audio conversion successful, created {len(wav_data)} bytes of WAV data")
                    return wav_data
                except Exception as e2:
                    logger.warning(f"Failed to convert as raw audio, trying with ffmpeg: {e2}")
            
            # Save incoming audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as temp_file:
                temp_file.write(webrtc_audio)
                temp_input_path = temp_file.name
            
            # Create temp output file
            temp_output_handle, temp_output_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_output_handle)

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
                        '-i', temp_input_path,  # Input file
                        '-acodec', 'pcm_s16le',  # Force 16-bit PCM
                        temp_output_path  # Output file
                    ]
                else:
                    # Try to guess format for larger chunks
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file if it exists
                        '-f', 'auto',  # Try to auto-detect format
                        '-i', temp_input_path,  # Input file
                        '-ar', str(self.sample_rate),  # Sampling rate
                        '-ac', str(self.channels),  # Channels
                        '-acodec', 'pcm_s16le',  # Force 16-bit PCM
                        temp_output_path  # Output file
                    ]
                
                logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                
                # Execute conversion with verbose logging
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                # Log the command output for debugging
                if process.returncode == 0:
                    logger.info(f"ffmpeg conversion successful from {len(webrtc_audio)} bytes to WAV")
                    if stderr:
                        logger.debug(f"ffmpeg stderr: {stderr.decode()}")
                else:
                    logger.error(f"ffmpeg conversion failed: {stderr.decode()}")
                    logger.info("Trying alternative formats...")
                    
                    # Try a different format if the first attempt failed
                    alternate_formats = ['s16le', 'opus', 'webm', 'aac']
                    for fmt in alternate_formats:
                        logger.info(f"Trying ffmpeg with format {fmt}")
                        alt_cmd = [
                            'ffmpeg',
                            '-y',  # Overwrite output file
                            '-f', fmt,  # Try specific format
                            '-i', temp_input_path,  # Input file
                            '-ar', str(self.sample_rate),  # Sampling rate
                            '-ac', str(self.channels),  # Channels
                            temp_output_path  # Output file
                        ]
                        
                        alt_process = subprocess.Popen(
                            alt_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        alt_stdout, alt_stderr = alt_process.communicate()
                        
                        if alt_process.returncode == 0:
                            logger.info(f"Alternative format {fmt} succeeded")
                            break
                        else:
                            logger.debug(f"Format {fmt} failed: {alt_stderr.decode()}")
                    
                    # If all else failed, try one more fallback
                    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
                        logger.warning("All ffmpeg attempts failed, using last resort conversion")
                        return self._create_wav_from_pcm(webrtc_audio)
                
                # Check if the output file exists and has content
                if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                    # Read the converted WAV file
                    with open(temp_output_path, 'rb') as wav_file:
                        wav_data = wav_file.read()
                    
                    logger.info(f"Conversion successful, returning {len(wav_data)} bytes of WAV data")
                    return wav_data
                else:
                    logger.error("ffmpeg created an empty or missing output file")
                    # Last resort, try PCM again
                    return self._create_wav_from_pcm(webrtc_audio)
            
            except Exception as e:
                logger.error(f"Error in conversion process: {e}")
                # Try one more fallback approach
                try:
                    logger.info("Trying last resort conversion method")
                    return self._create_wav_from_pcm(webrtc_audio)
                except Exception as e2:
                    logger.error(f"All conversion methods failed: {e2}")
                    
                # If everything failed, create a synthetic audio file with a beep 
                # to indicate something was received
                logger.warning("Creating synthetic audio as last resort")
                return self._create_synthetic_wav()
            
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_input_path):
                        os.unlink(temp_input_path)
                    
                    if os.path.exists(temp_output_path):
                        os.unlink(temp_output_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temp files: {e}")
                
        except Exception as e:
            logger.error(f"Error converting WebRTC audio to WAV: {e}")
            # Return synthetic WAV as fallback
            return self._create_synthetic_wav()
            
    def _create_wav_from_pcm(self, pcm_data: bytes) -> bytes:
        """Create a WAV file from PCM data"""
        # Check if data looks like PCM (should be even-sized for 16-bit)
        if len(pcm_data) % 2 != 0:
            raise ValueError("PCM data must have even length for 16-bit samples")
        
        # Create an in-memory WAV file
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data)
        
        return wav_buffer.getvalue()
    
    def _create_wav_from_raw_audio(self, audio_data: bytes) -> bytes:
        """Try to convert raw audio data into WAV by guessing format"""
        # Try to detect bit depth from data pattern
        is_likely_16bit = self._detect_if_16bit(audio_data)
        sample_width = 2 if is_likely_16bit else 1 
        
        # Create an in-memory WAV file
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(sample_width)  
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        return wav_buffer.getvalue()
    
    def _detect_if_16bit(self, audio_data: bytes) -> bool:
        """
        Heuristically detect if data is likely 16-bit PCM by checking for patterns
        in the byte distribution
        """
        # Check if length is even (required for 16-bit)
        if len(audio_data) % 2 != 0:
            return False
            
        # Take a sample of the data to analyze 
        sample_size = min(1000, len(audio_data) // 2)
        
        # Check for non-zero bytes in every other position which would indicate 16-bit
        high_byte_count = 0
        
        for i in range(1, sample_size * 2, 2):
            if audio_data[i] != 0:
                high_byte_count += 1
        
        # If more than 10% of high bytes are non-zero, likely 16-bit
        return high_byte_count > (sample_size * 0.1)
    
    def _create_empty_wav(self) -> bytes:
        """
        Create an empty WAV file
        
        Returns:
            Empty WAV file as bytes
        """
        try:
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b'')  # Empty audio data
            
            return audio_io.getvalue()
        except Exception as e:
            logger.error(f"Error creating empty WAV: {e}")
            # Create a minimal valid WAV file
            return bytes.fromhex(
                '52494646' +  # "RIFF"
                '24000000' +  # Chunk size (36 + 0 = 36 bytes)
                '57415645' +  # "WAVE"
                '666D7420' +  # "fmt "
                '10000000' +  # Subchunk1 size (16 bytes)
                '0100' +      # Audio format (1 = PCM)
                '0100' +      # Number of channels (1)
                '80BB0000' +  # Sample rate (48000 Hz)
                '00770100' +  # Byte rate (48000 * 2 * 1 = 96000)
                '0200' +      # Block align (2 * 1 = 2)
                '1000' +      # Bits per sample (16)
                '64617461' +  # "data"
                '00000000'    # Subchunk2 size (0 bytes)
            )
    
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

    def _create_synthetic_wav(self) -> bytes:
        """
        Create a synthetic WAV file with a beep tone to ensure something is audible
        
        Returns:
            WAV file with a beep tone as bytes
        """
        try:
            # Generate a short 500 Hz beep tone (16-bit PCM, mono, 16kHz)
            duration_ms = 500  # 500 ms beep
            frequency = 500  # 500 Hz tone
            amplitude = 10000  # Moderate volume (16-bit range is -32768 to 32767)
            
            # Calculate values
            samples = int(self.sample_rate * duration_ms / 1000)
            
            # Generate the tone
            audio_data = bytearray(samples * 2)  # 2 bytes per sample for 16-bit
            for i in range(samples):
                # Generate sine wave
                sample_value = int(amplitude * math.sin(2 * math.pi * frequency * i / self.sample_rate))
                
                # Convert to 16-bit PCM (little endian)
                audio_data[i*2] = sample_value & 0xFF
                audio_data[i*2 + 1] = (sample_value >> 8) & 0xFF
            
            # Create WAV from the synthetic audio
            return self._create_wav_from_pcm(bytes(audio_data))
            
        except Exception as e:
            logger.error(f"Error creating synthetic WAV: {e}")
            # Fall back to empty WAV
            return self._create_empty_wav()

# Create a singleton instance
audio_processor = AudioProcessor() 