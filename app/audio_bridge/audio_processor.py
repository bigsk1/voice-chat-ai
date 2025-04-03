"""
Audio Processor
Handles conversion and processing of audio between WebRTC bridge and application
Provides an abstraction layer for audio input/output operations
"""

import io
import logging
import wave

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
            # In a real implementation, this would convert from WebRTC's format
            # (likely Opus) to raw PCM and then to WAV
            # This is a simplified version that assumes webrtc_audio is already PCM
            
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(webrtc_audio)
            
            return audio_io.getvalue()
        except Exception as e:
            logger.error(f"Error converting WebRTC audio to WAV: {e}")
            # Return empty WAV as fallback
            return self._create_empty_wav()
            
    def convert_wav_to_webrtc_audio(self, wav_data: bytes) -> bytes:
        """
        Convert WAV audio to format suitable for WebRTC
        
        Args:
            wav_data: Audio data in WAV format
            
        Returns:
            Audio data suitable for WebRTC
        """
        try:
            # In a real implementation, this would extract PCM from WAV
            # and convert to a WebRTC-friendly format (likely Opus)
            # This is a simplified version that just extracts PCM
            
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                # Extract raw PCM data
                frames = wav_file.readframes(wav_file.getnframes())
                return frames
        except Exception as e:
            logger.error(f"Error converting WAV to WebRTC audio: {e}")
            return b''
            
    def _create_empty_wav(self, duration_ms: int = 500) -> bytes:
        """
        Create an empty WAV file of specified duration
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            Empty WAV file as bytes
        """
        # Calculate number of frames
        num_frames = int((duration_ms / 1000) * self.sample_rate)
        
        # Create silent audio data (all zeros)
        silent_data = b'\x00' * (num_frames * self.sample_width * self.channels)
        
        # Create WAV file
        audio_io = io.BytesIO()
        with wave.open(audio_io, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(silent_data)
        
        return audio_io.getvalue()
        
    def get_audio_properties(self, wav_data: bytes) -> dict:
        """
        Extract properties from WAV audio data
        
        Args:
            wav_data: Audio data in WAV format
            
        Returns:
            Dictionary with audio properties
        """
        try:
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                return {
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'frame_rate': wav_file.getframerate(),
                    'n_frames': wav_file.getnframes(),
                    'duration': wav_file.getnframes() / wav_file.getframerate()
                }
        except Exception as e:
            logger.error(f"Error getting audio properties: {e}")
            return {
                'error': str(e)
            }

# Initialize the processor
audio_processor = AudioProcessor() 