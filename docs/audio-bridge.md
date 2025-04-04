# Audio Bridge Technical Documentation

## Overview

The Audio Bridge is a WebRTC-based system that enables remote audio capture and playback between the browser and the server. It allows users to interact with the application using their microphone and speakers without needing to install additional software. The bridge facilitates real-time audio streaming through WebRTC, providing a seamless audio experience.

The primary purpose of the audio bridge is to enable using the application from another device in a local network, where a local device on the network can connect via a browser and use its own microphone and speakers to interact with the application. This allows users to access the application from any device (phones, tablets, laptops) on the local network without needing to directly access the machine where the server is running.

## Architecture

The Audio Bridge consists of several key components:

1. **Server-Side Components**:
   - `AudioBridgeServer`: Main server class that manages WebRTC connections, signaling, and audio processing
   - `AudioTrackProcessor`: Processes audio frames from WebRTC connections
   - `AudioProcessor`: Handles audio format conversion between WebRTC and WAV

2. **Client-Side Components**:
   - `AudioBridgeClient.js`: JavaScript client that establishes WebRTC connections with the server
   - WebRTC connection handling (PeerConnection, DataChannel, etc.)
   - Audio capture and playback functionality

3. **Communication Channels**:
   - WebSocket for signaling (connection establishment, ICE candidates, offers/answers)
   - WebRTC for audio data transmission
   - DataChannel for control messages

## Flow of Operation

1. **Initialization**:
   - Server starts and initializes the Audio Bridge server component
   - Client loads and creates an instance of the AudioBridgeClient class
   - Client connects to the server via WebSocket

2. **Connection Establishment**:
   - Client initiates a WebRTC connection by creating an offer
   - Server receives the offer and creates an answer
   - ICE candidates are exchanged to establish the optimal connection path
   - Once connected, the data channel is established for control messages

3. **Audio Capture**:
   - Client accesses the user's microphone through `getUserMedia()`
   - Audio data is sent to the server through the WebRTC connection
   - Server processes and buffers the audio data

4. **Audio Playback**:
   - Server sends audio data to the client through the WebRTC connection
   - Client plays the audio through the user's speakers

## Server Components in Detail

### AudioBridgeServer

The `AudioBridgeServer` class is the central component that manages the WebRTC connections. It:

- Maintains a registry of connected clients
- Handles signaling messages for WebRTC connection establishment
- Processes incoming audio data from clients
- Stores audio data for later retrieval
- Distributes audio to connected clients for playback

```python
# Example of key functionality
class AudioBridgeServer:
    def __init__(self):
        self.clients_set = set()
        self.connections = {}
        self.data_channels = {}
        self.client_audio = defaultdict(deque)
        # ...

    async def run_server(self):
        """Start the audio bridge server"""
        # Initialize and run the aiohttp server
        # Handle WebSocket connections
        # ...
        
    async def handle_signaling(self, client_id, message):
        """Process WebRTC signaling messages"""
        # Handle offer/answer exchange
        # Process ICE candidates
        # ...
```

### AudioTrackProcessor

The `AudioTrackProcessor` extends the `MediaStreamTrack` class from the aiortc library to process audio frames from the WebRTC connection. It:

- Receives audio frames from the client's microphone
- Processes and converts the audio data
- Stores the audio data for later retrieval
- Monitors audio levels for quality control

```python
class AudioTrackProcessor(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, client_id, server):
        # ...
        
    async def recv(self):
        # Process incoming audio frames
        # ...
```

### Audio File Handling

Audio files are saved in the `outputs` directory for debugging and analysis. The file naming includes timestamps for easy identification:

- Raw audio data: `webrtc_raw_{timestamp}.raw`
- Converted WAV files: `webrtc_wav_{timestamp}.wav`
- Audio frames: `audio_frame_{client_id}_{timestamp}.wav`

## Client Components in Detail

### AudioBridgeClient.js

The `AudioBridgeClient.js` file contains the JavaScript code that runs in the browser to establish the WebRTC connection with the server. It:

- Manages the WebSocket connection for signaling
- Creates and maintains the RTCPeerConnection
- Handles the exchange of SDP offers/answers and ICE candidates
- Captures audio from the user's microphone
- Plays back audio received from the server

```javascript
class AudioBridgeClient {
    constructor() {
        // Initialize properties
        // ...
    }
    
    _initWebSocket() {
        // Set up WebSocket connection for signaling
        // ...
    }
    
    createPeerConnection() {
        // Set up WebRTC peer connection
        // Handle ICE candidates
        // ...
    }
    
    // Other methods for audio capture, playback, etc.
}
```

## ICE Candidate Handling

The ICE (Interactive Connectivity Establishment) framework is crucial for establishing a direct connection between the client and server, especially when NAT (Network Address Translation) or firewalls are involved.

### Server-Side ICE Handling

The server receives ICE candidates from the client and parses them to create `RTCIceCandidate` objects:

```python
# The aiortc library expects RTCIceCandidate to be created with specific parameters
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
```

### Client-Side ICE Handling

The client collects ICE candidates from the local RTCPeerConnection and sends them to the server:

```javascript
// Handle ICE candidates
this.peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
        this.sendSignalingMessage({
            type: 'ice-candidate',
            candidate: {
                candidate: event.candidate.candidate,
                sdpMid: event.candidate.sdpMid,
                sdpMLineIndex: event.candidate.sdpMLineIndex
            }
        });
    }
};
```

## HTTP vs HTTPS Considerations

### Local Development

For local development, the Audio Bridge can be configured to use either HTTP or HTTPS. There are important considerations for each approach:

#### Using HTTP for Local Development

**Configuration**:
- Set `ENABLE_AUDIO_BRIDGE=true`
- Do not set `ENABLE_HTTPS=true`

**Pros**:
- Works with browsers that reject self-signed certificates (like Firefox)
- Simpler setup without SSL certificate generation
- Faster development iteration

**Cons**:
- Only works for local development (localhost)
- Some browsers restrict WebRTC features in non-secure contexts
- Not suitable for production

#### Using HTTPS for Local Development

**Configuration**:
- Set `ENABLE_AUDIO_BRIDGE=true`
- Set `ENABLE_HTTPS=true`

**Pros**:
- Mirrors production environment more closely
- All WebRTC features are available
- Prepares code for production deployment

**Cons**:
- Requires generating self-signed certificates
- Firefox and some browsers reject self-signed certificates
- More complex setup

### Production Deployment

For production, HTTPS is mandatory due to browser security requirements:

**Configuration**:
- Set `ENABLE_AUDIO_BRIDGE=true`
- Set `ENABLE_HTTPS=true`
- Use properly signed SSL certificates from a trusted certificate authority

**Considerations**:
- WebRTC is restricted to secure contexts (HTTPS) in modern browsers
- All audio capture functionality requires secure context
- User data security is enhanced with encrypted connections

## Mixed Environment Setup

A common development setup uses:
- HTTPS for the main application (port 8080) with self-signed certificates
- HTTP for the Audio Bridge server (port 8081)

This works because:
1. The main application runs on HTTPS, satisfying browser security requirements
2. The Audio Bridge runs on HTTP, avoiding issues with self-signed certificates
3. The main application's server-side code acts as a proxy between the client and the Audio Bridge, bypassing CORS issues

This mixed setup is achieved by:
- Setting `ENABLE_AUDIO_BRIDGE=true`
- Not setting `ENABLE_HTTPS=true`
- Using FastAPI endpoints to proxy requests to the Audio Bridge

## Environment Variables

The Audio Bridge behavior is controlled through these environment variables:

| Variable | Purpose | Values |
|----------|---------|--------|
| `ENABLE_AUDIO_BRIDGE` | Enables/disables the entire Audio Bridge | `true` or `false` |
| `ENABLE_HTTPS` | Configures the Audio Bridge to use SSL | `true` or `false` |
| `AUDIO_BRIDGE_PORT` | Port for the Audio Bridge server | Default: `8081` |
| `DEBUG_AUDIO_BRIDGE` | Enables verbose logging | `true` or `false` |

## Application Modes

### Local Mode (Default)

When the Audio Bridge is disabled (`ENABLE_AUDIO_BRIDGE=false` or not set):
- The application uses the microphone and speakers of the device where the server is running
- Audio recording and playback happen directly on the server machine
- No WebRTC connection is established
- Suitable for using the application only on the same device where it's running

### Remote Mode (Audio Bridge Enabled)

When the Audio Bridge is enabled (`ENABLE_AUDIO_BRIDGE=true`):
- The application can be accessed from any browser on other devices in the network
- Remote devices use their own microphones and speakers to interact with the application
- Audio from the client browser's microphone is sent to the server for processing
- The server's audio responses are sent back to be played on the client browser's speakers
- Enables using the application from mobile devices, tablets, laptops, etc., without needing to access the server machine directly
- Perfect for docker deployments or headless servers, as no audio hardware is required on the server

## SSL Certificate Setup

### Generating Self-Signed Certificates

The application includes a utility script for generating self-signed SSL certificates for local development:

1. Run the `generate_certs.py` script at the root of the project:
   ```bash
   python generate_certs.py
   ```

2. This script creates a `certs` folder containing:
   - `cert.pem`: The SSL certificate
   - `key.pem`: The private key
   - `openssl.cnf`: OpenSSL configuration file

3. The main application (port 8080) will automatically use these certificates when HTTPS is enabled.

4. For the audio bridge, you can choose to use HTTP or HTTPS as discussed in the HTTP vs HTTPS Considerations section.

### Certificate Considerations

- Self-signed certificates are intended for development purposes only
- Modern browsers will display security warnings with self-signed certificates
- Firefox has stricter policies and may reject WebRTC connections over self-signed certificates
- For production, proper certificates from a trusted certificate authority should be used

### Using the Certificates

To use the generated certificates:
1. Set `ENABLE_HTTPS=true` in your `.env` file for the main application
2. For the Audio Bridge, configure as needed based on browser compatibility

When running with a properly configured HTTPS setup and `ENABLE_AUDIO_BRIDGE=true`, the application can be accessed securely from other devices in your network or even publicly if your server is exposed to the internet.

## Debugging

### Audio Files

The system saves audio files to the `outputs` directory for debugging:
- Input audio from clients
- Synthetic silence when no audio is detected
- Processed audio frames
- Converted WAV files

### Logging

Log messages include:
- Connection states
- Audio levels
- ICE candidate processing
- WebRTC negotiation
- Error conditions

Example log message for very low audio:
```
Very low audio level detected from client client_1234567890: 0.47
```

## Common Issues and Solutions

### WebRTC Connection Failures

If clients cannot connect:
- Check ICE candidate exchange in logs
- Ensure network allows UDP traffic
- Verify STUN/TURN servers if using them
- Check browser console for errors

### Audio Quality Issues

If audio quality is poor:
- Check audio levels in logs
- Verify microphone permissions
- Look for packet loss in WebRTC stats
- Ensure adequate network bandwidth

### HTTP/HTTPS Mixed Content

If seeing mixed content warnings:
- Ensure all resources are loaded over the same protocol
- Use relative URLs where possible
- Check the browser console for specific warnings

## Conclusion

The Audio Bridge provides a powerful way to capture and process audio directly from the browser. Its WebRTC-based architecture enables real-time audio streaming with low latency and high quality. The system is designed to be flexible, supporting both development and production environments with appropriate security considerations. 