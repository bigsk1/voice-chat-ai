/**
 * Audio Bridge Client
 * Handles WebRTC audio streaming between client browser and server
 */

class AudioBridgeClient {
    constructor() {
        // Configuration
        this.clientId = null;
        this.isEnabled = false;
        this.isInitialized = false;
        this.isConnected = false;
        this.peerConnection = null;
        this.dataChannel = null;
        this.localStream = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.wsConnection = null;
        this.pendingAudio = null;
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.isSafari = this.detectSafari();
        this.isFirefox = this.detectFirefox();
        this.useFallbackMode = false;
        this.isSecureContext = window.location.protocol === 'https:' || window.location.hostname === 'localhost';
        this.checkingStatus = false;
        this.dashboardMode = window.location.pathname === '/';
        this.isRecording = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this.lastStatusCheck = 0;
        this.statusCheckMinInterval = 180000; // 3 minutes minimum
        
        // Event callbacks
        this.onStatusChange = null;
        this.onAudioReceived = null;
        this.onError = null;
        
        // Try to restore client ID from local storage
        this.clientId = localStorage.getItem('audio_bridge_client_id');
        if (this.clientId) {
            console.log(`Restored audio bridge client ID from storage: ${this.clientId}`);
            // Mark as already connected if we have a client ID
            this.isInitialized = true;
            this.isConnected = true;
        }
        
        // Do an initial status check after loading
        setTimeout(() => {
            this.checkEnabled().then(enabled => {
                if (enabled && this.clientId) {
                    // If we have a client ID and bridge is enabled, try to reconnect
                    console.log('Attempting to reconnect to audio bridge with saved client ID');
                    this._pingServer().then(success => {
                        if (success) {
                            // Re-initialize the microphone and connection
                            this.requestMicrophonePermission().then(granted => {
                                if (granted) {
                                    this.isInitialized = true;
                                    this.isConnected = true;
                                    this.setupWebSocket();
                                    console.log('Successfully reconnected to audio bridge');
                                    
                                    // Notify status change
                                    if (this.onStatusChange) {
                                        this.onStatusChange({
                                            enabled: this.isEnabled,
                                            connected: true,
                                            status: 'active',
                                            fallback: this.useFallbackMode
                                        });
                                    }
                                }
                            });
                        }
                    });
                }
            });
        }, 2000); // Wait 2 seconds before initial status check
        
        // Setup periodic status checking, much less frequently (every 60 seconds)
        this.statusInterval = setInterval(() => this.checkEnabledWithThrottle(), 60000);
        
        // Reduce polling frequency when the page is not active
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Clear the existing interval
                clearInterval(this.statusInterval);
                // Check much less frequently when page is hidden (every 5 minutes)
                this.statusInterval = setInterval(() => this.checkEnabledWithThrottle(), 300000);
            } else {
                // Clear the existing interval
                clearInterval(this.statusInterval);
                // Check more frequently when page is visible (every 60 seconds)
                this.statusInterval = setInterval(() => this.checkEnabledWithThrottle(), 60000);
                // Only do an immediate status check if we haven't checked recently
                if (Date.now() - this.lastStatusCheck > this.statusCheckMinInterval) {
                    this.checkEnabled();
                }
            }
        });
        
        // On dashboard page, check for the start button and mic icon
        if (this.dashboardMode) {
            this.setupDashboardIntegration();
        }
    }
    
    // Setup integration with dashboard
    setupDashboardIntegration() {
        console.log('Setting up dashboard integration for audio bridge');
        
        // Look for the start button after a short delay to ensure DOM is loaded
        setTimeout(() => {
            const startButton = document.getElementById('start-conversation-btn');
            const stopButton = document.getElementById('stop-conversation-btn');
            const micIcon = document.getElementById('mic-icon');
            
            if (startButton && stopButton) {
                console.log('Found dashboard controls, setting up audio bridge integration');
                
                // Monitor recording state based on mic icon
                const observeMicIcon = () => {
                    if (micIcon) {
                        const isRecording = micIcon.classList.contains('mic-on');
                        
                        // If recording state changed, update our state
                        if (isRecording !== this.isRecording) {
                            this.isRecording = isRecording;
                            
                            if (this.isRecording && this.isConnected) {
                                // If we've started recording and we're connected, activate the audio bridge
                                console.log('Dashboard recording started - activating audio bridge');
                                this.startRecording();
                            } else if (!this.isRecording && this.isConnected) {
                                // If we've stopped recording and we're connected, stop the audio bridge
                                console.log('Dashboard recording stopped - deactivating audio bridge');
                                this.stopRecording();
                            }
                        }
                    }
                };
                
                // Check every 250ms for recording state changes
                setInterval(observeMicIcon, 250);
            }
        }, 1000);
    }

    /**
     * Detect Safari browser
     */
    detectSafari() {
        return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    }

    /**
     * Detect Firefox browser
     */
    detectFirefox() {
        return navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    }
    
    /**
     * Polyfill getUserMedia for older browsers
     */
    getUserMediaWithPolyfill(constraints) {
        // Check for secure context - WebRTC typically requires HTTPS
        if (!this.isSecureContext) {
            console.warn('getUserMedia requires a secure context (HTTPS or localhost)');
            // Instead of rejecting, we'll set fallback mode and resolve with null
            this.useFallbackMode = true;
            return Promise.resolve(null);
        }
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Modern browsers
            return navigator.mediaDevices.getUserMedia(constraints)
                .catch(err => {
                    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || 
                        err.name === 'NotFoundError' || err.name === 'NotSupportedError') {
                        // Enable fallback mode and return null instead of rejecting
                        console.warn(`Using fallback mode due to error: ${err.message}`);
                        this.useFallbackMode = true;
                        return null;
                    }
                    return Promise.reject(err);
                });
        }
        
        // Legacy versions of Firefox, Chrome, etc.
        const getUserMedia = navigator.getUserMedia || 
                            navigator.webkitGetUserMedia || 
                            navigator.mozGetUserMedia ||
                            navigator.msGetUserMedia;
        
        if (!getUserMedia) {
            // Enable fallback mode and return null instead of rejecting
            console.warn('getUserMedia is not supported, using fallback mode');
            this.useFallbackMode = true;
            return Promise.resolve(null);
        }
        
        return new Promise((resolve, reject) => {
            getUserMedia.call(navigator, constraints, resolve, function(err) {
                // Enable fallback mode and return null instead of rejecting
                console.warn(`Using fallback mode due to error: ${err.message || err.name}`);
                this.useFallbackMode = true;
                resolve(null);
            });
        });
    }
    
    /**
     * Check if the audio bridge is enabled with throttling
     */
    checkEnabledWithThrottle() {
        // Only check if we haven't checked recently (at least 60s between checks)
        const now = Date.now();
        if (now - this.lastStatusCheck >= this.statusCheckMinInterval) {
            return this.checkEnabled();
        }
        return Promise.resolve(this.isEnabled);
    }
    
    /**
     * Check if the audio bridge is enabled
     */
    async checkEnabled() {
        if (this.checkingStatus) {
            return this.isEnabled;
        }
        
        this.checkingStatus = true;
        this.lastStatusCheck = Date.now();
        
        try {
            const response = await fetch('/audio-bridge/status');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Only log status occasionally to reduce console spam
            if (Math.random() < 0.1) { // Only log about 10% of checks
                console.log('Audio bridge status check result:', data);
            }
            
            // Set isEnabled based on the 'status' field being 'active'
            this.isEnabled = data.status === 'active';
            
            // Set a long interval before next check can run (3 minutes minimum)
            this.statusCheckMinInterval = 180000; // 3 minutes minimum
            
            // Important: If we have a client ID, we should consider ourselves connected
            // even if the server doesn't see us as an active client yet
            const wasConnected = this.isConnected;
            const hasActiveClient = data.active_clients > 0;
            
            // Only update connection status if we have consistent information
            if (this.clientId && this.isInitialized) {
                // If we were connected and have a client ID, maintain that state
                // This prevents flickering between states
                this.isConnected = true;
                
                // If server reports no active clients but we think we're connected, ping to remind server
                if (!hasActiveClient && wasConnected) {
                    console.log('Pinging server to maintain connection state');
                    this._pingServer();
                }
            } else if (!this.clientId) {
                // If we don't have a client ID, we're definitely not connected
                this.isConnected = false;
            }
            
            if (this.onStatusChange) {
                this.onStatusChange({
                    enabled: this.isEnabled,
                    connected: this.isConnected,
                    status: data.status,
                    activeClients: data.active_clients,
                    totalClients: data.total_clients,
                    fallback: this.useFallbackMode
                });
            }
            
            this.checkingStatus = false;
            return this.isEnabled;
        } catch (error) {
            console.error('Error checking audio bridge status:', error);
            this.isEnabled = false;
            
            if (this.onError) {
                this.onError(`Failed to check audio bridge status: ${error.message}`);
            }
            
            this.checkingStatus = false;
            return false;
        }
    }
    
    /**
     * Ping the server to update our active status
     */
    async _pingServer() {
        if (!this.clientId) return false;
        
        try {
            // Just reregister to ping the server
            const response = await fetch('/audio-bridge/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ client_id: this.clientId })
            });
            
            if (!response.ok) {
                console.error(`Server returned error code ${response.status}`);
                return false;
            }
            
            const data = await response.json();
            console.log('Pinged server:', data);
            
            return data.status === 'success';
        } catch (error) {
            console.error('Error pinging server:', error);
            return false;
        }
    }
    
    /**
     * Initialize the audio bridge client
     */
    async initialize() {
        // Always check if the bridge is enabled before proceeding
        await this.checkEnabled();
        
        if (!this.isEnabled) {
            console.warn('Cannot initialize audio bridge - disabled on server');
            if (this.onError) {
                this.onError('Audio bridge is disabled on the server. Please check your .env configuration.');
            }
            return false;
        }
        
        if (this.isInitialized) {
            console.warn('Audio bridge already initialized');
            return true;
        }
        
        try {
            // Register client with server
            const response = await fetch('/audio-bridge/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                const errMsg = `Server returned error code ${response.status}`;
                console.error(errMsg);
                if (this.onError) this.onError(errMsg);
                return false;
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.clientId = data.client_id;
                
                // Store client ID in local storage to persist across page refreshes
                localStorage.setItem('audio_bridge_client_id', this.clientId);
                
                // Request microphone permission - simple approach from webrtc_realtime.js
                const micPermissionGranted = await this.requestMicrophonePermission();
                if (!micPermissionGranted) {
                    console.error('Failed to get microphone permission');
                    
                    if (this.onError) {
                        this.onError('Microphone permission denied. The audio bridge requires microphone access to function.');
                    }
                    
                    return false;
                }
                
                this.isInitialized = true;
                this.isConnected = true;
                console.log(`Audio bridge initialized with client ID: ${this.clientId}`);
                
                // Setup WebSocket for communication
                this.setupWebSocket();
                
                if (this.onStatusChange) {
                    this.onStatusChange({
                        enabled: this.isEnabled,
                        connected: true,
                        status: 'active',
                        fallback: this.useFallbackMode
                    });
                }
                
                // Force a status update to notify the server we're connected
                setTimeout(() => this._pingServer(), 1000);
                
                return true;
                
            } else {
                console.error('Failed to initialize audio bridge:', data.message);
                
                if (this.onError) {
                    this.onError(`Failed to initialize audio bridge: ${data.message}`);
                }
                
                return false;
            }
        } catch (error) {
            console.error('Error initializing audio bridge:', error);
            
            if (this.onError) {
                this.onError(`Error initializing audio bridge: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Setup WebSocket connection
     */
    setupWebSocket() {
        // Close existing connection if any
        if (this.wsConnection) {
            this.wsConnection.close();
        }
        
        // Create new WebSocket connection
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/audio-bridge/ws/${this.clientId}`;
        this.wsConnection = new WebSocket(wsUrl);
        
        this.wsConnection.onopen = () => {
            console.log('WebSocket connection established');
            this.isConnected = true;
            
            if (this.onStatusChange) {
                this.onStatusChange({
                    enabled: this.isEnabled,
                    connected: this.isConnected
                });
            }
        };
        
        this.wsConnection.onclose = () => {
            console.log('WebSocket connection closed');
            this.isConnected = false;
            
            if (this.onStatusChange) {
                this.onStatusChange({
                    enabled: this.isEnabled,
                    connected: this.isConnected
                });
            }
        };
        
        this.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            
            if (this.onError) {
                this.onError('WebSocket connection error');
            }
        };
        
        this.wsConnection.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleSignalingMessage(message);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
    }
    
    /**
     * Handle signaling messages from server
     */
    async handleSignalingMessage(message) {
        if (!message.type) {
            console.warn('Received message with no type');
            return;
        }
        
        switch (message.type) {
            case 'answer':
                try {
                    const remoteDesc = new RTCSessionDescription({
                        type: 'answer',
                        sdp: message.sdp
                    });
                    
                    await this.peerConnection.setRemoteDescription(remoteDesc);
                    console.log('Set remote description from answer');
                } catch (error) {
                    console.error('Error setting remote description:', error);
                }
                break;
                
            case 'ice-candidate':
                try {
                    if (message.candidate) {
                        const candidate = new RTCIceCandidate(message.candidate);
                        await this.peerConnection.addIceCandidate(candidate);
                        console.log('Added ICE candidate');
                    }
                } catch (error) {
                    console.error('Error adding ICE candidate:', error);
                }
                break;
                
            case 'error':
                console.error('Signaling error:', message.message);
                
                if (this.onError) {
                    this.onError(`Signaling error: ${message.message}`);
                }
                break;
                
            default:
                console.warn(`Unhandled message type: ${message.type}`);
        }
    }
    
    /**
     * Start recording audio
     */
    startRecording() {
        if (!this.isInitialized || !this.localStream) {
            console.warn('Cannot start recording - not initialized or no microphone stream');
            return false;
        }
        
        try {
            // Set up MediaRecorder if not already set up
            if (!this.mediaRecorder) {
                this.setupMediaRecorder();
            }
            
            // Start recording
            this.audioChunks = [];
            this.mediaRecorder.start();
            console.log('Started recording');
            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            
            if (this.onError) {
                this.onError(`Error starting recording: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Stop recording audio
     */
    stopRecording() {
        if (!this.isInitialized || !this.mediaRecorder) {
            console.warn('Cannot stop recording - not initialized or no media recorder');
            return false;
        }
        
        if (this.mediaRecorder.state !== 'recording') {
            console.warn('Not recording');
            return false;
        }
        
        try {
            this.mediaRecorder.stop();
            console.log('Stopped recording');
            return true;
        } catch (error) {
            console.error('Error stopping recording:', error);
            
            if (this.onError) {
                this.onError(`Error stopping recording: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Setup MediaRecorder
     */
    setupMediaRecorder() {
        if (!this.localStream) {
            console.warn('Cannot setup media recorder - no microphone stream');
            return false;
        }
        
        try {
            // Check for MediaRecorder support
            if (!window.MediaRecorder) {
                throw new Error('MediaRecorder is not supported in this browser');
            }
            
            // Set up MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.localStream);
            
            // Handle dataavailable event
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // Handle stop event
            this.mediaRecorder.onstop = async () => {
                if (this.audioChunks.length === 0) {
                    console.warn('No audio data recorded');
                    return;
                }
                
                // Create blob from chunks
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                
                // Send audio to server
                await this.sendAudioToServer(audioBlob);
                
                // Clear chunks
                this.audioChunks = [];
            };
            
            return true;
        } catch (error) {
            console.error('Error setting up media recorder:', error);
            
            if (this.onError) {
                this.onError(`Error setting up media recorder: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Send recorded audio to server
     */
    async sendAudioToServer(audioBlob) {
        if (!this.isConnected || !this.clientId) {
            console.warn('Cannot send audio - not connected');
            return false;
        }
        
        try {
            // Use WebRTC data channel if available
            if (this.dataChannel && this.dataChannel.readyState === 'open') {
                console.log('Sending audio via WebRTC data channel');
                // Convert blob to ArrayBuffer and send over data channel
                const reader = new FileReader();
                reader.onload = () => {
                    if (reader.result) {
                        try {
                            this.dataChannel.send(reader.result);
                            return true;
                        } catch (error) {
                            console.error('Error sending audio via data channel:', error);
                            if (this.onError) {
                                this.onError(`Error sending audio via data channel: ${error.message}`);
                            }
                        }
                    }
                };
                reader.readAsArrayBuffer(audioBlob);
                return true;
            } else {
                console.warn('Data channel not available for sending audio');
                if (this.onError) {
                    this.onError('WebRTC data channel not available for sending audio');
                }
                return false;
            }
        } catch (error) {
            console.error('Error sending audio to server:', error);
            if (this.onError) {
                this.onError(`Error sending audio to server: ${error.message}`);
            }
            return false;
        }
    }
    
    /**
     * Handle audio data received from server
     */
    handleAudioData(data) {
        if (this.onAudioReceived) {
            this.onAudioReceived(data);
        } else {
            // Queue for playback if no handler is set
            this.audioQueue.push(data);
            this.playNextAudio();
        }
    }
    
    /**
     * Play next audio in queue
     */
    async playNextAudio() {
        if (this.isPlaying || this.audioQueue.length === 0) {
            return;
        }
        
        this.isPlaying = true;
        const audioData = this.audioQueue.shift();
        
        try {
            const arrayBuffer = await audioData.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            
            source.onended = () => {
                this.isPlaying = false;
                this.playNextAudio();
            };
            
            source.start(0);
        } catch (error) {
            console.error('Error playing audio:', error);
            this.isPlaying = false;
            this.playNextAudio();
        }
    }
    
    /**
     * Disconnect from the audio bridge
     */
    async disconnect() {
        console.log('Disconnecting from audio bridge...');
        
        try {
            // If we have a client ID, unregister it
            if (this.clientId) {
                const response = await fetch('/audio-bridge/unregister', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        client_id: this.clientId
                    })
                });
                
                const data = await response.json();
                console.log('Unregister response:', data);
                
                // Clear stored client ID
                localStorage.removeItem('audio_bridge_client_id');
                this.clientId = null;
            }
            
            // Close any active connections
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            if (this.wsConnection) {
                this.wsConnection.close();
                this.wsConnection = null;
            }
            
            // Stop any active streams
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => track.stop());
                this.localStream = null;
            }
            
            // Reset state
            this.isInitialized = false;
            this.isConnected = false;
            this.reconnectAttempts = 0;
            
            console.log('Successfully disconnected from audio bridge');
            return true;
        } catch (error) {
            console.error('Error disconnecting from audio bridge:', error);
            return false;
        }
    }
    
    /**
     * Request microphone permissions explicitly, can be called before initialize
     */
    async requestMicrophonePermission() {
        try {
            console.log('Requesting microphone permission...');
            
            // Check for secure context first - WebRTC requires HTTPS
            if (!window.isSecureContext) {
                // Not a secure context - this will fail on most browsers
                console.error('WebRTC requires HTTPS. Current protocol:', window.location.protocol);
                
                // Check if we're not on localhost - localhost is allowed as a secure context
                if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                    if (this.onError) {
                        this.onError(`WebRTC requires HTTPS. Please access this site using HTTPS instead of HTTP. If this is a development environment, you can generate a self-signed certificate.`);
                    }
                    return false;
                }
            }
            
            // For Firefox - try to detect and handle permission issues better
            const isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
            
            // Configure audio constraints for best results
            const audioConstraints = {
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            };
            
            // Handle Firefox differently if needed
            if (isFirefox) {
                console.log('Using Firefox-specific microphone request');
                audioConstraints.audio = true; // Simpler constraints for Firefox
            }
            
            // Check if mediaDevices API is available
            if (!navigator.mediaDevices) {
                const errorMsg = 'navigator.mediaDevices is not available in this browser. This may be because you are using HTTP instead of HTTPS, or your browser does not support WebRTC.';
                console.error(errorMsg);
                if (this.onError) {
                    this.onError(errorMsg);
                }
                return false;
            }
            
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);
            
            if (!stream) {
                throw new Error('Stream is null after getUserMedia');
            }
            
            // Check if we actually got audio tracks
            if (stream.getAudioTracks().length === 0) {
                throw new Error('No audio tracks received from microphone');
            }
            
            // Keep a reference to the stream
            this.localStream = stream;
            
            console.log('Microphone permission granted successfully');
            return true;
        } catch (error) {
            console.error('Failed to get microphone permission:', error);
            
            let errorMessage = `Error accessing microphone (${error.name}): ${error.message}`;
            
            // Provide more specific error messages based on the error
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                errorMessage = 'Microphone permission was denied. Please grant permission when prompted.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone found. Please connect a microphone and try again.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Microphone is already in use by another application.';
            } else if (error.name === 'SecurityError') {
                errorMessage = 'Microphone access is blocked due to security restrictions.';
            } else if (error.name === 'AbortError') {
                errorMessage = 'Microphone permission request was aborted.';
            } else if (error.name === 'TypeError' && !navigator.mediaDevices) {
                errorMessage = 'Your browser does not support WebRTC or you are using HTTP instead of HTTPS. Please use a compatible browser or switch to HTTPS.';
            }
            
            console.error(errorMessage);
            if (this.onError) {
                this.onError(errorMessage);
            }
            
            return false;
        }
    }
}

// Export for ES modules
if (typeof exports !== 'undefined') {
    exports.AudioBridgeClient = AudioBridgeClient;
}

// Create global instance
window.audioBridge = new AudioBridgeClient(); 