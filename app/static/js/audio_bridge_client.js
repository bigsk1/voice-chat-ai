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
        this.messageQueue = []; // Initialize messageQueue for signaling messages
        
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
                                    this._initWebSocket();
                                    
                                    // Also establish the peer connection
                                    this.createPeerConnection();
                                    
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
     * Generate a unique client ID
     * @private
     * @returns {string} A unique ID for this client
     */
    _generateClientId() {
        // Generate a random ID using timestamp and random number
        const timestamp = new Date().getTime();
        const random = Math.floor(Math.random() * 1000000);
        const clientId = `client_${timestamp}_${random}`;
        
        // Store the client ID in local storage for persistence
        localStorage.setItem('audio_bridge_client_id', clientId);
        
        console.log(`Generated new client ID: ${clientId}`);
        return clientId;
    }
    
    /**
     * Initialize the client
     */
    async initialize() {
        if (this.isInitialized) {
            console.log('Audio bridge client already initialized');
            return this.isInitialized;
        }
        
        // Generate a unique client ID if not provided
        if (!this.clientId) {
            this.clientId = this._generateClientId();
        }
        
        // Initialize WebSocket for signaling
        this._initWebSocket();
        
        try {
            // Request microphone permission
            const micPermission = await this.requestMicrophonePermission();
            if (!micPermission) {
                throw new Error('Microphone permission denied');
            }
            
            // Create peer connection
            const peerConnected = await this.createPeerConnection();
            if (!peerConnected) {
                throw new Error('Failed to create peer connection');
            }
            
            this.isInitialized = true;
            this.isEnabled = true;
            
            if (this.onStatusChange) {
                this.onStatusChange({
                    enabled: this.isEnabled,
                    connected: this.isConnected,
                    status: 'initialized'
                });
            }
            
            return true;
        } catch (error) {
            console.error('Initialization error:', error);
            
            if (this.onError) {
                this.onError(`Initialization failed: ${error.message}`);
            }
            
            this.isInitialized = false;
            return false;
        }
    }
    
    /**
     * Process any queued signaling messages
     * @private
     */
    _processQueuedMessages() {
        if (this.messageQueue.length > 0) {
            console.log(`Processing ${this.messageQueue.length} queued messages`);
            
            for (const message of this.messageQueue) {
                this.sendSignalingMessage(message);
            }
            
            // Clear the queue
            this.messageQueue = [];
        }
    }
    
    /**
     * Handle incoming signaling message
     * @private
     */
    _handleSignalingMessage(data) {
        try {
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            
            console.log('Processing signaling message:', message.type);
            
            if (message.type === 'welcome') {
                // Connection established
                this.isConnected = true;
                
                if (this.onStatusChange) {
                    this.onStatusChange({
                        enabled: this.isEnabled,
                        connected: this.isConnected,
                        status: 'connected'
                    });
                }
            } else if (message.type === 'answer') {
                // Handle WebRTC answer
                if (this.peerConnection) {
                    try {
                        const remoteDesc = new RTCSessionDescription({
                            type: 'answer',
                            sdp: message.sdp
                        });
                        
                        this.peerConnection.setRemoteDescription(remoteDesc)
                            .then(() => console.log('Set remote description from answer'))
                            .catch(error => console.error('Error setting remote description:', error));
                    } catch (error) {
                        console.error('Error creating RTCSessionDescription:', error);
                    }
                }
            } else if (message.type === 'ice-candidate') {
                // Handle ICE candidate
                if (this.peerConnection && message.candidate) {
                    this.peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate))
                        .then(() => console.log('Added ICE candidate'))
                        .catch(error => console.error('Error adding ICE candidate:', error));
                }
            } else if (message.type === 'error') {
                // Handle error message
                console.error('Received error from server:', message.message);
                
                if (this.onError) {
                    this.onError(message.message);
                }
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    }
    
    /**
     * Refresh the microphone connection to attempt to fix low audio
     * @private
     */
    async _refreshMicrophone() {
        console.log('Refreshing microphone connection due to low audio levels');
        
        // Stop existing tracks
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
        }
        
        try {
            // Request microphone again with higher gain
            const constraints = {
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: true,
                    channelCount: 1,
                    sampleRate: 44100
                },
                video: false
            };
            
            this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Replace the track in the RTCPeerConnection
            if (this.peerConnection) {
                const audioTracks = this.localStream.getAudioTracks();
                if (audioTracks.length > 0) {
                    const sender = this.peerConnection.getSenders().find(s => 
                        s.track && s.track.kind === 'audio'
                    );
                    
                    if (sender) {
                        console.log('Replacing audio track in RTCPeerConnection');
                        await sender.replaceTrack(audioTracks[0]);
                    } else {
                        console.log('No sender found, adding new track');
                        this.peerConnection.addTrack(audioTracks[0], this.localStream);
                    }
                }
            }
            
            // Restart audio monitoring
            this._setupAudioMonitoring();
            
            console.log('Microphone refreshed successfully');
        } catch (error) {
            console.error('Error refreshing microphone:', error);
        }
    }
    
    /**
     * Initialize WebSocket connection
     * @private
     */
    _initWebSocket() {
        if (this.wsConnection) {
            this.wsConnection.close();
            this.wsConnection = null;
        }
        
        // Generate a client ID if we don't have one
        if (!this.clientId) {
            this.clientId = 'client_' + Date.now() + '_' + Math.floor(Math.random() * 1000000);
            console.log('Generated new client ID: ' + this.clientId);
            // Save to local storage
            localStorage.setItem('audio_bridge_client_id', this.clientId);
        }
        
        // Use a relative URL to connect through the same origin/proxy as the main app
        // This avoids direct connection to port 8081 which causes certificate errors
        const wsUrl = ((window.location.protocol === 'https:') ? 'wss://' : 'ws://') + 
                      window.location.host + 
                      '/audio-bridge/ws/' + this.clientId;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        try {
            this.wsConnection = new WebSocket(wsUrl);
            
            this.wsConnection.onopen = () => {
                console.log('WebSocket connection opened');
                this.wsConnected = true;
                
                // Execute any queued messages
                this._processQueuedMessages();
            };
            
            this.wsConnection.onclose = (event) => {
                console.log('WebSocket connection closed', event.code, event.reason);
                this.wsConnected = false;
                
                // Try to reconnect after a delay, unless this was a clean close
                if (event.code !== 1000) {
                    console.log('Attempting to reconnect WebSocket in 2 seconds...');
                    setTimeout(() => this._initWebSocket(), 2000);
                }
            };
            
            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
                // Error will be followed by onclose event, so no need to reconnect here
            };
            
            this.wsConnection.onmessage = (event) => {
                console.log('Received WebSocket message:', event.data);
                try {
                    const data = JSON.parse(event.data);
                    this._handleSignalingMessage(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            // Try to reconnect after a longer delay
            setTimeout(() => this._initWebSocket(), 5000);
        }
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
                        console.log('Received ICE candidate from server:', message.candidate);
                        
                        const formattedCandidate = {
                            // Standard RTCIceCandidate format for browsers
                            candidate: message.candidate.candidate,
                            sdpMid: message.candidate.sdpMid,
                            sdpMLineIndex: message.candidate.sdpMLineIndex
                        };
                        
                        await this.peerConnection.addIceCandidate(new RTCIceCandidate(formattedCandidate));
                        console.log('Added ICE candidate from server');
                    }
                } catch (error) {
                    console.error('Error adding ICE candidate from server:', error);
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
     * Record audio from the microphone and send it to the server using WebRTC
     */
    startRecording() {
        if (!this.isConnected || !this.peerConnection) {
            console.error('Cannot start recording - not connected to audio bridge');
            return false;
        }
        
        console.log('Starting to record audio via WebRTC');
        
        // Check if we have a local stream
        if (!this.localStream) {
            console.warn('No microphone stream available - requesting again');
            this.requestMicrophonePermission().then(granted => {
                if (granted) {
                    // Try to start recording again with the new stream
                    this._continueStartRecording();
                } else {
                    console.error('Could not access microphone for recording');
                    this._simulateAudioInput(); // Simulate audio if we can't get microphone
                }
            });
        } else {
            this._continueStartRecording();
        }
        
        return true;
    }
    
    /**
     * Continue the recording process after microphone access is confirmed
     * @private
     */
    _continueStartRecording() {
        if (!this.localStream) {
            console.error('No local stream available for recording');
            this._simulateAudioInput(); // Simulate audio if no stream is available
            return;
        }
        
        // Add the local audio track to the peer connection
        try {
            const audioTracks = this.localStream.getAudioTracks();
            if (audioTracks.length > 0) {
                // Remove any existing senders
                const senders = this.peerConnection.getSenders();
                senders.forEach(sender => {
                    if (sender.track && sender.track.kind === 'audio') {
                        this.peerConnection.removeTrack(sender);
                    }
                });
                
                // Add the audio track to the peer connection
                this.peerConnection.addTrack(audioTracks[0], this.localStream);
                console.log('Added audio track to peer connection');
                
                // Start monitoring for no-audio situation
                this._startNoAudioDetection();
            } else {
                console.error('No audio tracks available in stream');
                this._simulateAudioInput(); // Simulate audio if no tracks found
            }
        } catch (error) {
            console.error('Error adding audio track to peer connection:', error);
            this._simulateAudioInput(); // Simulate audio on error
        }
    }
    
    /**
     * Start detection of no-audio situations and simulate input if needed
     * @private
     */
    _startNoAudioDetection() {
        // Clear any existing timer
        if (this.noAudioTimer) {
            clearTimeout(this.noAudioTimer);
        }
        
        // Detect if audio is not flowing after 10 seconds
        this.noAudioTimer = setTimeout(() => {
            // Check if audio levels have been very low
            if (this.lastAudioLevel < 5) {
                console.warn('No significant audio detected for 10 seconds - simulating audio input');
                this._simulateAudioInput();
            }
        }, 10000); // 10 second timeout
    }
    
    /**
     * Simulate audio input when microphone doesn't appear to be working
     * This sends a special signal to the server to initiate processing even without real audio
     * @private
     */
    _simulateAudioInput() {
        console.log('Simulating audio input to keep the system responsive');
        
        if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
            console.error('Cannot simulate audio - data channel not open');
            return;
        }
        
        try {
            // Send a special command to the server to indicate we're simulating audio
            const message = JSON.stringify({
                type: 'simulate_audio',
                client_id: this.clientId,
                timestamp: Date.now()
            });
            
            this.dataChannel.send(message);
            console.log('Sent simulate_audio command to server');
            
            // Also try to create and send a small audio sample directly
            this._sendSyntheticAudioSample();
            
        } catch (error) {
            console.error('Error simulating audio input:', error);
        }
    }
    
    /**
     * Create and send a synthetic audio sample
     * @private
     */
    _sendSyntheticAudioSample() {
        try {
            // Create an audio context if we don't have one
            if (!this.audioContext) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (!AudioContext) {
                    console.error('AudioContext not supported');
                    return;
                }
                this.audioContext = new AudioContext();
            }
            
            // Create a 1-second buffer of "silence with some noise"
            const sampleRate = 16000;
            const duration = 1;
            const buffer = this.audioContext.createBuffer(1, sampleRate * duration, sampleRate);
            const channelData = buffer.getChannelData(0);
            
            // Fill with very low random noise
            for (let i = 0; i < channelData.length; i++) {
                // Very small random values (-0.01 to 0.01)
                channelData[i] = (Math.random() * 0.02) - 0.01;
            }
            
            // Convert to 16-bit PCM
            const pcmData = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                // Scale to int16 range and convert
                pcmData[i] = Math.floor(channelData[i] * 32767);
            }
            
            // Convert to a blob and send as an "uploadAudio" command
            const blob = new Blob([pcmData], { type: 'audio/raw' });
            
            // Send as fallback upload
            const reader = new FileReader();
            reader.onload = (event) => {
                if (this.dataChannel && this.dataChannel.readyState === 'open') {
                    // Send in chunks to avoid size limitations
                    const data = event.target.result;
                    const chunkSize = 16000;  // Send in ~1 second chunks
                    for (let i = 0; i < data.byteLength; i += chunkSize) {
                        const chunk = data.slice(i, i + chunkSize);
                        this.dataChannel.send(chunk);
                    }
                    console.log(`Sent synthetic audio sample (${data.byteLength} bytes)`);
                }
            };
            reader.readAsArrayBuffer(blob);
            
        } catch (error) {
            console.error('Error creating synthetic audio sample:', error);
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
            
            // Set up MediaRecorder with options for smaller chunks and higher frequency
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            };
            
            try {
                this.mediaRecorder = new MediaRecorder(this.localStream, options);
            } catch (e) {
                console.warn('MediaRecorder with options failed, using default settings:', e);
                // Fallback to default settings
                this.mediaRecorder = new MediaRecorder(this.localStream);
            }
            
            // Handle dataavailable event
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    
                    // Send audio chunks immediately when recording (don't wait for stop)
                    if (this.mediaRecorder.state === 'recording' && event.data.size > 0) {
                        this.sendAudioToServer(event.data);
                    }
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
     * Request microphone permissions and set up audio stream
     */
    async requestMicrophonePermission() {
        try {
            // Check if we're in a secure context (required for getUserMedia)
            if (!window.isSecureContext) {
                throw new Error('WebRTC requires a secure context (HTTPS or localhost)');
            }
            
            // Set up audio constraints with specific settings for better audio quality
            const constraints = {
                audio: {
                    echoCancellation: false,  // Turn off echo cancellation to get raw audio
                    noiseSuppression: false,  // Turn off noise suppression
                    autoGainControl: true,    // Enable auto gain to boost quiet audio
                    channelCount: 1,          // Mono audio
                    sampleRate: 44100,        // Higher sample rate for better quality
                    sampleSize: 16,           // 16-bit audio
                    volume: 1.0               // Maximum volume
                },
                video: false
            };
            
            console.log('Requesting microphone access with constraints:', constraints);
            
            // Request the user's microphone stream
            this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Check if we got audio tracks
            const audioTracks = this.localStream.getAudioTracks();
            if (audioTracks.length === 0) {
                throw new Error('No audio tracks available from microphone');
            }
            
            // Enable the audio track explicitly
            audioTracks[0].enabled = true;
            
            // Set audio track to high volume
            try {
                const settings = audioTracks[0].getSettings();
                console.log('Microphone settings:', settings);
                
                // Try to apply custom constraints if possible to boost audio
                await audioTracks[0].applyConstraints({
                    autoGainControl: true,
                    echoCancellation: false,
                    noiseSuppression: false
                });
            } catch (settingsError) {
                console.warn('Could not apply custom audio constraints:', settingsError);
            }
            
            console.log('Got microphone stream with tracks:', audioTracks.length);
            
            // Set up audio monitoring for diagnostics
            this._setupAudioMonitoring();
            
            return true;
        } catch (error) {
            console.error('Error accessing microphone:', error);
            
            if (this.onError) {
                this.onError(`Could not access microphone: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Set up audio level monitoring
     * @private
     */
    _setupAudioMonitoring() {
        if (!this.localStream) return;
        
        try {
            // Create AudioContext for monitoring
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContext();
            
            // Create analyzer node
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 1024;
            this.analyser.smoothingTimeConstant = 0.8;
            
            // Connect the stream to the analyzer
            this.source = this.audioContext.createMediaStreamSource(this.localStream);
            this.source.connect(this.analyser);
            
            // Create a gain node to boost the signal for monitoring
            this.gainNode = this.audioContext.createGain();
            this.gainNode.gain.value = 5.0; // Boost the signal
            this.analyser.connect(this.gainNode);
            
            // Start monitoring
            this.monitorInterval = setInterval(() => {
                this._checkAudioLevels();
            }, 500);
            
            console.log('Audio monitoring started');
        } catch (error) {
            console.error('Error setting up audio monitoring:', error);
        }
    }
    
    /**
     * Check audio levels from the microphone
     * @private
     */
    _checkAudioLevels() {
        if (!this.analyser) return;
        
        try {
            const bufferLength = this.analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            this.analyser.getByteFrequencyData(dataArray);
            
            // Calculate average level
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            const average = sum / bufferLength;
            this.lastAudioLevel = average;
            
            // Show a warning if audio level is too low
            if (average < 5) {
                if (!this.lowAudioWarningShown && this.isConnected) {
                    console.warn('Very low audio levels detected from microphone:', average);
                    this.lowAudioWarningShown = true;
                    
                    if (this.onStatusChange) {
                        this.onStatusChange({
                            enabled: this.isEnabled,
                            connected: this.isConnected,
                            status: 'low-audio',
                            level: average
                        });
                    }
                    
                    // Start a timer to refresh the microphone after a delay
                    setTimeout(() => this._refreshMicrophone(), 3000);
                }
            } else {
                this.lowAudioWarningShown = false;
            }
            
            // Log audio level every few seconds
            if (this.isConnected && Date.now() - (this.lastAudioLogTime || 0) > 5000) {
                console.log(`Current microphone audio level: ${average.toFixed(2)}`);
                this.lastAudioLogTime = Date.now();
            }
        } catch (error) {
            console.error('Error monitoring audio levels:', error);
        }
    }
    
    /**
     * Create WebRTC peer connection
     */
    async createPeerConnection() {
        if (!this.clientId) {
            console.error('Cannot create peer connection without client ID');
            return false;
        }
        
        try {
            // Define ICE servers for WebRTC
            const iceServers = [
                { 
                    urls: [
                        'stun:stun.l.google.com:19302',
                        'stun:stun1.l.google.com:19302',
                        'stun:stun2.l.google.com:19302'
                    ] 
                }
            ];
            
            // Create peer connection
            this.peerConnection = new RTCPeerConnection({
                iceServers: iceServers
            });
            
            // Log connection state changes for debugging
            this.peerConnection.onconnectionstatechange = () => {
                console.log(`Connection state: ${this.peerConnection.connectionState}`);
                
                if (this.peerConnection.connectionState === 'connected') {
                    console.log('WebRTC peer connection established');
                    this.isConnected = true;
                    
                    if (this.onStatusChange) {
                        this.onStatusChange({
                            enabled: this.isEnabled,
                            connected: this.isConnected,
                            status: 'connected'
                        });
                    }
                } else if (this.peerConnection.connectionState === 'failed' || 
                          this.peerConnection.connectionState === 'disconnected' ||
                          this.peerConnection.connectionState === 'closed') {
                    console.log('WebRTC peer connection closed');
                    this.isConnected = false;
                    
                    if (this.onStatusChange) {
                        this.onStatusChange({
                            enabled: this.isEnabled,
                            connected: this.isConnected,
                            status: 'disconnected'
                        });
                    }
                }
            };
            
            // Handle ICE candidates
            this.peerConnection.onicecandidate = (event) => {
                if (event.candidate) {
                    console.log('Generated ICE candidate:', event.candidate);
                    
                    // Send the complete candidate string which includes all the required parameters
                    // for aiortc's RTCIceCandidate constructor
                    this.sendSignalingMessage({
                        type: 'ice-candidate',
                        candidate: {
                            candidate: event.candidate.candidate, // Contains all required params in the string
                            sdpMid: event.candidate.sdpMid,
                            sdpMLineIndex: event.candidate.sdpMLineIndex
                        }
                    });
                }
            };
            
            // Create data channel for control messages
            this.dataChannel = this.peerConnection.createDataChannel('audio', {
                ordered: true
            });
            
            this.dataChannel.onopen = () => {
                console.log('Data channel opened');
            };
            
            this.dataChannel.onclose = () => {
                console.log('Data channel closed');
            };
            
            this.dataChannel.onmessage = (event) => {
                console.log('Received data channel message:', event.data);
            };
            
            // Create offer and set local description
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: false,  // We're sending audio only
                offerToReceiveVideo: false
            });
            
            await this.peerConnection.setLocalDescription(offer);
            
            // Send offer to server
            // Use relative URL with the proxy endpoint instead of direct connection to port 8081
            const offerUrl = `/audio-bridge/offer`;
            
            const response = await fetch(offerUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'offer',
                    sdp: offer.sdp,
                    client_id: this.clientId
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to send offer: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.type === 'answer') {
                const remoteDesc = new RTCSessionDescription({
                    type: 'answer',
                    sdp: data.sdp
                });
                
                await this.peerConnection.setRemoteDescription(remoteDesc);
                console.log('Set remote description from answer');
            } else if (data.type === 'error') {
                throw new Error(`Signaling error: ${data.message}`);
            }
            
            console.log('WebRTC peer connection created');
            return true;
        } catch (error) {
            console.error('Error creating peer connection:', error);
            
            if (this.onError) {
                this.onError(`Error creating WebRTC connection: ${error.message}`);
            }
            
            return false;
        }
    }
    
    /**
     * Send signaling message to the server via WebSocket
     */
    sendSignalingMessage(message) {
        // Add client_id to the message if not already present
        if (!message.client_id && this.clientId) {
            message.client_id = this.clientId;
        }
        
        if (!this.wsConnection || this.wsConnection.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not open, queuing signaling message');
            // Queue message for later delivery
            this.messageQueue.push(message);
            
            // Initialize WebSocket if it doesn't exist or is closed
            if (!this.wsConnection || this.wsConnection.readyState === WebSocket.CLOSED) {
                console.log('Initializing WebSocket connection');
                this._initWebSocket();
            }
            
            return false;
        }
        
        try {
            const messageJson = JSON.stringify(message);
            this.wsConnection.send(messageJson);
            console.log('Sent signaling message:', message.type);
            return true;
        } catch (error) {
            console.error('Error sending signaling message:', error);
            // Queue the message for retry
            this.messageQueue.push(message);
            return false;
        }
    }
    
    /**
     * Create an emergency audio track if all else fails
     * @private
     */
    _createEmergencyAudioTrack() {
        try {
            console.log('Creating emergency audio track');
            
            // Create an audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create an oscillator
            const oscillator = audioContext.createOscillator();
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // 440 Hz tone
            
            // Create a gain node to control volume
            const gainNode = audioContext.createGain();
            gainNode.gain.setValueAtTime(0.01, audioContext.currentTime); // Very low volume
            
            // Connect the nodes
            oscillator.connect(gainNode);
            
            // Get the stream from the gain node
            const destination = audioContext.createMediaStreamDestination();
            gainNode.connect(destination);
            
            // Start the oscillator
            oscillator.start();
            
            // Save the stream and add the track to the peer connection
            this.emergencyStream = destination.stream;
            const audioTrack = this.emergencyStream.getAudioTracks()[0];
            
            if (audioTrack) {
                console.log('Adding emergency audio track to peer connection');
                this.peerConnection.addTrack(audioTrack, this.emergencyStream);
            }
        } catch (e) {
            console.error('Failed to create emergency audio track:', e);
        }
    }
}

// Export for ES modules
if (typeof exports !== 'undefined') {
    exports.AudioBridgeClient = AudioBridgeClient;
}

// Create global instance
window.audioBridge = new AudioBridgeClient(); 