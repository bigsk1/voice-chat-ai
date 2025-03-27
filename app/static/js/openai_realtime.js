document.addEventListener("DOMContentLoaded", function() {
    // DOM elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const micBtn = document.getElementById('micBtn');
    const testMicBtn = document.getElementById('testMicBtn');
    const micTestResult = document.getElementById('micTestResult');
    const micStatus = document.getElementById('micStatus');
    const characterSelect = document.getElementById('characterSelect');
    const voiceSelect = document.getElementById('voiceSelect');
    const transcript = document.getElementById('transcript');
    const audioPlayer = document.getElementById('audioPlayer');
    const sessionStatus = document.getElementById('session-status');
    const userVoiceVisualization = document.getElementById('userVoiceVisualization');
    const aiVoiceVisualization = document.getElementById('aiVoiceVisualization');
    const themeToggle = document.getElementById('theme-toggle');
    const testTextInput = document.getElementById('testTextInput');
    const sendTextBtn = document.getElementById('sendTextBtn');
    
    // Load characters from API
    loadCharacters();
    
    // WebSocket and audio context
    let socket = null;
    let mediaRecorder = null;
    let audioContext = null; // Initialize later
    let mediaStream = null;
    let isRecording = false;
    let isSessionActive = false;
    let sessionId = null;
    let userIsSpeaking = false;
    let aiIsSpeaking = false;
    let audioQueue = [];
    let audioProcessor = null;
    let isListening = false;
    let currentSessionId = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 1000; // 1 second
    let isPlayingAudio = false;
    
    // Constants
    const WEBSOCKET_URL = `ws://${window.location.host}/ws_openai_realtime`;
    
    // Set dark mode as default
    setDarkModeDefault();
    
    // Hide visualizations initially
    updateUserVoiceActivity(false);
    updateAIVoiceActivity(false);
    
    // Connect to WebSocket
    connectWebSocket();
    
    // Initialize voice visualizations early
    initializeVoiceVisualizations();
    
    // Add a debug panel to the UI
    addDebugPanel();
    
    // Initialize audio on first user interaction
    document.body.addEventListener('click', function() {
        initAudio();
    }, { once: true });
    
    // Test Microphone button
    if (testMicBtn) {
        testMicBtn.addEventListener('click', function() {
            testMicrophone();
        });
    }
    
    // Send Text button
    if (sendTextBtn && testTextInput) {
        sendTextBtn.addEventListener('click', function() {
            sendTextMessage();
        });
        
        // Also allow pressing Enter to send
        testTextInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendTextMessage();
            }
        });
    }
    
    function connectWebSocket() {
        // Close existing connection if any
        if (socket && socket.readyState !== WebSocket.CLOSED) {
            socket.close();
        }
        
        socket = new WebSocket(WEBSOCKET_URL);
        
        socket.onopen = function(event) {
            console.log("WebSocket connection established");
            // Don't disable the start button until a session is actually created
            startBtn.disabled = false;
            stopBtn.disabled = true;
            micBtn.disabled = true; // Keep mic disabled until session is created
            sessionStatus.textContent = 'Connected';
            sessionStatus.className = 'session-status inactive';
            addTranscriptMessage("Connected to server", "system");
            addTranscriptMessage("Click 'Start Session' to begin", "system");
        };
        
        socket.onmessage = function(event) {
            try {
                let data;
                try {
                    data = JSON.parse(event.data);
                } catch (e) {
                    console.error("Invalid JSON:", event.data);
                    data = { message: event.data, type: "error" };
                }
                
                console.log("WebSocket message received:", data);
                
                // If it's a debug message, handle it separately
                if (data.type === 'debug_info') {
                    handleDebugMessage(data);
                    return;
                }
                
                // Process message based on type
                if (data.type === "session_created") {
                    console.log("Session created:", data.session_id);
                    sessionId = data.session_id; // Make sure to store session ID
                    sessionStatus.textContent = "Active";
                    sessionStatus.classList.remove("badge-warning");
                    sessionStatus.classList.add("badge-success");
                    
                    // Set session as active and enable microphone
                    isSessionActive = true;
                    
                    // Update UI
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    micBtn.disabled = false;
                    
                    // Initialize audio contexts
                    initAudio();
                    
                    // Start voice capture automatically
                    startVoiceCapture();
                    
                    // Add session started message to transcript
                    addTranscriptMessage(`Session started with ${characterSelect.options[characterSelect.selectedIndex].text}`, "system");
                } else if (data.type === "session_closed") {
                    console.log("Session closed");
                    sessionStatus.textContent = "Inactive";
                    sessionStatus.classList.remove("badge-success");
                    sessionStatus.classList.add("badge-secondary");
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    micBtn.disabled = true;
                    micBtn.classList.remove('listening');
                    micStatus.textContent = 'Click to speak';
                    micStatus.style.color = ''; // Reset color
                    userVoiceVisualization.classList.add('hidden');
                    
                    // Add session closed message to transcript
                    addTranscriptMessage("Session ended", "system");
                    
                    // Stop all audio processing
                    stopVoiceCapture();
                    updateUserVoiceActivity(false);
                    updateAIVoiceActivity(false);
                } else if (data.type === "connected") {
                    console.log("WebSocket connection confirmed by server");
                } else if (data.type === "transcript" && data.text) {
                    // Transcription from AI's speech
                    addTranscriptMessage(data.text, data.role || "ai");
                    
                    if (data.is_final) {
                        // Suggest the AI has finished speaking
                        updateAIVoiceActivity(false);
                    }
                } else if (data.type === "audio_data") {
                    // Play the audio data
                    playAudioData(data.audio_data);
                } else if (data.type === "voice_activity") {
                    // Update voice activity visualizations
                    updateVoiceActivity(data.speaker, data.active);
                } else if (data.type === "error") {
                    console.error("Error:", data.message);
                    addTranscriptMessage(data.message, "error");
                }
            } catch (e) {
                console.error("Error handling WebSocket message:", e);
            }
        };
        
        socket.onclose = function(event) {
            console.log("WebSocket connection closed", event);
            
            // Update UI
            startBtn.disabled = true;
            stopBtn.disabled = true;
            micBtn.disabled = true;
            isSessionActive = false;
            
            // Stop all audio processing
            stopVoiceCapture();
            updateUserVoiceActivity(false);
            updateAIVoiceActivity(false);
            
            // Reset session status
            sessionId = null;
            sessionStatus.textContent = "Disconnected";
            sessionStatus.classList.remove("active");
            sessionStatus.classList.add("inactive");
            
            // Try to reconnect if not closed cleanly and not exceeding max attempts
            if (!event.wasClean && reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                const delay = reconnectDelay * reconnectAttempts;
                console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts}) in ${delay}ms...`);
                addTranscriptMessage(`Connection lost. Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`, "system");
                setTimeout(connectWebSocket, delay);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                addTranscriptMessage("Failed to connect to server after multiple attempts. Please refresh the page.", "error");
            }
        };
        
        socket.onerror = function(event) {
            console.error("WebSocket error:", event);
            addTranscriptMessage("Connection error. Please try again later.", "error");
            
            // Stop all audio processing
            stopVoiceCapture();
            updateUserVoiceActivity(false);
            updateAIVoiceActivity(false);
        };
    }
    
    // Start session button
    startBtn.addEventListener('click', function() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            const character = characterSelect.value;
            const voice = voiceSelect.value;
            
            // Send message to start a new session
            socket.send(JSON.stringify({
                type: 'start_session',
                character: character,
                voice: voice
            }));
            
            addTranscriptMessage(`Starting session with ${character.replace('_', ' ')}...`, "system");
            
            // Initialize audio context now to ensure permissions
            initAudio();
            
            // Set session as active immediately for better UX
            sessionStatus.textContent = "Starting...";
            sessionStatus.classList.remove("badge-secondary");
            sessionStatus.classList.add("badge-warning");
            
            // Disable start button during initialization
            startBtn.disabled = true;
        } else {
            addTranscriptMessage("WebSocket connection not available. Trying to reconnect...", "error");
            connectWebSocket();
        }
    });
    
    // Stop session button
    stopBtn.addEventListener('click', function() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            // Stop any ongoing recording
            stopVoiceCapture();
            
            // Send message to stop session
            socket.send(JSON.stringify({
                type: 'stop_session',
                session_id: sessionId
            }));
            
            addTranscriptMessage("Stopping session...", "system");
        }
    });
    
    // Clear transcript button
    clearBtn.addEventListener('click', function() {
        // Clear transcript display
        transcript.innerHTML = '';
        
        // Send clear history message
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'clear_history',
                session_id: sessionId
            }));
        }
        
        addTranscriptMessage("Transcript cleared", "system");
    });
    
    // Microphone button
    micBtn.addEventListener('mousedown', function() {
        // Start audio context if not initialized
        initAudio();
        
        if (!isSessionActive) {
            addTranscriptMessage("Please start a session first by clicking the 'Start Session' button", "error");
            return;
        }
        
        if (!isRecording) {
            startVoiceCapture();
        } else {
            stopVoiceCapture();
        }
    });
    
    // Start voice capture and streaming
    function startVoiceCapture() {
        if (!isSessionActive || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addTranscriptMessage("Media devices not supported or session not active", "error");
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        })
            .then(function(stream) {
                mediaStream = stream;
                
                // Create AudioContext after we have the stream to match the sample rate
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    console.log(`Using sample rate: ${audioContext.sampleRate} Hz`);
                }
                
                mediaRecorder = new MediaRecorder(stream);
                isRecording = true;
                isSessionActive = true; // Make sure session is marked as active
                
                // We need these UI updates to happen immediately
                micBtn.classList.add('listening');
                micStatus.textContent = "Listening... (speak now)";
                micStatus.style.color = "#ff3300";
                userVoiceVisualization.classList.remove('hidden');
                
                // Set up audio processing
                const audioInput = audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                audioInput.connect(analyser);
                
                // Create a script processor node
                let audioProcessor;
                if (audioContext.createScriptProcessor) {
                    // Old method
                    audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                } else if (audioContext.createJavaScriptNode) {
                    // Even older method
                    audioProcessor = audioContext.createJavaScriptNode(4096, 1, 1);
                } else {
                    // Use AudioWorklet if available (modern browsers)
                    console.log("Using audio worklet processing");
                    window.debugLog && window.debugLog("Using AudioWorklet for processing");
                    
                    // Create a worklet or fallback
                    const bufferSize = 4096;
                    audioProcessor = new AudioWorkletNode(audioContext, 'audio-processor');
                }
                
                // Connect the audio processor
                audioInput.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination);
                
                // Check that visualizations are working
                console.log("Audio visualization setup:", {
                    visualizationElement: userVoiceVisualization,
                    bars: userVoiceVisualization.querySelectorAll('.voice-bar'),
                    hidden: userVoiceVisualization.classList.contains('hidden')
                });
                
                // Animation frame for visualization - make sure this runs continuously
                function updateVisualization() {
                    if (!isRecording) return;
                    
                    // Get audio data for visualization
                    const dataArray = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(dataArray);
                    
                    // Calculate average volume
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) {
                        sum += dataArray[i];
                    }
                    const average = sum / dataArray.length;
                    
                    // Add some logging to check values
                    if (Math.random() < 0.05) { // Log occasionally
                        console.log(`Audio level: ${average.toFixed(2)}/255`);
                        window.debugLog && window.debugLog(`Audio level: ${average.toFixed(2)}/255`);
                    }
                    
                    // Make sure visualization updates even with low signal
                    const visualValue = Math.max(0.05, average / 255); // Ensure minimum visibility
                    updateUserVoiceVisualization(visualValue);
                    
                    // Lower the threshold to detect speaking
                    const speakingThreshold = 0.03; // Lower threshold to be more sensitive
                    const isSpeakingNow = average / 255 > speakingThreshold;
                    
                    // Only update if state changed to reduce messages
                    if (isSpeakingNow !== userIsSpeaking) {
                        userIsSpeaking = isSpeakingNow;
                        if (userIsSpeaking) {
                            micStatus.textContent = "Speaking...";
                            micStatus.style.color = "#ff3300";
                        } else {
                            micStatus.textContent = "Listening... (speak now)";
                            micStatus.style.color = "#0066ff";
                        }
                        
                        // Send voice activity update if session is active
                        if (socket && socket.readyState === WebSocket.OPEN && isSessionActive) {
                            socket.send(JSON.stringify({
                                type: 'voice_activity',
                                active: userIsSpeaking,
                                speaker: 'user',
                                session_id: sessionId
                            }));
                        }
                    }
                    
                    // Continue animation
                    requestAnimationFrame(updateVisualization);
                }
                
                // Start visualization
                updateVisualization();
                
                // Make sure the visualization bars exist
                const barsCount = userVoiceVisualization.querySelectorAll('.voice-bar').length;
                if (barsCount === 0) {
                    for (let i = 0; i < 8; i++) {
                        const bar = document.createElement('div');
                        bar.className = 'voice-bar';
                        userVoiceVisualization.appendChild(bar);
                    }
                    console.log("Created visualization bars");
                }
                
                // Process audio - make sure this creates valid audio chunks
                audioProcessor.onaudioprocess = function(e) {
                    if (isRecording && socket && socket.readyState === WebSocket.OPEN && isSessionActive) {
                        try {
                            // Get audio data
                            const inputData = e.inputBuffer.getChannelData(0);
                            
                            // Check if audio data is valid
                            let hasSound = false;
                            let maxAmp = 0;
                            for (let i = 0; i < inputData.length; i++) {
                                const abs = Math.abs(inputData[i]);
                                if (abs > 0.01) hasSound = true;
                                maxAmp = Math.max(maxAmp, abs);
                            }
                            
                            // Log audio levels occasionally
                            if (Math.random() < 0.01) {
                                console.log(`Max amplitude: ${maxAmp.toFixed(4)}, Has sound: ${hasSound}`);
                                window.debugLog && window.debugLog(`Max amplitude: ${maxAmp.toFixed(4)}`);
                            }
                            
                            // Create audio data in the format expected by the server
                            const pcmData = convertFloat32ToInt16(inputData);
                            
                            // Send audio data as JSON message with base64 encoding
                            const message = {
                                type: 'audio',
                                audio_data: arrayBufferToBase64(pcmData.buffer),
                                session_id: sessionId
                            };
                            
                            socket.send(JSON.stringify(message));
                        } catch (e) {
                            console.error("Error processing audio:", e);
                            window.debugLog && window.debugLog("Error processing audio: " + e.message, true);
                        }
                    }
                };
                
                addTranscriptMessage('Voice capture started', 'system');
                
                // Add real-time conversation instructions to the transcript
                addTranscriptMessage("Real-time conversation started. You can speak at the same time as the AI - it's fully interactive!", "system");
            })
            .catch(function(err) {
                console.error("Error accessing microphone:", err);
                addTranscriptMessage(`Microphone error: ${err.message}`, "error");
                window.debugLog && window.debugLog("Microphone error: " + err.message, true);
            });
    }
    
    // Stop voice capture
    function stopVoiceCapture() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        
        if (audioProcessor) {
            audioProcessor.disconnect();
            audioProcessor = null;
        }
        
        if (mediaRecorder) {
            mediaRecorder = null;
        }
        
        isRecording = false;
        micBtn.classList.remove('listening');
        micStatus.textContent = 'Click to speak';
        micStatus.style.color = ''; // Reset color
        userVoiceVisualization.classList.add('hidden');
        addTranscriptMessage('Voice capture stopped', 'system');
    }
    
    // Update user voice activity visualization
    function updateUserVoiceActivity(active) {
        const bars = userVoiceVisualization.querySelectorAll('.voice-bar');
        bars.forEach(bar => {
            if (active) {
                bar.classList.add('active-bar');
            } else {
                bar.classList.remove('active-bar');
                bar.style.height = '5px';
            }
        });
    }
    
    // Update AI voice activity visualization
    function updateAIVoiceActivity(active) {
        const bars = aiVoiceVisualization.querySelectorAll('.voice-bar');
        bars.forEach(bar => {
            if (active) {
                bar.classList.add('active-bar');
            } else {
                bar.classList.remove('active-bar');
                bar.style.height = '5px';
            }
        });
    }
    
    // Update UI based on voice activity
    function updateVoiceActivity(speaker, active) {
        if (speaker === 'user') {
            userIsSpeaking = active;
            if (active) {
                userVoiceVisualization.classList.remove('hidden');
            } else {
                userVoiceVisualization.classList.add('hidden');
            }
        } else if (speaker === 'assistant') {
            aiIsSpeaking = active;
            if (active) {
                aiVoiceVisualization.classList.remove('hidden');
            } else {
                aiVoiceVisualization.classList.add('hidden');
            }
        }
    }
    
    // Animate the user voice visualization with minimum heights
    function updateUserVoiceVisualization(energy) {
        const bars = userVoiceVisualization.querySelectorAll('.voice-bar');
        
        // Create bars if they don't exist
        if (bars.length === 0) {
            for (let i = 0; i < 8; i++) {
                const bar = document.createElement('div');
                bar.className = 'voice-bar';
                userVoiceVisualization.appendChild(bar);
            }
        }
        
        // Update each bar with some randomness for natural look
        const maxHeight = 50; // Maximum height in pixels
        const minHeight = 3;  // Minimum height so bars are always visible
        
        bars.forEach(bar => {
            // Ensure minimum height and add randomness
            const randomFactor = 0.5 + Math.random();
            const height = Math.max(minHeight, Math.min(maxHeight, energy * maxHeight * randomFactor));
            bar.style.height = `${height}px`;
            
            // Make sure bars are visible
            bar.classList.add('active-bar');
        });
    }
    
    // Animate the AI voice visualization
    function animateAIVoiceVisualization() {
        if (!aiIsSpeaking) return;
        
        const bars = aiVoiceVisualization.querySelectorAll('.voice-bar');
        const maxHeight = 50; // Maximum height in pixels
        
        bars.forEach(bar => {
            const height = Math.min(maxHeight, maxHeight * Math.random());
            bar.style.height = `${height}px`;
        });
        
        if (aiIsSpeaking) {
            requestAnimationFrame(animateAIVoiceVisualization);
        }
    }
    
    // Calculate audio energy (volume level)
    function calculateAudioEnergy(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += Math.abs(audioData[i]);
        }
        return sum / audioData.length;
    }
    
    // Convert Float32Array to Int16Array for audio processing
    function convertFloat32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }
    
    // Convert ArrayBuffer to Base64
    function arrayBufferToBase64(buffer) {
        const binary = [];
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary.push(String.fromCharCode(bytes[i]));
        }
        return btoa(binary.join(''));
    }
    
    // Convert Base64 to ArrayBuffer
    function base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    // Add message to transcript
    function addTranscriptMessage(message, type) {
        const messageElement = document.createElement('div');
        
        if (type === "system") {
            messageElement.className = "system-message";
            messageElement.textContent = message;
        } else if (type === "error") {
            messageElement.className = "error-message";
            messageElement.textContent = message;
        } else if (type === "user") {
            messageElement.className = "user-speech";
            messageElement.textContent = `You: ${message}`;
        } else if (type === "ai") {
            messageElement.className = "ai-speech";
            const characterName = characterSelect.options[characterSelect.selectedIndex].text;
            messageElement.textContent = `${characterName}: ${message}`;
        }
        
        transcript.appendChild(messageElement);
        messageElement.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Theme management functions
    
    // Update theme toggle icon based on current mode
    function updateThemeToggleIcon() {
        if (!themeToggle) return; // Skip if theme toggle doesn't exist
        
        const isDarkMode = document.body.classList.contains('dark-mode');
        themeToggle.innerHTML = isDarkMode 
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    }
    
    // Set dark mode as default or load from localStorage
    function setDarkModeDefault() {
        const isDarkMode = localStorage.getItem('darkMode');
        if (isDarkMode === null) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'true');
        } else {
            document.body.classList.toggle('dark-mode', isDarkMode === 'true');
        }
        updateThemeToggleIcon();
    }
    
    // Toggle theme when clicking theme button
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            updateThemeToggleIcon();
            localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
        });
    }
    
    // Send a text message using the proper OpenAI Realtime API format
    function sendTextMessage() {
        const text = testTextInput.value.trim();
        if (!text) return;
        
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            addTranscriptMessage("WebSocket not connected", "error");
            window.debugLog && window.debugLog("Cannot send text: WebSocket not connected", true);
            return;
        }
        
        if (!isSessionActive) {
            addTranscriptMessage("Please start a session first", "error");
            window.debugLog && window.debugLog("Cannot send text: No active session", true);
            return;
        }
        
        // Add to transcript first (for immediate feedback)
        addTranscriptMessage(text, "user");
        
        // Send message with the appropriate format for the server
        socket.send(JSON.stringify({
            type: 'text',
            text: text,
            session_id: sessionId
        }));
        
        // Log in debug
        window.debugLog && window.debugLog(`Sent text message: ${text}`);
        
        // Clear input
        testTextInput.value = '';
    }
    
    // Load characters from API
    function loadCharacters() {
        fetch('/characters')
            .then(response => response.json())
            .then(data => {
                if (data.characters && data.characters.length > 0) {
                    // Clear existing options
                    characterSelect.innerHTML = '';
                    
                    // Add each character
                    data.characters.forEach(character => {
                        const option = document.createElement('option');
                        option.value = character;
                        option.textContent = character.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                        characterSelect.appendChild(option);
                    });
                    
                    console.log(`Loaded ${data.characters.length} characters`);
                } else {
                    console.warn("No characters found or invalid response");
                    
                    // Add fallback character
                    const option = document.createElement('option');
                    option.value = 'assistant';
                    option.textContent = 'Assistant';
                    characterSelect.appendChild(option);
                }
            })
            .catch(error => {
                console.error("Error loading characters:", error);
                
                // Add fallback character on error
                characterSelect.innerHTML = '';
                const option = document.createElement('option');
                option.value = 'assistant';
                option.textContent = 'Assistant';
                characterSelect.appendChild(option);
            });
    }
    
    // Add a debug panel to the UI
    function addDebugPanel() {
        // Create debug panel container
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debugPanel';
        debugPanel.className = 'debug-panel';
        debugPanel.style.display = 'none'; // Hidden by default
        debugPanel.innerHTML = `
            <div class="debug-header">
                <h3>Debug Panel</h3>
                <button id="closeDebugBtn" class="btn btn-sm btn-danger">Close</button>
            </div>
            <div class="debug-controls">
                <button id="statusBtn" class="btn btn-sm btn-primary">Session Status</button>
                <button id="toggleAudioDebugBtn" class="btn btn-sm btn-info">Toggle Audio Debug</button>
                <button id="toggleWsDebugBtn" class="btn btn-sm btn-info">Toggle WS Debug</button>
            </div>
            <div class="debug-log" id="debugLog">
                <p>Debug information will appear here.</p>
            </div>
        `;
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .debug-panel {
                position: fixed;
                bottom: 0;
                right: 0;
                width: 400px;
                max-height: 300px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                z-index: 1000;
                overflow: auto;
                font-family: monospace;
                font-size: 12px;
            }
            .debug-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .debug-controls {
                display: flex;
                gap: 5px;
                margin-bottom: 10px;
            }
            .debug-log {
                background-color: #000;
                color: #0f0;
                padding: 5px;
                height: 200px;
                overflow-y: auto;
                border-radius: 3px;
            }
            .debug-log p {
                margin: 0;
                padding: 2px 0;
            }
        `;
        
        document.head.appendChild(style);
        document.body.appendChild(debugPanel);
        
        // Add event listeners
        document.getElementById('closeDebugBtn').addEventListener('click', () => {
            debugPanel.style.display = 'none';
        });
        
        document.getElementById('statusBtn').addEventListener('click', () => {
            sendDebugCommand('status');
        });
        
        document.getElementById('toggleAudioDebugBtn').addEventListener('click', () => {
            sendDebugCommand('toggle_audio_debug');
        });
        
        document.getElementById('toggleWsDebugBtn').addEventListener('click', () => {
            sendDebugCommand('toggle_websocket_debug');
        });
        
        // Add a toggle shortcut (Ctrl+Shift+D)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                debugPanel.style.display = debugPanel.style.display === 'none' ? 'block' : 'none';
            }
        });
        
        // Log function
        window.debugLog = function(message, isError = false) {
            const log = document.getElementById('debugLog');
            const p = document.createElement('p');
            p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            if (isError) {
                p.style.color = '#f55';
            }
            log.appendChild(p);
            log.scrollTop = log.scrollHeight;
        };
        
        window.debugLog('Debug panel initialized. Press Ctrl+Shift+D to toggle.');
    }
    
    // Function to send debug commands
    function sendDebugCommand(command) {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            window.debugLog('WebSocket not connected', true);
            return;
        }
        
        const message = {
            type: 'debug',
            command: command
        };
        
        socket.send(JSON.stringify(message));
        window.debugLog(`Sent debug command: ${command}`);
    }
    
    // Add this to the end of the onmessage handler for the WebSocket
    function handleDebugMessage(data) {
        if (data.type === 'debug_info') {
            if (data.message) {
                window.debugLog(data.message);
            } else {
                // For status response, format it nicely
                const status = JSON.stringify(data, null, 2);
                window.debugLog(`Session status:\n${status}`);
            }
        }
    }
    
    // Initialize audio context
    function initAudio() {
        if (!audioContext) {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log("Audio context initialized");
                if (window.debugLog) {
                    window.debugLog("Audio context initialized, sample rate: " + audioContext.sampleRate);
                }
            } catch (e) {
                console.error("Error initializing audio context:", e);
                if (window.debugLog) {
                    window.debugLog("Failed to initialize audio context: " + e.message, true);
                }
            }
        }
    }
    
    // Function to handle audio playback
    async function playAudioData(base64Audio) {
        if (!audioContext) {
            console.error("Audio context not initialized");
            if (window.debugLog) {
                window.debugLog("Cannot play audio: Audio context not initialized", true);
            }
            return;
        }
        
        try {
            // Decode base64 to array buffer
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Debug audio size
            console.log(`Decoded audio size: ${bytes.length} bytes`);
            if (window.debugLog) {
                window.debugLog(`Received audio chunk: ${bytes.length} bytes`);
            }
            
            // Skip empty chunks
            if (bytes.length < 100) {
                console.warn("Audio chunk too small, skipping");
                if (window.debugLog) {
                    window.debugLog("Audio chunk too small, skipping", true);
                }
                return;
            }
            
            // Queue the audio
            audioQueue.push(bytes.buffer);
            
            // Play if not already playing
            if (!isPlayingAudio) {
                playNextInQueue();
            }
        } catch (e) {
            console.error("Error processing audio data:", e);
            if (window.debugLog) {
                window.debugLog("Error processing audio: " + e.message, true);
            }
        }
    }
    
    // Play audio from queue
    async function playNextInQueue() {
        if (audioQueue.length === 0) {
            isPlayingAudio = false;
            return;
        }
        
        isPlayingAudio = true;
        const audioBuffer = audioQueue.shift();
        
        try {
            // Decode the audio data
            const decodedData = await audioContext.decodeAudioData(audioBuffer);
            
            // Create source node
            const source = audioContext.createBufferSource();
            source.buffer = decodedData;
            source.connect(audioContext.destination);
            
            // Play and handle completion
            source.onended = playNextInQueue;
            source.start(0);
            
            if (window.debugLog) {
                window.debugLog(`Playing audio: ${decodedData.duration.toFixed(2)}s, ${decodedData.length} samples`);
            }
        } catch (e) {
            console.error("Error playing audio:", e);
            if (window.debugLog) {
                window.debugLog("Error playing audio: " + e.message, true);
            }
            
            // Continue with next in queue even if there's an error
            playNextInQueue();
        }
    }
    
    // Initialize voice visualizations early
    function initializeVoiceVisualizations() {
        // Create visualization bars if they don't exist yet
        for (const container of [userVoiceVisualization, aiVoiceVisualization]) {
            if (container && container.querySelectorAll('.voice-bar').length === 0) {
                for (let i = 0; i < 8; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'voice-bar';
                    container.appendChild(bar);
                }
            }
        }
    }
    
    // Function to test microphone
    function testMicrophone() {
        micTestResult.style.display = 'block';
        micTestResult.innerHTML = 'Testing microphone...';
        micTestResult.className = 'alert alert-info';
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            micTestResult.innerHTML = 'Error: Media devices not supported in this browser.';
            micTestResult.className = 'alert alert-danger';
            window.debugLog && window.debugLog('Media devices not supported', true);
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                // We got microphone access - create audio context and analyzer
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const analyser = audioContext.createAnalyser();
                const microphone = audioContext.createMediaStreamSource(stream);
                const javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
                
                // Connect the nodes together
                microphone.connect(analyser);
                analyser.connect(javascriptNode);
                javascriptNode.connect(audioContext.destination);
                
                // Variables for analysis
                let soundDetected = false;
                let maxVolume = 0;
                let testDuration = 0;
                const startTime = Date.now();
                
                // Create visual feedback element
                const volumeBar = document.createElement('div');
                volumeBar.style.height = '20px';
                volumeBar.style.width = '0%';
                volumeBar.style.backgroundColor = '#4CAF50';
                volumeBar.style.marginTop = '10px';
                volumeBar.style.transition = 'width 0.1s';
                micTestResult.appendChild(volumeBar);
                
                // Process audio
                javascriptNode.onaudioprocess = function(e) {
                    const array = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(array);
                    
                    // Calculate volume
                    let values = 0;
                    for (let i = 0; i < array.length; i++) {
                        values += array[i];
                    }
                    
                    const average = values / array.length;
                    maxVolume = Math.max(maxVolume, average);
                    
                    // Update UI
                    volumeBar.style.width = average + '%';
                    
                    // Detect sounds above threshold
                    if (average > 5) {
                        soundDetected = true;
                    }
                    
                    // Check if test is complete
                    testDuration = (Date.now() - startTime) / 1000;
                    if (testDuration >= 3) {
                        // Test complete
                        stream.getTracks().forEach(track => track.stop());
                        javascriptNode.disconnect();
                        analyser.disconnect();
                        microphone.disconnect();
                        
                        displayMicTestResults(soundDetected, maxVolume);
                    }
                };
                
                // Update status
                micTestResult.innerHTML = 'Testing microphone... Please speak now.';
                volumeBar.style.width = '0%';
                
            })
            .catch(function(err) {
                micTestResult.innerHTML = 'Error accessing microphone: ' + err.message;
                micTestResult.className = 'alert alert-danger';
                window.debugLog && window.debugLog('Microphone access error: ' + err.message, true);
            });
    }
    
    function displayMicTestResults(soundDetected, maxVolume) {
        if (soundDetected && maxVolume > 10) {
            micTestResult.innerHTML = `✓ Microphone working properly! (Peak volume: ${maxVolume.toFixed(1)})`;
            micTestResult.className = 'alert alert-success';
            window.debugLog && window.debugLog(`Microphone test successful - Peak volume: ${maxVolume.toFixed(1)}`);
        } else if (soundDetected) {
            micTestResult.innerHTML = `⚠️ Microphone detected sound but signal is weak. (Peak volume: ${maxVolume.toFixed(1)})`;
            micTestResult.className = 'alert alert-warning';
            window.debugLog && window.debugLog(`Microphone test completed - Weak signal detected: ${maxVolume.toFixed(1)}`);
        } else {
            micTestResult.innerHTML = `✗ No sound detected. Check your microphone settings.`;
            micTestResult.className = 'alert alert-danger';
            window.debugLog && window.debugLog('Microphone test failed - No sound detected');
        }
    }
}); 