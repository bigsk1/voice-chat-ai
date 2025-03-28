/**
 * WebRTC implementation for OpenAI's Realtime API
 * Based on OpenAI's example at https://platform.openai.com/docs/guides/realtime
 */

document.addEventListener("DOMContentLoaded", function() {
    // DOM elements
    const startButton = document.getElementById('startBtn');
    const stopButton = document.getElementById('stopBtn');
    const clearButton = document.getElementById('clearBtn');
    const testMicButton = document.getElementById('testMicBtn');
    const micButton = document.getElementById('micBtn');
    const micStatus = document.getElementById('micStatus');
    const transcript = document.getElementById('transcript');
    const sessionStatus = document.getElementById('session-status');
    const characterSelect = document.getElementById('character-select');
    const voiceSelect = document.getElementById('voice-select');
    const userVoiceVisualization = document.getElementById('userVoiceVisualization');
    const aiVoiceVisualization = document.getElementById('aiVoiceVisualization');
    const waitingIndicator = document.getElementById('waitingIndicator');
    const themeToggle = document.getElementById('theme-toggle');
    
    // Global state
    let peerConnection = null;
    let dataChannel = null;
    let micStream = null;
    let isSessionActive = false;
    let ephemeralKey = null;
    let audioPlayer = new Audio();
    
    // Set dark mode as default
    setDarkModeDefault();
    
    // Initialize debug panel
    if (typeof window.debugLog !== 'function') {
        // Simple debug log function if debug panel doesn't exist
        window.debugLog = (message, type) => {
            if (type === 'error') {
                console.error(message);
            } else {
                console.log(message);
            }
        };
    }
    
    // Setup event listeners
    startButton.addEventListener('click', startSession);
    stopButton.addEventListener('click', stopSession);
    micButton.addEventListener('click', toggleMicrophone);
    clearButton.addEventListener('click', clearTranscript);
    testMicButton.addEventListener('click', testMicrophone);
    
    // Info box toggle
    const infoToggleBtn = document.getElementById('infoToggleBtn');
    const infoBox = document.getElementById('infoBox');
    if (infoToggleBtn && infoBox) {
        infoToggleBtn.addEventListener('click', function() {
            const isVisible = infoBox.style.display !== 'none';
            if (isVisible) {
                infoBox.style.display = 'none';
                infoToggleBtn.innerHTML = '<i class="fas fa-info-circle"></i> Show Usage Guide';
            } else {
                infoBox.style.display = 'block';
                infoToggleBtn.innerHTML = '<i class="fas fa-info-circle"></i> Hide Usage Guide';
            }
        });
    }
    
    // Voice selection change handler
    voiceSelect.addEventListener('change', function() {
        const selectedVoice = voiceSelect.value;
        if (isSessionActive && dataChannel && dataChannel.readyState === 'open') {
            sendVoicePreference(selectedVoice);
            addTranscriptMessage(`Voice changed to ${selectedVoice}`, "system");
        }
    });
    
    // Character selection change handler
    characterSelect.addEventListener('change', function() {
        const selectedCharacter = characterSelect.value;
        addTranscriptMessage(`Character changed to ${selectedCharacter}`, "system");
        
        // If session is active, recommend restarting for new character to take effect
        if (isSessionActive) {
            addTranscriptMessage("Please stop and restart the session for the new character to take effect", "system");
        }
    });
    
    // Add transcript message
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
    
    // Test microphone function
    async function testMicrophone() {
        try {
            console.log("Testing microphone...");
            debugLog("Starting microphone test...", "info");
            
            // Use direct debug messaging without showing panel
            if (window.forceDebugMessage) {
                window.forceDebugMessage("Starting microphone test...", "info", false);
            }
            
            const micTestResult = document.getElementById('micTestResult');
            if (!micTestResult) {
                console.error("micTestResult element not found in DOM");
                debugLog("Error: micTestResult element not found in DOM", "error");
                if (window.forceDebugMessage) {
                    window.forceDebugMessage("Error: micTestResult element not found in DOM", "error", false);
                }
                return;
            }
            
            micTestResult.textContent = "Testing microphone...";
            micTestResult.style.display = "block";
            
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            debugLog("Microphone access granted", "success");
            
            // Create audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const microphone = audioContext.createMediaStreamSource(stream);
            const scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
            
            analyser.smoothingTimeConstant = 0.8;
            analyser.fftSize = 1024;
            
            microphone.connect(analyser);
            analyser.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
            
            let peakVolume = 0;
            let testDuration = 0;
            const testStartTime = Date.now();
            const testTimeout = 3000;
            
            debugLog("Audio processing pipeline ready", "info");
            
            scriptProcessor.onaudioprocess = function() {
                const array = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(array);
                
                // Get average volume
                let values = 0;
                for (let i = 0; i < array.length; i++) {
                    values += array[i];
                }
                const average = values / array.length;
                
                // Update peak volume
                if (average > peakVolume) {
                    peakVolume = average;
                    debugLog(`New peak volume: ${Math.round(peakVolume)}`, "data");
                }
                
                // Animate mic test UI
                micTestResult.textContent = `Testing microphone... Current level: ${Math.round(average)}`;
                
                // Check if test duration is reached
                testDuration = Date.now() - testStartTime;
                if (testDuration >= testTimeout) {
                    // Clean up and show results
                    scriptProcessor.onaudioprocess = null;
                    stream.getTracks().forEach(track => track.stop());
                    microphone.disconnect();
                    analyser.disconnect();
                    scriptProcessor.disconnect();
                    
                    // Display results
                    if (peakVolume < 10) {
                        micTestResult.textContent = `Microphone test complete. Peak volume: ${Math.round(peakVolume)}. WARNING: Very low volume detected. Check your microphone.`;
                        micTestResult.style.color = "#ff0000";
                        debugLog(`Microphone test complete. Peak volume: ${Math.round(peakVolume)}. WARNING: Very low volume detected.`, "warning");
                    } else if (peakVolume < 30) {
                        micTestResult.textContent = `Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Low volume detected. Consider adjusting your microphone.`;
                        micTestResult.style.color = "#ff8800";
                        debugLog(`Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Low volume detected.`, "warning");
                    } else {
                        micTestResult.textContent = `Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Your microphone is working properly.`;
                        micTestResult.style.color = "#00cc00";
                        debugLog(`Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Microphone is working properly.`, "success");
                    }
                    
                    // Close audio context
                    if (audioContext.state !== 'closed') {
                        audioContext.close();
                    }
                }
            };
            
            // Set timeout as a fallback
            setTimeout(() => {
                if (testDuration < testTimeout) {
                    // Test didn't complete properly, force cleanup
                    scriptProcessor.onaudioprocess = null;
                    stream.getTracks().forEach(track => track.stop());
                    microphone.disconnect();
                    analyser.disconnect();
                    scriptProcessor.disconnect();
                    
                    micTestResult.textContent = "Microphone test timed out. Try again.";
                    micTestResult.style.color = "#ff0000";
                    debugLog("Microphone test timed out", "error");
                    
                    // Close audio context
                    if (audioContext.state !== 'closed') {
                        audioContext.close();
                    }
                }
            }, testTimeout + 500);
            
        } catch (error) {
            console.error("Microphone test error:", error);
            const micTestResult = document.getElementById('micTestResult');
            if (micTestResult) {
                micTestResult.textContent = `Microphone access error: ${error.message}`;
                micTestResult.style.display = "block";
                micTestResult.style.color = "#ff0000";
            }
            
            debugLog(`Microphone test error: ${error.message}`, "error");
        }
    }
    
    // Clear transcript
    function clearTranscript() {
        if (transcript) {
            // Remove all messages except the system messages at the beginning
            while (transcript.childNodes.length > 2) {
                transcript.removeChild(transcript.lastChild);
            }
            
            // Add clear message
            addTranscriptMessage("Transcript cleared", "system");
            
            debugLog("Transcript cleared", "info");
        }
    }
    
    // Start WebRTC session
    async function startSession() {
        try {
            console.log("Starting session...");
            // Update UI
            sessionStatus.textContent = "Connecting...";
            sessionStatus.classList.remove("badge-secondary");
            sessionStatus.classList.add("badge-warning");
            startButton.disabled = true;
            
            console.log("Fetching ephemeral key...");
            // Get ephemeral key from server
            const response = await fetch('/openai_ephemeral_key');
            console.log("Ephemeral key response status:", response.status);
            if (!response.ok) {
                throw new Error(`Failed to get ephemeral key: ${response.status} - ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log("Received data from server:", data);
            ephemeralKey = data.client_secret?.value;
            
            if (!ephemeralKey) {
                throw new Error("Invalid ephemeral key received from server: " + JSON.stringify(data));
            }
            
            // Log success
            console.log("Received valid ephemeral key from server");
            
            // Create peer connection with STUN servers to help with NAT traversal
            console.log("Creating RTCPeerConnection with STUN servers...");
            peerConnection = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            });
            
            // Add event listeners for connection status
            peerConnection.oniceconnectionstatechange = () => {
                console.log("ICE connection state changed to:", peerConnection.iceConnectionState);
                debugLog(`ICE connection state changed to: ${peerConnection.iceConnectionState}`, "event");
            };
            
            peerConnection.onicecandidate = (event) => {
                console.log("ICE candidate event:", event.candidate ? "new candidate" : "all candidates gathered");
                if (event.candidate) {
                    debugLog("New ICE candidate gathered", "info");
                } else {
                    debugLog("All ICE candidates gathered", "success");
                }
            };
            
            // Setup audio playback
            audioPlayer.autoplay = true;
            peerConnection.ontrack = e => {
                console.log(`Received ${e.track.kind} track from OpenAI`);
                debugLog(`Received ${e.track.kind} track from OpenAI`, "success");
                audioPlayer.srcObject = e.streams[0];
            };
            
            // Setup microphone
            console.log("Setting up microphone...");
            const micSetupSuccess = await setupMicrophone();
            if (!micSetupSuccess) {
                throw new Error("Failed to set up microphone");
            }
            
            // Setup data channel
            console.log("Setting up data channel...");
            dataChannel = peerConnection.createDataChannel("oai-events");
            
            // Handle data channel events
            setupDataChannel();
            
            // Create offer and set local description
            console.log("Creating offer...");
            const offer = await peerConnection.createOffer();
            console.log("Setting local description...");
            await peerConnection.setLocalDescription(offer);
            
            // Wait for ICE gathering to complete with a reasonable timeout
            let iceGatheringComplete = false;
            try {
                await new Promise((resolve, reject) => {
                    const iceGatheringTimeout = setTimeout(() => {
                        if (!iceGatheringComplete) {
                            console.warn("ICE gathering timed out, continuing anyway");
                            resolve();
                        }
                    }, 2000); // 2 second timeout
                    
                    const checkState = () => {
                        if (peerConnection.iceGatheringState === 'complete') {
                            iceGatheringComplete = true;
                            clearTimeout(iceGatheringTimeout);
                            peerConnection.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    
                    if (peerConnection.iceGatheringState === 'complete') {
                        iceGatheringComplete = true;
                        clearTimeout(iceGatheringTimeout);
                        resolve();
                    } else {
                        peerConnection.addEventListener('icegatheringstatechange', checkState);
                    }
                });
            } catch (error) {
                console.warn("Error during ICE gathering:", error);
                // Continue anyway
            }
            
            // Get SDP from local description
            if (!peerConnection.localDescription) {
                throw new Error("No local description available");
            }
            
            const sdp = peerConnection.localDescription.sdp;
            if (!sdp) {
                throw new Error("Failed to get SDP from local description");
            }
            
            // Configuration/Settings
            const defaultModel = "gpt-4o-realtime-preview-2024-12-17"; // Default fallback model
            
            // Fetch model from server-side configuration or use default
            let model = rtcConfig.model || defaultModel;
            debugLog(`Using model: ${model}`, 'info');
            
            // Construct URL for our proxy endpoint instead of the direct OpenAI endpoint
            const proxyUrl = `/openai_realtime_proxy?model=${model}`;
            
            // Log SDP being sent
            console.log(`Sending SDP to proxy endpoint: ${proxyUrl}`);
            
            // Send SDP to our proxy endpoint
            console.log("Making fetch request to proxy...");
            try {
                const sdpResponse = await fetch(proxyUrl, {
                    method: "POST",
                    body: sdp,
                    headers: {
                        "Content-Type": "application/sdp"
                    }
                });
                
                // Check response
                console.log("Proxy response status:", sdpResponse.status);
                if (!sdpResponse.ok) {
                    let errorText = "";
                    try {
                        errorText = await sdpResponse.text();
                    } catch (e) {
                        errorText = "Could not read error response";
                    }
                    console.error("Error response from proxy:", errorText);
                    throw new Error(`Failed to create session: ${sdpResponse.status} - ${errorText}`);
                }
                
                // Get answer SDP
                const answerSdp = await sdpResponse.text();
                console.log("Received answer SDP from proxy");
                
                // Set remote description
                console.log("Setting remote description...");
                const answer = {
                    type: "answer",
                    sdp: answerSdp
                };
                await peerConnection.setRemoteDescription(answer);
                console.log("Remote description set successfully");
                
                // Update UI only after successful connection
                isSessionActive = true;
                sessionStatus.textContent = "Active";
                sessionStatus.classList.remove("badge-warning");
                sessionStatus.classList.add("badge-success");
                startButton.disabled = true;
                stopButton.disabled = false;
                micButton.disabled = false;
                
                // Add session message
                addTranscriptMessage("Session started with " + characterSelect.value, "system");
                addTranscriptMessage("You can now speak", "system");
                
                // Set mic icon to waiting state
                updateHeaderMicIcon(false);
                
                // Start microphone automatically
                toggleMicrophone();
                console.log("Session started successfully");
            } catch (fetchError) {
                console.error("Fetch error:", fetchError);
                throw new Error(`Proxy request failed: ${fetchError.message}`);
            }
            
        } catch (error) {
            console.error("Error starting session:", error);
            // Update UI for error state
            sessionStatus.textContent = "Error";
            sessionStatus.classList.remove("badge-warning");
            sessionStatus.classList.add("badge-danger");
            startButton.disabled = false;
            
            // Add error message
            addTranscriptMessage(`Failed to start session: ${error.message}`, "error");
            
            // Clean up resources
            stopSession();
            
            // Provide helpful error message
            if (error.message.includes('CORS')) {
                console.error("This appears to be a CORS issue. The WebRTC implementation requires direct communication with OpenAI's servers, which browsers restrict for security reasons. Consider using the WebSocket implementation instead, which proxies through your server.");
                addTranscriptMessage("CORS error: This browser implementation can't directly connect to OpenAI. Try the WebSocket implementation instead.", "error");
            }
        }
    }
    
    // Setup microphone
    async function setupMicrophone() {
        try {
            console.log("Requesting microphone access...");
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log("Microphone access granted, adding audio track to peer connection...");
            // Add audio track to peer connection
            const audioTrack = micStream.getAudioTracks()[0];
            console.log("Audio track:", audioTrack.label, "- enabled:", audioTrack.enabled);
            peerConnection.addTrack(audioTrack, micStream);
            
            // Add debug logging for microphone activity
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const microphone = audioContext.createMediaStreamSource(micStream);
            const scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
            
            analyser.smoothingTimeConstant = 0.8;
            analyser.fftSize = 1024;
            
            microphone.connect(analyser);
            analyser.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
            
            let lastLogTime = 0;
            
            scriptProcessor.onaudioprocess = function() {
                const array = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(array);
                
                // Get average volume
                let values = 0;
                for (let i = 0; i < array.length; i++) {
                    values += array[i];
                }
                const average = values / array.length;
                
                // Update header mic icon based on volume
                if (average > 10) {
                    // Speaking - make mic icon pulsing red
                    updateHeaderMicIcon(true);
                } else {
                    // Silent - turn off mic icon pulse
                    updateHeaderMicIcon(false);
                }
                
                // Log microphone activity every second if volume is above threshold
                const now = Date.now();
                if (average > 10 && now - lastLogTime > 1000) {
                    // Use direct debug messaging without showing panel
                    if (window.forceDebugMessage) {
                        window.forceDebugMessage(`Microphone active - volume level: ${Math.round(average)}`, "info", false);
                    }
                    
                    debugLog(`Microphone active - volume level: ${Math.round(average)}`, "info");
                    lastLogTime = now;
                }
            };
            
            console.log("Microphone setup complete");
            return true;
        } catch (error) {
            console.error("Error accessing microphone:", error);
            debugLog(`Error accessing microphone: ${error.message}`, "error");
            addTranscriptMessage(`Microphone error: ${error.message}`, "error");
            return false;
        }
    }
    
    // Setup data channel events
    function setupDataChannel() {
        dataChannel.onopen = () => {
            debugLog("Data channel opened", "success");
            
            // Get character and voice settings
            const character = characterSelect.value;
            const voice = voiceSelect.value;
            
            // Set voice preference
            sendVoicePreference(voice);
            
            // Fetch and use the actual character prompt
            fetchCharacterPrompt(character)
                .then(instructions => {
                    // Send the instructions immediately
                    sendInstructions(instructions);
                    
                    // Log the instructions
                    debugLog(`Using character instructions for ${character}`, "info");
                })
                .catch(error => {
                    // Fallback to simple instructions if fetch fails
                    const characterName = characterSelect.options[characterSelect.selectedIndex].text;
                    const fallbackInstructions = `You are ${characterName}. Speak in the style of ${characterName}. Keep your responses concise and engaging.`;
                    
                    sendInstructions(fallbackInstructions);
                    debugLog(`Using fallback instructions for ${characterName}`, "warning");
                    console.error("Error fetching character prompt:", error);
                });
        };
        
        dataChannel.onmessage = (event) => {
            try {
                // Parse the message
                const data = JSON.parse(event.data);
                const messageType = data.type;
                
                // Hide waiting indicator whenever we get a response
                showWaitingIndicator(false);
                
                // Use direct message display without showing the panel
                if (window.forceDebugMessage) {
                    window.forceDebugMessage(`Received message type: ${messageType}`, "info", false);
                    window.forceDebugMessage(`Message data: ${JSON.stringify(data).substring(0, 100)}...`, "data", false);
                } else {
                    // Fallback to regular debug log
                    debugLog(`Received message type: ${messageType}`, "info");
                }
                
                // Log all messages for debugging
                debugLog(`Received message: ${JSON.stringify(data)}`, "data");
                
                // Handle different message types
                if (messageType === "conversation.item.text.created") {
                    // AI text response
                    const content = data.content || {};
                    const text = content.text || "";
                    
                    if (text) {
                        addTranscriptMessage(text, "ai");
                        // Add explicit debug log for AI speech
                        debugLog(`AI responded: ${text}`, "success");
                    }
                    
                    // Show AI voice visualization during speech
                    aiVoiceVisualization.classList.remove('hidden');
                    // Simulate voice bars animation
                    animateVoiceBars('aiVoiceVisualization');
                    
                } else if (messageType === "conversation.item.message.completed") {
                    // Message completed
                    debugLog("AI message completed", "success");
                    // Hide AI voice visualization
                    aiVoiceVisualization.classList.add('hidden');
                    
                } else if (messageType === "conversation.item.text.delta") {
                    // Text delta - partial text updates
                    const content = data.delta || {};
                    const text = content.text || "";
                    
                    if (text) {
                        debugLog(`Text delta: ${text}`, "info");
                        // We could update the UI incrementally here if desired
                    }
                    
                } else if (messageType === "audio_buffer.meta.received") {
                    // Audio buffer meta info
                    debugLog("Audio received by API - user is speaking", "info");
                    // Show user voice visualization
                    userVoiceVisualization.classList.remove('hidden');
                    // Simulate voice bars animation
                    animateVoiceBars('userVoiceVisualization');
                    
                } else if (messageType === "audio_buffer.committed") {
                    // Audio buffer committed
                    debugLog("Audio buffer committed - user finished speaking", "success");
                    // Hide user voice visualization
                    userVoiceVisualization.classList.add('hidden');
                    
                } else if (messageType === "session.updated") {
                    // Session update confirmation
                    const session = data.session || {};
                    if (session.voice) {
                        debugLog(`Voice set to: ${session.voice}`, "success");
                    }
                    if (session.instructions) {
                        debugLog("Instructions updated", "success");
                    }
                    
                } else if (messageType === "error") {
                    // Error message
                    const error = data.error || {};
                    const errorMessage = error.message || "Unknown error";
                    debugLog(`Error from OpenAI: ${errorMessage}`, "error");
                    addTranscriptMessage(`Error: ${errorMessage}`, "error");
                    
                } else {
                    // Unhandled message type
                    debugLog(`Unhandled message type: ${messageType}`, "warning");
                }
            } catch (error) {
                console.error("Error parsing message:", error);
                debugLog(`Error parsing message: ${error.message}`, "error");
            }
        };
        
        dataChannel.onerror = (error) => {
            console.error("Data channel error:", error);
            debugLog(`Data channel error: ${error.message || "Unknown error"}`, "error");
        };
        
        dataChannel.onclose = () => {
            debugLog("Data channel closed", "warning");
        };
    }
    
    // Helper function to animate voice visualization bars
    function animateVoiceBars(elementId) {
        const voiceVisualization = document.getElementById(elementId);
        if (!voiceVisualization) return;
        
        // Check if we already have voice bars, if not create them
        if (voiceVisualization.querySelectorAll('.voice-bar').length === 0) {
            // Create 8 voice bars
            for (let i = 0; i < 8; i++) {
                const bar = document.createElement('div');
                bar.classList.add('voice-bar');
                voiceVisualization.appendChild(bar);
            }
        }
        
        const bars = voiceVisualization.querySelectorAll('.voice-bar');
        
        // Clear any existing animation
        bars.forEach(bar => {
            bar.style.height = '5px';
            bar.classList.remove('active-bar');
        });
        
        // Start new animation
        const animate = () => {
            if (voiceVisualization.classList.contains('hidden')) {
                // Stop animation if visualization is hidden
                bars.forEach(bar => {
                    bar.style.height = '5px';
                    bar.classList.remove('active-bar');
                });
                return;
            }
            
            bars.forEach(bar => {
                // Random height between 5 and 40px
                const height = Math.floor(Math.random() * 36) + 5;
                bar.style.height = `${height}px`;
                
                // Add active class for high bars
                if (height > 20) {
                    bar.classList.add('active-bar');
                } else {
                    bar.classList.remove('active-bar');
                }
            });
            
            // Continue animation
            requestAnimationFrame(animate);
        };
        
        // Start animation
        animate();
    }
    
    // Function to show/hide waiting indicator
    function showWaitingIndicator(show) {
        if (show) {
            waitingIndicator.classList.remove('hidden');
        } else {
            waitingIndicator.classList.add('hidden');
        }
    }
    
    // Function to fetch character prompt from server
    async function fetchCharacterPrompt(characterName) {
        debugLog(`Fetching character prompt for: ${characterName}`, "info");
        try {
            const response = await fetch(`/api/character/${characterName}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch character prompt: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            
            debugLog(`Received prompt for ${characterName}: ${data.prompt.length} chars`, "success");
            console.log(`Character prompt for ${characterName}:`, data.prompt);
            return data.prompt;
        } catch (error) {
            console.error("Error fetching character prompt:", error);
            debugLog(`Error fetching character prompt: ${error.message}`, "error");
            throw error;
        }
    }
    
    // Send voice preference
    function sendVoicePreference(voice) {
        if (!isSessionActive || !dataChannel || dataChannel.readyState !== "open") {
            return;
        }
        
        try {
            const message = {
                event_id: `event_${Date.now()}`,
                type: "session.update",
                session: {
                    voice: voice
                }
            };
            
            dataChannel.send(JSON.stringify(message));
            debugLog(`Set voice preference: ${voice}`);
        } catch (error) {
            console.error("Error setting voice:", error);
            debugLog(`Error setting voice: ${error.message}`, true);
        }
    }
    
    // Send instructions to the API
    function sendInstructions(instructions) {
        if (!dataChannel || dataChannel.readyState !== 'open') {
            console.error("Data channel not open");
            debugLog("Data channel not open, can't send instructions", "error");
            return;
        }
        
        console.log("Sending instructions to API:", instructions);
        debugLog(`Sending instructions (${instructions.length} chars)`, "info");
        
        const message = {
            type: "session.update",
            session: {
                instructions: instructions
            }
        };
        
        dataChannel.send(JSON.stringify(message));
    }
    
    // Toggle microphone
    function toggleMicrophone() {
        console.log("Toggle microphone called, session active:", isSessionActive);
        
        // Use direct message display to ensure visibility without showing panel
        if (window.forceDebugMessage) {
            window.forceDebugMessage(`Toggling microphone. Session active: ${isSessionActive}`, "event", false);
        }
        
        debugLog("Toggling microphone. Session active: " + isSessionActive, "event");
        
        if (!isSessionActive) {
            addTranscriptMessage("Cannot use microphone: No active session", "error");
            debugLog("Cannot use microphone: No active session", "error");
            return;
        }
        
        if (micStream) {
            // Toggle mic state
            const audioTracks = micStream.getAudioTracks();
            console.log("Audio tracks:", audioTracks.length);
            debugLog(`Found ${audioTracks.length} audio tracks`, "info");
            
            if (audioTracks.length > 0) {
                const isEnabled = audioTracks[0].enabled;
                console.log("Current mic state:", isEnabled ? "enabled" : "disabled", "- toggling to", !isEnabled ? "enabled" : "disabled");
                debugLog(`Current mic state: ${isEnabled ? "enabled" : "disabled"} - toggling to ${!isEnabled ? "enabled" : "disabled"}`, "info");
                audioTracks[0].enabled = !isEnabled;
                
                // Update UI
                if (!isEnabled) {
                    // Enabling
                    micButton.classList.add('listening');
                    micStatus.textContent = "Listening... (speak now)";
                    micStatus.style.color = "#ff3300";
                    console.log("Microphone enabled");
                    debugLog("Microphone enabled - speak now to test audio pipeline", "success");
                    
                    // Update header mic icon to waiting state
                    updateHeaderMicIcon(false);
                } else {
                    // Disabling
                    micButton.classList.remove('listening');
                    micStatus.textContent = "Click to speak";
                    micStatus.style.color = "";
                    console.log("Microphone disabled");
                    debugLog("Microphone disabled", "warning");
                    
                    // Update header mic icon to off state when mic disabled
                    updateHeaderMicIcon(false);
                }
            } else {
                console.error("No audio tracks found in mic stream");
                debugLog("No audio tracks found in microphone stream", "error");
            }
        } else {
            // Need to set up microphone
            console.log("No mic stream found, setting up microphone...");
            debugLog("No microphone stream found, setting up microphone...", "warning");
            setupMicrophone().then(success => {
                console.log("Microphone setup result:", success);
                debugLog(`Microphone setup result: ${success ? "SUCCESS" : "FAILED"}`, success ? "success" : "error");
                if (success) {
                    micButton.classList.add('listening');
                    micStatus.textContent = "Listening... (speak now)";
                    micStatus.style.color = "#ff3300";
                    debugLog("Microphone ready - speak now to test audio pipeline", "success");
                }
            });
        }
    }
    
    // Stop session
    function stopSession() {
        // Clean up resources
        if (dataChannel) {
            dataChannel.close();
            dataChannel = null;
        }
        
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
        
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }
        
        if (audioPlayer.srcObject) {
            audioPlayer.srcObject = null;
        }
        
        // Update state
        isSessionActive = false;
        
        // Update UI
        sessionStatus.textContent = "Inactive";
        sessionStatus.classList.remove("badge-success", "badge-warning", "badge-danger");
        sessionStatus.classList.add("badge-secondary");
        startButton.disabled = false;
        stopButton.disabled = true;
        micButton.disabled = true;
        
        // Reset mic UI
        micButton.classList.remove('listening');
        micStatus.textContent = "Click to speak";
        micStatus.style.color = "";
        
        // Reset header mic icon
        updateHeaderMicIcon(false);
        
        // Add status message
        if (transcript) {
            addTranscriptMessage("Session ended", "system");
        }
        
        debugLog("Session stopped");
    }
    
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
    
    // Add CSS for microphone icon states
    const styleEl = document.createElement('style');
    styleEl.textContent = `
        #mic-icon.active {
            color: #e74c3c;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        #mic-icon.mic-off {
            color: #6c757d;
        }
    `;
    document.head.appendChild(styleEl);
    
    // Function to update header mic icon
    function updateHeaderMicIcon(isActive) {
        const micIcon = document.getElementById('mic-icon');
        if (!micIcon) return;
        
        // Check if mic is currently enabled (via the central mic button)
        const isMicEnabled = micStream && micStream.getAudioTracks().length > 0 && 
                            micStream.getAudioTracks()[0].enabled;
        
        if (isActive && isMicEnabled) {
            // Speaking AND mic is enabled - make icon pulse red
            micIcon.classList.remove('mic-off', 'mic-waiting');
            micIcon.classList.add('mic-on', 'pulse-animation');
        } else if (isSessionActive && isMicEnabled) {
            // Session active and mic enabled but not speaking - waiting state
            micIcon.classList.remove('mic-off', 'mic-on', 'pulse-animation');
            micIcon.classList.add('mic-waiting');
        } else if (isSessionActive && !isMicEnabled) {
            // Session active but mic is disabled - show as off
            micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
            micIcon.classList.add('mic-off');
        } else {
            // Session not active - off state
            micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
            micIcon.classList.add('mic-off');
        }
    }
}); 