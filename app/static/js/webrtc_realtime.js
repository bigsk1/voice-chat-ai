/**
 * WebRTC implementation for OpenAI's Realtime API
 * Based on OpenAI's example at https://platform.openai.com/docs/guides/realtime
 */

document.addEventListener("DOMContentLoaded", function() {
    // DOM elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const transcript = document.getElementById('transcript');
    const characterSelect = document.getElementById('characterSelect');
    const voiceSelect = document.getElementById('voiceSelect');
    const sessionStatus = document.getElementById('session-status');
    const testTextInput = document.getElementById('testTextInput');
    const sendTextBtn = document.getElementById('sendTextBtn');
    const micBtn = document.getElementById('micBtn');
    const micStatus = document.getElementById('micStatus');
    const testMicBtn = document.getElementById('testMicBtn');
    
    // Global state
    let peerConnection = null;
    let dataChannel = null;
    let micStream = null;
    let isSessionActive = false;
    let ephemeralKey = null;
    let audioPlayer = new Audio();
    
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
    startBtn.addEventListener('click', startSession);
    stopBtn.addEventListener('click', stopSession);
    sendTextBtn.addEventListener('click', sendTextMessage);
    micBtn.addEventListener('click', toggleMicrophone);
    clearBtn.addEventListener('click', clearTranscript);
    testMicBtn.addEventListener('click', testMicrophone);
    
    // Send text message with Enter key
    testTextInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendTextMessage();
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
            const micTestResult = document.getElementById('micTestResult');
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
            const testTimeout = 3000; // 3 seconds
            
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
                    } else if (peakVolume < 30) {
                        micTestResult.textContent = `Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Low volume detected. Consider adjusting your microphone.`;
                        micTestResult.style.color = "#ff8800";
                    } else {
                        micTestResult.textContent = `Microphone test complete. Peak volume: ${Math.round(peakVolume)}. Your microphone is working properly.`;
                        micTestResult.style.color = "#00cc00";
                    }
                    
                    debugLog(`Microphone test complete. Peak volume: ${Math.round(peakVolume)}`, "info");
                    
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
                    
                    // Close audio context
                    if (audioContext.state !== 'closed') {
                        audioContext.close();
                    }
                }
            }, testTimeout + 500);
            
        } catch (error) {
            console.error("Microphone test error:", error);
            const micTestResult = document.getElementById('micTestResult');
            micTestResult.textContent = `Microphone access error: ${error.message}`;
            micTestResult.style.display = "block";
            micTestResult.style.color = "#ff0000";
            
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
            startBtn.disabled = true;
            
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
            };
            
            peerConnection.onicecandidate = (event) => {
                console.log("ICE candidate event:", event.candidate ? "new candidate" : "all candidates gathered");
            };
            
            // Setup audio playback
            audioPlayer.autoplay = true;
            peerConnection.ontrack = e => {
                console.log(`Received ${e.track.kind} track from OpenAI`);
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
            
            // Construct URL for our proxy endpoint instead of the direct OpenAI endpoint
            const model = "gpt-4o-realtime-preview-2024-12-17";
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
                startBtn.disabled = true;
                stopBtn.disabled = false;
                micBtn.disabled = false;
                sendTextBtn.disabled = false;
                
                // Add session message
                addTranscriptMessage("Session started with " + characterSelect.value, "system");
                addTranscriptMessage("You can now speak or type messages", "system");
                
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
            startBtn.disabled = false;
            
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
            
            // Example: Send character instructions
            const character = characterSelect.value;
            const voice = voiceSelect.value;
            
            // Set voice preference
            sendVoicePreference(voice);
            
            // Use a default instruction string formatted for the character
            const characterName = characterSelect.options[characterSelect.selectedIndex].text;
            const defaultInstructions = `You are ${characterName}. Speak in the style of ${characterName}. Keep your responses concise and engaging.`;
            
            // Send the instructions immediately
            sendInstructions(defaultInstructions);
            
            // Log the instructions
            debugLog(`Using character instructions for ${characterName}`, "info");
        };
        
        dataChannel.onmessage = (event) => {
            try {
                // Parse the message
                const data = JSON.parse(event.data);
                const messageType = data.type;
                
                // Log all messages for debugging
                debugLog(`Received message: ${JSON.stringify(data)}`, "data");
                
                // Handle different message types
                if (messageType === "conversation.item.text.created") {
                    // AI text response
                    const content = data.content || {};
                    const text = content.text || "";
                    
                    if (text) {
                        addTranscriptMessage(text, "ai");
                    }
                    
                    // Show AI voice visualization during speech
                    document.getElementById('aiVoiceVisualization').classList.remove('hidden');
                    // Simulate voice bars animation
                    animateVoiceBars('aiVoiceVisualization');
                    
                } else if (messageType === "conversation.item.message.completed") {
                    // Message completed
                    debugLog("AI message completed", "success");
                    // Hide AI voice visualization
                    document.getElementById('aiVoiceVisualization').classList.add('hidden');
                    
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
                    debugLog("Audio received by API", "info");
                    // Show user voice visualization
                    document.getElementById('userVoiceVisualization').classList.remove('hidden');
                    // Simulate voice bars animation
                    animateVoiceBars('userVoiceVisualization');
                    
                } else if (messageType === "audio_buffer.committed") {
                    // Audio buffer committed
                    debugLog("Audio buffer committed", "success");
                    // Hide user voice visualization
                    document.getElementById('userVoiceVisualization').classList.add('hidden');
                    
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
    
    // Send text message
    function sendTextMessage() {
        if (!isSessionActive || !dataChannel || dataChannel.readyState !== "open") {
            addTranscriptMessage("Cannot send message: No active session", "error");
            return;
        }
        
        const text = testTextInput.value.trim();
        if (!text) return;
        
        try {
            // Add to transcript
            addTranscriptMessage(text, "user");
            
            // Create the message
            const message = {
                event_id: `event_${Date.now()}`,
                type: "conversation.item.create",
                item: {
                    type: "message",
                    role: "user",
                    content: [
                        {
                            type: "input_text",
                            text: text
                        }
                    ]
                }
            };
            
            // Send the message
            dataChannel.send(JSON.stringify(message));
            debugLog(`Sent text message: ${text}`);
            
            // Clear input
            testTextInput.value = '';
        } catch (error) {
            console.error("Error sending message:", error);
            debugLog(`Error sending message: ${error.message}`, "error");
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
    
    // Send character instructions
    function sendInstructions(instructions) {
        if (!isSessionActive || !dataChannel || dataChannel.readyState !== "open") {
            return;
        }
        
        try {
            const message = {
                event_id: `event_${Date.now()}`,
                type: "session.update",
                session: {
                    instructions: instructions
                }
            };
            
            dataChannel.send(JSON.stringify(message));
            debugLog(`Set instructions: ${instructions.substring(0, 50)}...`);
        } catch (error) {
            console.error("Error setting instructions:", error);
            debugLog(`Error setting instructions: ${error.message}`, true);
        }
    }
    
    // Toggle microphone
    function toggleMicrophone() {
        console.log("Toggle microphone called, session active:", isSessionActive);
        if (!isSessionActive) {
            addTranscriptMessage("Cannot use microphone: No active session", "error");
            return;
        }
        
        if (micStream) {
            // Toggle mic state
            const audioTracks = micStream.getAudioTracks();
            console.log("Audio tracks:", audioTracks.length);
            if (audioTracks.length > 0) {
                const isEnabled = audioTracks[0].enabled;
                console.log("Current mic state:", isEnabled ? "enabled" : "disabled", "- toggling to", !isEnabled ? "enabled" : "disabled");
                audioTracks[0].enabled = !isEnabled;
                
                // Update UI
                if (!isEnabled) {
                    // Enabling
                    micBtn.classList.add('listening');
                    micStatus.textContent = "Listening... (speak now)";
                    micStatus.style.color = "#ff3300";
                    console.log("Microphone enabled");
                } else {
                    // Disabling
                    micBtn.classList.remove('listening');
                    micStatus.textContent = "Click to speak";
                    micStatus.style.color = "";
                    console.log("Microphone disabled");
                }
            } else {
                console.error("No audio tracks found in mic stream");
            }
        } else {
            // Need to set up microphone
            console.log("No mic stream found, setting up microphone...");
            setupMicrophone().then(success => {
                console.log("Microphone setup result:", success);
                if (success) {
                    micBtn.classList.add('listening');
                    micStatus.textContent = "Listening... (speak now)";
                    micStatus.style.color = "#ff3300";
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
        startBtn.disabled = false;
        stopBtn.disabled = true;
        micBtn.disabled = true;
        sendTextBtn.disabled = true;
        
        // Reset mic UI
        micBtn.classList.remove('listening');
        micStatus.textContent = "Click to speak";
        micStatus.style.color = "";
        
        // Add status message
        if (transcript) {
            addTranscriptMessage("Session ended", "system");
        }
        
        debugLog("Session stopped");
    }
}); 