document.addEventListener("DOMContentLoaded", function() {
    let websocket;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 2000; // 2 seconds
    
    const micIcon = document.getElementById('mic-icon');
    const themeToggle = document.getElementById('theme-toggle');
    const downloadButton = document.getElementById('download-button');
    const conversation = document.getElementById('conversation');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const characterSelect = document.getElementById('characterSelect');
    const voiceSelect = document.getElementById('voiceSelect');
    const modelSelect = document.getElementById('modelSelect');
    const ttsModelSelect = document.getElementById('ttsModelSelect');
    const transcriptionModelSelect = document.getElementById('transcriptionModelSelect');

    // Default speed value (since we removed the speedSelect dropdown)
    const defaultSpeed = "1.0";

    let isRecording = false;
    let hasStarted = false;
    let listeningIndicator = null;
    
    // For message queue management (like the main page)
    let aiMessageQueue = [];
    let isAISpeaking = false;
    
    function connectWebSocket() {
        // Close existing connection if any
        if (websocket && websocket.readyState !== WebSocket.CLOSED) {
            websocket.close();
        }
        
        websocket = new WebSocket(`ws://${window.location.hostname}:8000/ws_enhanced`);
        
        websocket.onopen = function(event) {
            console.log("WebSocket connection established");
            startBtn.disabled = false;
            reconnectAttempts = 0; // Reset reconnect counter on successful connection
            displayMessage("Connected to server", "system-message");
        };
        
        websocket.onmessage = function(event) {
            console.log("Message received:", event.data);
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                console.error("Error parsing WebSocket message:", e);
                data = { message: event.data, action: "error" };
            }
            
            if (data.action === "waiting_for_speech") {
                isRecording = false;
                micIcon.classList.remove('mic-on');
                micIcon.classList.add('mic-waiting');
                // Show listening message with animation
                showListeningIndicator(data.message || "Waiting for speech...");
            } else if (data.action === "recording_started") {
                isRecording = true;
                micIcon.classList.remove('mic-off', 'mic-waiting');
                micIcon.classList.add('mic-on');
                micIcon.classList.add('pulse-animation');
                hideListeningIndicator();
                displayMessage("Recording...", "system-message");
            } else if (data.action === "recording_stopped") {
                isRecording = false;
                micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
                micIcon.classList.add('mic-off');
                hideListeningIndicator();
                displayMessage("Processing your message...", "system-message");
            } else if (data.action === "audio_actually_playing") {
                // Set speaking flag and show animation
                isAISpeaking = true;
                showVoiceWaveAnimation();
                // Process any queued messages after a slight delay
                setTimeout(processQueuedMessages, 100);
            } else if (data.action === "ai_start_speaking") {
                // The server is preparing to speak, but audio hasn't started yet
                console.log("AI preparing to speak");
            } else if (data.action === "ai_stop_speaking") {
                // Audio finished playing
                isAISpeaking = false;
                hideVoiceWaveAnimation();
                // Process any queued messages
                processQueuedMessages();
            } else if (data.action === "conversation_stopped") {
                hasStarted = false;
                stopBtn.disabled = true;
                startBtn.disabled = false;
                micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
                micIcon.classList.add('mic-off');
                hideListeningIndicator();
                hideVoiceWaveAnimation();
                isAISpeaking = false;
                processQueuedMessages(); // Process any remaining messages
                console.log("Conversation stopped");
            } else if (data.action === "error") {
                console.error("Error:", data.message);
                displayMessage(data.message, "error-message");
                // Reset mic icon on error
                micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
                micIcon.classList.add('mic-off');
                hideListeningIndicator();
                hideVoiceWaveAnimation();
                isAISpeaking = false;
                processQueuedMessages(); // Process any remaining messages
            } else if (data.action === "connected") {
                console.log("WebSocket connection confirmed by server");
            } else if (data.message) {
                if (data.message.startsWith("You:")) {
                    // User messages are displayed immediately
                    displayMessage(data.message);
                } else if (data.type === "system-message") {
                    // System messages like character selection are displayed with system styling
                    displayMessage(data.message, "system-message");
                } else {
                    // Queue AI messages to display after audio finishes
                    aiMessageQueue.push(data.message);
                    if (!isAISpeaking) {
                        processQueuedMessages();
                    }
                }
            }
        };
        
        websocket.onclose = function(event) {
            console.log("WebSocket connection closed", event);
            startBtn.disabled = true;
            stopBtn.disabled = true;
            
            // Reset mic icon on disconnect
            micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
            micIcon.classList.add('mic-off');
            hideListeningIndicator();
            hideVoiceWaveAnimation(); // Hide voice animation on disconnect
            
            // Try to reconnect if not closed cleanly and not exceeding max attempts
            if (!event.wasClean && reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                const delay = reconnectDelay * reconnectAttempts;
                console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts}) in ${delay}ms...`);
                displayMessage(`Connection lost. Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`, "system-message");
                setTimeout(connectWebSocket, delay);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                displayMessage("Failed to connect to server after multiple attempts. Please refresh the page.", "error-message");
            }
        };
        
        websocket.onerror = function(event) {
            console.error("WebSocket error:", event);
            displayMessage("Connection error. Please try again later.", "error-message");
            
            // Reset mic icon on error
            micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
            micIcon.classList.add('mic-off');
            hideListeningIndicator();
            hideVoiceWaveAnimation(); // Hide voice animation on error
        };
    }
    
    function processQueuedMessages() {
        while (aiMessageQueue.length > 0 && !isAISpeaking) {
            displayMessage(aiMessageQueue.shift());
        }
    }
    
    function displayMessage(message, className = "") {
        const messagesContainer = document.getElementById('messages');
        const messageElement = document.createElement("div");
        
        if (className) {
            messageElement.className = className;
        } else if (message.startsWith("You:")) {
            messageElement.className = "user-message";
            message = message.substring(4).trim();
        } else {
            messageElement.className = "ai-message";
        }
        
        messageElement.textContent = message;
        messagesContainer.appendChild(messageElement);
        conversation.scrollTop = conversation.scrollHeight;
        adjustScrollPosition();
    }
    
    function adjustScrollPosition() {
        // Ensure the conversation is scrolled down even with voice animation
        setTimeout(() => {
            conversation.scrollTop = conversation.scrollHeight;
        }, 10);
    }
    
    function showListeningIndicator(message) {
        hideListeningIndicator(); // Remove any existing indicator
        
        const messagesContainer = document.getElementById('messages');
        listeningIndicator = document.createElement("div");
        listeningIndicator.className = "listening-indicator";
        
        const textSpan = document.createElement("span");
        textSpan.textContent = message;
        
        const dotsContainer = document.createElement("span");
        dotsContainer.className = "listening-dots";
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            dot.className = "dot";
            dot.style.animationDelay = `${i * 0.3}s`;
            dotsContainer.appendChild(dot);
        }
        
        listeningIndicator.appendChild(textSpan);
        listeningIndicator.appendChild(dotsContainer);
        
        messagesContainer.appendChild(listeningIndicator);
        conversation.scrollTop = conversation.scrollHeight;
    }
    
    function hideListeningIndicator() {
        if (listeningIndicator && listeningIndicator.parentNode) {
            listeningIndicator.parentNode.removeChild(listeningIndicator);
            listeningIndicator = null;
        }
    }
    
    function showVoiceWaveAnimation() {
        const voiceWave = document.getElementById('voiceWaveAnimation');
        if (voiceWave) {
            voiceWave.classList.remove('hidden');
            adjustScrollPosition();
        }
    }
    
    function hideVoiceWaveAnimation() {
        const voiceWave = document.getElementById('voiceWaveAnimation');
        if (voiceWave) {
            voiceWave.classList.add('hidden');
            adjustScrollPosition();
        }
    }
    
    startBtn.addEventListener('click', function() {
        // Check if WebSocket is connected
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            displayMessage("Not connected to server. Attempting to reconnect...", "system-message");
            connectWebSocket();
            return;
        }
        
        // Disable start button and enable stop button
        startBtn.disabled = true;
        stopBtn.disabled = false;
        hasStarted = true;
        
        // Clear any previous state
        micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
        micIcon.classList.add('mic-off'); // Will be updated by the server
        
        // Get all the selected settings
        const settings = {
            character: characterSelect.value,
            voice: voiceSelect.value,
            speed: defaultSpeed,
            model: modelSelect.value,
            ttsModel: ttsModelSelect.value,
            transcriptionModel: transcriptionModelSelect.value
        };
        
        // Don't display starting message - keep UI cleaner
        console.log("Starting enhanced conversation with settings:", settings);
        
        // Send the start command with settings
        fetch('/start_enhanced_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            console.log("Start conversation response:", data);
        })
        .catch(error => {
            console.error("Error starting conversation:", error);
            displayMessage("Error starting conversation. The server may be unresponsive.", "error-message");
            // Reset buttons on error
            startBtn.disabled = false;
            stopBtn.disabled = true;
            hasStarted = false;
        });
    });
    
    stopBtn.addEventListener('click', function() {
        // Disable stop button and enable start button
        stopBtn.disabled = true;
        startBtn.disabled = false;
        
        console.log("Stopping enhanced conversation");
        
        // Send the stop command
        fetch('/stop_enhanced_conversation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log("Stop conversation response:", data);
            hasStarted = false;
        })
        .catch(error => {
            console.error("Error stopping conversation:", error);
            displayMessage("Error stopping conversation. The server may be unresponsive.", "error-message");
            // Still enable start button even if error
            startBtn.disabled = false;
        });
    });
    
    clearBtn.addEventListener('click', function() {
        // Clear the conversation by emptying the messages div 
        // (don't remove the div itself)
        const messagesContainer = document.getElementById('messages');
        messagesContainer.innerHTML = '';
        
        console.log("Clearing conversation");
        
        // Send clear command to the server
        fetch('/clear_history', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log("Clear conversation response:", data);
            // Display a confirmation message
            displayMessage("Conversation history has been cleared.", "system-message");
        })
        .catch(error => {
            console.error("Error clearing conversation:", error);
            displayMessage("Error clearing conversation history", "error-message");
        });
    });
    
    // Update theme toggle functionality
    function updateThemeToggleIcon() {
        const isDarkMode = document.body.classList.contains('dark-mode');
        themeToggle.innerHTML = isDarkMode 
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    }
    
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        updateThemeToggleIcon();
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });
    
    // Load theme preference
    function loadThemePreference() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        document.body.classList.toggle('dark-mode', isDarkMode);
        updateThemeToggleIcon();
    }
    
    // Download conversation history
    downloadButton.addEventListener('click', function() {
        fetch('/download_enhanced_history')
            .then(response => {
                if (response.ok) {
                    return response.blob();
                }
                throw new Error('Failed to download history');
            })
            .then(blob => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'conversation_history.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Error downloading history:', error);
                displayMessage("Failed to download conversation history", "error-message");
            });
    });
    
    // Fetch available characters
    function fetchCharacters() {
        fetch('/characters')
            .then(response => response.json())
            .then(data => {
                characterSelect.innerHTML = '';
                
                // Sort the characters alphabetically
                data.characters.sort((a, b) => a.localeCompare(b));
                
                data.characters.forEach(character => {
                    const option = document.createElement('option');
                    option.value = character;
                    option.textContent = character.replace(/_/g, ' '); // Replace all underscores with spaces
                    characterSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching characters:', error);
                displayMessage("Failed to load characters", "error-message");
            });
    }
    
    // Fetch default settings from server
    function fetchDefaultSettings() {
        fetch('/enhanced_defaults')
            .then(response => response.json())
            .then(data => {
                // Wait a moment to ensure the character dropdown is populated
                setTimeout(() => {
                    // Set default character
                    if (data.character && characterSelect.querySelector(`option[value="${data.character}"]`)) {
                        characterSelect.value = data.character;
                    }
                    
                    // Set default voice
                    if (data.voice && voiceSelect.querySelector(`option[value="${data.voice}"]`)) {
                        voiceSelect.value = data.voice;
                    }
                    
                    // Set default model
                    if (data.model && modelSelect.querySelector(`option[value="${data.model}"]`)) {
                        modelSelect.value = data.model;
                    }
                    
                    // Set default TTS model
                    if (data.tts_model && ttsModelSelect.querySelector(`option[value="${data.tts_model}"]`)) {
                        ttsModelSelect.value = data.tts_model;
                    }
                    
                    // Set default transcription model
                    if (data.transcription_model && transcriptionModelSelect.querySelector(`option[value="${data.transcription_model}"]`)) {
                        transcriptionModelSelect.value = data.transcription_model;
                    }
                }, 300); // Small delay to ensure dropdowns are populated
            })
            .catch(error => {
                console.error('Error fetching default settings:', error);
            });
    }
    
    // Add a simple heartbeat to keep the connection alive
    function startHeartbeat() {
        setInterval(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                // Send a ping to keep the connection alive
                try {
                    websocket.send(JSON.stringify({action: "ping"}));
                } catch (e) {
                    console.log("Error sending heartbeat", e);
                }
            }
        }, 30000); // Every 30 seconds
    }
    
    // Initialize
    loadThemePreference();
    fetchCharacters();
    fetchDefaultSettings();
    connectWebSocket();
    startHeartbeat();
    stopBtn.disabled = true;
});