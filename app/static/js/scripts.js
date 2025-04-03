document.addEventListener("DOMContentLoaded", function() {
    // Use secure WebSocket (wss://) if the page is loaded over HTTPS
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsPort = window.location.protocol === 'https:' ? '8080' : '8000';
    const websocket = new WebSocket(`${wsProtocol}//${window.location.hostname}:${wsPort}/ws`);
    const themeToggle = document.getElementById('theme-toggle');
    const downloadButton = document.getElementById('download-button');
    const body = document.body;
    const voiceAnimation = document.getElementById('voice-animation');
    const startButton = document.getElementById('start-conversation-btn');
    const stopButton = document.getElementById('stop-conversation-btn');
    const clearButton = document.getElementById('clear-conversation-btn');
    const messages = document.getElementById('messages');
    const micIcon = document.getElementById('mic-icon');
    const characterSelect = document.getElementById('character-select');
    const providerSelect = document.getElementById('provider-select');
    const ttsSelect = document.getElementById('tts-select');
    const openaiVoiceSelect = document.getElementById('openai-voice-select');
    const elevenLabsVoiceSelect = document.getElementById('elevenlabs-voice-select');
    const openaiModelSelect = document.getElementById('openai-model-select');
    const ollamaModelSelect = document.getElementById('ollama-model-select');
    const xaiModelSelect = document.getElementById('xai-model-select');
    const xttsSpeedSelect = document.getElementById('xtts-speed-select');
    const transcriptionSelect = document.getElementById('transcription-select');

    let aiMessageQueue = [];
    let isAISpeaking = false;

    // Fetch and populate characters as soon as page loads
    fetchCharacters();
    
    // Fetch Ollama models if that's the current provider
    if (providerSelect.value === 'ollama') {
        fetchOllamaModels();
    }

    // Function to fetch available characters
    async function fetchCharacters() {
        console.log("Fetching characters...");
        try {
            const response = await fetch('/characters');
            if (response.ok) {
                const data = await response.json();
                console.log(`Successfully fetched ${data.characters.length} characters:`, data.characters);
                
                if (data.characters && data.characters.length > 0) {
                    populateCharacterSelect(data.characters);
                } else {
                    console.error('No characters found in response');
                    // Create a fallback option if no characters were returned
                    characterSelect.innerHTML = '';
                    const option = document.createElement('option');
                    option.value = 'default';
                    option.textContent = 'Default Character';
                    characterSelect.appendChild(option);
                }
            } else {
                console.error('Failed to fetch characters:', response.status, response.statusText);
                
                // Try to get more detailed error information
                try {
                    const errorData = await response.text();
                    console.error('Error response:', errorData);
                } catch (parseError) {
                    console.error('Could not parse error response');
                }
                
                // Create a fallback option if fetch failed
                characterSelect.innerHTML = '';
                const option = document.createElement('option');
                option.value = 'default';
                option.textContent = 'Default Character';
                characterSelect.appendChild(option);
            }
        } catch (error) {
            console.error('Error fetching characters:', error);
            
            // Create a fallback option if fetch failed
            characterSelect.innerHTML = '';
            const option = document.createElement('option');
            option.value = 'default';
            option.textContent = 'Default Character';
            characterSelect.appendChild(option);
        }
    }
    
    // Function to fetch available Ollama models
    async function fetchOllamaModels() {
        try {
            const response = await fetch('/ollama_models');
            if (response.ok) {
                const data = await response.json();
                
                if (data.error) {
                    console.warn('Ollama API warning:', data.error);
                }
                
                if (data.models && data.models.length > 0) {
                    populateOllamaModelSelect(data.models);
                } else {
                    console.warn('No Ollama models found');
                }
            } else {
                console.error('Failed to fetch Ollama models:', response.statusText);
            }
        } catch (error) {
            console.error('Error fetching Ollama models:', error);
        }
    }
    
    // Function to populate Ollama model select dropdown
    function populateOllamaModelSelect(models) {
        // Save the current selection
        const currentValue = ollamaModelSelect.value;
        
        // Clear the select element
        ollamaModelSelect.innerHTML = '';
        
        // Sort the models alphabetically
        models.sort((a, b) => a.localeCompare(b));
        
        // Add each model as an option
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            ollamaModelSelect.appendChild(option);
        });
        
        // Try to restore the previous selection
        if (models.includes(currentValue)) {
            ollamaModelSelect.value = currentValue;
        } else if (models.includes('llama3.2')) {
            // Default to llama3.2 if available
            ollamaModelSelect.value = 'llama3.2';
        } else if (models.length > 0) {
            // Otherwise select the first available model
            ollamaModelSelect.value = models[0];
        }
    }

    // Function to populate character select dropdown
    function populateCharacterSelect(characters) {
        characterSelect.innerHTML = '';
        
        // Sort the characters alphabetically
        characters.sort((a, b) => a.localeCompare(b));
        
        characters.forEach(character => {
            const option = document.createElement('option');
            option.value = character;
            
            // Fix the display name to handle potential encoding issues
            let displayName = character;
            try {
                // Replace underscores with spaces and capitalize first letter of each word
                displayName = character.replace(/_/g, ' ')
                    .replace(/\b\w/g, c => c.toUpperCase());
            } catch (e) {
                console.warn(`Error formatting character name: ${character}`, e);
            }
            
            option.textContent = displayName;
            characterSelect.appendChild(option);
        });
        
        // Try to set the default character
        const defaultCharacter = document.querySelector('meta[name="default-character"]')?.getAttribute('content');
        if (defaultCharacter && characters.includes(defaultCharacter)) {
            characterSelect.value = defaultCharacter;
        } else if (characters.length > 0) {
            // Set the first character as default if the specified default isn't available
            characterSelect.value = characters[0];
        }
        
        console.log(`Populated ${characters.length} characters in dropdown`);
    }

    websocket.onopen = function(event) {
        console.log("WebSocket is open now.");
        startButton.disabled = false;
    };

    websocket.onclose = function(event) {
        console.log("WebSocket is closed now.");
        startButton.disabled = true;
    };

    websocket.onerror = function(event) {
        console.error("WebSocket error observed:", event);
        startButton.disabled = true;
    };

    websocket.onmessage = function(event) {
        let data;
        
        // First check if the data is already a string that should be displayed directly
        if (typeof event.data === 'string' && !event.data.startsWith('{') && !event.data.startsWith('[')) {
            displayMessage(event.data);
            return;
        }
        
        // Try to parse as JSON
        try {
            data = JSON.parse(event.data);
            console.log("Received message:", data);
        } catch (e) {
            console.log("Received non-JSON message:", event.data);
            // Don't treat this as an error if it's just a plain text message
            if (event.data && typeof event.data === 'string') {
                displayMessage(event.data);
                return;
            }
            console.error("Error parsing JSON:", e);
            data = { message: event.data };
        }

        if (data.action === "ai_start_speaking") {
            isAISpeaking = true;
            showVoiceAnimation();
            setTimeout(processQueuedMessages, 100);
        } else if (data.action === "ai_stop_speaking") {
            isAISpeaking = false;
            hideVoiceAnimation();
            processQueuedMessages();
        } else if (data.action === "error") {
            console.error("Error from server:", data.message);
            displayMessage(data.message, 'error-message');
        } else if (data.action === "waiting_for_speech") {
            // Show the listening indicator for waiting_for_speech action
            showListeningIndicator();
        } else if (data.message) {
            if (data.message.startsWith('You:')) {
                displayMessage(data.message);
                // Hide the listening indicator when user's message is received
                hideListeningIndicator();
            } else {
                aiMessageQueue.push(data.message);
                if (!isAISpeaking) {
                    processQueuedMessages();
                }
            }
        } else if (data.action === "recording_started") {
            micIcon.classList.remove('mic-off');
            micIcon.classList.add('mic-on');
            micIcon.classList.add('pulse-animation');
            // Show the listening indicator when recording starts
            showListeningIndicator();
        } else if (data.action === "recording_stopped") {
            micIcon.classList.remove('mic-on');
            micIcon.classList.remove('pulse-animation');
            micIcon.classList.add('mic-off');
            // Hide the listening indicator when recording stops
            hideListeningIndicator();
        }
    };

    function processQueuedMessages() {
        while (aiMessageQueue.length > 0 && !isAISpeaking) {
            displayMessage(aiMessageQueue.shift());
        }
    }

    // Function to create and show the listening indicator with animated dots
    function showListeningIndicator() {
        // Remove any existing listening indicator
        hideListeningIndicator();
        
        // Create the listening indicator
        const listeningIndicator = document.createElement('div');
        listeningIndicator.className = "listening-indicator";
        listeningIndicator.id = "listening-indicator";
        
        // Add the text
        listeningIndicator.textContent = "Listening";
        
        // Create dots container
        const dotsContainer = document.createElement('div');
        dotsContainer.className = "listening-dots";
        
        // Create three animated dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = "dot";
            dot.style.animationDelay = `${i * 0.2}s`;
            dotsContainer.appendChild(dot);
        }
        
        // Add dots to the indicator
        listeningIndicator.appendChild(dotsContainer);
        
        // Add indicator to messages
        messages.appendChild(listeningIndicator);
        adjustScrollPosition();
        
        // Also add animation to mic icon
        micIcon.classList.add('mic-waiting');
    }

    // Function to hide the listening indicator
    function hideListeningIndicator() {
        const existingIndicator = document.getElementById('listening-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        // Remove animation from mic icon
        micIcon.classList.remove('mic-waiting');
    }

    function adjustScrollPosition() {
        const conversation = document.getElementById('conversation');
        if (isAISpeaking) {
            // Add some buffer space to ensure animation is visible
            conversation.scrollTop = conversation.scrollHeight - 250;
        } else {
            // When not speaking, scroll to bottom but leave some space
            conversation.scrollTop = conversation.scrollHeight - 100;
        }
    }

    function showVoiceAnimation() {
        voiceAnimation.classList.remove('hidden');
        adjustScrollPosition();
    }

    function hideVoiceAnimation() {
        voiceAnimation.classList.add('hidden');
        // Only scroll back to bottom with buffer after animation is hidden
        setTimeout(() => {
            // Short delay to ensure smooth transition
            adjustScrollPosition();
            processQueuedMessages();
        }, 100);
    }

    function displayMessage(message, className = '') {
        let formattedMessage = message;
        
        // Strip out <think>...</think> blocks
        formattedMessage = formattedMessage.replace(/<think>[\s\S]*?<\/think>/g, '');
        
        const messageElement = document.createElement('div');
        if (className) {
            messageElement.className = className;
        } else if (formattedMessage.startsWith('You:')) {
            messageElement.className = 'user-message';
            formattedMessage = formattedMessage.replace('You:', '').trim();
        } else {
            messageElement.className = 'ai-message';
        }
        
        // Handle code blocks
        if (formattedMessage.includes('```')) {
            // Split by code blocks and process each segment
            let segments = formattedMessage.split(/(```(?:.*?)```)/gs);
            segments.forEach(segment => {
                if (segment.startsWith('```') && segment.endsWith('```')) {
                    // This is a code block
                    const codeContent = segment.slice(3, -3).trim();
                    const preElement = document.createElement('pre');
                    const codeElement = document.createElement('code');
                    codeElement.textContent = codeContent;
                    preElement.appendChild(codeElement);
                    messageElement.appendChild(preElement);
                } else if (segment.trim()) {
                    // This is regular text
                    // Handle newlines in regular text
                    segment.split('\n').forEach((line, index) => {
                        if (index > 0) {
                            messageElement.appendChild(document.createElement('br'));
                        }
                        messageElement.appendChild(document.createTextNode(line));
                    });
                }
            });
        } else {
            // Handle newlines in the message (no code blocks)
            if (formattedMessage.includes('\n')) {
                formattedMessage.split('\n').forEach((line, index) => {
                    if (index > 0) {
                        messageElement.appendChild(document.createElement('br'));
                    }
                    messageElement.appendChild(document.createTextNode(line));
                });
            } else {
                messageElement.textContent = formattedMessage;
            }
        }
        
        messages.appendChild(messageElement);
        // Adjust scroll position whenever a message is added
        setTimeout(() => adjustScrollPosition(), 10);
    }

    startButton.addEventListener('click', function() {
        const selectedCharacter = document.getElementById('character-select').value;
        websocket.send(JSON.stringify({ action: "start", character: selectedCharacter }));
        console.log("Start conversation message sent");
    });

    stopButton.addEventListener('click', function() {
        websocket.send(JSON.stringify({ action: "stop" }));
        console.log("Stop conversation message sent");
    });

    clearButton.addEventListener('click', async function() {
        messages.innerHTML = '';
        try {
            const response = await fetch('/clear_history', { method: 'POST' });
            const data = await response.json();
            console.log("Conversation history cleared.");
            // Add a confirmation message
            displayMessage("Conversation history has been cleared.", "system-message");
        } catch (error) {
            console.error("Error clearing history:", error);
            displayMessage("Error clearing conversation history", "error-message");
        }
    });
    

    messages.addEventListener('scroll', function() {
        if (isAISpeaking) {
            const conversation = document.getElementById('conversation');
            const isScrolledToBottom = conversation.scrollHeight - conversation.clientHeight <= conversation.scrollTop + 1;
            voiceAnimation.style.opacity = isScrolledToBottom ? '1' : '0';
        }
    });

    function setProvider() {
        const provider = providerSelect.value;
        websocket.send(JSON.stringify({ action: "set_provider", provider: provider }));
        
        // When Ollama is selected, fetch available models
        if (provider === 'ollama') {
            fetchOllamaModels();
        }
    }

    function setTTS() {
        const selectedTTS = document.getElementById('tts-select').value;
        websocket.send(JSON.stringify({ action: "set_tts", tts: selectedTTS }));
    }

    function setOpenAIVoice() {
        const selectedVoice = document.getElementById('openai-voice-select').value;
        websocket.send(JSON.stringify({ action: "set_openai_voice", voice: selectedVoice }));
    }

    function setOpenAIModel() {
        const selectedModel = document.getElementById('openai-model-select').value;
        websocket.send(JSON.stringify({ action: "set_openai_model", model: selectedModel }));
    }

    function setOllamaModel() {
        const selectedModel = document.getElementById('ollama-model-select').value;
        websocket.send(JSON.stringify({ action: "set_ollama_model", model: selectedModel }));
    }

    function setXAIModel() {
        const selectedModel = document.getElementById('xai-model-select').value;
        websocket.send(JSON.stringify({ action: "set_xai_model", model: selectedModel }));
    }

    function setAnthropicModel() {
        const selectedModel = document.getElementById('anthropic-model-select').value;
        websocket.send(JSON.stringify({ action: "set_anthropic_model", model: selectedModel }));
    }

    function setXTTSSpeed() {
        const selectedSpeed = document.getElementById('xtts-speed-select').value;
        websocket.send(JSON.stringify({ action: "set_xtts_speed", speed: selectedSpeed }));
    }

    function setElevenLabsVoice() {
        const selectedVoice = document.getElementById('elevenlabs-voice-select').value;
        websocket.send(JSON.stringify({ action: "set_elevenlabs_voice", voice: selectedVoice }));
    }

    characterSelect.addEventListener('change', function() {
        const selectedCharacter = this.value;
        console.log(`Character selected: ${selectedCharacter}`);
        
        // Clear existing conversation display
        messages.innerHTML = '';
        
        // Set the selected character
        fetch('/set_character', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ character: selectedCharacter })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Character set response:', data);
            
            // Check if this is a story/game character and fetch history
            if (selectedCharacter.startsWith('story_') || selectedCharacter.startsWith('game_')) {
                // Fetch history for this character
                fetch('/get_character_history')
                    .then(response => response.json())
                    .then(historyData => {
                        if (historyData.status === 'success' && historyData.history) {
                            // Display the history
                            const historyLines = historyData.history.split('\n');
                            let currentSpeaker = null;
                            let currentMessage = '';
                            
                            // Process each line
                            historyLines.forEach(line => {
                                if (line.startsWith('User:')) {
                                    // Display previous message if exists
                                    if (currentSpeaker && currentMessage) {
                                        if (currentSpeaker === 'User') {
                                            displayMessage(`You: ${currentMessage}`);
                                        } else {
                                            displayMessage(currentMessage);
                                        }
                                    }
                                    
                                    // Start new user message
                                    currentSpeaker = 'User';
                                    currentMessage = line.substring(5).trim();
                                } else if (line.startsWith('Assistant:')) {
                                    // Display previous message if exists
                                    if (currentSpeaker && currentMessage) {
                                        if (currentSpeaker === 'User') {
                                            displayMessage(`You: ${currentMessage}`);
                                        } else {
                                            displayMessage(currentMessage);
                                        }
                                    }
                                    
                                    // Start new assistant message
                                    currentSpeaker = 'Assistant';
                                    currentMessage = line.substring(10).trim();
                                } else if (line.trim() && currentSpeaker) {
                                    // Continuation of current message
                                    currentMessage += '\n' + line;
                                }
                            });
                            
                            // Display the last message
                            if (currentSpeaker && currentMessage) {
                                if (currentSpeaker === 'User') {
                                    displayMessage(`You: ${currentMessage}`);
                                } else {
                                    displayMessage(currentMessage);
                                }
                            }
                            
                            // Add a note that this is previous history
                            displayMessage(`Previous conversation history loaded for ${selectedCharacter.replace('_', ' ')}. Press Start to continue.`, "system-message");
                            
                            // Scroll to bottom to show latest messages
                            conversation.scrollTop = conversation.scrollHeight;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching character history:', error);
                    });
            }
        })
        .catch(error => console.error('Error setting character:', error));
    });

    providerSelect.addEventListener('change', setProvider);
    ttsSelect.addEventListener('change', setTTS);
    openaiVoiceSelect.addEventListener('change', setOpenAIVoice);
    openaiModelSelect.addEventListener('change', setOpenAIModel);
    ollamaModelSelect.addEventListener('change', setOllamaModel);
    xaiModelSelect.addEventListener('change', setXAIModel);
    const anthropicModelSelect = document.getElementById('anthropic-model-select');
    if (anthropicModelSelect) {
        anthropicModelSelect.addEventListener('change', setAnthropicModel);
    }
    xttsSpeedSelect.addEventListener('change', setXTTSSpeed);
    elevenLabsVoiceSelect.addEventListener('change', setElevenLabsVoice);

    transcriptionSelect.addEventListener('change', function() {
        fetch('/set_transcription_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: this.value })
        });
    });

    async function downloadHistory() {
        const response = await fetch('/download_history');
        if (response.status === 200) {
            const historyText = await response.text();
            const blob = new Blob([historyText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'conversation_history.txt';
            a.click();
            URL.revokeObjectURL(url);
        } else {
            alert("Failed to download conversation history.");
        }
    }

    downloadButton.addEventListener('click', downloadHistory);
    
    // Theme toggle functionality
    function setDarkModeDefault() {
        const isDarkMode = localStorage.getItem('darkMode');
        if (isDarkMode === null) {
            body.classList.add('dark-mode');
        } else {
            body.classList.toggle('dark-mode', isDarkMode === 'true');
        }
        updateThemeIcon();
    }

    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-mode');
        updateThemeIcon();
        saveThemePreference();
    });

    function updateThemeIcon() {
        const isDarkMode = body.classList.contains('dark-mode');
        themeToggle.innerHTML = isDarkMode 
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    }

    function saveThemePreference() {
        const isDarkMode = body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isDarkMode);
    }

    function loadThemePreference() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        body.classList.toggle('dark-mode', isDarkMode);
        updateThemeIcon();
    }

    loadThemePreference();
    setDarkModeDefault();

    // Initialize audio bridge status checking
    let lastStatusCheckTime = 0;
    const STATUS_CHECK_MIN_INTERVAL = 180000; // Minimum 3 minutes between checks
    
    function checkAudioBridgeStatus(force = false) {
        // Throttle checks to prevent excessive requests
        const now = Date.now();
        if (!force && now - lastStatusCheckTime < STATUS_CHECK_MIN_INTERVAL) {
            return;
        }
        
        lastStatusCheckTime = now;
        
        // Check for mini display first
        const audioBridgeMini = document.querySelector('.audio-bridge-mini');
        const audioBridgeSection = document.getElementById('audio-bridge-section');
        const miniStatus = document.getElementById('audio-bridge-status-mini');
        
        if (!audioBridgeMini || !audioBridgeSection || !miniStatus) {
            return;
        }
        
        // Get the status elements
        const bridgeStatus = document.getElementById('audio-bridge-status');
        const clientsCount = document.getElementById('audio-bridge-clients');
        const bridgeDetails = document.getElementById('audio-bridge-details');
        const actionsSection = document.querySelector('.audio-bridge-actions');
        const miniButton = document.getElementById('init-bridge-btn-mini');
        
        // Make sure the elements exist
        if (!bridgeStatus || !clientsCount || !bridgeDetails) {
            console.warn('Audio bridge status elements not found');
            return;
        }
        
        // Show checking state only for forced checks
        if (force) {
            bridgeStatus.textContent = 'Checking...';
            bridgeStatus.className = 'status-checking';
            miniStatus.textContent = 'Checking...';
            miniStatus.className = 'status-checking';
        }
        
        // Fetch the status from the server
        fetch('/audio-bridge/status')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Log only occasionally to reduce console spam
                if (Math.random() < 0.1 || force) {
                    console.log("Audio bridge status:", data);
                }
                
                // Update the status indicators
                if (data.status === "active") {
                    bridgeStatus.textContent = "Enabled";
                    bridgeStatus.className = "status-enabled";
                    
                    // Check if we have a client ID stored in localStorage - this means we're connected
                    const storedClientId = localStorage.getItem('audio_bridge_client_id');
                    
                    if (storedClientId) {
                        // We have a client ID, so we should show as connected regardless of active clients
                        miniStatus.textContent = "Connected";
                        miniStatus.className = "status-enabled";
                    } else {
                        // No client ID, show as ready or connected based on active clients
                        miniStatus.textContent = data.active_clients > 0 ? "Connected" : "Ready";
                        miniStatus.className = data.active_clients > 0 ? "status-enabled" : "status-warning";
                    }
                } else {
                    bridgeStatus.textContent = "Disabled";
                    bridgeStatus.className = "status-disabled";
                    miniStatus.textContent = "Disabled";
                    miniStatus.className = "status-disabled";
                }
                
                // Update mini button
                if (miniButton) {
                    const storedClientId = localStorage.getItem('audio_bridge_client_id');
                    
                    if (storedClientId || data.active_clients > 0) {
                        miniButton.textContent = "View";
                    } else {
                        miniButton.textContent = "Connect";
                    }
                }
                
                // Update clients count
                clientsCount.textContent = `Active Clients: ${data.active_clients}`;
                
                // Update details section with more information
                let detailsHtml = `<strong>Status:</strong> ${data.status}<br>`;
                detailsHtml += `<strong>Total Clients:</strong> ${data.total_clients}<br>`;
                detailsHtml += `<strong>Active Clients:</strong> ${data.active_clients}<br>`;
                
                if (data.active_clients > 0) {
                    detailsHtml += `<div class="alert alert-success mt-2">Audio bridge is active and receiving audio!</div>`;
                } else if (data.status === "active") {
                    detailsHtml += `<div class="alert alert-warning mt-2">No active clients. Please initialize the audio bridge.</div>`;
                } else {
                    detailsHtml += `<div class="alert alert-secondary mt-2">Audio bridge is disabled. Set ENABLE_AUDIO_BRIDGE=true in your .env file to use this feature.</div>`;
                }
                
                bridgeDetails.innerHTML = detailsHtml;
                
                // Update the actions section
                if (actionsSection) {
                    const storedClientId = localStorage.getItem('audio_bridge_client_id');
                    const isConnected = storedClientId || data.active_clients > 0;
                    
                    if (isConnected) {
                        // Show test and disconnect options when connected
                        actionsSection.innerHTML = `
                            <button id="test-bridge-btn" class="btn btn-primary">Test Connection</button>
                        `;
                    } else if (data.status === "active") {
                        // Show initialize button when not connected but bridge is active
                        actionsSection.innerHTML = `
                            <button id="init-bridge-btn" class="btn btn-primary">Connect Bridge</button>
                        `;
                    } else {
                        // Show disabled message when bridge is inactive
                        actionsSection.innerHTML = `
                            <div class="alert alert-secondary">Bridge is disabled. Set ENABLE_AUDIO_BRIDGE=true in your .env file.</div>
                        `;
                    }
                    
                    // Add event listeners to buttons
                    setupAudioBridgeButtons();
                }
            })
            .catch(error => {
                console.error('Error checking audio bridge status:', error);
                bridgeStatus.textContent = 'Error';
                bridgeStatus.className = 'status-error';
                miniStatus.textContent = 'Error';
                miniStatus.className = 'status-error';
                clientsCount.textContent = 'Unknown';
                bridgeDetails.innerHTML = `<div class="alert alert-danger">Error checking audio bridge status: ${error.message}</div>`;
                
                // Show reconnect option
                if (actionsSection) {
                    actionsSection.innerHTML = `
                        <button id="init-bridge-btn" class="btn btn-primary">Initialize Bridge</button>
                        <button id="retry-status-btn" class="btn btn-primary">Retry</button>
                    `;
                    
                    // Add event listeners to buttons
                    setupAudioBridgeButtons();
                }
            });
    }
    
    // Setup event listeners for audio bridge buttons
    function setupAudioBridgeButtons() {
        // Main panel buttons
        const initBtn = document.getElementById('init-bridge-btn');
        if (initBtn) {
            initBtn.addEventListener('click', function() {
                // Instead of opening a new tab, show the dialog
                showConnectDialog();
            });
        }
        
        const testBtn = document.getElementById('test-bridge-btn');
        if (testBtn) {
            testBtn.addEventListener('click', function() {
                // Show test dialog instead of connect dialog
                showTestDialog();
            });
        }
        
        const retryBtn = document.getElementById('retry-status-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', checkAudioBridgeStatus);
        }
        
        // Mini display button
        const miniBridgeBtn = document.getElementById('init-bridge-btn-mini');
        if (miniBridgeBtn) {
            miniBridgeBtn.addEventListener('click', function() {
                // Get current status
                fetch('/audio-bridge/status')
                    .then(response => response.json())
                    .then(data => {
                        const storedClientId = localStorage.getItem('audio_bridge_client_id');
                        
                        if (storedClientId || data.active_clients > 0) {
                            // If already connected, just show panel
                            toggleAudioBridgePanel(true);
                        } else if (data.status === 'active') {
                            // If status is active but not connected, try to connect
                            showConnectDialog();
                        } else {
                            // If disabled, just show panel
                            toggleAudioBridgePanel(true);
                        }
                    })
                    .catch(error => {
                        console.error('Error checking status:', error);
                        toggleAudioBridgePanel(true);
                    });
            });
        }
        
        // Close panel button
        const closeBtn = document.getElementById('close-bridge-panel');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                const panel = document.getElementById('audio-bridge-section');
                if (panel) {
                    panel.style.display = 'none';
                }
            });
        }
    }
    
    // Show test dialog for already connected clients
    function showTestDialog() {
        // Remove any existing dialog first to prevent duplicates
        const existingDialog = document.querySelector('.audio-bridge-confirm-dialog');
        if (existingDialog) {
            document.body.removeChild(existingDialog);
        }
        
        // Create a custom dialog
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'audio-bridge-confirm-dialog';
        confirmDialog.innerHTML = `
            <div class="confirm-dialog-content">
                <h3>Audio Bridge Options</h3>
                <p>You are connected to the audio bridge. What would you like to do?</p>
                <div class="dialog-buttons">
                    <button id="goto-test-page-btn" class="btn btn-primary">View Test Page</button>
                    <button id="disconnect-btn" class="btn">Disconnect</button>
                    <button id="cancel-test-btn" class="btn">Cancel</button>
                </div>
            </div>
        `;
        document.body.appendChild(confirmDialog);
        
        // Add event listeners to the buttons
        document.getElementById('goto-test-page-btn').addEventListener('click', function() {
            // Remove the dialog
            document.body.removeChild(confirmDialog);
            
            // Open the test page in the same tab instead of a new tab
            window.location.href = '/audio-bridge-test';
        });
        
        document.getElementById('disconnect-btn').addEventListener('click', function() {
            // Remove the dialog
            document.body.removeChild(confirmDialog);
            
            // Disconnect from audio bridge
            if (window.audioBridge) {
                window.audioBridge.disconnect().then(() => {
                    localStorage.removeItem('audio_bridge_client_id');
                    checkAudioBridgeStatus(true); // Update status to show disconnected
                    alert('Disconnected from audio bridge.');
                });
            }
        });
        
        document.getElementById('cancel-test-btn').addEventListener('click', function() {
            // Remove the dialog
            document.body.removeChild(confirmDialog);
        });
    }
    
    // Show connection dialog
    function showConnectDialog() {
        // Check if already connected first
        const storedClientId = localStorage.getItem('audio_bridge_client_id');
        if (storedClientId) {
            // If already connected, show test dialog instead
            showTestDialog();
            return;
        }
        
        // Remove any existing dialog first to prevent duplicates
        const existingDialog = document.querySelector('.audio-bridge-confirm-dialog');
        if (existingDialog) {
            document.body.removeChild(existingDialog);
        }
        
        // Create a custom dialog
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'audio-bridge-confirm-dialog';
        confirmDialog.innerHTML = `
            <div class="confirm-dialog-content">
                <h3>Connect to Audio Bridge</h3>
                <p>Would you like to connect to the audio bridge? This requires microphone permission.</p>
                <div class="dialog-buttons">
                    <button id="confirm-connect-btn" class="btn btn-primary">Connect</button>
                    <button id="cancel-connect-btn" class="btn">Cancel</button>
                </div>
            </div>
        `;
        document.body.appendChild(confirmDialog);
        
        // Add event listeners to the buttons
        document.getElementById('confirm-connect-btn').addEventListener('click', function() {
            // Remove the dialog
            document.body.removeChild(confirmDialog);
            
            // Request microphone permission and initialize
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({audio: true})
                    .then(stream => {
                        console.log('Microphone permission granted, initializing bridge...');
                        // Stop the stream we just used for permission
                        stream.getTracks().forEach(track => track.stop());
                        
                        // Initialize the bridge
                        if (window.audioBridge) {
                            window.audioBridge.initialize()
                                .then(success => {
                                    if (success) {
                                        console.log('Audio bridge initialized from dashboard!');
                                        checkAudioBridgeStatus(true); // Update status display
                                        
                                        // Show the panel
                                        toggleAudioBridgePanel(true);
                                    } else {
                                        alert('Failed to initialize audio bridge. Please try again.');
                                    }
                                });
                        } else {
                            alert('Audio bridge not available. Please refresh the page and try again.');
                        }
                    })
                    .catch(err => {
                        console.warn('Microphone permission denied:', err);
                        alert('Microphone permission is required for the audio bridge to work.');
                    });
            } else {
                alert('Your browser does not support microphone access.');
            }
        });
        
        document.getElementById('cancel-connect-btn').addEventListener('click', function() {
            // Remove the dialog
            document.body.removeChild(confirmDialog);
            // Show the panel anyway
            toggleAudioBridgePanel(true);
        });
    }

    function toggleAudioBridgePanel(forceShow) {
        const panel = document.getElementById('audio-bridge-section');
        if (panel) {
            if (forceShow) {
                panel.style.display = 'block';
                checkAudioBridgeStatus(true); // Refresh status when showing panel, force update
            } else if (panel.style.display === 'block') {
                panel.style.display = 'none';
            } else {
                panel.style.display = 'block';
                checkAudioBridgeStatus(true); // Refresh status when showing panel, force update
            }
        }
    }

    // Check status on page load
    setTimeout(() => checkAudioBridgeStatus(true), 1000); // Initial check with 1 second delay
    
    // Setup audio bridge buttons
    setupAudioBridgeButtons();

    // Check status periodically but much less frequently (5 minutes instead of 2 minutes)
    let statusInterval = setInterval(() => checkAudioBridgeStatus(), 300000);
    
    // Reduce polling frequency when the page is not active
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            // Page is hidden, clear the frequent interval
            clearInterval(statusInterval);
            // Check less frequently (once every 10 minutes)
            statusInterval = setInterval(() => checkAudioBridgeStatus(), 600000);
        } else {
            // Page is visible again, resume more frequent checks
            clearInterval(statusInterval);
            statusInterval = setInterval(() => checkAudioBridgeStatus(), 300000);
            // Check when page becomes visible only if we haven't checked recently
            const now = Date.now();
            if (now - lastStatusCheckTime > STATUS_CHECK_MIN_INTERVAL) {
                checkAudioBridgeStatus(true);
            }
        }
    });

    // Show dialog with audio bridge status
    function showAudioBridgeStatus(statusData) {
        // Create the panel if it doesn't exist
        let panel = document.querySelector('.audio-bridge-panel');
        if (!panel) {
            panel = document.createElement('div');
            panel.className = 'audio-bridge-panel';
            panel.innerHTML = `
                <div class="audio-bridge-status-header">
                    <h3>Audio Bridge Status</h3>
                    <button class="close-btn" id="close-status-btn" aria-label="Close">×</button>
                </div>
                <div class="audio-bridge-status-content">
                    <div class="status-row">
                        <span class="status-label">Status:</span>
                        <span id="status-value" class="status-value status-enabled">Enabled</span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Clients:</span>
                        <span id="clients-value" class="status-value">Active Clients: 0</span>
                    </div>
                    
                    <div class="status-details" id="status-details">
                        <div class="status-row no-border">
                            <span class="status-label">Status:</span> 
                            <span id="detail-status" class="status-value">active</span>
                        </div>
                        <div class="status-row no-border">
                            <span class="status-label">Total Clients:</span> 
                            <span id="detail-total-clients" class="status-value">0</span>
                        </div>
                        <div class="status-row no-border">
                            <span class="status-label">Active Clients:</span> 
                            <span id="detail-active-clients" class="status-value">0</span>
                        </div>
                    </div>
                    
                    <div id="status-message" class="alert"></div>
                    
                    <div class="audio-bridge-actions">
                        <button id="test-bridge-btn" class="btn btn-primary">Test Connection</button>
                    </div>
                </div>
            `;
            document.body.appendChild(panel);
            
            // Add event listener to close button
            document.getElementById('close-status-btn').addEventListener('click', function() {
                panel.style.display = 'none';
            });
            
            // Add event listener to test bridge button
            document.getElementById('test-bridge-btn').addEventListener('click', function() {
                panel.style.display = 'none';
                
                // Check for stored client ID to determine if we're connected
                const clientId = localStorage.getItem('audio_bridge_client_id');
                if (clientId && statusData.active_clients > 0) {
                    // We're connected, show test dialog
                    showTestDialog();
                } else {
                    // Not connected, show connect dialog
                    showConnectDialog();
                }
            });
        }
        
        // Update status information
        document.getElementById('status-value').textContent = statusData.status === 'active' ? 'Enabled' : 'Disabled';
        document.getElementById('status-value').className = statusData.status === 'active' ? 'status-value status-enabled' : 'status-value status-disabled';
        
        document.getElementById('clients-value').textContent = `Active Clients: ${statusData.active_clients}`;
        
        document.getElementById('detail-status').textContent = statusData.status;
        document.getElementById('detail-total-clients').textContent = statusData.total_clients;
        document.getElementById('detail-active-clients').textContent = statusData.active_clients;
        
        // Show appropriate message based on status
        const statusMessage = document.getElementById('status-message');
        
        if (statusData.status === 'active') {
            if (statusData.active_clients > 0) {
                statusMessage.className = 'alert alert-success';
                statusMessage.innerHTML = `
                    Audio bridge is active and receiving audio!
                `;
            } else {
                statusMessage.className = 'alert alert-warning';
                statusMessage.innerHTML = `
                    No active clients. Please initialize the audio bridge.
                `;
            }
        } else {
            statusMessage.className = 'alert alert-danger';
            statusMessage.innerHTML = `
                Audio bridge is disabled. Please check server settings.
            `;
        }
        
        // Display the panel
        panel.style.display = 'block';
    }
});