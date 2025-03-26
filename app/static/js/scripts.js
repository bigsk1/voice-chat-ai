document.addEventListener("DOMContentLoaded", function() {
    const websocket = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
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
        try {
            const response = await fetch('/characters');
            if (response.ok) {
                const data = await response.json();
                populateCharacterSelect(data.characters);
            } else {
                console.error('Failed to fetch characters:', response.statusText);
            }
        } catch (error) {
            console.error('Error fetching characters:', error);
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
            option.textContent = character.replace(/_/g, ' '); // Replace all underscores with spaces
            characterSelect.appendChild(option);
        });
        
        // Try to set the default character
        const defaultCharacter = document.querySelector('meta[name="default-character"]')?.getAttribute('content');
        if (defaultCharacter) {
            characterSelect.value = defaultCharacter;
        }
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

    function showVoiceAnimation() {
        voiceAnimation.classList.remove('hidden');
        adjustScrollPosition();
    }

    function hideVoiceAnimation() {
        voiceAnimation.classList.add('hidden');
        adjustScrollPosition();
        processQueuedMessages();
    }

    function adjustScrollPosition() {
        const conversation = document.getElementById('conversation');
        if (isAISpeaking) {
            conversation.scrollTop = conversation.scrollHeight;
        }
    }

    function displayMessage(message, className = '') {
        let formattedMessage = message;
        
        // Strip out <think>...</think> blocks
        formattedMessage = formattedMessage.replace(/<think>[\s\S]*?<\/think>/g, '');
        
        // Format code blocks
        if (formattedMessage.includes('```')) {
            formattedMessage = formattedMessage.replace(/```(.*?)```/gs, function(match, p1) {
                return `<pre><code>${p1}</code></pre>`;
            });
        }
    
        const messageElement = document.createElement('p');
        if (className) {
            messageElement.className = className;
        } else if (formattedMessage.startsWith('You:')) {
            messageElement.className = 'user-message';
            formattedMessage = formattedMessage.replace('You:', '').trim();
        } else {
            messageElement.className = 'ai-message';
        }
        messageElement.innerHTML = formattedMessage;
        messages.appendChild(messageElement);
        adjustScrollPosition();
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

    function setXTTSSpeed() {
        const selectedSpeed = document.getElementById('xtts-speed-select').value;
        websocket.send(JSON.stringify({ action: "set_xtts_speed", speed: selectedSpeed }));
    }

    function setElevenLabsVoice() {
        const selectedVoice = document.getElementById('elevenlabs-voice-select').value;
        websocket.send(JSON.stringify({ action: "set_elevenlabs_voice", voice: selectedVoice }));
    }

    characterSelect.addEventListener('change', function() {
        fetch('/set_character', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character: this.value })
        });
    });

    providerSelect.addEventListener('change', setProvider);
    ttsSelect.addEventListener('change', setTTS);
    openaiVoiceSelect.addEventListener('change', setOpenAIVoice);
    openaiModelSelect.addEventListener('change', setOpenAIModel);
    ollamaModelSelect.addEventListener('change', setOllamaModel);
    xaiModelSelect.addEventListener('change', setXAIModel);
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
});