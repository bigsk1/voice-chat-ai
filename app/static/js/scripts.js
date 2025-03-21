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

    let aiMessageQueue = [];
    let isAISpeaking = false;

    // Fetch and populate characters as soon as page loads
    fetchCharacters();

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

    // Function to populate character select dropdown
    function populateCharacterSelect(characters) {
        characterSelect.innerHTML = '';
        characters.forEach(character => {
            const option = document.createElement('option');
            option.value = character;
            option.textContent = character;
            characterSelect.appendChild(option);
        });
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
        console.log("Received message:", event.data);
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
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
        } else if (data.message) {
            if (data.message.startsWith('You:')) {
                displayMessage(data.message);
            } else {
                aiMessageQueue.push(data.message);
                if (!isAISpeaking) {
                    processQueuedMessages();
                }
            }
        } else if (data.action === "recording_started") {
            micIcon.classList.remove('mic-off');
            micIcon.classList.add('mic-on');
        } else if (data.action === "recording_stopped") {
            micIcon.classList.remove('mic-on');
            micIcon.classList.add('mic-off');
        } else {
            displayMessage(event.data);
        }
    };

    function processQueuedMessages() {
        while (aiMessageQueue.length > 0 && !isAISpeaking) {
            displayMessage(aiMessageQueue.shift());
        }
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
        await fetch('/clear_history', { method: 'POST' });
        console.log("Conversation history cleared.");
    });
    

    messages.addEventListener('scroll', function() {
        if (isAISpeaking) {
            const conversation = document.getElementById('conversation');
            const isScrolledToBottom = conversation.scrollHeight - conversation.clientHeight <= conversation.scrollTop + 1;
            voiceAnimation.style.opacity = isScrolledToBottom ? '1' : '0';
        }
    });

    function setProvider() {
        const selectedProvider = document.getElementById('provider-select').value;
        websocket.send(JSON.stringify({ action: "set_provider", provider: selectedProvider }));
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

    document.getElementById('character-select').addEventListener('change', function() {
        const selectedCharacter = this.value;
        websocket.send(JSON.stringify({ action: "set_character", character: selectedCharacter }));
    });

    document.getElementById('provider-select').addEventListener('change', setProvider);
    document.getElementById('tts-select').addEventListener('change', setTTS);
    document.getElementById('openai-voice-select').addEventListener('change', setOpenAIVoice);
    document.getElementById('openai-model-select').addEventListener('change', setOpenAIModel);
    document.getElementById('ollama-model-select').addEventListener('change', setOllamaModel);
    document.getElementById('xai-model-select').addEventListener('change', setXAIModel);
    document.getElementById('xtts-speed-select').addEventListener('change', setXTTSSpeed);
    document.getElementById('elevenlabs-voice-select').addEventListener('change', setElevenLabsVoice);

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