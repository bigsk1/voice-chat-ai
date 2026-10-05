document.addEventListener("DOMContentLoaded", function() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const websocket = new WebSocket(`${wsProtocol}://${window.location.host}/ws`);
    const themeToggle = document.getElementById('theme-toggle');
    const downloadButton = document.getElementById('download-button');
    const body = document.body;
    const voiceAnimation = document.getElementById('voice-animation');
    const startButton = document.getElementById('start-conversation-btn');
    const pauseAudioButton = document.getElementById('pause-audio-btn');
    const stopButton = document.getElementById('stop-conversation-btn');
    const clearButton = document.getElementById('clear-conversation-btn');
    const messages = document.getElementById('messages');
    const micIcon = document.getElementById('mic-icon');
    const characterSelect = document.getElementById('character-select');

    const providerSelect = document.getElementById('provider-select');
    const modelSelect = document.getElementById('model-select');
    const modelHint = document.getElementById('model-hint');
    const ttsSelect = document.getElementById('tts-select');
    const voiceSelect = document.getElementById('voice-select');
    const voiceHint = document.getElementById('voice-hint');
    const voiceSpeedSelect = document.getElementById('voice-speed-select');
    const transcriptionSelect = document.getElementById('transcription-select');

    const modelActions = {
        openai: 'set_openai_model',
        ollama: 'set_ollama_model',
        xai: 'set_xai_model',
        anthropic: 'set_anthropic_model'
    };
    const voiceActions = {
        openai: 'set_openai_voice',
        elevenlabs: 'set_elevenlabs_voice',
        kokoro: 'set_kokoro_voice',
        xai: 'set_xai_tts_voice',
        typecast: 'set_typecast_voice',
        speechify: 'set_speechify_voice'
    };
    const selectedModels = Object.create(null);
    const selectedVoices = Object.create(null);
    let activeModelProvider = providerSelect.value;
    let activeTTSProvider = ttsSelect.value;
    let modelRequestVersion = 0;
    let voiceRequestVersion = 0;

    let aiMessageQueue = [];
    let isAISpeaking = false;
    let isAudioPaused = false;

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
    

    function sendSetting(action, field, value) {
        if (websocket.readyState === WebSocket.OPEN && action && value) {
            websocket.send(JSON.stringify({ action, [field]: value }));
            return true;
        }
        return false;
    }

    function updateStartAvailability() {
        const hasModel = providerSelect.value && !modelSelect.disabled && modelSelect.value;
        const hasVoice = ttsSelect.value === 'sparktts' || (!voiceSelect.disabled && voiceSelect.value);
        startButton.disabled = websocket.readyState !== WebSocket.OPEN ||
            !ttsSelect.value || !hasModel || !hasVoice;
    }

    function showPlaceholder(select, label) {
        select.replaceChildren(new Option(label, ''));
        select.disabled = true;
        updateStartAvailability();
    }

    function templateOptions(kind, provider) {
        const template = document.getElementById(`${kind}-options-${provider}`);
        return template ? Array.from(template.content.querySelectorAll('option'), option => ({
            id: option.value,
            name: option.textContent,
            disabled: option.disabled
        })) : [];
    }

    function configuredSelection(kind, provider) {
        const selected = kind === 'model' ? selectedModels : selectedVoices;
        const template = document.getElementById(`${kind}-options-${provider}`);
        return selected[provider] || (template && template.dataset.initial) || '';
    }

    function fillSelect(select, options, preferred, emptyLabel, allowConfigured = true) {
        select.replaceChildren();
        const usable = options.filter(option => option.id && !option.disabled);
        if (preferred && allowConfigured && !usable.some(option => option.id === preferred)) {
            options = [...options, { id: preferred, name: `${preferred} (configured)` }];
        }
        for (const item of options) {
            const option = new Option(item.name || item.id, item.id);
            option.disabled = Boolean(item.disabled);
            select.add(option);
        }
        const first = Array.from(select.options).find(option => !option.disabled && option.value);
        if (!first) {
            showPlaceholder(select, emptyLabel);
            return false;
        }
        select.value = preferred && Array.from(select.options).some(
            option => option.value === preferred && !option.disabled
        ) ? preferred : first.value;
        select.disabled = false;
        updateStartAvailability();
        return true;
    }

    function matchOllamaModel(models, preferred) {
        if (!preferred) return '';
        if (models.includes(preferred)) return preferred;
        const base = preferred.replace(/:latest$/, '');
        return models.find(model =>
            model.replace(/:latest$/, '') === base || model.startsWith(`${base}:`)
        ) || '';
    }

    async function fetchVoiceData(path) {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`Voice list request failed: ${response.status}`);
        return response.json();
    }

    async function renderModelSelect() {
        const provider = providerSelect.value;
        const requestVersion = ++modelRequestVersion;
        modelHint.textContent = '';
        if (!provider) {
            showPlaceholder(modelSelect, 'Configure a model provider in .env');
            return;
        }

        const preferred = configuredSelection('model', provider);
        if (provider === 'ollama') {
            showPlaceholder(modelSelect, 'Loading Ollama models…');
            try {
                const response = await fetch('/ollama_models');
                if (!response.ok) throw new Error('Ollama model request failed');
                const data = await response.json();
                if (providerSelect.value !== provider || requestVersion !== modelRequestVersion) return;
                if (data.error) {
                    showPlaceholder(modelSelect, 'Ollama models unavailable');
                    modelHint.textContent = 'Check the Ollama server connection.';
                    return;
                }
                const models = Array.isArray(data.models) ? data.models : [];
                const matched = matchOllamaModel(models, preferred);
                fillSelect(
                    modelSelect,
                    models.sort((a, b) => a.localeCompare(b)).map(id => ({ id, name: id })),
                    matched,
                    'No Ollama models available',
                    false
                );
                if (!models.length) modelHint.textContent = 'Start Ollama and install a model.';
            } catch (error) {
                if (providerSelect.value !== provider || requestVersion !== modelRequestVersion) return;
                console.error('Error fetching Ollama models:', error);
                showPlaceholder(modelSelect, 'Ollama models unavailable');
                modelHint.textContent = 'Check the Ollama server connection.';
            }
        } else {
            fillSelect(
                modelSelect,
                templateOptions('model', provider),
                preferred,
                'No models available'
            );
        }
        if (providerSelect.value === provider && !modelSelect.disabled) {
            selectedModels[provider] = modelSelect.value;
            sendSetting(modelActions[provider], 'model', modelSelect.value);
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
        sendDashboardState();
        updateStartAvailability();
        pauseAudioButton.disabled = true;
    };

    websocket.onclose = function(event) {
        console.log("WebSocket is closed now.");
        startButton.disabled = true;
        pauseAudioButton.disabled = true;
    };

    websocket.onerror = function(event) {
        console.error("WebSocket error observed:", event);
        startButton.disabled = true;
        pauseAudioButton.disabled = true;
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
            resetPauseAudioButton(false);
            showVoiceAnimation();
            setTimeout(processQueuedMessages, 100);
        } else if (data.action === "ai_stop_speaking") {
            isAISpeaking = false;
            resetPauseAudioButton(true);
            hideVoiceAnimation();
            processQueuedMessages();
        } else if (data.action === "audio_paused") {
            setPauseAudioState(true);
        } else if (data.action === "audio_resumed") {
            setPauseAudioState(false);
        } else if (data.action === "conversation_stopped") {
            aiMessageQueue = [];
            isAISpeaking = false;
            resetPauseAudioButton(true);
            hideVoiceAnimation();
            hideListeningIndicator();
            micIcon.classList.remove('mic-on', 'mic-waiting', 'pulse-animation');
            micIcon.classList.add('mic-off');
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
        voiceAnimation.classList.remove('paused');
        voiceAnimation.classList.remove('hidden');
        adjustScrollPosition();
    }

    function hideVoiceAnimation() {
        voiceAnimation.classList.remove('paused');
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

    function setPauseAudioState(paused) {
        isAudioPaused = paused;
        pauseAudioButton.textContent = paused ? 'Resume' : 'Pause';
        pauseAudioButton.disabled = !isAISpeaking;
        voiceAnimation.classList.toggle('paused', paused);
    }

    function resetPauseAudioButton(disabled) {
        isAudioPaused = false;
        pauseAudioButton.textContent = 'Pause';
        pauseAudioButton.disabled = disabled;
        voiceAnimation.classList.remove('paused');
    }

    startButton.addEventListener('click', function() {
        const selectedCharacter = document.getElementById('character-select').value;
        resetPauseAudioButton(true);
        websocket.send(JSON.stringify({ action: "start", character: selectedCharacter }));
        console.log("Start conversation message sent");
    });

    pauseAudioButton.addEventListener('click', function() {
        if (!isAISpeaking) {
            return;
        }

        if (isAudioPaused) {
            websocket.send(JSON.stringify({ action: "resume_audio" }));
            console.log("Resume audio message sent");
        } else {
            websocket.send(JSON.stringify({ action: "pause_audio" }));
            console.log("Pause audio message sent");
        }
    });

    stopButton.addEventListener('click', function() {
        resetPauseAudioButton(true);
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

    function syncProviderSelection() {
        const provider = providerSelect.value;
        if (provider && provider !== providerSelect.dataset.configured &&
            sendSetting('set_provider', 'provider', provider)) {
            providerSelect.dataset.configured = provider;
        }
    }

    function syncTTSSelection() {
        const provider = ttsSelect.value;
        if (provider && provider !== ttsSelect.dataset.configured &&
            sendSetting('set_tts', 'tts', provider)) {
            ttsSelect.dataset.configured = provider;
        }
    }

    function sendDashboardState() {
        syncProviderSelection();
        syncTTSSelection();
        if (!modelSelect.disabled) {
            sendSetting(modelActions[providerSelect.value], 'model', modelSelect.value);
        }
        if (!voiceSelect.disabled) {
            sendSetting(voiceActions[ttsSelect.value], 'voice', voiceSelect.value);
        }
    }

    function setProvider() {
        if (activeModelProvider && modelSelect.value) {
            selectedModels[activeModelProvider] = modelSelect.value;
        }
        activeModelProvider = providerSelect.value;
        syncProviderSelection();
        renderModelSelect();
    }

    function setTTS() {
        if (activeTTSProvider && voiceSelect.value) {
            selectedVoices[activeTTSProvider] = voiceSelect.value;
        }
        activeTTSProvider = ttsSelect.value;
        syncTTSSelection();
        renderVoiceSelect();
    }

    async function renderVoiceSelect() {
        const provider = ttsSelect.value;
        const requestVersion = ++voiceRequestVersion;
        const isCurrent = () => ttsSelect.value === provider &&
            requestVersion === voiceRequestVersion;
        voiceHint.textContent = '';
        if (!provider) {
            showPlaceholder(voiceSelect, 'Configure a TTS provider in .env');
            return;
        }
        if (provider === 'sparktts') {
            showPlaceholder(voiceSelect, 'Uses selected character voice');
            voiceHint.textContent = 'Spark-TTS clones the selected character’s reference audio.';
            return;
        }

        const preferred = configuredSelection('voice', provider);
        let options = [];
        let emptyLabel = 'No voices available';
        showPlaceholder(voiceSelect, 'Loading voices…');
        try {
            if (provider === 'openai') {
                const data = await fetchVoiceData('/openai_tts_voices');
                if (!isCurrent()) return;
                if (data.local) {
                    options = Array.isArray(data.voices) ? data.voices : [];
                    voiceHint.textContent = data.error
                        ? 'Local voice list unavailable; using the configured voice if available.'
                        : 'Voices from your OpenAI-compatible TTS server.';
                } else {
                    options = templateOptions('voice', provider);
                }
            } else if (provider === 'xai') {
                options = templateOptions('voice', provider);
            } else {
                const paths = {
                    elevenlabs: '/elevenlabs_voices',
                    kokoro: '/kokoro_voices',
                    typecast: '/typecast_voices',
                    speechify: '/speechify_voices'
                };
                const data = await fetchVoiceData(paths[provider]);
                if (!isCurrent()) return;
                const voices = Array.isArray(data.voices)
                    ? data.voices
                    : Object.entries(data.voices || {}).map(([name, id]) => ({ id, name }));
                options = voices.map(voice => ({
                    id: voice.id,
                    name: voice.name || voice.id,
                    disabled: String(voice.id).startsWith('separator_')
                }));
                if (data.error) {
                    voiceHint.textContent = 'Voice list unavailable; using the configured voice if available.';
                }
            }
        } catch (error) {
            if (!isCurrent()) return;
            console.error(`Error fetching ${provider} voices:`, error);
            voiceHint.textContent = 'Voice list unavailable; using the configured voice if available.';
            emptyLabel = 'Voices unavailable';
        }

        if (!isCurrent()) return;
        fillSelect(voiceSelect, options, preferred, emptyLabel);
        if (!voiceSelect.disabled) {
            selectedVoices[provider] = voiceSelect.value;
            sendSetting(voiceActions[provider], 'voice', voiceSelect.value);
        }
    }

    function setVoiceSpeed() {
        sendSetting('set_voice_speed', 'speed', voiceSpeedSelect.value);
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
    modelSelect.addEventListener('change', function() {
        selectedModels[providerSelect.value] = modelSelect.value;
        sendSetting(modelActions[providerSelect.value], 'model', modelSelect.value);
        updateStartAvailability();
    });
    voiceSelect.addEventListener('change', function() {
        selectedVoices[ttsSelect.value] = voiceSelect.value;
        sendSetting(voiceActions[ttsSelect.value], 'voice', voiceSelect.value);
        updateStartAvailability();
    });
    voiceSpeedSelect.addEventListener('change', setVoiceSpeed);

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

    renderModelSelect();
    renderVoiceSelect();

    window.addEventListener('pageshow', function(event) {
        if (event.persisted) {
            renderModelSelect();
            renderVoiceSelect();
        }
    });
});
