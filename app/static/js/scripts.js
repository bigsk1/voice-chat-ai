document.addEventListener("DOMContentLoaded", function() {
    const websocket = new WebSocket("ws://localhost:8000/ws");

    websocket.onopen = function(event) {
        console.log("WebSocket is open now.");
    };

    websocket.onclose = function(event) {
        console.log("WebSocket is closed now.");
    };

    websocket.onerror = function(event) {
        console.error("WebSocket error observed:", event);
    };

    websocket.onmessage = function(event) {
        const messages = document.getElementById('messages');
        console.log("Received message:", event.data); // Debugging line

        // Check if the message contains a code block and format it
        let formattedMessage = event.data;
        if (formattedMessage.includes('```')) {
            formattedMessage = formattedMessage.replace(/```(.*?)```/gs, function(match, p1) {
                return `<pre><code>${p1}</code></pre>`;
            });
        }

        // Determine if the message is from the AI or the user
        const userMessagePrefix = "You";
        const messageParts = formattedMessage.split(":");
        if (messageParts.length > 1) {
            const sender = messageParts[0].trim();
            if (sender === userMessagePrefix) {
                messages.innerHTML += `<p class="user-message">${formattedMessage}</p>`;
            } else {
                messages.innerHTML += `<p class="ai-message">${formattedMessage}</p>`;
            }
        } else {
            messages.innerHTML += `<p class="user-message">${formattedMessage}</p>`;
        }
    };

    document.getElementById('start-conversation-btn').addEventListener('click', function() {
        const selectedCharacter = document.getElementById('character-select').value;
        websocket.send(JSON.stringify({ action: "start", character: selectedCharacter }));
    });

    document.getElementById('stop-conversation-btn').addEventListener('click', function() {
        websocket.send(JSON.stringify({ action: "stop" }));
    });

    document.getElementById('clear-conversation-btn').addEventListener('click', function() {
        document.getElementById('messages').innerHTML = '';
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

    function setXTTSSpeed() {
        const selectedSpeed = document.getElementById('xtts-speed-select').value;
        websocket.send(JSON.stringify({ action: "set_xtts_speed", speed: selectedSpeed }));
    }

    function setElevenLabsVoice() {
        const selectedVoice = document.getElementById('elevenlabs-voice-select').value;
        websocket.send(JSON.stringify({ action: "set_elevenlabs_voice", voice: selectedVoice }));
    }

    document.getElementById('provider-select').addEventListener('change', setProvider);
    document.getElementById('tts-select').addEventListener('change', setTTS);
    document.getElementById('openai-voice-select').addEventListener('change', setOpenAIVoice);
    document.getElementById('openai-model-select').addEventListener('change', setOpenAIModel);
    document.getElementById('ollama-model-select').addEventListener('change', setOllamaModel);
    document.getElementById('xtts-speed-select').addEventListener('change', setXTTSSpeed);
    document.getElementById('elevenlabs-voice-select').addEventListener('change', setElevenLabsVoice);
});
