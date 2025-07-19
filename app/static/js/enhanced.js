document.addEventListener('DOMContentLoaded', () => {
  const micIcon = document.getElementById('mic-icon');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const clearBtn = document.getElementById('clearBtn');
  const themeToggle = document.getElementById('theme-toggle');
  const downloadButton = document.getElementById('download-button');
  const conversation = document.getElementById('conversation');
  const characterSelect = document.getElementById('characterSelect');
  const voiceSelect = document.getElementById('voiceSelect');
  const modelSelect = document.getElementById('modelSelect');
  const ttsModelSelect = document.getElementById('ttsModelSelect');
  const transcriptionModelSelect = document.getElementById('transcriptionModelSelect');
  const apiKeyInput = document.getElementById('openai-api-key');

  function getApiKey() {
    return apiKeyInput ? apiKeyInput.value.trim() : '';
  }

  let mediaRecorder;
  let audioChunks = [];
  let isAISpeaking = false;
  let listeningIndicator = null;

  function displayMessage(message, className = '') {
    const messagesContainer = document.getElementById('messages');
    const el = document.createElement('div');
    if (className) {
      el.className = className;
    } else if (message.startsWith('You:')) {
      el.className = 'user-message';
      message = message.substring(4).trim();
    } else {
      el.className = 'ai-message';
    }
    message.split('\n').forEach((line, idx) => {
      if (idx > 0) el.appendChild(document.createElement('br'));
      el.appendChild(document.createTextNode(line));
    });
    messagesContainer.appendChild(el);
    setTimeout(() => { conversation.scrollTop = conversation.scrollHeight; }, 10);
  }

  function showListeningIndicator(text) {
    hideListeningIndicator();
    const messagesContainer = document.getElementById('messages');
    listeningIndicator = document.createElement('div');
    listeningIndicator.className = 'listening-indicator';
    listeningIndicator.textContent = text;
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
    const wave = document.getElementById('voiceWaveAnimation');
    wave && wave.classList.remove('hidden');
  }

  function hideVoiceWaveAnimation() {
    const wave = document.getElementById('voiceWaveAnimation');
    wave && wave.classList.add('hidden');
  }

  async function startBrowserConversation() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        micIcon.classList.remove('mic-on', 'pulse-animation');
        hideListeningIndicator();
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = [];
        const formData = new FormData();
        formData.append('file', blob, 'speech.webm');
        const headers = {};
        const key = getApiKey();
        if (key) headers['Authorization'] = `Bearer ${key}`;
        const transRes = await fetch('/api/transcribe', { method: 'POST', headers, body: formData });
        const transData = await transRes.json();
        if (transData.text) {
          displayMessage('You: ' + transData.text);
          const chatHeaders = { 'Content-Type': 'application/json', ...headers };
          const chatRes = await fetch('/api/chat', { method: 'POST', headers: chatHeaders, body: JSON.stringify({ text: transData.text }) });
          const chatData = await chatRes.json();
          if (chatData.text) {
            displayMessage(chatData.text);
            const synthRes = await fetch('/api/synthesize', { method: 'POST', headers: chatHeaders, body: JSON.stringify({ text: chatData.text }) });
            const audioBuffer = await synthRes.arrayBuffer();
            const url = URL.createObjectURL(new Blob([audioBuffer], { type: 'audio/wav' }));
            const audio = new Audio(url);
            isAISpeaking = true;
            showVoiceWaveAnimation();
            audio.onended = () => { isAISpeaking = false; hideVoiceWaveAnimation(); };
            audio.play();
          }
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
      };
      mediaRecorder.start();
      micIcon.classList.add('mic-on', 'pulse-animation');
      showListeningIndicator('Listening');
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } catch (err) {
      console.error('Error accessing microphone:', err);
      displayMessage('Microphone access error', 'error-message');
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  function stopBrowserConversation() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  clearBtn.addEventListener('click', () => {
    document.getElementById('messages').innerHTML = '';
    fetch('/clear_history', { method: 'POST' });
  });

  startBtn.addEventListener('click', startBrowserConversation);
  stopBtn.addEventListener('click', stopBrowserConversation);

  // Theme toggle
  function updateThemeIcon() {
    const isDarkMode = document.body.classList.contains('dark-mode');
    themeToggle.innerHTML = isDarkMode
      ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
      : '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
  }

  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    updateThemeIcon();
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
  });

  function loadThemePreference() {
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    document.body.classList.toggle('dark-mode', isDarkMode);
    updateThemeIcon();
  }

  downloadButton.addEventListener('click', () => {
    fetch('/download_enhanced_history')
      .then(res => res.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'conversation_history.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
  });

  function fetchCharacters() {
    fetch('/characters')
      .then(r => r.json())
      .then(data => {
        characterSelect.innerHTML = '';
        data.characters.sort((a,b) => a.localeCompare(b));
        data.characters.forEach(c => {
          const o = document.createElement('option');
          o.value = c;
          o.textContent = c.replace(/_/g, ' ');
          characterSelect.appendChild(o);
        });
      });
  }

  function fetchDefaultSettings() {
    fetch('/enhanced_defaults')
      .then(r => r.json())
      .then(d => {
        setTimeout(() => {
          if (d.character && characterSelect.querySelector(`option[value="${d.character}"]`)) characterSelect.value = d.character;
          if (d.voice && voiceSelect.querySelector(`option[value="${d.voice}"]`)) voiceSelect.value = d.voice;
          if (d.model && modelSelect.querySelector(`option[value="${d.model}"]`)) modelSelect.value = d.model;
          if (d.tts_model && ttsModelSelect.querySelector(`option[value="${d.tts_model}"]`)) ttsModelSelect.value = d.tts_model;
          if (d.transcription_model && transcriptionModelSelect.querySelector(`option[value="${d.transcription_model}"]`)) transcriptionModelSelect.value = d.transcription_model;
        }, 300);
      });
  }

  characterSelect.addEventListener('change', () => {
    const selectedCharacter = characterSelect.value;
    document.getElementById('messages').innerHTML = '';
    fetch('/set_character', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character: selectedCharacter })
    }).then(() => {
      if (selectedCharacter.startsWith('story_') || selectedCharacter.startsWith('game_')) {
        fetch('/get_character_history')
          .then(res => res.json())
          .then(h => {
            if (h.status === 'success' && h.history) {
              h.history.split('\n').forEach(line => {
                if (line.startsWith('User:')) {
                  displayMessage('You: ' + line.substring(5).trim());
                } else if (line.startsWith('Assistant:')) {
                  displayMessage(line.substring(10).trim());
                }
              });
              displayMessage(`Previous conversation history loaded for ${selectedCharacter.replace('_',' ')}. Press Start to continue.`, 'system-message');
            }
          });
      }
    });
  });

  loadThemePreference();
  fetchCharacters();
  fetchDefaultSettings();
  stopBtn.disabled = true;
});

