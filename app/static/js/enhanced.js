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
  let loopCount   = 0;
  const MAX_LOOPS = 5;
  let isAutoLoop  = false;
  startBtn.onclick = () => {
    loopCount = 0;          // ここでリセット
    isAutoLoop = true;
    startBrowserConversation();        // ← こっちを呼ぶ
    startBtn.disabled = true;
    stopBtn.disabled  = false;
  };
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
  /**
   * ユーザ入力テキストを /api/chat → /api/synthesize に流して
   * AI の応答音声を再生する
   */
  async function sendChatAndSynthesize(userText) {
    const key = getApiKey();
    const headers = { 'Content-Type': 'application/json' };
    if (key) headers['Authorization'] = `Bearer ${key}`;

    // 1) チャット問い合わせ
    const chatRes = await fetch('/api/chat', {
      method: 'POST',
      headers,
      body: JSON.stringify({ text: userText })
    });
    if (!chatRes.ok) {
      console.error('[Enhanced] /api/chat エラー:', await chatRes.text());
      return;
    }
    const chatData = await chatRes.json();
    if (!chatData.text) return;
    displayMessage(chatData.text);

    // 2a) TTS エンジンを切り替え
    const ttsEngine = document.getElementById('ttsModelSelect').value;
    if (ttsEngine === 'web-speech') {
      // クライアント側 Web Speech API で再生
      const utter = new SpeechSynthesisUtterance(chatData.text);
      utter.lang = 'ja-JP';
      utter.onend = () => {
        // 自動ループ制御
        if (isAutoLoop && ++loopCount < MAX_LOOPS) {
          startBrowserConversation();
        } else if (isAutoLoop) {
          displayMessage(
            `自動対話は最大${MAX_LOOPS}回に達したため終了しました…`,
            'system-message'
          );
          isAutoLoop = false;
          startBtn.disabled = false;
          stopBtn.disabled  = true;
          hideListeningIndicator();
        }
      };
      speechSynthesis.speak(utter);
      return;  // 以降のサーバー TTS 呼び出しをスキップ
    }

    // 2) 音声合成
    const synthRes = await fetch('/api/synthesize', {
      method: 'POST',
      headers,
      body: JSON.stringify({ text: chatData.text })
    });
    if (!synthRes.ok) {
      console.error('[Enhanced] /api/synthesize エラー:', await synthRes.text());
      return;
    }
    const audioBuffer = await synthRes.arrayBuffer();
    const url = URL.createObjectURL(new Blob([audioBuffer], { type: 'audio/wav' }));
    const audio = new Audio(url);
    audio.onended = () => {
      isAISpeaking = false;
      hideVoiceWaveAnimation();

      // ── 自動ループ制御 ──
      if (isAutoLoop) {
        loopCount += 1;
        console.log(`[AutoLoop] 回数: ${loopCount}/${MAX_LOOPS}`);
        if (loopCount < MAX_LOOPS) {
          console.log('[AutoLoop] 次のリスンを開始');
          startBrowserConversation();
        } else {
          console.log(`[AutoLoop] 最大回数(${MAX_LOOPS})到達。自動ループ終了`);
          displayMessage(
            `自動対話は最大${MAX_LOOPS}回に達したため終了しました。放置による無駄な消費を防ぐため会話を終了します。`,
            'system-message'
          );
          isAutoLoop = false;
          startBtn.disabled = false;
          stopBtn.disabled  = true;
          hideListeningIndicator();
        }
      }
      // ───────────────────
    };
    audio.play();

  }
  async function startBrowserConversation() {
    // ① どの文字起こしエンジンを使うか分岐
    const engine = transcriptionModelSelect.value;
    if (engine === 'web-speech') {
      // ────────────────
      // Web Speech API パス
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) {
        displayMessage('このブラウザはWeb Speech APIに対応していません', 'error-message');
        return;
      }
      const recog = new SpeechRec();
      recog.continuous = false;
      recog.interimResults = false;
      recog.lang = 'ja-JP';

      recog.onstart = () => showListeningIndicator('Listening (WebSpeech)');
      recog.onresult = async (e) => {
        const text = Array.from(e.results)
                          .map(r => r[0].transcript)
                          .join('');
        displayMessage('You: ' + text);
        await sendChatAndSynthesize(text);
      };
      recog.onerror = (err) => {
        console.error('[WebSpeech] error', err);
        displayMessage('SpeechRecognition エラー', 'error-message');
      };
      recog.onend = () => hideListeningIndicator();
      recog.start();
      return;
      // ────────────────
    }

    // ② 既存の MediaRecorder → /api/transcribe パス
    try {
      console.log('[Enhanced] startBrowserConversation 呼ばれました');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // ——————————————
      // Web Audio API でサイレンス検知の準備
      const audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const analyser   = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      sourceNode.connect(analyser);
      const dataArray       = new Uint8Array(analyser.fftSize);
      const SILENCE_THRESH  = 5;    // 調整可：振幅の閾値
      const SILENCE_PERIOD  = 1500;  // ms：この時間沈黙で自動停止
      const CHECK_INTERVAL  = 100;   // ms ごとにチェック
      let   silenceStart    = Date.now();
      let   silenceChecker; 
      // ——————————————
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      // ——————————————
      // 定期的に振幅をチェックして沈黙なら stop() を呼ぶ
      silenceChecker = setInterval(() => {
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let v of dataArray) {
          const x = v - 128;
          sum += x * x;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        if (rms < SILENCE_THRESH) {
          if (Date.now() - silenceStart > SILENCE_PERIOD) {
            console.log('[Enhanced] 自動沈黙検知 – stop()');
            clearInterval(silenceChecker);
            mediaRecorder.stop();
          }
        } else {
          silenceStart = Date.now();
        }
      }, CHECK_INTERVAL);
      // ——————————————

      mediaRecorder.ondataavailable = e => {
        console.log('[Enhanced] dataavailable:', e.data, 'size=', e.data.size);
        audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // サイレンス検知タイマーを止める
        clearInterval(silenceChecker);
        console.log('[Enhanced] メディア録音停止 – ブロブを送信します');
        micIcon.classList.remove('mic-on', 'pulse-animation');
        hideListeningIndicator();

        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log('[Enhanced] Blob size=', blob.size, 'type=', blob.type);
        audioChunks = [];

        const formData = new FormData();
        formData.append('file', blob, 'speech.webm');
        formData.append('model', transcriptionModelSelect.value);

        const headers = {};
        const key = getApiKey();
        if (key) headers['Authorization'] = `Bearer ${key}`;

        console.log('[Enhanced] /api/transcribe へリクエスト:', { headers });
        let transRes;
        try {
          transRes = await fetch('/api/transcribe', { method: 'POST', headers, body: formData });
        } catch (err) {
          console.error('[Enhanced] fetch エラー (transcribe):', err);
          displayMessage('Transcribe リクエストでエラーが発生しました', 'error-message');
          startBtn.disabled = false;
          stopBtn.disabled = true;
          return;
        }
        console.log('[Enhanced] /api/transcribe レスポンス:', transRes.status);

        let transData;
        try {
          transData = await transRes.json();
        } catch (err) {
          console.error('[Enhanced] JSON 解析エラー (transData):', err);
          displayMessage('Transcribe レスポンスの解析に失敗しました', 'error-message');
          startBtn.disabled = false;
          stopBtn.disabled = true;
          return;
        }
        console.log('[Enhanced] transData:', transData);

        if (transData.text) {
          displayMessage('You: ' + transData.text);

          const chatHeaders = { 'Content-Type': 'application/json' };
          if (key) chatHeaders['Authorization'] = `Bearer ${key}`;

          console.log('[Enhanced] /api/chat へリクエスト:', { headers: chatHeaders, body: transData.text });
          const chatRes = await fetch('/api/chat', {
            method: 'POST',
            headers: chatHeaders,
            body: JSON.stringify({ text: transData.text })
          });
          console.log('[Enhanced] /api/chat レスポンス:', chatRes.status);

          let chatData;
          try {
            chatData = await chatRes.json();
          } catch (err) {
            console.error('[Enhanced] JSON 解析エラー (chatData):', err);
            displayMessage('Chat レスポンスの解析に失敗しました', 'error-message');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            return;
          }
          console.log('[Enhanced] chatData:', chatData);

          if (chatData.text) {
            displayMessage(chatData.text);

            console.log('[Enhanced] /api/synthesize へリクエスト:', { headers: chatHeaders });
            const synthRes = await fetch('/api/synthesize', {
              method: 'POST',
              headers: chatHeaders,
              body: JSON.stringify({ text: chatData.text })
            });
            console.log('[Enhanced] /api/synthesize レスポンス:', synthRes.status);

            let audioBuffer;
            try {
              audioBuffer = await synthRes.arrayBuffer();
            } catch (err) {
              console.error('[Enhanced] AudioBuffer 取得エラー:', err);
              displayMessage('音声合成データの取得に失敗しました', 'error-message');
              startBtn.disabled = false;
              stopBtn.disabled = true;
              return;
            }
            console.log('[Enhanced] audioBuffer byteLength:', audioBuffer.byteLength);

            const url = URL.createObjectURL(new Blob([audioBuffer], { type: 'audio/wav' }));
            const audio = new Audio(url);

            audio.play();
            isAISpeaking = true;
            showVoiceWaveAnimation();
            audio.onended = () => {
              isAISpeaking = false;
              hideVoiceWaveAnimation();
 
              // ── 自動ループ制御 ──
              if (isAutoLoop) {
                loopCount += 1;
                console.log(`[AutoLoop] 回数: ${loopCount}/${MAX_LOOPS}`);
                if (loopCount < MAX_LOOPS) {
                  console.log('[AutoLoop] 次のリスンを開始');
                  startBrowserConversation();
                } else {
                  console.log(`[AutoLoop] 最大回数(${MAX_LOOPS})到達。自動ループ終了`);
                  displayMessage(
                    `自動対話は最大${MAX_LOOPS}回に達したため終了しました。放置による無駄な消費を防ぐため会話を終了します。`,
                    'system-message'
                  );
                  isAutoLoop = false;
                  startBtn.disabled = false;
                  stopBtn.disabled  = true;
                  hideListeningIndicator();
                }
              }
              // ───────────────────
            };
           audio.play();
          }
        }

        startBtn.disabled = false;
        stopBtn.disabled = true;
      };

      // 録音開始
      mediaRecorder.start();
      micIcon.classList.add('mic-on', 'pulse-animation');
      showListeningIndicator('Listening');
      startBtn.disabled = true;
      stopBtn.disabled = false;

    } catch (err) {
      console.error('[Enhanced] getUserMedia エラー:', err);
      displayMessage('Microphone access error', 'error-message');
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  function stopBrowserConversation() {
    // ── まず、自動ループを止めるフラグを下ろす ──
    isAutoLoop = false;

    // ── UI を「停止済み」状態に更新 ──
    stopBtn.disabled  = true;
    startBtn.disabled = false;
    hideListeningIndicator();

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }

    // ── Web Speech API を使っているなら認識も止める ──
    if (typeof recog !== 'undefined' && recog) {
      recog.stop();
    }

    // ── サイレンス検知のタイマーが回っていればクリア ──
    if (typeof silenceChecker !== 'undefined' && silenceChecker) {
      clearInterval(silenceChecker);
      silenceChecker = null;
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

