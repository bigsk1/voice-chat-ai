document.addEventListener('DOMContentLoaded', () => {
  // ---- 要素取得 ----
  const sttLangSel = document.getElementById('stt-language-select');
  const startBtn   = document.getElementById('start-rec');
  const stopBtn    = document.getElementById('stop-rec');
  const statusEl   = document.getElementById('stt-status');
  const resultEl   = document.getElementById('transcript');

  const ttsLangSel = document.getElementById('tts-language-select');
  const rateInput  = document.getElementById('tts-rate');
  const pitchInput = document.getElementById('tts-pitch');
  const volInput   = document.getElementById('tts-volume');
  const rateVal    = document.getElementById('rate-value');
  const pitchVal   = document.getElementById('pitch-value');
  const volVal     = document.getElementById('volume-value');
  const ttsInput   = document.getElementById('tts-input');
  const playBtn    = document.getElementById('play-tts');

  // ---- 認識設定 ----
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recog     = new SpeechRec();
  recog.continuous     = true;
  recog.interimResults = true;

  recog.onstart = () => { statusEl.textContent = '認識中...'; };
  recog.onend   = () => { statusEl.textContent = '待機中'; };
  recog.onerror = e => { statusEl.textContent = `エラー: ${e.error}`; };

  recog.onresult = e => {
    let finalT = '', interimT = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalT += t;
      else interimT += t;
    }
    resultEl.textContent = finalT || interimT;
  };

  startBtn.onclick = () => {
    recog.lang = sttLangSel.value;
    recog.start();
  };
  stopBtn.onclick  = () => { recog.stop(); };

  // ---- 合成パラメータ同期 ----
  rateInput.oninput  = () => { rateVal.textContent  = rateInput.value;  };
  pitchInput.oninput = () => { pitchVal.textContent = pitchInput.value; };
  volInput.oninput   = () => { volVal.textContent   = volInput.value;   };

  // ---- 音声合成実行 ----
  playBtn.onclick = () => {
    const utt = new SpeechSynthesisUtterance(ttsInput.value);
    utt.lang   = ttsLangSel.value;
    utt.rate   = parseFloat(rateInput.value);
    utt.pitch  = parseFloat(pitchInput.value);
    utt.volume = parseFloat(volInput.value);
    // 対応音声選択 (必要なら)
    const voice = speechSynthesis.getVoices()
                  .find(v => v.lang === utt.lang);
    if (voice) utt.voice = voice;

    window.speechSynthesis.speak(utt);
  };
});

