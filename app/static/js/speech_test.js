document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('start-rec');
  const stopBtn = document.getElementById('stop-rec');
  const resultEl = document.getElementById('transcript');
  const speakBtn = document.getElementById('play-tts');
  const inputEl = document.getElementById('tts-input');

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRec) {
    const recog = new SpeechRec();
    recog.lang = 'ja-JP';
    recog.continuous = true;
    recog.interimResults = true;

    recog.onresult = e => {
      let finalT = '', interimT = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const transcript = e.results[i][0].transcript;
        if (e.results[i].isFinal) {
          finalT += transcript;
        } else {
          interimT += transcript;
        }
      }
      resultEl.textContent = finalT || interimT;
    };

    recog.onend = () => recog.start();

    startBtn.onclick = () => recog.start();
    stopBtn.onclick = () => recog.stop();
  }

  speakBtn.onclick = () => {
    const utt = new SpeechSynthesisUtterance(inputEl.value);
    utt.lang = 'ja-JP';
    window.speechSynthesis.speak(utt);
  };
});
