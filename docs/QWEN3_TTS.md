# Qwen3 TTS (local, OpenAI-compatible) — Optional

Use [Qwen3-TTS-Openai-Fastapi](https://github.com/bigsk1/Qwen3-TTS-Openai-Fastapi) as a **separate local TTS server** with Voice Chat AI. It exposes the same style of API as OpenAI’s `/v1/audio/speech` endpoint, so you get fast local speech, **28+ preset cloned voices**, and the ability to **add your own voices** without installing Spark-TTS inside this app.

## Why use it?

| | Spark-TTS (built-in) | Kokoro (HTTP) | Qwen3 TTS (HTTP) |
|---|----------------------|---------------|------------------|
| Runs | Inside Voice Chat AI | Separate server | Separate server |
| Custom voices | Character `.wav` per folder | Fixed presets | Presets + your own samples on the TTS server |
| Typical speed (GPU) | Often slow | Fast | Usually faster than Spark |
| GPU sharing | Stays loaded in app process | Separate process | Separate process + optional VRAM auto-unload |

Good fit if you want **local TTS faster than Spark-TTS** and **more voice choice than Kokoro**, and you’re fine running one extra service (Docker or native).

## What Voice Chat AI supports today

Voice Chat AI does **not** have a dedicated `qwen` TTS provider. Integration uses the existing **`openai` TTS path** by pointing `OPENAI_TTS_URL` at your local Qwen server. That works for **normal conversation mode** (dashboard / CLI when `TTS_PROVIDER=openai`).

**STT timing** (`STT_SILENCE_DURATION`, `STT_SILENCE_THRESHOLD`) and **TTS HTTP timeouts** are the same whether `OPENAI_TTS_URL` is cloud or custom — only the host changes; silence detection and `timeout=30` on speech requests are unchanged.

**OpenAI Enhanced Mode** always calls OpenAI’s cloud TTS API. Local Qwen is **not** used there unless you change that mode in code later.

Character `.wav` files under `characters/` are used for **Spark-TTS cloning only**. For Qwen, register voices on the **Qwen server** (see below), then set `OPENAI_TTS_VOICE` to that name.

---

## 1. Install and run the Qwen3 TTS server

Follow the upstream repo: [bigsk1/Qwen3-TTS-Openai-Fastapi](https://github.com/bigsk1/Qwen3-TTS-Openai-Fastapi).

### Docker (recommended)

```bash
git clone https://github.com/bigsk1/Qwen3-TTS-Openai-Fastapi.git
cd Qwen3-TTS-Openai-Fastapi
docker compose up -d qwen3-tts-gpu
docker compose logs -f qwen3-tts-gpu
```

Default base URL: `http://localhost:8881/v1` (port **8881** unless you change `PORT` in the Qwen project).

### Check the server

```bash
curl http://localhost:8881/health
curl http://localhost:8881/v1/voices
```

Swagger UI (if enabled): `http://localhost:8881/docs`

First synthesis after idle may be slow while the model loads (~4.5GB VRAM). The Qwen server can auto-unload GPU memory when idle so other apps (Whisper, Ollama, etc.) can use the GPU.

---

## 2. Configure Voice Chat AI (`.env`)

Use the **OpenAI TTS provider** with a **local URL**. No OpenAI API key is required for local Qwen; any non-empty value satisfies the client.

```env
# Use the OpenAI-compatible client in Voice Chat AI
TTS_PROVIDER=openai

# Point at your Qwen server (not api.openai.com)
OPENAI_TTS_URL=http://localhost:8881/v1/audio/speech
OPENAI_API_KEY=not-needed

# Qwen expects model name tts-1 (or tts-1-hd if your server supports it)
OPENAI_MODEL_TTS=tts-1

# Pick a voice exposed by GET /v1/voices on the Qwen server
OPENAI_TTS_VOICE=Jarvis

# Optional: 0.7–1.2 (same as other providers)
VOICE_SPEED=1.0
```

### Example preset voices (on the Qwen server)

**Male:** Jarvis, Paddington, Professor, Josh, John, Mark, Adam, Russell, Curt, Eustis, General_Joe, Grandpa, Nigel, Richard, Valentino, Wildebeest  

**Female:** Lucy, Carmen, Caroline, Joanne, Victoria, Natasha, Bianca, Cecile, Emmaline, Monika, Tally, Villain  

Run `curl http://localhost:8881/v1/voices` for the authoritative list on your install.

### Docker / remote host

If Voice Chat AI runs in Docker and Qwen runs on the host:

```env
OPENAI_TTS_URL=http://host.docker.internal:8881/v1/audio/speech
```

Adjust host/port to match your network.

### UI

In the dashboard, set **TTS Provider** to **openai** (not kokoro or sparktts).

When `OPENAI_TTS_URL` is **not** the default OpenAI cloud speech URL, the app fetches voices from your server (`GET …/v1/voices`) and fills the **TTS Voice (local)** dropdown automatically. Your `.env` voice (e.g. `Jarvis`) is selected when it exists in that list.

---

## 3. Custom voices (per character / persona)

On the **Qwen server** (not in Voice Chat AI):

1. Record **3–10 seconds** of clear speech (`.wav`).
2. Name the file `VoiceName.wav` (e.g. `Wizard.wav`).
3. Place it in the Qwen project’s voice samples directory (see upstream README — e.g. `sample-voices-xtts/` or `VOICE_SAMPLES_DIR` in Docker).
4. Restart the Qwen container/service.
5. In Voice Chat AI `.env`: `OPENAI_TTS_VOICE=Wizard` (match the name, without `.wav`).

To use different characters with different voices, either:

- change `OPENAI_TTS_VOICE` when you switch characters, or  
- register one Qwen voice per character and document the mapping yourself.

There is no automatic “use `characters/wizard/wizard.wav`” wiring for Qwen like Spark-TTS provides.

---

## 4. Quick test (before starting Voice Chat AI)

```bash
curl -X POST http://localhost:8881/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d "{\"model\":\"tts-1\",\"voice\":\"Jarvis\",\"input\":\"Hello from Qwen.\",\"response_format\":\"wav\"}" \
  --output test.wav
```

If `test.wav` plays, use the same URL and voice settings in `.env` above.

---

## 5. Troubleshooting

### No audio / HTTP errors

- Confirm Qwen is up: `curl http://localhost:8881/health`
- Confirm `OPENAI_TTS_URL` ends with `/v1/audio/speech`
- Confirm `OPENAI_TTS_VOICE` exists on the server (`/v1/voices`)
- Use `OPENAI_MODEL_TTS=tts-1` (not `gpt-4o-mini-tts` — that is for OpenAI cloud, Qwen expects tts-1)
- Restart Voice Chat AI after `.env` changes

### First reply is very slow

- Cold start or model reload after VRAM auto-unload. Warm up with the curl test or `POST /admin/reload` on the Qwen server (see upstream docs).

### OpenAI voice dropdown empty or wrong

- Only auto-fills when `OPENAI_TTS_URL` is not `https://api.openai.com/v1/audio/speech`
- Confirm `curl http://YOUR_HOST:8881/v1/voices` returns your voice names
- Restart Voice Chat AI after changing `OPENAI_TTS_URL`

![Image](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/7bacd8bb-254c-4d9c-414b-df448c342d00/public)

### Still using Spark or Kokoro

- `TTS_PROVIDER` must be `openai` for this setup
- Kokoro mode sends `model: "kokoro"` — that does **not** match the Qwen API; use the `openai` provider instead

### Enhanced Mode still uses cloud voices

- Expected: Enhanced Mode uses OpenAI cloud TTS. Use normal conversation mode for local Qwen.

### GPU out of memory

- Stop Spark-TTS in the same machine (`TTS_PROVIDER` not `sparktts`)
- Use Qwen’s idle unload / `POST /admin/unload` (upstream)
- Avoid loading large Whisper + Qwen + Spark at the same time without unload settings

---

## 6. Related docs

- [Qwen3-TTS-Openai-Fastapi](https://github.com/bigsk1/Qwen3-TTS-Openai-Fastapi) — install, Docker, benchmarks, VRAM
- [Spark-TTS (built-in)](SPARKTTS.md) — in-process cloning from character `.wav`
- [README — Kokoro TTS](../README.md#kokoro-tts-for-local-voices---optional) — another local HTTP TTS option

## Summary `.env` block

```env
TTS_PROVIDER=openai
OPENAI_TTS_URL=http://localhost:8881/v1/audio/speech
OPENAI_API_KEY=not-needed
OPENAI_MODEL_TTS=tts-1
OPENAI_TTS_VOICE=Jarvis
VOICE_SPEED=1.0
```
