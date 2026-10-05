# Speechify TTS - Optional

[Speechify](https://speechify.ai) is a cloud text-to-speech API. Voice Chat AI calls its streaming endpoint (`POST /v1/audio/stream`) directly, so there is nothing extra to install: it uses the `aiohttp` and `requests` packages the app already ships with.

## 1. Get an API key

Sign up at [platform.speechify.ai](https://platform.speechify.ai) and create an API key.

## 2. Configure `.env`

```env
TTS_PROVIDER=speechify
SPEECHIFY_API_KEY=your_api_key_here
SPEECHIFY_TTS_VOICE=geffen_32
SPEECHIFY_TTS_MODEL=simba-3.2
# Optional
# SPEECHIFY_TTS_LANGUAGE=fr-FR
# SPEECHIFY_TTS_TIMEOUT=60
# SPEECHIFY_BASE_URL=https://api.speechify.ai
```

| Variable | Default | Notes |
|---|---|---|
| `SPEECHIFY_API_KEY` | none | Required. Sent as `Authorization: Bearer <key>`. |
| `SPEECHIFY_TTS_VOICE` | `geffen_32` | Voice id. The dashboard lists every voice your key can use. |
| `SPEECHIFY_TTS_MODEL` | `simba-3.2` | `simba-3.2` is the recommended English model. Use `simba-3.0` for German, Spanish, French, Italian or Portuguese. |
| `SPEECHIFY_TTS_LANGUAGE` | blank | Language of the text, for example `de-DE`. Leave blank to use the voice's own locale. |
| `SPEECHIFY_TTS_TIMEOUT` | `60` | Seconds to wait for a full response. Raise it for long story or game replies. |
| `SPEECHIFY_BASE_URL` | `https://api.speechify.ai` | Only change this if you were given a different endpoint. |

Once `SPEECHIFY_API_KEY` is set, **Speechify** shows up in the dashboard TTS Provider dropdown.

## 3. Pick a voice

In the Web UI, choose **Speechify** as the TTS Provider. The Voice dropdown loads the voices for your `SPEECHIFY_TTS_MODEL` from `GET /v1/voices`, including any voices you cloned in your Speechify account (marked "your voice"). The voice you pick is used right away; `SPEECHIFY_TTS_VOICE` is only the startup default.

In the CLI (`python cli.py`), set `SPEECHIFY_TTS_VOICE` in `.env`.

To see the voice list yourself:

```bash
curl -s "https://api.speechify.ai/v1/voices?model=simba-3.2" \
  -H "Authorization: Bearer $SPEECHIFY_API_KEY"
```

## How it works

- The app streams raw 16-bit, 24 kHz mono PCM (`Accept: audio/pcm`) and wraps it in a WAV header, so playback does not need ffmpeg.
- It works in the normal conversation mode, screenshot analysis, and the CLI. OpenAI Enhanced Mode and OpenAI Realtime use OpenAI voices as before.
- `MAX_CHAR_LENGTH` applies as it does for the other cloud providers. The stream endpoint itself accepts up to 20,000 characters per request.
- `VOICE_SPEED` is not applied to Speechify voices.

## Troubleshooting

- **Speechify is missing from the TTS dropdown:** `SPEECHIFY_API_KEY` is unset or still `your_api_key_here`. Restart the app after editing `.env`.
- **HTTP 401:** the key is wrong or was revoked.
- **HTTP 404 `voice_not_found`:** the voice id does not exist for your key. Pick a voice from the dashboard list.
- **HTTP 400:** usually a model name that is not valid. Use `simba-3.2` or `simba-3.0`.
- **HTTP 429:** you hit your plan's rate or concurrency limit. Wait a moment and try again.

API reference: [docs.speechify.ai](https://docs.speechify.ai)
