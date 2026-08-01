# Local Realtime

Voice Chat AI can use a self-hosted, OpenAI GA-compatible WebRTC speech-to-speech server as an alternative to hosted OpenAI Realtime. Hosted OpenAI remains the default.

The validated deployment uses the Realtime implementation derived from [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech), pinned at commit `5b443c8011349f7b5a7138c2ad29fe1c1e65b8bd`, with Silero VAD, Faster Whisper, an Ollama LLM, and Qwen3-TTS. Custom fork I made here: [bigsk1/realtime](https://github.com/bigsk1/realtime)

## Configuration

Add these values to `.env`:

```env
OPENAI_REALTIME_PROVIDER=local
LOCAL_REALTIME_URL=http://192.0.2.10:8765/v1/realtime/calls
LOCAL_REALTIME_VOICE=Aiden
LOCAL_REALTIME_VAD_THRESHOLD=0.6
LOCAL_REALTIME_SILENCE_MS=1000
```

`OPENAI_REALTIME_MODEL` is ignored in local mode. The local Realtime connection does not require `OPENAI_API_KEY`; other enabled OpenAI features may still require it.

The local voice dropdown supports the Qwen CustomVoice speakers `Aiden`, `Ryan`, `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, and `Sohee`. `LOCAL_REALTIME_VOICE` selects the initial speaker; invalid values fall back to `Aiden`. Stop the session before switching voices because the selected voice is applied through `session.update` when a call starts.

Restart Voice Chat AI after changing `.env`. Open the Realtime page, select a character, start the session, and click the microphone button to speak. Stop the session before changing its character or voice.

Voice Chat AI overrides the local server's very short default silence duration with a 1000 ms pause while retaining its validated Silero threshold of 0.6. Increase `LOCAL_REALTIME_SILENCE_MS` to `1200` or `1500` if you pause frequently; this adds the corresponding delay before a reply begins. Raise `LOCAL_REALTIME_VAD_THRESHOLD` if background noise triggers speech, or lower it cautiously if quiet speech is consistently missed.

### Faster Whisper and premature responses

When the local server uses Faster Whisper, disable its live-transcription mode in the server's Compose command:

```yaml
- --enable_live_transcription
- "False"
```

At the pinned upstream commit, live transcription is enabled by default and sends progressive audio to STT every 0.5 seconds. The Faster Whisper adapter treats that progressive chunk as a completed transcription rather than a partial result. It can therefore start the LLM from the first fraction of an utterance, drop the remainder of the same revision, and hallucinate short silence phrases such as “Thank you.” This is a server startup setting and cannot be changed by the WebRTC client's `session.update` event. Recreate the local Realtime container after changing it.

To return to hosted OpenAI:

```env
OPENAI_REALTIME_PROVIDER=openai
```

## Protocol behavior

The browser continues to send its SDP offer to Voice Chat AI. In local mode, the backend forwards raw `application/sdp` to `LOCAL_REALTIME_URL` without an Authorization header. Media uses WebRTC RTP, while GA-compatible JSON events use the `oai-events` data channel.

After the channel opens, Voice Chat AI sends the selected character instructions and local Qwen voice in `session.update`. During barge-in, it sends `output_audio_buffer.clear` and resets queued browser playback. When the session stops, it uses the call ID returned in the upstream `Location` header to explicitly delete the local call and release its pipeline.

The integration handles the server's supported events, including speech start/stop, completed input transcription, response creation, output audio, output audio transcript, response completion, and errors. It does not assume hosted-only Realtime features.

## Docker and networking

Docker Compose already loads the repository `.env`, so no Compose-file change is required. For a server on another LAN machine, use its LAN IP exactly as shown above; that address is requested by the Voice Chat AI backend from inside the container.

If the local server's optional Caddy UI proxies `/v1/*`, its calls endpoint can be available at an address such as `https://192.0.2.10:8443/v1/realtime/calls` after the Caddy local CA is trusted. Voice Chat AI should nevertheless use the direct backend endpoint, such as `http://192.0.2.10:8765/v1/realtime/calls`. Replace the documentation-only IP with the Realtime server's actual LAN address. The browser communicates with Voice Chat AI over its own HTTPS origin, while the Voice Chat AI backend performs this trusted-LAN HTTP request, avoiding browser CORS and mixed-content restrictions.

Do not use `127.0.0.1` or `localhost` for a service running outside the Voice Chat AI container. On Docker Desktop for macOS or Windows, use `host.docker.internal` when the Realtime service runs directly on the Docker host. A separate LAN host should continue to use its LAN IP.

The browser must also be able to reach the ICE candidates advertised by the Realtime server. Permit TCP 8765 and the configured WebRTC UDP range (the validated deployment uses UDP 32768–60999) only from trusted LAN networks. The local server does not provide authentication, TLS, quotas, or public-ingress protection.

The validated deployment has one concurrent pipeline. A second session may receive `session_limit_reached` until the first call has been released.

## Duplicate or continued dialogue

Small local LLMs may continue the transcript by inventing the user's next line instead of stopping after the assistant answer. This can look like two immediate replies and may include dialogue separators or phrases the user never said. Voice Chat AI adds local turn instructions, combines multiple transcript-completion chunks into one assistant entry, and cancels a second response cycle for the same VAD turn.

If invented dialogue continues, select a local LLM with stronger instruction and chat-template adherence in the Realtime server configuration. This model is owned by the local server; `OPENAI_REALTIME_MODEL` does not change it.

## Optional health check

The validated server exposes pool status at:

```text
http://192.0.2.10:8765/v1/pool
```

This endpoint is useful for checking availability and whether the single pipeline is occupied. Voice Chat AI does not poll it continuously; session-start errors are displayed in the Realtime activity log.

## Live validation checklist

1. Confirm `/v1/pool` is reachable from the Voice Chat AI host or container.
2. Start a Realtime session and confirm the activity log receives `session.created`.
3. Enable the microphone and confirm speech start/stop and a completed transcription.
4. Confirm audible Qwen output and an output transcript.
5. Interrupt the response and confirm queued audio stops promptly.
6. Stop the session and confirm the pool becomes available again.
