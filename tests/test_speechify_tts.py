import asyncio
import wave

from fastapi.testclient import TestClient

import app.app as app_module
import app.main as main_module
from app.main import _speechify_voice_options, app


class FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, status=200, chunks=None, body=None, text=""):
        self.status = status
        self.content = FakeContent(chunks or [])
        self._body = body
        self._text = text

    async def json(self):
        return self._body

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, responses, calls):
        self._responses = responses
        self.calls = calls

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def fake_session_factory(responses, calls):
    return lambda *args, **kwargs: FakeSession(responses, calls)


def test_voice_options_label_and_dedupe():
    voices = _speechify_voice_options([
        {"id": "geffen_32", "display_name": "Geffen", "locale": "en-US", "gender": "female", "type": "shared"},
        {"id": "dominic", "display_name": "Dominic", "locale": "en-US", "gender": "male", "type": "shared"},
        {"id": "dominic_32", "display_name": "Dominic", "locale": "en-US", "gender": "male", "type": "shared"},
        {"id": "abc-123", "display_name": "My Clone", "locale": "en-US", "gender": "male", "type": "personal"},
        {"display_name": "No id"},
    ])
    assert voices == [
        {"id": "geffen_32", "name": "Geffen (en-US, female)"},
        {"id": "dominic", "name": "Dominic (en-US, male)"},
        {"id": "dominic_32", "name": "Dominic (en-US, male) [dominic_32]"},
        {"id": "abc-123", "name": "My Clone (en-US, male) - your voice"},
    ]


def test_speechify_voices_requires_key(monkeypatch):
    monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
    response = TestClient(app).get("/speechify_voices")
    assert response.json() == {"voices": [], "error": "SPEECHIFY_API_KEY not set"}


def test_speechify_voices_follows_cursor_pages(monkeypatch):
    monkeypatch.setenv("SPEECHIFY_API_KEY", "test-key")
    monkeypatch.setenv("SPEECHIFY_TTS_MODEL", "simba-3.2")
    calls = []
    responses = [
        FakeResponse(body={
            "voices": [{"id": "geffen_32", "display_name": "Geffen", "locale": "en-US", "gender": "female"}],
            "has_more": True,
            "next_cursor": "page2",
        }),
        FakeResponse(body={
            "voices": [{"id": "harper_32", "display_name": "Harper", "locale": "en-US", "gender": "female"}],
            "has_more": False,
            "next_cursor": None,
        }),
    ]
    monkeypatch.setattr(main_module.aiohttp, "ClientSession", fake_session_factory(responses, calls))

    response = TestClient(app).get("/speechify_voices")

    assert [v["id"] for v in response.json()["voices"]] == ["geffen_32", "harper_32"]
    assert calls[0][2]["headers"] == {"Authorization": "Bearer test-key"}
    assert calls[0][2]["params"]["model"] == "simba-3.2"
    assert calls[1][2]["params"]["cursor"] == "page2"


def test_speechify_voices_accepts_plain_list(monkeypatch):
    monkeypatch.setenv("SPEECHIFY_API_KEY", "test-key")
    calls = []
    responses = [FakeResponse(body=[{"id": "geffen_32", "display_name": "Geffen"}])]
    monkeypatch.setattr(main_module.aiohttp, "ClientSession", fake_session_factory(responses, calls))

    response = TestClient(app).get("/speechify_voices")

    assert response.json() == {"voices": [{"id": "geffen_32", "name": "Geffen"}]}
    assert len(calls) == 1


def test_speechify_tts_payload(monkeypatch):
    monkeypatch.setattr(app_module, "SPEECHIFY_TTS_VOICE", "geffen_32")
    monkeypatch.setattr(app_module, "SPEECHIFY_TTS_MODEL", "simba-3.2")
    monkeypatch.setattr(app_module, "SPEECHIFY_TTS_LANGUAGE", "")
    assert app_module.build_speechify_tts_payload("Hello") == {
        "input": "Hello",
        "voice_id": "geffen_32",
        "model": "simba-3.2",
    }

    monkeypatch.setattr(app_module, "SPEECHIFY_TTS_LANGUAGE", "fr-FR")
    assert app_module.build_speechify_tts_payload("Bonjour")["language"] == "fr-FR"


def test_speechify_tts_writes_wav_from_pcm_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "SPEECHIFY_API_KEY", "test-key")
    pcm = b"\x01\x00" * 2400  # 0.1s of 24 kHz mono 16-bit audio
    calls = []
    responses = [FakeResponse(chunks=[pcm[:1000], pcm[1000:]])]
    monkeypatch.setattr(app_module.aiohttp, "ClientSession", fake_session_factory(responses, calls))
    output_path = tmp_path / "out.wav"

    assert asyncio.run(app_module.speechify_text_to_speech("Hi", str(output_path)))

    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", "https://api.speechify.ai/v1/audio/stream")
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["headers"]["Accept"] == "audio/pcm"
    with wave.open(str(output_path), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(wav.getnframes()) == pcm


def test_speechify_tts_reports_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "SPEECHIFY_API_KEY", "test-key")
    sent = []

    async def fake_send(message):
        sent.append(message)

    monkeypatch.setattr(app_module, "send_message_to_clients", fake_send)
    responses = [FakeResponse(status=401, text='{"error": "unauthorized"}')]
    monkeypatch.setattr(app_module.aiohttp, "ClientSession", fake_session_factory(responses, []))
    output_path = tmp_path / "out.wav"

    assert not asyncio.run(app_module.speechify_text_to_speech("Hi", str(output_path)))
    assert not output_path.exists()
    assert "Speechify TTS error: 401" in sent[0]
