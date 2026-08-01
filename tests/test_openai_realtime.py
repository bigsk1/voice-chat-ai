import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.openai_realtime import (
    DEFAULT_LOCAL_REALTIME_CALLS_URL,
    DEFAULT_LOCAL_REALTIME_SILENCE_MS,
    DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD,
    DEFAULT_OPENAI_REALTIME_MODEL,
    LOCAL_REALTIME_VOICES,
    build_local_realtime_call_url,
    build_openai_realtime_session,
    extract_local_realtime_call_id,
    normalize_local_realtime_url,
    normalize_local_realtime_silence_ms,
    normalize_local_realtime_vad_threshold,
    normalize_local_realtime_voice,
    normalize_openai_realtime_model,
    normalize_openai_realtime_voice,
    normalize_realtime_provider,
)


@pytest.mark.parametrize(
    ("retired_model", "replacement"),
    [
        ("gpt-4o-realtime-preview-2024-12-17", "gpt-realtime-2.1"),
        ("gpt-4o-mini-realtime-preview-2024-12-17", "gpt-realtime-2.1-mini"),
        ("gpt-realtime", "gpt-realtime-2.1"),
        ("gpt-realtime-mini", "gpt-realtime-2.1-mini"),
    ],
)
def test_retired_realtime_models_are_normalized(retired_model, replacement):
    assert normalize_openai_realtime_model(retired_model) == replacement


def test_current_or_custom_realtime_model_is_preserved():
    assert normalize_openai_realtime_model("gpt-realtime-2.1") == "gpt-realtime-2.1"
    assert normalize_openai_realtime_model("local-realtime") == "local-realtime"
    assert normalize_openai_realtime_model(None) == DEFAULT_OPENAI_REALTIME_MODEL


def test_invalid_voice_uses_recommended_default():
    assert normalize_openai_realtime_voice("unknown") == "marin"
    assert normalize_openai_realtime_voice("CEDAR") == "cedar"


def test_local_realtime_configuration_is_normalized_separately():
    assert normalize_realtime_provider("LOCAL") == "local"
    assert normalize_realtime_provider("unknown") == "openai"
    assert normalize_local_realtime_voice(" Aiden ") == "Aiden"
    assert normalize_local_realtime_voice("uncle_fu") == "Uncle_Fu"
    assert normalize_local_realtime_voice("ONO_ANNA") == "Ono_Anna"
    assert normalize_local_realtime_voice("unsupported") == "Aiden"
    assert normalize_local_realtime_voice(None) == "Aiden"
    assert normalize_local_realtime_url(None) == DEFAULT_LOCAL_REALTIME_CALLS_URL


def test_local_realtime_voice_catalog_matches_qwen_custom_voices():
    assert LOCAL_REALTIME_VOICES == (
        "Aiden",
        "Ryan",
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ono_Anna",
        "Sohee",
    )


def test_local_realtime_page_renders_all_voices_and_configured_default(monkeypatch):
    from app.main import app

    monkeypatch.setenv("OPENAI_REALTIME_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_REALTIME_VOICE", "sohee")

    response = TestClient(app).get("/webrtc_realtime")

    assert response.status_code == 200
    for voice in LOCAL_REALTIME_VOICES:
        assert f'value="{voice}"' in response.text
    assert 'value="Sohee" selected' in response.text


def test_local_realtime_url_requires_calls_endpoint():
    with pytest.raises(ValueError, match="/v1/realtime/calls"):
        normalize_local_realtime_url("http://192.0.2.10:8765/v1/pool")


def test_local_realtime_vad_configuration_is_normalized():
    assert normalize_local_realtime_vad_threshold("0.45") == 0.45
    assert normalize_local_realtime_vad_threshold("invalid") == (
        DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD
    )
    assert normalize_local_realtime_vad_threshold("1.1") == (
        DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD
    )
    assert DEFAULT_LOCAL_REALTIME_VAD_THRESHOLD == 0.6
    assert normalize_local_realtime_silence_ms("1200") == 1200
    assert normalize_local_realtime_silence_ms("50") == (
        DEFAULT_LOCAL_REALTIME_SILENCE_MS
    )
    assert normalize_local_realtime_silence_ms("invalid") == (
        DEFAULT_LOCAL_REALTIME_SILENCE_MS
    )


def test_local_call_id_is_only_accepted_from_configured_server():
    calls_url = "http://192.0.2.10:8765/v1/realtime/calls"
    assert (
        extract_local_realtime_call_id(
            "/v1/realtime/calls/call-123", calls_url
        )
        == "call-123"
    )
    assert (
        extract_local_realtime_call_id(
            "http://attacker.invalid/v1/realtime/calls/call-123", calls_url
        )
        is None
    )
    assert build_local_realtime_call_url(calls_url, "call-123") == (
        "http://192.0.2.10:8765/v1/realtime/calls/call-123"
    )


def test_ga_session_uses_nested_audio_configuration():
    session = build_openai_realtime_session("gpt-realtime-2.1", "cedar")

    assert session == {
        "type": "realtime",
        "model": "gpt-realtime-2.1",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                }
            },
            "output": {"voice": "cedar"},
        },
    }
    assert "modalities" not in session
    assert "voice" not in session


def test_proxy_uses_ga_calls_endpoint_and_multipart_session(monkeypatch):
    from app.main import proxy_openai_realtime

    captured = {}

    class FakeResponse:
        content = b"v=0\r\n"
        status_code = 201
        headers = {"content-type": "application/sdp"}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["request_kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_REALTIME_PROVIDER", "openai")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    body = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\n"
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_realtime_proxy",
            "query_string": b"model=gpt-4o-realtime-preview-2024-12-17&voice=CEDAR",
            "headers": [(b"content-type", b"application/sdp")],
        },
        receive,
    )

    response = asyncio.run(proxy_openai_realtime(request))

    assert response.status_code == 201
    assert response.body == b"v=0\r\n"
    assert captured["url"] == "https://api.openai.com/v1/realtime/calls"
    request_kwargs = captured["request_kwargs"]
    assert request_kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert "OpenAI-Beta" not in request_kwargs["headers"]
    assert request_kwargs["files"]["sdp"] == (
        None,
        body.decode(),
        "application/sdp",
    )
    session = json.loads(request_kwargs["files"]["session"][1])
    assert session["model"] == "gpt-realtime-2.1"
    assert session["audio"]["output"]["voice"] == "cedar"


def test_local_proxy_uses_raw_sdp_without_openai_auth(monkeypatch):
    from app.main import proxy_openai_realtime

    captured = {}

    class FakeResponse:
        content = b"v=0\r\n"
        status_code = 201
        headers = {
            "content-type": "application/sdp",
            "location": "/v1/realtime/calls/local-call-123",
        }

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["request_kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setenv("OPENAI_REALTIME_PROVIDER", "local")
    monkeypatch.setenv(
        "LOCAL_REALTIME_URL", "http://192.0.2.10:8765/v1/realtime/calls"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    body = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\n"
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_realtime_proxy",
            "query_string": b"model=ignored&voice=marin",
            "headers": [(b"content-type", b"application/sdp")],
        },
        receive,
    )

    response = asyncio.run(proxy_openai_realtime(request))

    assert response.status_code == 201
    assert response.headers["x-realtime-call-id"] == "local-call-123"
    assert captured["url"] == "http://192.0.2.10:8765/v1/realtime/calls"
    assert captured["request_kwargs"] == {
        "content": body.decode(),
        "headers": {"Content-Type": "application/sdp"},
    }


def test_local_proxy_explicitly_deletes_call(monkeypatch):
    from app.main import close_local_realtime_call

    captured = {}

    class FakeResponse:
        content = b""
        status_code = 204
        headers = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def delete(self, url, **kwargs):
            captured["url"] = url
            captured["request_kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setenv("OPENAI_REALTIME_PROVIDER", "local")
    monkeypatch.setenv(
        "LOCAL_REALTIME_URL", "http://192.0.2.10:8765/v1/realtime/calls"
    )
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = asyncio.run(close_local_realtime_call("local-call-123"))

    assert response.status_code == 204
    assert captured["url"] == (
        "http://192.0.2.10:8765/v1/realtime/calls/local-call-123"
    )
    assert captured["request_kwargs"] == {}
