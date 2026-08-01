import asyncio
import json

import httpx
import pytest
from starlette.requests import Request

from app.openai_realtime import (
    DEFAULT_OPENAI_REALTIME_MODEL,
    build_openai_realtime_session,
    normalize_openai_realtime_model,
    normalize_openai_realtime_voice,
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
