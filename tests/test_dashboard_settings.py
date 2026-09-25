import re

import pytest
from fastapi.testclient import TestClient

from app.main import _dashboard_model_providers, _dashboard_tts_providers, app


DASHBOARD_ENV_KEYS = (
    "MODEL_PROVIDER",
    "TTS_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_TTS_URL",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "TYPECAST_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "KOKORO_BASE_URL",
    "KOKORO_TTS_VOICE",
    "SPARKTTS_MODEL_DIR",
)


@pytest.fixture
def clean_dashboard_env(monkeypatch):
    for name in DASHBOARD_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def select_values(html: str, select_id: str) -> list[str]:
    match = re.search(
        rf'<select id="{select_id}"[^>]*>(.*?)</select>', html, re.DOTALL
    )
    assert match is not None
    return re.findall(r'<option value="([^"]*)"', match.group(1))


def test_dashboard_hides_unconfigured_providers(clean_dashboard_env, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TTS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")
    monkeypatch.setenv("XAI_API_KEY", "your_api_key_here")

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert select_values(response.text, "provider-select") == [""]
    assert select_values(response.text, "tts-select") == [""]
    assert response.text.count('id="model-select"') == 1
    assert response.text.count('id="voice-select"') == 1
    assert 'id="start-conversation-btn" disabled' in response.text


def test_dashboard_lists_configured_cloud_providers(clean_dashboard_env, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_api_key_here")
    monkeypatch.setenv("TYPECAST_API_KEY", "your_api_key_here")

    response = TestClient(app).get("/")

    assert select_values(response.text, "provider-select") == ["xai"]
    assert select_values(response.text, "tts-select") == ["xai", "elevenlabs"]


def test_dashboard_supports_local_and_custom_tts_without_cloud_key(
    clean_dashboard_env, monkeypatch
):
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("TTS_PROVIDER", "typecast")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
    monkeypatch.setenv("OPENAI_TTS_URL", "http://localhost:8881/v1/audio/speech")

    response = TestClient(app).get("/")

    assert select_values(response.text, "provider-select") == ["ollama"]
    assert select_values(response.text, "tts-select") == ["openai", "kokoro"]
    assert 'id="provider-select" data-initial="ollama" data-configured="anthropic"' in response.text
    assert 'id="tts-select" data-initial="openai" data-configured="typecast"' in response.text


def test_spark_tts_requires_configured_model_files(clean_dashboard_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKTTS_MODEL_DIR", str(tmp_path))
    assert ("sparktts", "Spark-TTS (Local)") not in _dashboard_tts_providers()

    for part in ("LLM", "BiCodec"):
        folder = tmp_path / part
        folder.mkdir()
        (folder / "model.safetensors").touch()

    assert ("sparktts", "Spark-TTS (Local)") in _dashboard_tts_providers()
    assert _dashboard_model_providers() == []
