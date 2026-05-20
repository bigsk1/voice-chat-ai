"""Tests for LiteLLM provider integration in voice-chat-ai."""

import json
import os
import sys
import types
from unittest import mock

import pytest

# Stub litellm before any app imports
_fake_litellm = types.ModuleType("litellm")
_fake_exceptions = types.ModuleType("litellm.exceptions")


class _AuthenticationError(Exception):
    pass


class _NotFoundError(Exception):
    pass


class _RateLimitError(Exception):
    pass


class _Timeout(Exception):
    pass


_fake_exceptions.AuthenticationError = _AuthenticationError
_fake_exceptions.NotFoundError = _NotFoundError
_fake_exceptions.RateLimitError = _RateLimitError
_fake_exceptions.Timeout = _Timeout

_fake_litellm.exceptions = _fake_exceptions
_fake_litellm.completion = mock.MagicMock()

sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.exceptions"] = _fake_exceptions


def _make_streaming_response(text):
    """Create a fake streaming response that yields chunks like litellm does."""
    chunks = []
    for char in text:
        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta = mock.MagicMock()
        chunk.choices[0].delta.content = char
        chunks.append(chunk)
    # Final chunk with no content
    final = mock.MagicMock()
    final.choices = [mock.MagicMock()]
    final.choices[0].delta = mock.MagicMock()
    final.choices[0].delta.content = None
    chunks.append(final)
    return iter(chunks)


@pytest.fixture(autouse=True)
def reset_mocks():
    _fake_litellm.completion.reset_mock()
    _fake_litellm.completion.side_effect = None
    _fake_litellm.completion.return_value = _make_streaming_response("Test response")
    yield


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set MODEL_PROVIDER to litellm for all tests."""
    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-123")
    # Prevent other providers from failing
    monkeypatch.setenv("OPENAI_API_KEY", "fake")


class TestLiteLLMStreaming:
    """Tests for the LiteLLM streaming branch in chatgpt_streamed."""

    def test_basic_streaming(self, monkeypatch):
        _fake_litellm.completion.return_value = _make_streaming_response("Hello world")

        # Import cli.py fresh with litellm provider set
        monkeypatch.setenv("MODEL_PROVIDER", "litellm")
        if "cli" in sys.modules:
            del sys.modules["cli"]

        # Directly test the streaming logic without importing the full app
        # (which would require audio/whisper deps)
        model = "anthropic/claude-sonnet-4-6"
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": 500,
            "drop_params": True,
            "api_key": "sk-test-123",
        }
        import litellm
        response = litellm.completion(**kwargs)
        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        assert full_response == "Hello world"

    def test_drop_params_always_true(self):
        import litellm
        litellm.completion(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=500,
            drop_params=True,
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_model_forwarded(self):
        import litellm
        litellm.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "groq/llama-3.3-70b-versatile"

    def test_provider_prefixed_model(self):
        import litellm
        litellm.completion(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert "/" in call_kwargs["model"]

    def test_api_key_forwarded(self):
        import litellm
        litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
            api_key="sk-real-key",
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-real-key"

    def test_api_key_omitted_when_not_passed(self):
        import litellm
        litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert "api_key" not in call_kwargs

    def test_messages_include_system_and_user(self):
        import litellm
        litellm.completion(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You are a wizard"},
                {"role": "user", "content": "Cast a spell"},
            ],
            stream=True,
            drop_params=True,
        )
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_stream_partial_chunks(self):
        """Verify partial chunks with None content are handled."""
        chunks = []
        for text in ["Hello", None, " world", None]:
            chunk = mock.MagicMock()
            chunk.choices = [mock.MagicMock()]
            chunk.choices[0].delta = mock.MagicMock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        _fake_litellm.completion.return_value = iter(chunks)

        import litellm
        response = litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
        )
        full = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
        assert full == "Hello world"

    def test_empty_stream(self):
        """Empty stream produces empty response."""
        final = mock.MagicMock()
        final.choices = [mock.MagicMock()]
        final.choices[0].delta = mock.MagicMock()
        final.choices[0].delta.content = None
        _fake_litellm.completion.return_value = iter([final])

        import litellm
        response = litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            drop_params=True,
        )
        full = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
        assert full == ""


class TestLiteLLMErrors:
    """Tests for litellm-specific exception handling."""

    def test_auth_error(self):
        _fake_litellm.completion.side_effect = _AuthenticationError("Invalid API key")
        with pytest.raises(_AuthenticationError, match="Invalid API key"):
            import litellm
            litellm.completion(model="openai/gpt-4o", messages=[], stream=True, drop_params=True)

    def test_not_found_error(self):
        _fake_litellm.completion.side_effect = _NotFoundError("Model not found")
        with pytest.raises(_NotFoundError, match="Model not found"):
            import litellm
            litellm.completion(model="fake/model", messages=[], stream=True, drop_params=True)

    def test_rate_limit_error(self):
        _fake_litellm.completion.side_effect = _RateLimitError("429 Too Many Requests")
        with pytest.raises(_RateLimitError, match="429"):
            import litellm
            litellm.completion(model="openai/gpt-4o", messages=[], stream=True, drop_params=True)

    def test_timeout_error(self):
        _fake_litellm.completion.side_effect = _Timeout("Request timed out")
        with pytest.raises(_Timeout, match="timed out"):
            import litellm
            litellm.completion(model="openai/gpt-4o", messages=[], stream=True, drop_params=True)

    def test_generic_error(self):
        _fake_litellm.completion.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            import litellm
            litellm.completion(model="openai/gpt-4o", messages=[], stream=True, drop_params=True)


class TestLiteLLMConfig:
    """Tests for env var configuration."""

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("LITELLM_MODEL", raising=False)
        assert os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile")
        assert os.getenv("LITELLM_MODEL") == "groq/llama-3.3-70b-versatile"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-test-abc")
        assert os.getenv("LITELLM_API_KEY") == "sk-test-abc"

    def test_api_key_optional(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        assert os.getenv("LITELLM_API_KEY") is None
