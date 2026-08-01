import pytest

from app.openai_compat import build_openai_chat_payload


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2"],
)
def test_low_latency_models_disable_reasoning(model):
    payload = build_openai_chat_payload(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=100,
        stream=True,
    )

    assert payload["reasoning_effort"] == "none"
    assert payload["max_completion_tokens"] == 100
    assert payload["stream"] is True
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_legacy_models_do_not_receive_unsupported_reasoning_setting():
    payload = build_openai_chat_payload(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=100,
    )

    assert "reasoning_effort" not in payload
    assert payload["stream"] is False
