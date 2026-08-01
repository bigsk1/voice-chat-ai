import pytest

from app.xai_compat import build_xai_chat_payload, normalize_xai_model, xai_chat_timeout


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    [
        ("grok-4.3", "none"),
        ("grok-4.3-latest", "none"),
        ("grok-latest", "none"),
        ("grok-4.5", "low"),
    ],
)
def test_current_models_receive_low_latency_reasoning_settings(model, reasoning_effort):
    payload = build_xai_chat_payload(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        stream=True,
    )

    assert payload["reasoning_effort"] == reasoning_effort
    assert payload["max_tokens"] == 100
    assert payload["stream"] is True
    assert "temperature" not in payload


def test_unknown_models_do_not_receive_reasoning_setting():
    payload = build_xai_chat_payload(
        model="custom-xai-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    assert "reasoning_effort" not in payload
    assert payload["stream"] is False


def test_reasoning_model_gets_longer_default_timeout():
    assert xai_chat_timeout("grok-4.3") == 45
    assert xai_chat_timeout("grok-4.5") == 120


def test_configured_timeout_overrides_model_default():
    assert xai_chat_timeout("grok-4.5", "180") == 180


@pytest.mark.parametrize(
    "retired_model",
    ["grok-4-1-fast-non-reasoning", "grok-2-vision-1212"],
)
def test_retired_models_are_normalized_to_grok_4_3(retired_model):
    assert normalize_xai_model(retired_model) == "grok-4.3"


def test_custom_model_is_preserved():
    assert normalize_xai_model("custom-xai-model") == "custom-xai-model"
