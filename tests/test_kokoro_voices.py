from app.main import (
    _build_kokoro_voice_dropdown,
    _ensure_kokoro_configured_voice,
    _normalize_kokoro_voice_entries,
)


def test_normalize_legacy_string_voices():
    data = {"voices": ["af_bella", "am_adam"]}
    entries = _normalize_kokoro_voice_entries(data)
    assert entries == [
        {"id": "af_bella", "label": "af_bella"},
        {"id": "am_adam", "label": "am_adam"},
    ]


def test_normalize_v04_object_voices():
    data = {
        "voices": [
            {"id": "af_heart", "name": "af_heart"},
            {"id": "my_custom_voice", "name": "My Custom Voice"},
        ]
    }
    entries = _normalize_kokoro_voice_entries(data)
    assert entries == [
        {"id": "af_heart", "label": "af_heart"},
        {"id": "my_custom_voice", "label": "My Custom Voice"},
    ]


def test_build_dropdown_uses_custom_voice_label():
    entries = [
        {"id": "af_bella", "label": "af_bella"},
        {"id": "my_custom_voice", "label": "My Custom Voice"},
    ]
    voices = _build_kokoro_voice_dropdown(entries)
    custom = next(v for v in voices if v["id"] == "my_custom_voice")
    assert custom["name"] == "My Custom Voice"


def test_build_dropdown_keeps_combo_voice_id():
    entries = [{"id": "af_bella+af_sky", "label": "af_bella+af_sky"}]
    voices = _build_kokoro_voice_dropdown(entries)
    combo = next(v for v in voices if v["id"] == "af_bella+af_sky")
    assert combo["name"] == "af_bella+af_sky"


def test_build_dropdown_keeps_weighted_combo_voice():
    entries = [{"id": "af_bella(2)+af_sky(1)", "label": "af_bella(2)+af_sky(1)"}]
    voices = _build_kokoro_voice_dropdown(entries)
    combo = next(v for v in voices if v["id"] == "af_bella(2)+af_sky(1)")
    assert combo["name"] == "af_bella(2)+af_sky(1)"


def test_ensure_configured_voice_injects_combo_mix():
    voices = [{"id": "af_bella", "name": "Bella (Female) - American English"}]
    result = _ensure_kokoro_configured_voice(voices, "af_bella+af_alloy")
    assert any(v["id"] == "af_bella+af_alloy" for v in result)
