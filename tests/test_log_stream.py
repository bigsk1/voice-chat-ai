import asyncio
import logging

from app import log_stream


@pytest.fixture(autouse=True)
def reset_log_stream_state(monkeypatch):
    monkeypatch.setenv("UI_SERVER_LOGS", "true")
    log_stream.log_clients.clear()
    log_stream._ring_buffer.clear()
    log_stream._installed = False
    log_stream._event_loop = None
    log_stream._ws_handler = None


def test_ui_server_logs_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("UI_SERVER_LOGS", raising=False)
    assert log_stream.ui_server_logs_enabled() is True


def test_ui_server_logs_can_be_disabled(monkeypatch):
    monkeypatch.setenv("UI_SERVER_LOGS", "false")
    assert log_stream.ui_server_logs_enabled() is False


def test_strip_ansi():
    assert log_stream.strip_ansi("\x1b[92mWaiting for speech...\x1b[0m") == "Waiting for speech..."


def test_format_log_entry_truncates_long_lines():
    entry = log_stream.format_log_entry("x" * 5000, tag="APP")
    assert len(entry["message"]) <= 4099
    assert entry["message"].endswith("...")


def test_broadcast_log_reaches_connected_client():
    sent = []

    class FakeWebSocket:
        async def send_text(self, payload):
            sent.append(payload)

    async def runner():
        client = FakeWebSocket()
        log_stream.add_log_client(client)
        entry = log_stream.format_log_entry("Speech detected", tag="APP")
        await log_stream.broadcast_log(entry)
        assert len(sent) == 1
        assert "Speech detected" in sent[0]

    asyncio.run(runner())


def test_install_log_streaming_disabled(monkeypatch):
    monkeypatch.setenv("UI_SERVER_LOGS", "false")

    async def runner():
        log_stream.install_log_streaming(asyncio.get_running_loop())

    asyncio.run(runner())
    assert log_stream._installed is False
