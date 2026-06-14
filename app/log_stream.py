"""
Stream server stdout/stderr and logging to dashboard clients via /ws/logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import deque
from datetime import datetime
from typing import Any

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m(?:\x1b\]?[-\d;]*[A-Za-z])?")
MAX_LINE_LENGTH = 4096
RING_BUFFER_SIZE = 2000

log_clients: set[Any] = set()
_ring_buffer: deque[dict] = deque(maxlen=RING_BUFFER_SIZE)
_event_loop: asyncio.AbstractEventLoop | None = None
_installed = False
_ws_handler: logging.Handler | None = None


def ui_server_logs_enabled() -> bool:
    return os.getenv("UI_SERVER_LOGS", "true").lower() not in ("false", "0", "no")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def format_log_entry(message: str, tag: str = "APP", stream: str = "stdout") -> dict:
    cleaned = strip_ansi(message).strip()
    if len(cleaned) > MAX_LINE_LENGTH:
        cleaned = cleaned[:MAX_LINE_LENGTH] + "..."
    return {
        "action": "server_log",
        "ts": datetime.now().strftime("%H:%M:%S"),
        "tag": tag,
        "stream": stream,
        "message": cleaned,
    }


def get_log_history() -> list[dict]:
    return list(_ring_buffer)


def add_log_client(client) -> None:
    log_clients.add(client)


def remove_log_client(client) -> None:
    log_clients.discard(client)


async def broadcast_log(entry: dict) -> None:
    if not log_clients:
        return

    payload = json.dumps(entry)
    dead = []
    for client in list(log_clients):
        try:
            await client.send_text(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        log_clients.discard(client)


def _append_and_broadcast(entry: dict) -> None:
    if not entry.get("message"):
        return

    _ring_buffer.append(entry)
    loop = _event_loop
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_log(entry), loop)


def schedule_log(message: str, tag: str = "APP", stream: str = "stdout") -> None:
    if not ui_server_logs_enabled() or not _installed:
        return
    if not message or not message.strip():
        return
    _append_and_broadcast(format_log_entry(message, tag=tag, stream=stream))


class WebSocketLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            schedule_log(message, tag=record.levelname, stream="logging")
        except Exception:
            self.handleError(record)


class TeeStream:
    def __init__(self, original, stream_name: str, default_tag: str = "APP"):
        self._original = original
        self._stream_name = stream_name
        self._default_tag = default_tag
        self._buffer = ""

    def write(self, data) -> int:
        if not isinstance(data, str):
            data = str(data)

        written = self._original.write(data)
        if not data or not ui_server_logs_enabled() or not _installed:
            return written if written is not None else len(data)

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                schedule_log(line, tag=self._default_tag, stream=self._stream_name)
        return written if written is not None else len(data)

    def flush(self) -> None:
        if self._buffer.strip():
            schedule_log(self._buffer.rstrip(), tag=self._default_tag, stream=self._stream_name)
            self._buffer = ""
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


def install_log_streaming(loop: asyncio.AbstractEventLoop) -> None:
    global _installed, _event_loop, _ws_handler

    if _installed or not ui_server_logs_enabled():
        return

    _event_loop = loop
    sys.stdout = TeeStream(sys.stdout, "stdout", "APP")
    sys.stderr = TeeStream(sys.stderr, "stderr", "ERR")

    _ws_handler = WebSocketLogHandler()
    _ws_handler.setFormatter(logging.Formatter("%(message)s"))

    for logger_name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.addHandler(_ws_handler)

    _installed = True
    _append_and_broadcast(format_log_entry("Log streaming enabled", "SYS", "sys"))


async def send_log_history(websocket) -> None:
    await websocket.send_json({"action": "log_history", "lines": get_log_history()})
    await websocket.send_json(
        format_log_entry("Connected - showing new log entries", "SYS", "sys")
    )


async def close_log_clients() -> None:
    for client in list(log_clients):
        try:
            await client.close()
        except Exception:
            pass
    log_clients.clear()
