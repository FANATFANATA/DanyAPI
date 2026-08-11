"""Минимальный парсер Server-Sent Events для стрима /api/v0/chat/completion."""

from __future__ import annotations

import json
from typing import Any, Optional


class SSEEvent:
    __slots__ = ("event", "data")

    def __init__(self, event: Optional[str], data: Any) -> None:
        self.event = event
        self.data = data

    def __repr__(self) -> str:  # pragma: no cover
        return f"SSEEvent({self.event!r}, {self.data!r})"


def parse_sse(data: str) -> list[SSEEvent]:
    """Разбирает сырой буфер SSE в список событий.

    События разделяются пустой строкой; поля `event:` и `data:`.
    Если имя события отсутствует, но есть data - это delta-событие.
    """
    events: list[SSEEvent] = []
    event_name: Optional[str] = None
    data_lines: list[str] = []
    for raw_line in data.split("\n"):
        line = raw_line.strip("\r")
        if line == "":
            if data_lines:
                raw_data = "\n".join(data_lines)
                payload: Any
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    payload = raw_data
                events.append(SSEEvent(event_name, payload))
                event_name = None
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = raw_data
        events.append(SSEEvent(event_name, payload))
    return events
