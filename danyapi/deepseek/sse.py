from __future__ import annotations

import json
from typing import Any


class SSEEvent:
    __slots__ = ("data", "event")

    def __init__(self, event: str | None, data: Any) -> None:
        self.event = event
        self.data = data


def _decode(raw_data: str) -> Any:
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        return raw_data


def parse_sse(data: str) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in data.split("\n"):
        line = raw_line.strip("\r")
        if line == "":
            if data_lines:
                events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
                event_name = None
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        events.append(SSEEvent(event_name, _decode("\n".join(data_lines))))
    return events
