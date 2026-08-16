from __future__ import annotations

from .prompt import ToolCall


def format_tool_message(tool_calls: list[ToolCall], text: str, reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    message["tool_calls"] = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in tool_calls
    ]
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def tool_call_deltas(tool_calls: list[ToolCall], text: str | None = None) -> list[dict]:
    deltas: list[dict] = []
    if text:
        deltas.append({"role": "assistant", "content": text})
    for index, call in enumerate(tool_calls):
        deltas.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": ""},
                    }
                ],
            }
        )
        arguments = call.arguments
        if arguments:
            step = max(1, (len(arguments) + 5) // 6)
            for offset in range(0, len(arguments), step):
                deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments[offset : offset + step]}}]})
    return deltas
