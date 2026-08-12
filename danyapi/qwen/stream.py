from __future__ import annotations

from ..deepseek.sse import SSEEvent

ANSWER_PHASES = {"answer", "deep_research_answer", "ReportGeneration", "PdfMdGen"}
THINK_PHASES = {"think", "DeepThinking"}
SUMMARY_PHASE = "thinking_summary"


def _delta_text(delta: dict, key: str) -> str:
    value = delta.get(key)
    return value if isinstance(value, str) else ""


class QwenStreamReconstructor:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.content: str = ""
        self.reasoning: str = ""
        self.finished: bool = False
        self.error: dict | None = None
        self.usage: dict = {}
        self._prev_content: str = ""
        self._prev_reasoning: str = ""
        self._summary_mode: bool = False

    def handle(self, event: SSEEvent) -> None:
        data = event.data
        if not isinstance(data, dict):
            return
        created = data.get("response.created")
        if isinstance(created, dict) and created.get("response_id"):
            self.response_id = created["response_id"]
        if data.get("response.stopped") is not None:
            self.finished = True
        if data.get("done"):
            self.finished = True
        if data.get("response_id"):
            self.response_id = data["response_id"]
        if data.get("error"):
            error = data["error"]
            self.error = error if isinstance(error, dict) else {"code": "Internal_Server_Error", "details": error}
            return
        if isinstance(data.get("usage"), dict):
            self.usage = data["usage"]
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return
        if delta.get("status") == "finished":
            self.finished = True
        phase = delta.get("phase") or ""
        if phase in ANSWER_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self.content += text
        elif phase in THINK_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self.reasoning += text
                self._summary_mode = False
        elif phase == SUMMARY_PHASE:
            self._summary_mode = True
            extra = delta.get("extra")
            if isinstance(extra, dict):
                summary = extra.get("summary_thought")
                if isinstance(summary, dict):
                    items = summary.get("content")
                    if isinstance(items, list):
                        joined = "\n\n".join(str(item) for item in items if item)
                        if joined:
                            self.reasoning = joined

    def take_diffs(self) -> tuple[str, str]:
        content, reasoning = self.content, self.reasoning
        c_diff = content.removeprefix(self._prev_content)
        r_diff = reasoning.removeprefix(self._prev_reasoning)
        self._prev_content, self._prev_reasoning = content, reasoning
        return c_diff, r_diff

    @property
    def has_content(self) -> bool:
        return bool(self.content or self.reasoning)

    @property
    def usage_tokens(self) -> dict:
        return {
            "prompt_tokens": self.usage.get("input_tokens", 0),
            "completion_tokens": self.usage.get("output_tokens", 0),
            "total_tokens": self.usage.get("total_tokens", 0),
        }


def error_code(error: dict | None) -> str | None:
    if not error:
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None
