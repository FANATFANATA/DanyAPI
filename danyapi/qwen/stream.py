from __future__ import annotations

import re

from ..deepseek.sse import SSEEvent

ANSWER_PHASES = {"answer", "deep_research_answer", "ReportGeneration", "PdfMdGen"}
IMAGE_PHASES = {"image", "image_generation", "image_gen", "t2i"}
THINK_PHASES = {"think", "DeepThinking"}
SUMMARY_PHASE = "thinking_summary"

_IMAGE_URL_RE = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)|(https?://cdn\.qwenlm\.ai/[^\s)\"]+)")


def _delta_text(delta: dict, key: str) -> str:
    value = delta.get(key)
    return value if isinstance(value, str) else ""


def _extract_image_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in _IMAGE_URL_RE.finditer(text):
        url = match.group(1) or match.group(2)
        if url:
            urls.append(url)
    return urls


class QwenStreamReconstructor:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.content: str = ""
        self.reasoning: str = ""
        self.image_urls: list[str] = []
        self.finished: bool = False
        self.error: dict | None = None
        self.usage: dict = {}
        self._prev_content: str = ""
        self._prev_reasoning: str = ""
        self._seen_image_urls: set[str] = set()

    def _collect_image_urls(self, text: str) -> None:
        for url in _extract_image_urls(text):
            if url not in self._seen_image_urls:
                self._seen_image_urls.add(url)
                self.image_urls.append(url)

    def _extract_images_from_content(self, content: str) -> str:
        if "cdn.qwenlm.ai" not in content and "![" not in content:
            return content
        self._collect_image_urls(content)
        cleaned = _IMAGE_URL_RE.sub("", content)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

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
        if phase in IMAGE_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self._collect_image_urls(text)
                self.content += text
            image_field = delta.get("image_url") or delta.get("image")
            if isinstance(image_field, str) and image_field.startswith("http"):
                if image_field not in self._seen_image_urls:
                    self._seen_image_urls.add(image_field)
                    self.image_urls.append(image_field)
            extra = delta.get("extra")
            if isinstance(extra, dict):
                for key in ("image_url", "image_urls", "images", "url"):
                    val = extra.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        if val not in self._seen_image_urls:
                            self._seen_image_urls.add(val)
                            self.image_urls.append(val)
                    elif isinstance(val, list):
                        for item in val:
                            item_url = item if isinstance(item, str) else (item.get("url") if isinstance(item, dict) else None)
                            if isinstance(item_url, str) and item_url.startswith("http"):
                                if item_url not in self._seen_image_urls:
                                    self._seen_image_urls.add(item_url)
                                    self.image_urls.append(item_url)
        elif phase in ANSWER_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self.content += text
                self._collect_image_urls(text)
        elif phase in THINK_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self.reasoning += text
        elif phase == SUMMARY_PHASE:
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
        return bool(self.content or self.reasoning or self.image_urls)

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
