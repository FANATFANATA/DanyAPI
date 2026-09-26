from __future__ import annotations

import re
from typing import Any

from ..sseutil import SSEEvent

ANSWER_PHASES = {"answer", "deep_research_answer", "ReportGeneration", "PdfMdGen"}
IMAGE_PHASES = {"image", "image_generation", "image_gen", "t2i"}
THINK_PHASES = {"think", "DeepThinking"}
SUMMARY_PHASE = "thinking_summary"

_IMAGE_URL_RE = re.compile(r"!\[[^\]]*\]\((https?://[^\s)>'\"]+)\)|(https?://cdn\.qwenlm\.ai/[^\s)>'\"]+)")
_IMAGE_TAIL_LIMIT = 4096
_TRAILING_PUNCT = ".,;:!?"


def _delta_text(delta: dict, key: str) -> str:
    value = delta.get(key)
    return value if isinstance(value, str) else ""


def _trailing_open_url(text: str) -> int | None:
    match = _IMAGE_URL_RE.search(text)
    start: int | None = None
    for match in _IMAGE_URL_RE.finditer(text):
        if match.group(1) is None and match.end() == len(text) and text[-1] not in _TRAILING_PUNCT:
            start = match.start()
    return start


def _extract_image_urls(text: str, final: bool = False) -> list[str]:
    if not final:
        start = _trailing_open_url(text)
        if start is not None:
            text = text[:start]
    urls: list[str] = []
    for match in _IMAGE_URL_RE.finditer(text):
        url = (match.group(1) or match.group(2) or "").rstrip(_TRAILING_PUNCT)
        if url:
            urls.append(url)
    return urls


def _incomplete_tail(window: str) -> str:
    if len(window) > _IMAGE_TAIL_LIMIT:
        window = window[-_IMAGE_TAIL_LIMIT:]
    start = _trailing_open_url(window)
    tail_index = start if start is not None else len(window)
    for marker in ("![", "http"):
        found = window.rfind(marker, 0, tail_index)
        if found != -1 and found < tail_index:
            tail_index = found
    return window[tail_index:] if tail_index < len(window) else ""


def _summary_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "value", "summary"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


class QwenStreamReconstructor:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._content_joined_cache: str | None = None
        self._reasoning_joined_cache: str | None = None
        self._nonempty: bool = False
        self.image_urls: list[str] = []
        self.image_size: tuple[int, int] | None = None
        self.finished: bool = False
        self.error: dict | None = None
        self.usage: dict = {}
        self._content_pending: list[str] = []
        self._reasoning_pending: list[str] = []
        self._reasoning_committed: list[str] = []
        self._reasoning_replaced: bool = False
        self._image_scan_tail: str = ""
        self._seen_image_urls: set[str] = set()

    @property
    def content(self) -> str:
        if self._content_joined_cache is None:
            self._content_joined_cache = "".join(self._content_parts)
        return self._content_joined_cache

    @property
    def reasoning(self) -> str:
        if self._reasoning_joined_cache is None:
            self._reasoning_joined_cache = "".join(self._reasoning_parts)
        return self._reasoning_joined_cache

    def _collect_image_urls(self, text: str) -> None:
        if not text:
            return
        window = self._image_scan_tail + text
        for url in _extract_image_urls(window):
            if url not in self._seen_image_urls:
                self._seen_image_urls.add(url)
                self.image_urls.append(url)
                self._nonempty = True
        self._image_scan_tail = _incomplete_tail(window)

    def finalize(self) -> None:
        tail = self._image_scan_tail
        self._image_scan_tail = ""
        if not tail:
            return
        for url in _extract_image_urls(tail, final=True):
            if url not in self._seen_image_urls:
                self._seen_image_urls.add(url)
                self.image_urls.append(url)
                self._nonempty = True

    def handle(self, event: SSEEvent) -> None:
        data = event.data
        if not isinstance(data, dict):
            return
        created = data.get("response.created")
        if isinstance(created, dict) and created.get("response_id"):
            self.response_id = created["response_id"]
        if data.get("response.stopped"):
            self.finished = True
        if data.get("done"):
            self.finished = True
        if data.get("response_id"):
            self.response_id = data["response_id"]
        if isinstance(data.get("usage"), dict):
            self.usage = data["usage"]
        if data.get("error"):
            error = data["error"]
            self.error = error if isinstance(error, dict) else {"code": "Internal_Server_Error", "details": error}
            return
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
                self._content_parts.append(text)
                self._content_joined_cache = None
                self._nonempty = True
                self._content_pending.append(text)
                self._collect_image_urls(text)
            image_field = delta.get("image_url") or delta.get("image")
            if isinstance(image_field, str) and image_field.startswith("http"):
                if image_field not in self._seen_image_urls:
                    self._seen_image_urls.add(image_field)
                    self.image_urls.append(image_field)
                    self._nonempty = True
            extra = delta.get("extra")
            if isinstance(extra, dict):
                hw = extra.get("output_image_hw")
                if isinstance(hw, list) and hw:
                    pair = hw[0]
                    if isinstance(pair, list) and len(pair) >= 2:
                        try:
                            w, h = int(pair[0]), int(pair[1])
                        except (TypeError, ValueError):
                            w = h = 0
                        if w > 0 and h > 0:
                            self.image_size = (w, h)
                for key in ("image_url", "image_urls", "images", "url"):
                    val = extra.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        if val not in self._seen_image_urls:
                            self._seen_image_urls.add(val)
                            self.image_urls.append(val)
                            self._nonempty = True
                    elif isinstance(val, list):
                        for item in val:
                            item_url = item if isinstance(item, str) else (item.get("url") if isinstance(item, dict) else None)
                            if isinstance(item_url, str) and item_url.startswith("http"):
                                if item_url not in self._seen_image_urls:
                                    self._seen_image_urls.add(item_url)
                                    self.image_urls.append(item_url)
                                    self._nonempty = True
        elif phase in ANSWER_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self._content_parts.append(text)
                self._content_joined_cache = None
                self._nonempty = True
                self._content_pending.append(text)
                self._collect_image_urls(text)
        elif phase in THINK_PHASES:
            text = _delta_text(delta, "content")
            if text:
                self._reasoning_parts.append(text)
                self._reasoning_joined_cache = None
                self._nonempty = True
                self._reasoning_pending.append(text)
        elif phase == SUMMARY_PHASE:
            extra = delta.get("extra")
            if isinstance(extra, dict):
                summary = extra.get("summary_thought")
                if isinstance(summary, dict):
                    items = summary.get("content")
                    if isinstance(items, list):
                        joined = "\n\n".join(text for text in (_summary_text(item) for item in items) if text)
                        if joined:
                            self._reasoning_parts[:] = [joined]
                            self._reasoning_joined_cache = joined
                            self._nonempty = True
                            self._reasoning_pending[:] = [joined]
                            self._reasoning_replaced = True

    def take_diffs(self) -> tuple[str, str]:
        c_diff = "".join(self._content_pending)
        self._content_pending.clear()
        if self._reasoning_replaced:
            current_reasoning = "".join(self._reasoning_parts)
            r_diff = current_reasoning.removeprefix("".join(self._reasoning_committed))
            self._reasoning_committed = list(self._reasoning_parts)
            self._reasoning_pending.clear()
            self._reasoning_replaced = False
        else:
            self._reasoning_committed.extend(self._reasoning_pending)
            r_diff = "".join(self._reasoning_pending)
            self._reasoning_pending.clear()
        return c_diff, r_diff

    @property
    def has_content(self) -> bool:
        return self._nonempty

    @property
    def usage_tokens(self) -> dict:
        def _int(value: Any) -> int:
            if isinstance(value, bool):
                return 0
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if not isinstance(value, str):
                return 0
            try:
                return int(value.strip())
            except (TypeError, ValueError):
                pass
            try:
                return int(float(value.strip()))
            except (TypeError, ValueError):
                return 0

        return {
            "prompt_tokens": _int(self.usage.get("input_tokens")),
            "completion_tokens": _int(self.usage.get("output_tokens")),
            "total_tokens": _int(self.usage.get("total_tokens")),
        }


def error_code(error: dict | None) -> str | None:
    if not error:
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None
