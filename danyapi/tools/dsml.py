from __future__ import annotations

import re
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from functools import lru_cache

from .common import _XML_STRAY_TOOL_CLOSE_RE, _XML_WRAPPER_CLOSE_RE
from .jsonfix import _extract_calls, _extract_json_object

_DSML_RUN_MAX = 8
MAX_BUFFER_CHARS = 64 * 1024
_DSML_PIPE = r"|\u00a6\u01c0\u01c1\u05c0\u2016\u2223\u2502\u2551\u2758\ufe31\uff5c"
_DSML_CHAR = r"(?:[|]|[^\x00-\x7f])"
_DSML_RUN = rf"{_DSML_CHAR}{{1,{_DSML_RUN_MAX}}}"
_DSML_MARKER = rf"{_DSML_RUN}\s*DSML\s*{_DSML_RUN}"
_DSML_PIPE_ANGLE = rf"[{_DSML_PIPE}<>]"
_DSML_PIPE_ANGLE_RUN = rf"{_DSML_PIPE_ANGLE}{{1,{_DSML_RUN_MAX}}}"
_DSML_BLOCK = re.compile(
    rf"<{_DSML_PIPE_ANGLE_RUN}\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE_RUN}>\s*DSML\s*<{_DSML_PIPE_ANGLE_RUN}\s*[a-zA-Z_][^<>]*\s*{_DSML_PIPE_ANGLE_RUN}>",
    re.IGNORECASE,
)
_DSML_WRAP = re.compile(
    rf"{_DSML_RUN}\s*>\s*DSML\s*<\s*{_DSML_RUN}",
    re.IGNORECASE,
)
_DSML_XML_NORMALIZE = re.compile(rf"<\s*(/?)\s*{_DSML_MARKER}\s*([a-zA-Z_][^<>]*)>", re.IGNORECASE)
_DSML_TAG = re.compile(rf"<\s*/?\s*{_DSML_MARKER}\s*[^<>]*>", re.IGNORECASE)
_DSML_NAKED = re.compile(rf"{_DSML_MARKER}", re.IGNORECASE)
_DSML_PRESENT = re.compile(r"dsml", re.IGNORECASE)


def _dsml_present(text: str) -> bool:
    if "dsml" in text:
        return True
    if text.isascii():
        return "dsml" in text.lower()
    return _DSML_PRESENT.search(text) is not None


_DSML_TOOL_CALLS_BLOCK = re.compile(
    rf"<{_DSML_MARKER}\s*tool_calls\b[^<>]*>(.*?)</{_DSML_MARKER}\s*tool_calls\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE = re.compile(
    rf"<{_DSML_MARKER}\s*invoke\b[^>]*?\sname\s*=\s*([\"']?)([^\s>\"']+)\1[^>]*>(.*?)</{_DSML_MARKER}\s*invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAMETER = re.compile(
    rf"<{_DSML_MARKER}\s*parameter\s+name\s*=\s*([\"']?)([^\"']+)\1[^>]*>(.*?)</{_DSML_MARKER}\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_HIDDEN_NAMES = (
    r"thinking|reasoning|thought|analysis|summary|abbreviation|"
    r"ds_safety|ds_sensitive|ds_core|ds_middle|ds_end|ds_pii|ds_related|"
    r"ds_rephrase|ds_translate|ds_bilingual|ds_inner|ds_header|ds_web_search|"
    r"search|result|reference|quote"
)
_DSML_HIDDEN_TAGS = tuple(_DSML_HIDDEN_NAMES.split("|"))
_DSML_HIDDEN_PATS = tuple(re.compile(rf"<{_DSML_MARKER}\s*{name}\b[^<>]*>", re.IGNORECASE) for name in _DSML_HIDDEN_TAGS)
_DSML_HIDDEN_CLOSE_PATS = tuple(re.compile(rf"</{_DSML_MARKER}\s*{name}\s*>", re.IGNORECASE) for name in _DSML_HIDDEN_TAGS)
_DSML_HIDDEN_GUARDS = tuple(re.compile(rf"{name}", re.IGNORECASE) for name in _DSML_HIDDEN_TAGS)
_DSML_HIDDEN_NAKED_PATS = tuple(re.compile(rf"{_DSML_MARKER}\s*<{name}\b[^<>]*>", re.IGNORECASE) for name in _DSML_HIDDEN_TAGS)
_DSML_HIDDEN_NAKED_CLOSE_PATS = tuple(re.compile(rf"</{name}>\s*{_DSML_MARKER}", re.IGNORECASE) for name in _DSML_HIDDEN_TAGS)
_DSML_HIDDEN_NAKED_GUARDS = _DSML_HIDDEN_GUARDS
_DSML_EQUALS = r"=\uff1d"
_DSML_LAX_MARKER = rf"(?:{_DSML_MARKER}|{_DSML_RUN})"
_DSML_LAX_SKIP_TAGS = frozenset(
    {"parameter", "tool_calls", "tool_call", "function_call", "function_calls", "call", "calls", "tool", "functions", "tools", "name"}
    | set(_DSML_HIDDEN_NAMES.split("|"))
)
_DSML_LAX_TAG = re.compile(
    rf"</?{_DSML_LAX_MARKER}\s*[a-zA-Z_][a-zA-Z0-9_-]*\b[^<>]*>",
    re.IGNORECASE,
)
_DSML_LAX_BLOCK = re.compile(
    rf"<{_DSML_LAX_MARKER}\s*(?:tool_calls|calls)\b[^<>]*>(?P<body>.*?)</{_DSML_LAX_MARKER}\s*(?:tool_calls|calls)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_LAX_OPENANY = re.compile(
    rf"<(?P<sep>{_DSML_LAX_MARKER})\s*(?P<tagname>[a-zA-Z_][a-zA-Z0-9_-]*)\b(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)
_DSML_LAX_NAME_ATTR = re.compile(rf"\bname\s*[{_DSML_EQUALS}]\s*([\"'])([^\"']+)\1", re.IGNORECASE)
_DSML_LAX_TOOLNAME_TAIL = re.compile(rf"(?<!parameter\s)\bname\s*[{_DSML_EQUALS}]\s*([\"'])([^\"']+)\1", re.IGNORECASE)
_DSML_LAX_PARAMETER = re.compile(
    rf"(?:<{_DSML_LAX_MARKER}\s*)?parameter\b\s+name\s*[{_DSML_EQUALS}]\s*"
    rf"([\"']?)(?P<name>[^\"']+)\1[^>]*>(?P<value>.*?)"
    rf"(?:</?{_DSML_LAX_MARKER}\s*parameter\s*>|/?\s*parameter\s*>|</?parameter\s*>|/?parameter\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_DSML_LAX_PARAMETER_HEAD = re.compile(
    rf"(?:<{_DSML_LAX_MARKER}\s*)?parameter\b\s+name\s*[{_DSML_EQUALS}]\s*",
    re.IGNORECASE,
)
_DSML_LAX_PARAMETER_END = re.compile(
    rf"(?:</?{_DSML_LAX_MARKER}\s*parameter\s*>|/?\s*parameter\s*>|</?parameter\s*>|/?parameter\s*>)",
    re.IGNORECASE,
)
_XML_SELFCLOSE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_-]*)\b([^<>]*?)/>", re.DOTALL | re.IGNORECASE)
_XML_OPEN_TAG_SCAN = re.compile(r"<\s*([a-zA-Z_][a-zA-Z0-9_-]*)\b([^<>]*)>", re.IGNORECASE)


@lru_cache(maxsize=256)
def _xml_close_pattern(name: str, name_space: bool, tail_space: bool) -> re.Pattern[str]:
    gap = r"[ \t\r\n]*"
    return re.compile(rf"</{gap if name_space else ''}{re.escape(name.lower())}{gap if tail_space else ''}>", re.IGNORECASE)


def _find_xml_close(text: str, name: str, start: int, name_space: bool, tail_space: bool) -> tuple[int, int] | None:
    match = _xml_close_pattern(name, name_space, tail_space).search(text, start)
    if match is None:
        return None
    return match.start(), match.end()


def _next_at(positions: list[int], start: int) -> int:
    index = bisect_left(positions, start)
    return positions[index] if index < len(positions) else -1


def _lax_unquoted_candidates(name_start: int, limit: int, greater: list[int]) -> Iterator[tuple[int, int]]:
    first = bisect_left(greater, limit)
    if first < len(greater) and limit > name_start:
        yield limit, greater[first]
    for index in range(first - 1, -1, -1):
        gt = greater[index]
        if gt <= name_start:
            return
        yield gt, gt


def _lax_parameter_bounds(
    text: str,
    name_start: int,
    spans: list[tuple[int, int]],
    greater: list[int],
    quotes: dict[str, list[int]],
) -> tuple[str, int, int] | None:
    quote = text[name_start : name_start + 1]
    if quote in quotes:
        closing = _next_at(quotes[quote], name_start + 1)
        if closing < name_start + 2:
            return None
        greater_at = _next_at(greater, closing + 1)
        if greater_at == -1:
            return None
        name = text[name_start + 1 : closing]
        tag_end = greater_at + 1
        term = bisect_left(spans, (tag_end, 0))
        if term >= len(spans):
            return None
        return name, term, tag_end
    quote_at = -1
    for positions in quotes.values():
        found = _next_at(positions, name_start)
        if found != -1 and (quote_at == -1 or found < quote_at):
            quote_at = found
    limit = len(text) if quote_at == -1 else quote_at
    for name_end, gt in _lax_unquoted_candidates(name_start, limit, greater):
        tag_end = gt + 1
        term = bisect_left(spans, (tag_end, 0))
        if term < len(spans):
            return text[name_start:name_end], term, tag_end
    return None


def _iter_dsml_lax_parameters(text: str) -> Iterator[tuple[int, int, str, str]]:
    if _DSML_LAX_PARAMETER_HEAD.search(text) is None:
        return
    spans = [match.span() for match in _DSML_LAX_PARAMETER_END.finditer(text)]
    if not spans:
        return
    greater = [index for index, char in enumerate(text) if char == ">"]
    quotes = {char: [index for index, current in enumerate(text) if current == char] for char in ('"', "'")}
    resume = 0
    for head in _DSML_LAX_PARAMETER_HEAD.finditer(text):
        if head.start() < resume:
            continue
        bounds = _lax_parameter_bounds(text, head.end(), spans, greater, quotes)
        if bounds is None:
            continue
        name, term, tag_end = bounds
        end = spans[term][1]
        resume = end
        yield head.start(), end, name.strip(), text[tag_end : spans[term][0]]


def _scan_xml_pairs(
    text: str,
    name_filter: frozenset[str] | None = None,
    name_space: bool = False,
    tail_space: bool = False,
) -> Iterator[tuple[int, int, str, str, str]]:
    pos = 0
    length = len(text)
    last_lt = -1
    last_gt = -1
    close_memo: dict[str, tuple[int, tuple[int, int] | None]] = {}
    while pos < length:
        open_match = _XML_OPEN_TAG_SCAN.search(text, pos)
        if open_match is None:
            return
        name = open_match.group(1)
        lowered = name.lower()
        if name_filter is not None and lowered not in name_filter:
            pos = open_match.end()
            continue
        attrs = open_match.group(2)
        if (attrs if not attrs[-1:].isspace() else attrs.rstrip()).endswith("/"):
            pos = open_match.end()
            continue
        body_start = open_match.end()
        memo = close_memo.get(lowered)
        if memo is not None and memo[0] <= body_start and (memo[1] is None or memo[1][0] >= body_start):
            close = memo[1]
        else:
            close = _find_xml_close(text, name, body_start, name_space, tail_space)
            close_memo[lowered] = (body_start, close)
        if close is None:
            if name_filter is None:
                pos = body_start
                continue
            if last_lt < body_start:
                last_lt = text.rfind("<", body_start)
            if last_gt < body_start:
                last_gt = text.rfind(">", body_start)
            if last_lt > last_gt:
                pos = body_start
                continue
            wrapper_close = _XML_WRAPPER_CLOSE_RE.search(text, body_start)
            trunc = wrapper_close.start() if wrapper_close is not None else length
            if _XML_STRAY_TOOL_CLOSE_RE.search(text, body_start, trunc):
                pos = body_start
                continue
            close = (trunc, trunc)
        close_start, end = close
        yield open_match.start(), end, name, attrs, text[body_start:close_start]
        pos = end


class _IntervalSet:
    __slots__ = ("ends", "starts")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []

    def add(self, start: int, end: int) -> None:
        i = bisect_right(self.ends, start)
        if i > 0 and start <= self.ends[i - 1]:
            i -= 1
        if i < len(self.ends) and start <= self.ends[i]:
            start = min(start, self.starts[i])
            end = max(end, self.ends[i])
            while i + 1 < len(self.ends) and self.starts[i + 1] <= end:
                end = max(end, self.ends[i + 1])
                del self.starts[i + 1]
                del self.ends[i + 1]
            self.starts[i] = start
            self.ends[i] = end
        else:
            self.starts.insert(i, start)
            self.ends.insert(i, end)

    def contains(self, start: int, end: int) -> bool:
        i = bisect_right(self.starts, start) - 1
        return i >= 0 and self.ends[i] >= end


def _blanked(text: str, mask: bytearray) -> str:
    if 1 not in mask:
        return text
    parts: list[str] = []
    cursor = 0
    i = 0
    n = len(text)
    while i < n:
        if mask[i]:
            parts.append(text[cursor:i])
            while i < n and mask[i]:
                i += 1
            parts.append(" ")
            cursor = i
        else:
            i += 1
    parts.append(text[cursor:])
    return "".join(parts)


_XML_WRAPPER_OPEN = re.compile(
    r"<(?:tool_calls|tool_call|function_calls|function_call|tools|calls|_calls)\b[^>]*>",
    re.IGNORECASE,
)
_XML_SKIP_ELEMENTS = frozenset(
    {
        "calls",
        "_calls",
        "tool_calls",
        "tool_call",
        "function_calls",
        "function_call",
        "functions",
        "tools",
        "invoke",
        "use_tool",
        "tool_use",
        "tool",
        "function",
        "parameter",
        "thinking",
        "reasoning",
        "thought",
        "analysis",
    }
)
_XML_GENERIC_TOOL_TAGS = frozenset(
    {
        "invoke",
        "toolinvoke",
        "tool_invoke",
        "use_tool",
        "tool_use",
        "call",
        "tool_call",
        "function_call",
        "action",
        "run",
    }
)
_XML_OPEN_TAG = re.compile(
    r"<(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools|calls|_calls)\b[^>]*>",
    re.IGNORECASE,
)
_XML_CLOSE_TAG = re.compile(
    r"</(?:tool_calls|tool_call|function_calls|function_call|functions|function|tools|calls|_calls"
    r"|invoke|toolinvoke|tool_invoke|use_tool|tool_use|call|action|run)\s*>",
    re.IGNORECASE,
)
_XML_HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "address",
        "area",
        "article",
        "aside",
        "audio",
        "b",
        "base",
        "bdi",
        "bdo",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "dialog",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "mark",
        "menu",
        "meta",
        "meter",
        "nav",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "param",
        "picture",
        "pre",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "script",
        "section",
        "select",
        "slot",
        "small",
        "source",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "template",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "track",
        "u",
        "ul",
        "var",
        "video",
        "wbr",
    }
)


def _replace_dsml_tag(match: re.Match) -> str:
    tag = match.group(0)
    extracted = _extract_json_object(tag)
    if extracted is not None:
        obj, start, end = extracted
        if _extract_calls(obj) is not None:
            return tag[start : end + 1]
    return " "


_DSML_STRIP_ROUNDS = 10


def _strip_markers(
    text: str,
    drop_hidden_spans: bool,
    drop_tail: bool,
    normalise_xml: bool,
    tag_replacement: str | Callable[[re.Match[str]], str],
) -> str:
    if not text:
        return text
    if not _dsml_present(text):
        return _drop_dangling(text) if drop_tail else text
    result = text
    for _ in range(_DSML_STRIP_ROUNDS):
        updated = _drop_spans(result, _hidden_spans(result)) if drop_hidden_spans else result
        updated = _DSML_BLOCK.sub(" ", updated)
        updated = _DSML_WRAP.sub(" ", updated)
        for pattern, guard, close_pattern in zip(_DSML_HIDDEN_PATS, _DSML_HIDDEN_GUARDS, _DSML_HIDDEN_CLOSE_PATS, strict=True):
            if guard.search(updated) is not None:
                updated = _drop_regex_spans(updated, pattern, close_pattern)
        for pattern, guard, close_pattern in zip(_DSML_HIDDEN_NAKED_PATS, _DSML_HIDDEN_NAKED_GUARDS, _DSML_HIDDEN_NAKED_CLOSE_PATS, strict=True):
            if guard.search(updated) is not None:
                updated = _drop_regex_spans(updated, pattern, close_pattern)
        if updated == result:
            break
        result = updated
        if not _dsml_present(result):
            break
    if drop_tail:
        result = _drop_dangling(result)
    if normalise_xml:
        result = _DSML_XML_NORMALIZE.sub(r"<\1\2>", result)
    result = _DSML_TAG.sub(tag_replacement, result)
    return _DSML_NAKED.sub(" ", result)


def _strip_dsml(text: str) -> str:
    return _strip_markers(
        text,
        drop_hidden_spans=False,
        drop_tail=False,
        normalise_xml=True,
        tag_replacement=_replace_dsml_tag,
    )


def strip_dsml(text: str) -> str:
    if not text:
        return text
    return _strip_output(text)


_DSML_SPACE = " \t\r\n"
_DSML_PIPE_RUN = "|\u00a6\u01c0\u01c1\u05c0\u2016\u2223\u2502\u2551\u2758\ufe31\uff5c"
_DSML_RUN_CHARS = _DSML_SPACE + "<>/" + _DSML_PIPE_RUN
_DSML_SIGNAL_CHARS = "<" + _DSML_PIPE_RUN
_DSML_CHAIN_MAX = 4
_DSML_STRADDLE_MAX = 64
_DSML_RUN_SET = frozenset(_DSML_RUN_CHARS)
_DSML_HIDDEN_NAME_SET = frozenset(_DSML_HIDDEN_NAMES.split("|"))
_DSML_SIGNAL_RE = re.compile(rf"[{re.escape(_DSML_SIGNAL_CHARS)}]")
_DSML_MARKER_RE = re.compile(rf"[{re.escape(_DSML_SIGNAL_CHARS)}]|[^\x00-\x7f]")
_DSML_PARTIAL_RE = re.compile(r"(?:DSM|DSML|DS|D)\Z", re.IGNORECASE)
_DSML_PARTIAL_LAST = frozenset("DdMmSsLl")
_DSML_TAG_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*")
_DSML_PIPE_RUN_RE = rf"[{_DSML_PIPE}]{{1,8}}"
_DSML_TAIL_RUN_RE = rf"\s*[{_DSML_PIPE}]{{0,8}}"
_DSML_DANGLING = re.compile(
    rf"(?:<[/]?\s*{_DSML_PIPE_RUN_RE}(?:\s*(?:DSM|DSML|DS|D))?{_DSML_TAIL_RUN_RE}"
    rf"(?:\s*[A-Za-z0-9_.:-]+(?:\s+[^\s<>]*)*)?"
    rf"|{_DSML_PIPE_RUN_RE}\s*(?:DSM|DSML|DS|D){_DSML_TAIL_RUN_RE})\Z",
    re.IGNORECASE,
)
_DSML_DANGLING_MAX = 40
_DSML_CLOSE_CACHE: dict[str, re.Pattern[str]] = {}
_DSML_CLOSE_CACHE_MAX = 64
_DSML_CLOSE_CACHE_LOCK = threading.Lock()


def _in_dsml_run(char: str) -> bool:
    return char in _DSML_RUN_SET or not char.isascii()


def _dsml_run_start(text: str, end: int, floor: int) -> int:
    limit = max(floor, end - _DSML_RUN_MAX)
    start = end
    while start > limit and _in_dsml_run(text[start - 1]):
        start -= 1
    return start


def _dsml_straddle_start(text: str, cut: int) -> int:
    size = len(text)
    if cut <= 0 or cut >= size:
        return cut
    low = max(0, cut - _DSML_STRADDLE_MAX)
    high = min(size, cut + _DSML_STRADDLE_MAX)
    found = cut
    for match in _DSML_NAKED.finditer(text, low, high):
        if match.start() < cut < match.end():
            found = match.start()
    return found


def _dsml_hold_start(text: str, floor: int = 0) -> int:
    size = len(text)
    if size <= floor:
        return size
    end = size
    signal = _DSML_SIGNAL_RE
    if text[size - 1] in _DSML_PARTIAL_LAST:
        partial = _DSML_PARTIAL_RE.search(text, floor)
        if partial is not None:
            end = partial.start()
            signal = _DSML_MARKER_RE
            if signal.search(text, floor, end) is None:
                return size
    start = _dsml_run_start(text, end, floor)
    if start == end:
        return end
    for _ in range(_DSML_CHAIN_MAX):
        lead = _DSML_PARTIAL_RE.search(text, floor, start)
        if lead is None:
            break
        back = _dsml_run_start(text, lead.start(), floor)
        if back >= lead.start() or _DSML_MARKER_RE.search(text, back, lead.start()) is None:
            break
        start = back
    if signal.search(text, start, end) is None:
        return end
    return start


def _dsml_tag_may_start(text: str, start: int) -> bool:
    size = len(text)
    index = start + 1
    if index < size and text[index] == "/":
        index += 1
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    if index >= size:
        return False
    return text[index] == "|" or text[index] > "\x7f"


def _dsml_tag_pending(text: str, start: int) -> bool:
    size = len(text)
    index = start + 1
    if index < size and text[index] == "/":
        index += 1
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    if index >= size:
        return True
    if text[index] != "|" and text[index] <= "\x7f":
        return False
    while index < size and (text[index] == "|" or text[index] > "\x7f"):
        index += 1
    if index >= size:
        return True
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    return "dsml".startswith(text[index : index + 4].lower())


def _dsml_dangling_pending(text: str, start: int) -> bool:
    size = len(text)
    index = start + 1
    if index < size and text[index] == "/":
        index += 1
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    if index >= size:
        return True
    return index >= size or text[index] in _DSML_PIPE_RUN


def _dsml_tag_at(text: str, start: int) -> tuple[int, str, bool, bool] | None:
    size = len(text)
    index = start + 1
    if index >= size:
        return None
    closing = False
    if text[index] == "/":
        closing = True
        index += 1
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    run = index
    while index < size and (text[index] == "|" or text[index] > "\x7f"):
        index += 1
    if index == run:
        return None
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    if text[index : index + 4].upper() != "DSML":
        return None
    index += 4
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    while index < size and (text[index] == "|" or text[index] > "\x7f"):
        index += 1
    while index < size and text[index] in _DSML_SPACE:
        index += 1
    if index < size and text[index] == ">":
        return index + 1, "", closing, False
    name_start = index
    while index < size and text[index] not in "<>":
        index += 1
    if index >= size or text[index] != ">":
        return None
    name = text[name_start:index].strip()
    self_closing = name.endswith("/")
    if self_closing:
        name = name[:-1].strip()
    found = _DSML_TAG_NAME_RE.match(name)
    return index + 1, found.group(0).lower() if found else "", closing, self_closing


def _dsml_close_pattern(name: str) -> re.Pattern[str]:
    pattern = _DSML_CLOSE_CACHE.get(name)
    if pattern is not None:
        return pattern
    with _DSML_CLOSE_CACHE_LOCK:
        pattern = _DSML_CLOSE_CACHE.get(name)
        if pattern is None:
            pattern = re.compile(rf"<\s*[^<>]*?\b{re.escape(name)}\b[^<>]*>", re.IGNORECASE)
            while len(_DSML_CLOSE_CACHE) >= _DSML_CLOSE_CACHE_MAX:
                _DSML_CLOSE_CACHE.pop(next(iter(_DSML_CLOSE_CACHE)))
            _DSML_CLOSE_CACHE[name] = pattern
    return pattern


def _dsml_matching_close(text: str, start: int, name: str) -> int | None:
    pattern = _dsml_close_pattern(name)
    depth = 1
    pos = start
    while True:
        match = pattern.search(text, pos)
        if match is None:
            return None
        parsed = _dsml_tag_at(text, match.start())
        if parsed is None:
            pos = match.end()
            continue
        end, tag_name, closing, self_closing = parsed
        if tag_name != name:
            pos = end
            continue
        if closing:
            depth -= 1
            if depth == 0:
                return end
        elif not self_closing:
            depth += 1
        pos = end


def _dsml_scan_cut(text: str, final: bool) -> int:
    size = len(text)
    pos = 0
    floor = 0
    while pos < size:
        found = text.find("<", pos)
        if found == -1:
            break
        index = found
        parsed = _dsml_tag_at(text, index)
        if parsed is None:
            if not final and _dsml_tag_may_start(text, index) and (_dsml_tag_pending(text, index) or _dsml_dangling_pending(text, index)):
                return index
            pos = index + 1
            continue
        end, name, closing, self_closing = parsed
        if not closing and not self_closing and name in _DSML_HIDDEN_NAME_SET:
            close = _dsml_matching_close(text, end, name)
            if close is None:
                if not final:
                    return index
                end = size
        pos = end
        floor = end
    return size if final else _dsml_hold_start(text, floor)


def _hidden_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    size = len(text)
    while pos < size:
        found = text.find("<", pos)
        if found == -1:
            break
        index = found
        parsed = _dsml_tag_at(text, index)
        if parsed is None:
            pos = index + 1
            continue
        end, name, closing, self_closing = parsed
        if closing or self_closing or name not in _DSML_HIDDEN_NAME_SET:
            pos = end
            continue
        close = _dsml_matching_close(text, end, name)
        if close is None:
            spans.append((index, size))
            return spans
        spans.append((index, close))
        pos = close
    return spans


def _drop_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            parts.append(text[cursor:start])
        parts.append(" ")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _drop_regex_spans(text: str, open_pattern: re.Pattern[str], close_pattern: re.Pattern[str]) -> str:
    closes = [(match.start(), match.end()) for match in close_pattern.finditer(text)]
    if not closes:
        return text
    spans: list[tuple[int, int]] = []
    pos = 0
    index = 0
    dead_from = len(text) + 1
    while True:
        open_match = open_pattern.search(text, pos)
        if open_match is None or open_match.end() >= dead_from:
            break
        while index < len(closes) and closes[index][0] < open_match.end():
            index += 1
        if index >= len(closes):
            dead_from = open_match.end()
            pos = open_match.start() + 1
            continue
        spans.append((open_match.start(), closes[index][1]))
        pos = closes[index][1]
    return _drop_spans(text, spans)


def _drop_dangling(text: str) -> str:
    look = len(text) - _DSML_DANGLING_MAX
    if look <= 0:
        return _DSML_DANGLING.sub(" ", text)
    if _DSML_DANGLING.search(text, look) is None:
        return text
    return text[:look] + _DSML_DANGLING.sub(" ", text[look:])


def _strip_output(text: str, drop_tail: bool = True) -> str:
    return _strip_markers(
        text,
        drop_hidden_spans=True,
        drop_tail=drop_tail,
        normalise_xml=False,
        tag_replacement=" ",
    )


class DsmlFilter:
    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str | None) -> str:
        if not text:
            return ""
        self._buf += text
        if len(self._buf) > MAX_BUFFER_CHARS:
            overflow = len(self._buf) - MAX_BUFFER_CHARS
            head = self._buf[:overflow]
            self._buf = self._buf[overflow:]
            return _strip_output(head, drop_tail=False)
        cut = _dsml_straddle_start(self._buf, _dsml_scan_cut(self._buf, False))
        if cut <= 0:
            return ""
        head = self._buf[:cut]
        self._buf = self._buf[cut:]
        return _strip_output(head, drop_tail=False)

    def flush(self) -> str:
        text = self._buf
        self._buf = ""
        if not text:
            return ""
        return _strip_output(text)
