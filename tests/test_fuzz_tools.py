from __future__ import annotations

import json
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from danyapi import tools as toolemu

_tool_call_json = st.one_of(
    st.fixed_dictionaries(
        {
            "tool_calls": st.lists(
                st.fixed_dictionaries(
                    {
                        "name": st.text(min_size=1, max_size=30),
                        "arguments": st.dictionaries(st.text(), st.none()),
                    }
                ),
                min_size=1,
                max_size=5,
            ),
        }
    ).map(json.dumps),
    st.fixed_dictionaries(
        {
            "function_call": st.fixed_dictionaries(
                {
                    "name": st.text(min_size=1, max_size=30),
                    "arguments": st.text(max_size=200),
                }
            ),
        }
    ).map(json.dumps),
    st.lists(
        st.fixed_dictionaries(
            {
                "name": st.text(min_size=1, max_size=30),
                "arguments": st.dictionaries(st.text(), st.none()),
            }
        ),
        min_size=1,
        max_size=5,
    ).map(json.dumps),
    st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=30),
            "arguments": st.dictionaries(st.text(), st.none()),
        }
    ).map(json.dumps),
)


def _prefix_join(prefix: str) -> st.SearchStrategy[str]:
    return _tool_call_json.map(lambda j: f"{prefix}\n\n{j}")


def _halved(j: str) -> st.SearchStrategy[str]:
    if len(j) > 4:
        return st.just(j[: len(j) // 2 + 1])
    return st.just(j)


_malformed_json = st.one_of(
    st.text(max_size=100).flatmap(_prefix_join),
    _tool_call_json.flatmap(_halved),
    _tool_call_json.map(lambda s: re.sub(r"\}(?=\])", "", s, count=1)),
    st.text(min_size=1, max_size=20).map(lambda name: f'<invoke name="{name}"><parameter name="city">Moscow</parameter></invoke>'),
)

_raw_text = st.one_of(
    _malformed_json,
    _tool_call_json.map(lambda j: f"```json\n{j}\n```"),
    _tool_call_json.map(lambda j: f"||DSML||\n{j}"),
    st.text(max_size=250),
)


@settings(max_examples=60, deadline=None)
@given(text=_raw_text)
def test_parse_tool_calls_no_crash(text: str) -> None:
    result = toolemu.parse_tool_calls(text)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


@settings(max_examples=60, deadline=None)
@given(text=_raw_text)
def test_parse_tool_calls_result_structure(text: str) -> None:
    result = toolemu.parse_tool_calls(text)
    if result is None:
        return
    calls, _wrapper = result
    assert isinstance(calls, list)
    for call in calls:
        assert hasattr(call, "name")
        assert isinstance(call.name, str) and len(call.name) > 0


@settings(max_examples=60, deadline=None)
@given(text=_tool_call_json)
def test_parse_valid_tool_calls_succeeds(text: str) -> None:
    result = toolemu.parse_tool_calls(text)
    assert result is not None, f"Failed to parse valid tool call JSON: {text[:100]}"
    calls, _ = result
    assert len(calls) >= 1


@settings(max_examples=60, deadline=None)
@given(raw=st.text(max_size=250))
def test_fix_unbalanced_json_no_crash(raw: str) -> None:
    result = toolemu._fix_unbalanced_json(raw)
    assert result is None or isinstance(result, str)


def _balanced_brackets(text: str) -> bool:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack or (ch == "]" and stack[-1] != "[") or (ch == "}" and stack[-1] != "{"):
                return False
            stack.pop()
    return not stack and not in_string


@settings(max_examples=60, deadline=None)
@given(raw=st.text(max_size=250))
def test_fix_unbalanced_json_produces_balanced_output(raw: str) -> None:
    result = toolemu._fix_unbalanced_json(raw)
    if result is None:
        return
    assert _balanced_brackets(result), f"Unbalanced brackets in: {result[:100]}"


@settings(max_examples=60, deadline=None)
@given(raw=st.text(max_size=200))
def test_loads_lenient_no_crash(raw: str) -> None:
    try:
        result = toolemu._loads_lenient(raw)
        assert isinstance(result, (dict, list, str, int, float, type(None)))
    except (ValueError, json.JSONDecodeError):
        pass


@settings(max_examples=60, deadline=None)
@given(raw=st.text(max_size=200))
def test_loads_lenient_idempotent(raw: str) -> None:
    try:
        r1 = toolemu._loads_lenient(raw)
        r2 = toolemu._loads_lenient(raw)
        assert json.dumps(r1, default=str, ensure_ascii=False) == json.dumps(r2, default=str, ensure_ascii=False)
    except (ValueError, json.JSONDecodeError):
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
