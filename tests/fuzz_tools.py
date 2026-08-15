"""Property-based fuzz tests for tool-call parsing.

Feeds random JSON-like text into parse_tool_calls and verifies:
1. No crash (no uncaught exception)
2. When it returns calls, each ToolCall has a valid name and arguments
3. _fix_unbalanced_json doesn't crash and produces balanced output on success
4. _loads_lenient is idempotent — calling twice gives the same result
"""

from __future__ import annotations

import json
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from danyapi import tools as toolemu

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Raw tool-call JSON that a model might emit — including malformed variants
_tool_call_json = st.one_of(
    # Valid tool_calls wrapper
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
    # Legacy function_call
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
    # Bare array of calls
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
    # Direct single call (no wrapper)
    st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=30),
            "arguments": st.dictionaries(st.text(), st.none()),
        }
    ).map(json.dumps),
)

# Text that looks like JSON but may be broken — unmatched braces, trailing commas, etc.
_malformed_json = st.one_of(
    # Valid JSON with random text prepended/appended (simulates model chatter)
    st.text(max_size=100).flatmap(lambda prefix, _=_tool_call_json: _tool_call_json.map(lambda j, p=prefix: f"{p}\n\n{j}")),
    # Truncated JSON — cut valid JSON in half
    _tool_call_json.flatmap(lambda j: st.just(j[: len(j) // 2 + 1]) if len(j) > 4 else st.just(j)),
    # JSON with missing closing brackets (the DeepSeek bug pattern)
    _tool_call_json.map(lambda s: re.sub(r"\}(?=\])", "", s, count=1)),
    # XML invoke format
    st.text(min_size=1, max_size=20).map(lambda name: f'<invoke name="{name}"><parameter name="city">Moscow</parameter></invoke>'),
)

# General raw text — anything a model might spit out
_raw_text = st.one_of(
    _malformed_json,
    # Markdown code fences around JSON
    _tool_call_json.map(lambda j: f"```json\n{j}\n```"),
    # DSML markers
    _tool_call_json.map(lambda j: f"||DSML||\n{j}"),
    st.text(max_size=500),  # Pure random text
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(text=_raw_text)
def test_parse_tool_calls_no_crash(text: str) -> None:
    """parse_tool_calls must never raise an uncaught exception."""
    result = toolemu.parse_tool_calls(text)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


@settings(max_examples=200, deadline=None)
@given(text=_raw_text)
def test_parse_tool_calls_result_structure(text: str) -> None:
    """When parse_tool_calls returns calls, each must have a valid name."""
    result = toolemu.parse_tool_calls(text)
    if result is None:
        return
    calls, _wrapper = result
    assert isinstance(calls, list)
    for call in calls:
        # ToolCall invariant: name is non-empty string
        assert hasattr(call, "name")
        assert isinstance(call.name, str) and len(call.name) > 0


@settings(max_examples=200, deadline=None)
@given(text=_tool_call_json)
def test_parse_valid_tool_calls_succeeds(text: str) -> None:
    """Well-formed tool call JSON should parse successfully."""
    result = toolemu.parse_tool_calls(text)
    assert result is not None, f"Failed to parse valid tool call JSON: {text[:100]}"
    calls, _ = result
    assert len(calls) >= 1


@settings(max_examples=200, deadline=None)
@given(raw=st.text(max_size=500))
def test_fix_unbalanced_json_no_crash(raw: str) -> None:
    """_fix_unbalanced_json must never crash on any input."""
    result = toolemu._fix_unbalanced_json(raw)
    assert result is None or isinstance(result, str)


@settings(max_examples=200, deadline=None)
@given(raw=st.text(max_size=500))
def test_fix_unbalanced_json_produces_balanced_output(raw: str) -> None:
    """When _fix_unbalanced_json returns a result, it should be balanced."""
    result = toolemu._fix_unbalanced_json(raw)
    if result is None:
        return
    # Count braces/brackets (ignoring strings — simplified check)
    stripped = re.sub(r'"[^"]*"', "", result)  # remove string literals
    assert stripped.count("{") == stripped.count("}"), f"Unbalanced {{}} in: {result[:100]}"
    assert stripped.count("[") == stripped.count("]"), f"Unbalanced [] in: {result[:100]}"


@settings(max_examples=200, deadline=None)
@given(raw=st.text(max_size=300))
def test_loads_lenient_no_crash(raw: str) -> None:
    """_loads_lenient must either parse or raise ValueError — nothing else."""
    try:
        result = toolemu._loads_lenient(raw)
        assert isinstance(result, (dict, list, str, int, float, type(None)))
    except (ValueError, json.JSONDecodeError):
        pass  # expected


@settings(max_examples=200, deadline=None)
@given(raw=st.text(max_size=300))
def test_loads_lenient_idempotent(raw: str) -> None:
    """Calling _loads_lenient twice on the same input should give the same result."""
    try:
        r1 = toolemu._loads_lenient(raw)
        r2 = toolemu._loads_lenient(raw)
        assert json.dumps(r1, default=str, ensure_ascii=False) == json.dumps(r2, default=str, ensure_ascii=False)
    except (ValueError, json.JSONDecodeError):
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
