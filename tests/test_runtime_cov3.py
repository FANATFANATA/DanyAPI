from __future__ import annotations

import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import danyapi.sessions as sessions_mod
import danyapi.sseutil as sseutil_mod
import danyapi.tokens as tokens_mod
import danyapi.usage as usage_mod
from danyapi import store as store_mod
from danyapi.sessions import SessionRegistry
from danyapi.sseutil import IncrementalSSE, MessageReconstructor, SSEEvent, StreamStopFilter
from danyapi.store import JsonStore
from danyapi.tokens import StreamBudget, count_message_tokens, estimate_tokens, trim_to_tokens
from danyapi.usage import UsageTracker, init_tracker, record_usage_dict, reset_tracker


class Chatty:
    def __init__(self, prefix: str = "srv") -> None:
        self.prefix = prefix
        self.created: list[str] = []

    async def create_session(self, **kwargs: Any) -> Any:
        self.created.append(self.prefix)
        return SimpleNamespace(id=f"{self.prefix}-{len(self.created)}", last_message_id=None, title="", accumulated_tokens=0)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(store_mod.settings, "cache_enabled", True)
    return tmp_path


def sessions_store(cache_dir: Any) -> JsonStore:
    return JsonStore("sessions", "default")


def test_as_int_rejects_booleans_and_non_finite_numbers():
    assert sessions_mod._as_int(True) == 0
    assert sessions_mod._as_int(False, 7) == 7
    assert sessions_mod._as_int(float("inf"), 5) == 5
    assert sessions_mod._as_int(float("nan")) == 0
    assert sessions_mod._as_int(2.9) == 2
    assert sessions_mod._as_int(3) == 3


def test_as_int_parses_a_numeric_string_then_falls_back_to_a_float():
    assert sessions_mod._as_int("42") == 42
    assert sessions_mod._as_int("12.9") == 12
    assert sessions_mod._as_int("-4") == -4
    assert sessions_mod._as_int("inf") == 0
    assert sessions_mod._as_int("nan") == 0
    assert sessions_mod._as_int("nonsense") == 0
    assert sessions_mod._as_int(None) == 0
    assert sessions_mod._as_int([1]) == 0


def test_a_restored_session_coerces_its_token_count(cache_dir):
    store = sessions_store(cache_dir)
    store.set("inf", {"id": "c-inf", "accumulated_tokens": float("inf")})
    store.set("str-inf", {"id": "c-str-inf", "accumulated_tokens": "inf"})
    store.set("str-num", {"id": "c-str-num", "accumulated_tokens": "12.9"})
    store.set("bool", {"id": "c-bool", "accumulated_tokens": True})
    reg = SessionRegistry(Chatty(), store=JsonStore("sessions", "default"))
    assert reg.get("inf").accumulated_tokens == 0
    assert reg.get("str-inf").accumulated_tokens == 0
    assert reg.get("str-num").accumulated_tokens == 12
    assert reg.get("bool").accumulated_tokens == 0


async def test_two_concurrent_obtains_on_one_key_create_one_session():
    client = Chatty()
    reg = SessionRegistry(client)
    gate = asyncio.Event()
    entered = asyncio.Event()
    creates: list[str] = []

    async def slow_create(**kwargs: Any) -> Any:
        creates.append("create")
        entered.set()
        await gate.wait()
        return await client.create_session(**kwargs)

    reg._create = slow_create
    first = asyncio.create_task(reg.obtain("same-key"))
    await entered.wait()
    assert reg._session_refs == {"same-key": 1}
    second = asyncio.create_task(reg.obtain("same-key"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not second.done()
    assert reg._session_refs == {"same-key": 2}
    assert creates == ["create"]
    gate.set()
    first_session, first_key = await first
    second_session, second_key = await second
    assert first_key == "same-key"
    assert second_key == "same-key"
    assert first_session is second_session
    assert first_session.id == "srv-1"
    assert creates == ["create"]
    assert client.created == ["srv"]
    assert reg._session_refs == {}
    assert reg._session_locks == {}


async def test_releasing_a_lock_that_was_never_taken_is_a_noop():
    reg = SessionRegistry(Chatty())
    await reg._release_session_lock("never-seen")
    assert reg._session_refs == {}
    assert reg._session_locks == {}


async def test_a_referenced_lock_survives_a_release_and_loses_its_last_reference():
    reg = SessionRegistry(Chatty())
    lock = await reg._session_lock("k")
    assert reg._session_refs == {"k": 1}
    await reg._session_lock("k")
    assert reg._session_refs == {"k": 2}
    assert reg._session_locks["k"] is lock
    await reg._release_session_lock("k")
    assert reg._session_refs == {"k": 1}
    assert reg._session_locks["k"] is lock
    await reg._release_session_lock("k")
    assert reg._session_refs == {}
    assert reg._session_locks == {}


def test_touch_last_message_ignores_a_blank_message_id(cache_dir):
    store = sessions_store(cache_dir)
    reg = SessionRegistry(Chatty(), store=store)
    _session, key = asyncio.run(reg.obtain(None))
    reg.touch_last_message(key, "m1")
    reg.touch_last_message(key, None)
    reg.touch_last_message(key, "")
    assert reg.get(key).last_message_id == "m1"
    assert store.get(key)["last_message_id"] == "m1"


def test_touch_last_message_serialises_each_update_under_one_lock(cache_dir):
    store = sessions_store(cache_dir)
    reg = SessionRegistry(Chatty(), store=store)
    session, key = asyncio.run(reg.obtain(None))
    written: list[str] = []
    real_set = store.set

    def recording_set(seen_key: str, value: Any) -> None:
        written.append(value["last_message_id"])
        real_set(seen_key, value)

    store.set = recording_set  # type: ignore[method-assign]
    first_entered = threading.Event()
    release_first = threading.Event()
    real_update = reg._update_last
    calls = {"n": 0}

    def blocking_update(target: Any, message_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            first_entered.set()
            release_first.wait(5)
        real_update(target, message_id)

    reg._update_last = blocking_update  # type: ignore[method-assign]

    def touch(message_id: str) -> None:
        reg.touch_last_message(key, message_id)

    first = threading.Thread(target=touch, args=("m1",))
    first.start()
    assert first_entered.wait(5)
    assert session.last_message_id is None
    assert written == []

    second = threading.Thread(target=touch, args=("m2",))
    second.start()
    second.join(0.05)
    assert second.is_alive()
    assert session.last_message_id is None
    assert written == []

    release_first.set()
    first.join(5)
    second.join(5)
    assert written == ["m1", "m2"]
    assert session.last_message_id == "m2"
    assert store.get(key)["last_message_id"] == "m2"


def test_the_base_reuse_accepts_any_arguments_and_always_confirms():
    reg = SessionRegistry(Chatty())
    session = SimpleNamespace(id="c1")
    assert reg._reuse(session, "c1") is True
    assert reg._reuse(None, "", model="whatever", extra=1) is True
    assert reg._reuse(object(), "other", unknown_kwarg=True) is True
    _session, key = asyncio.run(reg.obtain(None))
    assert reg.can_reuse(key) is True
    assert reg.can_reuse(key, model="a-different-model") is True


def test_close_all_discards_every_known_session_when_there_is_no_prefix(cache_dir):
    store = sessions_store(cache_dir)
    reg = SessionRegistry(Chatty(), store=store)
    _s1, k1 = asyncio.run(reg.obtain(None))
    _s2, k2 = asyncio.run(reg.obtain(None))
    assert k1 in store
    assert k2 in store
    reg.close_all()
    assert k1 not in store
    assert k2 not in store
    assert reg._sessions == {}


def test_flush_reaches_the_store(cache_dir):
    store = sessions_store(cache_dir)
    reg = SessionRegistry(Chatty(), store=store)
    _session, key = asyncio.run(reg.obtain(None))
    assert key in store
    reg.flush()
    assert json.loads((cache_dir / "sessions-default.json").read_text(encoding="utf-8"))[key]["id"] == f"{key}"


def test_restore_keeps_the_canonical_session_for_every_alias(cache_dir):
    store = sessions_store(cache_dir)
    store.set("c1", {"id": "c1", "title": "canonical"})
    store.set("alias-1", {"id": "c1", "title": "duplicate"})
    reg = SessionRegistry(Chatty(), store=JsonStore("sessions", "default"))
    via_alias = reg.get("alias-1")
    via_id = reg.get("c1")
    assert via_alias is via_id
    assert via_alias.title == "canonical"


def test_as_count_falls_back_from_int_to_float_and_then_to_zero():
    assert usage_mod._as_count("7") == 7
    assert usage_mod._as_count("7.9") == 7
    assert usage_mod._as_count(float("inf")) == 0
    assert usage_mod._as_count("nonsense") == 0
    assert usage_mod._as_count(None) == 0
    assert usage_mod._as_count(-3) == 0
    assert usage_mod._as_count(-3.5) == 0


async def test_loop_active_is_true_inside_a_running_loop():
    assert usage_mod._loop_active() is True


def test_loop_active_is_false_on_a_worker_thread():
    seen: list[bool] = []
    thread = threading.Thread(target=lambda: seen.append(usage_mod._loop_active()))
    thread.start()
    thread.join(5)
    assert seen == [False]


def test_record_usage_dict_reads_each_field_independently():
    reset_tracker()
    tracker = init_tracker()
    record_usage_dict("deepseek", "m", {})
    record_usage_dict("deepseek", "m", {"prompt_tokens": "3", "completion_tokens": None, "total_tokens": "nonsense"})
    record_usage_dict("deepseek", "m", {"prompt_tokens": 4.8, "completion_tokens": 2, "total_tokens": 0})
    totals = tracker.snapshot()["totals"]
    assert totals == {"requests": 3, "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    reset_tracker()


def test_record_usage_dict_ignores_a_payload_that_is_not_a_mapping():
    reset_tracker()
    tracker = init_tracker()
    payload: Any = ["prompt_tokens", 5]
    record_usage_dict("deepseek", "m", payload)
    assert tracker.snapshot()["totals"] == {"requests": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    reset_tracker()


def test_a_restore_evicts_the_oversized_dimension_maps(cache_dir):
    store = JsonStore("usage-overflow", "default")
    store.set("usage", {"by_model": {"m1": {"requests": 1}, "m2": {"requests": 2}}})
    tracker = UsageTracker(store=store, max_records=1)
    assert list(tracker.snapshot()["by_model"]) == ["m2"]


def test_every_dimension_map_shares_the_recent_record_cap(cache_dir):
    tracker = UsageTracker(max_records=2)
    for model in ("m1", "m2", "m3"):
        tracker.record("p", model, 1, 1, 2, user="alice")
    snapshot = tracker.snapshot()
    assert list(snapshot["by_model"]) == ["m2", "m3"]
    assert list(snapshot["by_user"]) == ["alice"]
    assert len(snapshot["recent"]) == 2
    assert snapshot["totals"]["requests"] == 3


def test_usage_recorded_off_the_loop_persists_the_totals_on_every_call(cache_dir):
    tracker = UsageTracker(store=JsonStore("usage-throttle", "default"))
    assert usage_mod._loop_active() is False
    for _ in range(3):
        tracker.record("p", "m", 1, 1, 2)
    restored = UsageTracker(store=JsonStore("usage-throttle", "default"))
    assert restored.snapshot()["totals"]["requests"] == 3
    assert restored.snapshot()["totals"]["total_tokens"] == 6
    assert len(restored.snapshot()["recent"]) == 1
    tracker.flush()
    assert len(UsageTracker(store=JsonStore("usage-throttle", "default")).snapshot()["recent"]) == 3


def test_a_failing_store_write_is_reported_and_does_not_propagate(cache_dir, caplog):
    class Exploding:
        def get(self, key: str, default: Any = None) -> Any:
            return default

        def set(self, key: str, value: Any) -> None:
            raise RuntimeError("store is gone")

        def discard(self, key: str) -> None:
            raise RuntimeError("store is gone")

        def flush(self) -> None:
            raise RuntimeError("store is gone")

    tracker = UsageTracker(store=Exploding())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="danyapi.usage"):
        tracker.record("p", "m", 1, 1, 2)
        tracker.flush()
        tracker.reset()
    assert [record.getMessage() for record in caplog.records] == [
        "usage store write failed: store is gone",
        "usage flush failed: store is gone",
        "usage store clear failed: store is gone",
    ]


def test_a_snapshot_and_a_flush_hand_out_copies(cache_dir):
    store = JsonStore("usage-copies", "default")
    tracker = UsageTracker(store=store)
    tracker.record("p", "m", 1, 2, 3, user="alice", session_id="s1")
    tracker.flush()
    snap = tracker.snapshot()
    snap["recent"][0]["total_tokens"] = 999
    snap["recent"][0]["user"] = "mallory"
    snap["totals"]["requests"] = 999
    snap["by_model"]["m"]["total_tokens"] = 999
    snap["recent"].clear()
    fresh = tracker.snapshot()
    assert len(fresh["recent"]) == 1
    assert fresh["recent"][0]["total_tokens"] == 3
    assert fresh["recent"][0]["user"] == "alice"
    assert fresh["totals"]["requests"] == 1
    assert fresh["by_model"]["m"]["total_tokens"] == 3
    tracker.flush()
    persisted = store.get("usage_recent")
    assert len(persisted) == 1
    assert persisted[0]["total_tokens"] == 3
    assert persisted[0]["user"] == "alice"


def _ready(message_id: str = "asst_1") -> SSEEvent:
    return SSEEvent(
        None,
        {
            "o": "SET",
            "p": "response",
            "v": {
                "response": {
                    "message_id": message_id,
                    "role": "assistant",
                    "accumulated_token_usage": 0,
                    "status": "WIP",
                    "fragments": [{"type": "THINK", "content": ""}, {"type": "RESPONSE", "content": ""}],
                }
            },
        },
    )


def _append(index: int, piece: str) -> SSEEvent:
    return SSEEvent(None, {"o": "APPEND", "p": f"response/fragments/{index}/content", "v": piece})


def _primed() -> MessageReconstructor:
    rec = MessageReconstructor()
    rec.handle(_ready())
    rec.take_diffs()
    return rec


def test_parse_sse_keeps_a_non_json_payload_and_its_event_name():
    events = sseutil_mod.parse_sse("event: delta\ndata: not json\n\n")
    assert len(events) == 1
    assert events[0].event == "delta"
    assert events[0].data == "not json"


def test_parse_sse_drops_blank_lines_and_empty_frames():
    assert sseutil_mod.parse_sse("\n\n\n") == []
    assert sseutil_mod.parse_sse("") == []


def test_incremental_sse_finishes_with_a_trailing_partial_frame():
    inc = IncrementalSSE()
    assert [event.data for event in inc.feed(b'data: {"a":1}\n\ndata: {"b":2}\n')] == [{"a": 1}]
    assert [event.data for event in inc.finish()] == [{"b": 2}]
    assert inc._buffer == bytearray()
    assert list(inc.finish()) == []


def test_incremental_sse_drops_a_trailing_frame_of_only_whitespace():
    inc = IncrementalSSE()
    assert [event.data for event in inc.feed(b"data: x\n\n  \n")] == ["x"]
    assert list(inc.finish()) == []


def test_split_stop_normalises_every_accepted_shape():
    assert sseutil_mod.split_stop(None) == []
    assert sseutil_mod.split_stop("STOP") == ["STOP"]
    assert sseutil_mod.split_stop("") == []
    assert sseutil_mod.split_stop(["a", "", 3, None, "b"]) == ["a", "b"]
    assert sseutil_mod.split_stop(7) == []
    assert sseutil_mod.split_stop({"a": 1}) == []


def test_a_stop_filter_cuts_at_the_earliest_marker():
    filt = StreamStopFilter(["<|end|>", "</s>"])
    assert filt.feed("hello") == ("", False)
    assert filt.flush() == "hello"
    assert filt.feed("a</s>b") == ("a", True)
    assert filt.feed("a<|end|>b") == ("a", True)
    assert filt.feed("") == ("", False)
    assert filt.flush() == ""


def test_a_single_character_marker_holds_nothing_back():
    filt = StreamStopFilter(["!"])
    assert filt.feed("ab") == ("ab", False)
    assert filt.flush() == ""
    assert filt.feed("a!b") == ("a", True)


def test_a_two_character_marker_holds_one_character_back():
    filt = StreamStopFilter(["ab"])
    assert filt.feed("xyz") == ("xy", False)
    assert filt.flush() == "z"
    assert filt.feed("q") == ("", False)
    assert filt.flush() == "q"
    assert filt.feed("zzab") == ("zz", True)


def test_navigate_rejects_a_bad_index_a_missing_key_and_a_scalar():
    node: dict = {"a": [1, 2], "b": 5}
    assert sseutil_mod._navigate(node, ["a", "1"]) == 2
    assert sseutil_mod._navigate(node, ["a", "9"]) is None
    assert sseutil_mod._navigate(node, ["a", "zz"]) is None
    assert sseutil_mod._navigate(node, ["missing"]) is None
    assert sseutil_mod._navigate(node, ["b", "x"]) is None
    assert sseutil_mod._navigate(node, []) is node


def test_set_path_writes_into_a_list_and_refuses_a_bad_index():
    target: dict = {"fragments": ["a", "b"]}
    sseutil_mod._set_path(target, ("fragments", "1"), "c")
    assert target["fragments"] == ["a", "c"]
    sseutil_mod._set_path(target, ("fragments", "zz"), "d")
    sseutil_mod._set_path(target, ("fragments", "9"), "d")
    assert target["fragments"] == ["a", "c"]
    listed: dict = {"fragments": ["a", "b"]}
    sseutil_mod._set_path(listed, ("fragments", "xx", "content"), "b")
    assert listed["fragments"] == ["a", "b"]
    scalar: dict = {"a": 1}
    sseutil_mod._set_path(scalar, ("a", "b", "c"), 2)
    assert scalar == {"a": {}}
    sseutil_mod._set_path(scalar, ("a", "b"), {"c": 1})
    assert scalar == {"a": {"b": {"c": 1}}}
    sseutil_mod._set_path(scalar, ("a", "b"), 1)
    assert scalar == {"a": {"b": 1}}
    sseutil_mod._set_path(scalar, ("a", "b", "c"), 3)
    assert scalar == {"a": {"b": {"c": 3}}}


def test_set_path_walks_a_list_and_refuses_a_scalar_leaf_and_a_bad_index():
    target: dict = {"rows": [{"name": "a"}]}
    sseutil_mod._set_path(target, ("rows", "0", "name"), "z")
    assert target["rows"] == [{"name": "z"}]
    sseutil_mod._set_path(target, ("rows", "9", "name"), "z")
    sseutil_mod._set_path(target, ("rows", "zz", "name"), "z")
    assert target["rows"] == [{"name": "z"}]
    sseutil_mod._set_path(target, ("rows", "0", "name", "deep"), "z")
    assert target["rows"] == [{"name": {"deep": "z"}}]


def test_set_path_creates_only_the_root_intermediate():
    target: dict = {}
    sseutil_mod._set_path(target, ("fragments", "0", "content"), "x")
    assert target == {"fragments": {}}
    sseutil_mod._set_path(target, ("a", "b", "c"), 1)
    assert target == {"fragments": {}, "a": {}}
    sseutil_mod._set_path(target, ("a", "zz", "c"), 1)
    assert target == {"fragments": {}, "a": {}}
    rooted: dict = {"rows": ["a"]}
    sseutil_mod._set_path(rooted, ("rows", "zz", "name"), "x")
    assert rooted == {"rows": ["a"]}
    sseutil_mod._set_path(rooted, ("rows", "0", "name", "deep"), "x")
    assert rooted == {"rows": ["a"]}
    assert sseutil_mod._navigate({"a": {"b": 1}}, ("a", "b")) == 1


def test_init_message_gives_each_reconstructor_its_own_copy():
    payload = {"response": {"fragments": [{"type": "RESPONSE", "content": "seed"}], "status": "WIP"}}
    first = MessageReconstructor()
    second = MessageReconstructor()
    for rec in (first, second):
        rec.handle(SSEEvent(None, {"o": "SET", "p": "response", "v": payload}))
    assert first.message["fragments"] is not second.message["fragments"]
    assert first.message["fragments"] is not payload["response"]["fragments"]
    first.message["fragments"].append({"type": "RESPONSE", "content": "extra"})
    assert len(second.message["fragments"]) == 1
    assert len(payload["response"]["fragments"]) == 1
    assert second.content == "seed"


def test_apply_delta_applies_a_batch_of_sub_deltas():
    message: dict = {}
    sseutil_mod._apply_delta(
        message,
        "BATCH",
        "response/fragments",
        [
            {"o": "SET", "p": "response/status", "v": "DONE"},
            "not-a-dict",
            {"o": "SET", "p": "response/accumulated_token_usage", "v": 7},
        ],
    )
    assert message == {"status": "DONE", "accumulated_token_usage": 7}
    sseutil_mod._apply_delta(message, "BATCH", "response", "not-a-list")
    assert message == {"status": "DONE", "accumulated_token_usage": 7}


def test_apply_delta_appends_into_lists_and_objects():
    message: dict = {"fragments": ["a"], "list": [1], "plain": 1, "text": "x"}
    sseutil_mod._apply_delta(message, "APPEND", "response/fragments/0", "b")
    sseutil_mod._apply_delta(message, "APPEND", "response/fragments/9", "b")
    sseutil_mod._apply_delta(message, "APPEND", "response/fragments/zz", "b")
    sseutil_mod._apply_delta(message, "APPEND", "response/list", [2, 3])
    sseutil_mod._apply_delta(message, "APPEND", "response/list", 4)
    sseutil_mod._apply_delta(message, "APPEND", "response/plain", 9)
    sseutil_mod._apply_delta(message, "APPEND", "response/missing", "v")
    sseutil_mod._apply_delta(message, "APPEND", "response/missing", "w")
    sseutil_mod._apply_delta(message, "APPEND", "response/text", 7)
    sseutil_mod._apply_delta(message, "APPEND", "response/text", 8)
    assert message["fragments"] == ["ab"]
    assert message["list"] == [1, 2, 3, 4]
    assert message["plain"] == 9
    assert message["missing"] == "vw"
    assert message["text"] == 8
    fresh: dict = {"a": 1}
    sseutil_mod._apply_delta(fresh, "APPEND", "response/a", "s")
    assert fresh["a"] == "s"
    sseutil_mod._apply_delta(fresh, "APPEND", "response/a", "t")
    assert fresh["a"] == "st"
    absent: dict = {}
    sseutil_mod._apply_delta(absent, "APPEND", "response/a", 5)
    assert absent == {"a": 5}


def test_apply_delta_seeds_a_message_from_a_root_set_and_ignores_other_roots():
    message: dict = {"stale": 1}
    sseutil_mod._apply_delta(message, "SET", "", {"response": {"id": "c1"}})
    assert message == {"id": "c1"}
    sseutil_mod._apply_delta(message, "SET", "other/id", "x")
    sseutil_mod._apply_delta(message, "APPEND", "response", "x")
    assert message == {"id": "c1"}
    sseutil_mod._apply_delta(message, "SET", "", "not-a-mapping")
    assert message == {"id": "c1"}


def test_apply_delta_prefixes_a_relative_sub_path_with_its_batch_path():
    message: dict = {}
    sseutil_mod._apply_delta(message, "BATCH", "response", [{"o": "SET", "p": "status", "v": "DONE"}])
    assert message == {"status": "DONE"}
    sseutil_mod._apply_delta(message, "BATCH", "response/fragments", [{"o": "SET", "p": "response/status", "v": "X"}])
    assert message == {"status": "X"}


def test_fragment_text_reads_every_content_shape():
    assert sseutil_mod._fragment_text("plain") == "plain"
    assert sseutil_mod._fragment_text(7) == ""
    assert sseutil_mod._fragment_text({}) == ""
    assert sseutil_mod._fragment_text({"content": "text"}) == "text"
    assert sseutil_mod._fragment_text({"content": 5}) == ""
    assert sseutil_mod._fragment_text({"content": ["a", {"text": "b"}, {"text": 3}, 9]}) == "ab"
    assert sseutil_mod._fragment_text({"content": None}) == ""


def test_touches_aggregated_fragment_walks_a_batch_and_a_bad_index():
    assert sseutil_mod._touches_aggregated_fragment("SET", "response/fragments/0/content", "x", 2) is True
    assert sseutil_mod._touches_aggregated_fragment("SET", "response/fragments/5/content", "x", 2) is False
    assert sseutil_mod._touches_aggregated_fragment("SET", "response/fragments/zz/content", "x", 2) is False
    assert sseutil_mod._touches_aggregated_fragment("SET", "response/status", "x", 2) is False
    assert sseutil_mod._touches_aggregated_fragment("SET", "other/fragments/0", "x", 2) is False
    assert sseutil_mod._touches_aggregated_fragment("BATCH", "response", "not-a-list", 2) is False
    assert sseutil_mod._touches_aggregated_fragment("BATCH", "response", ["nope"], 2) is False
    assert (
        sseutil_mod._touches_aggregated_fragment(
            "BATCH",
            "response/fragments",
            [{"o": "SET", "p": "1/content", "v": "x"}],
            2,
        )
        is True
    )


def test_a_reconstructor_reads_the_ready_hint_and_the_error_hint():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response", "v": {"response": {"id": "c1"}}}))
    rec.handle(SSEEvent("ready", {"response_message_id": "asst_9"}))
    assert rec.response_message_id == "asst_9"
    rec.handle(SSEEvent("ready", "not-a-mapping"))
    rec.handle(SSEEvent("toast", {"type": "error", "content": "quota", "finish_reason": "length"}))
    assert rec.hint_error == {"message": "quota", "finish_reason": "length"}
    rec.handle(SSEEvent("hint", {"type": "error", "message": "fallback"}))
    assert rec.hint_error == {"message": "fallback", "finish_reason": None}
    rec.handle(SSEEvent("toast", {"type": "info"}))
    rec.handle(SSEEvent("hint", "not-a-mapping"))
    rec.handle(SSEEvent("other", {"a": 1}))
    rec.handle(SSEEvent(None, "not-a-mapping"))
    rec.handle(SSEEvent(None, {"o": "APPEND", "p": "response", "v": "x"}))
    assert rec.take_diffs() == ("", "")


def test_a_delta_with_a_non_string_path_reuses_the_last_one():
    rec = _primed()
    rec.handle(_append(1, "Hi"))
    assert rec.take_diffs() == ("Hi", "")
    rec.handle(SSEEvent(None, {"o": "APPEND", "p": 17, "v": "!"}))
    assert rec.content == "Hi!"
    assert rec._last_path == "response/fragments/1/content"
    rec.handle(SSEEvent(None, {"o": 17, "p": "response/fragments/1/content", "v": "?"}))
    assert rec.content == "Hi!?"


def test_a_fast_append_bails_out_on_every_unexpected_shape():
    rec = _primed()
    assert rec._fast_append_tail("SET", "response/fragments/1/content", "x") is False
    assert rec._fast_append_tail("APPEND", "response/fragments/1/content", 5) is False
    assert rec._fast_append_tail("APPEND", "response/fragments/1/content", "x") is True
    assert rec._fast_append_tail("APPEND", "response/fragments/5/content", "c") is False
    assert rec._fast_append_tail("APPEND", "response/fragments/zz/content", "c") is False
    assert rec._fast_append_tail("APPEND", "response/fragments", "c") is False
    assert rec._fast_append_tail("APPEND", "response/fragments/1", "c") is False
    assert rec._fast_append_tail("APPEND", "response/status", "c") is False
    assert rec._fast_append_tail("APPEND", "other/fragments/1/content", "c") is False
    rec.message["fragments"] = ["not-a-dict"]
    rec._agg_fragments = rec.message["fragments"]
    rec._frag_idx = 1
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "c") is False
    rec.message["fragments"] = [{"type": "UNKNOWN", "content": ""}]
    rec._agg_fragments = rec.message["fragments"]
    rec._frag_idx = 1
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "c") is False
    rec.message["fragments"] = []
    rec._agg_fragments = rec.message["fragments"]
    rec._frag_idx = 0
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "c") is False
    stale = MessageReconstructor()
    stale.message = {"fragments": [{"type": "RESPONSE", "content": ""}]}
    assert stale._fast_append_tail("APPEND", "response/fragments/0/content", "c") is False
    detached = MessageReconstructor()
    detached.message = {"fragments": [{"type": "RESPONSE", "content": ""}]}
    detached._agg_fragments = [{"type": "RESPONSE", "content": ""}]
    detached._frag_idx = 1
    assert detached._fast_append_tail("APPEND", "response/fragments/0/content", "c") is False
    trailing = _primed()
    trailing.message["fragments"].append({"type": "RESPONSE", "content": "late"})
    assert trailing._fast_append_tail("APPEND", "response/fragments/1/content", "c") is False


def test_a_fast_append_sends_reasoning_fragments_to_the_reasoning_buffer():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response", "v": {"response": {"fragments": [{"type": "THINK", "content": ""}]}}}))
    rec.take_diffs()
    assert rec._fast_append_tail("APPEND", "response/fragments/0/content", "why") is True
    assert rec._reasoning_parts == ["why"]
    assert rec._content_parts == []
    assert rec.reasoning == "why"


def test_a_rebuild_from_a_replaced_fragment_list_reads_both_buffers():
    rec = _primed()
    rec.handle(_append(0, "think "))
    rec.handle(_append(1, "answer "))
    assert rec.content == "answer "
    assert rec.reasoning == "think "
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments", "v": [{"type": "THINK", "content": "only-think"}]}))
    assert (rec.content, rec.reasoning) == ("", "only-think")
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments", "v": "not-a-list"}))
    assert (rec.content, rec.reasoning) == ("", "")
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments", "v": [7, {"type": "RESPONSE", "content": "only-body"}]}))
    assert (rec.content, rec.reasoning) == ("only-body", "")
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments", "v": [7, "not-a-dict"]}))
    assert (rec.content, rec.reasoning) == ("", "")


def test_a_reconstructor_folds_fragments_that_appeared_after_its_last_read():
    rec = _primed()
    rec.handle(_append(0, "think "))
    rec.handle(_append(1, "answer "))
    rec.take_diffs()
    rec.message["fragments"].append({"type": "RESPONSE", "content": " more"})
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/accumulated_token_usage", "v": 5}))
    assert rec.content == "answer  more"
    assert rec.reasoning == "think "
    rec.message["fragments"].append({"type": "THINK", "content": " deeper"})
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/accumulated_token_usage", "v": 6}))
    assert rec.reasoning == "think  deeper"
    assert rec.accumulated_tokens == 6


def test_a_batch_that_rewrites_an_earlier_fragment_resets_the_aggregate():
    rec = _primed()
    rec.handle(_append(0, "think "))
    rec.handle(_append(1, "answer "))
    rec.take_diffs()
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response/fragments/0/content", "v": "rewritten"}))
    assert rec.reasoning == "rewritten"
    assert rec.content == "answer "
    assert rec.take_diffs() == ("", "rewritten")
    assert rec.take_diffs() == ("", "")


def test_take_diffs_folds_the_reasoning_base_over_two_rounds():
    rec = MessageReconstructor()
    rec.handle(SSEEvent(None, {"o": "SET", "p": "response", "v": {"response": {"fragments": [{"type": "THINK", "content": ""}]}}}))
    rec.take_diffs()
    rec.handle(_append(0, "a "))
    assert rec.take_diffs() == ("", "a ")
    assert rec._reasoning_base == ""
    assert rec._reported_r == 1
    rec.handle(_append(0, "b "))
    assert rec.take_diffs() == ("", "b ")
    assert rec._reasoning_base == "a "
    assert rec._reasoning_parts == ["b "]
    rec.handle(_append(0, "c "))
    assert rec.take_diffs() == ("", "c ")
    assert rec._reasoning_base == "a b "
    assert rec.reasoning == "a b c "
    assert rec._fold_reasoning() == ""
    assert rec._fold_content() == ""


def test_extend_with_merges_another_reconstructor():
    first = _primed()
    first.handle(_append(1, "hello "))
    first.take_diffs()
    second = _primed()
    second.handle(_append(1, "world"))
    second.take_diffs()
    second.message["id"] = "c2"
    second.message["status"] = "DONE"
    second.message["accumulated_token_usage"] = 12
    second.hint_error = {"message": "boom", "finish_reason": None}
    first.extend_with(second)
    assert first.content == "hello world"
    assert first.id == "c2"
    assert first.status == "DONE"
    assert first.accumulated_tokens == 12
    assert first.usage == {"prompt_tokens": 0, "completion_tokens": 12, "total_tokens": 12}
    assert first.hint_error == {"message": "boom", "finish_reason": None}
    assert first.take_diffs() == ("world", "")


def test_extend_with_adopts_a_whole_message_when_there_are_no_fragments_yet():
    first = MessageReconstructor()
    first.message = {"status": "WIP"}
    second = _primed()
    second.handle(_append(1, "tail"))
    first.extend_with(second)
    assert first.content == "tail"
    assert first.status == "WIP"
    assert first.id == "asst_1"
    bare = MessageReconstructor()
    first.extend_with(bare)
    assert first.id == "asst_1"
    assert first.accumulated_tokens == 0
    assert first.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert first.content == "tail"


def test_delta_helpers_normalise_paths_ops_and_diffs():
    assert sseutil_mod._path_parts("/a//b/") == ("a", "b")
    assert sseutil_mod._normalise_key("message_id") == "id"
    assert sseutil_mod._normalise_key("other") == "other"
    assert sseutil_mod._delta_op(None, "SET") == "SET"
    assert sseutil_mod._delta_op("APPEND", "SET") == "APPEND"
    assert sseutil_mod._diff_suffix("ab", "ab") == ""
    assert sseutil_mod._diff_suffix("ab", "abc") == "c"


def test_the_cjk_helpers_count_every_script():
    assert tokens_mod._is_cjk("你") is True
    assert tokens_mod._is_cjk("a") is False
    assert tokens_mod._cjk_count("abc") == 0
    assert tokens_mod._cjk_count("你好abc") == 2


def test_a_cjk_only_budget_counts_characters_not_quarters():
    budget = StreamBudget(2, trim_to_tokens)
    assert budget.feed("你") == "你"
    assert budget._count() == 1
    assert budget.text == "你"
    assert budget.done is False


def test_head_length_and_trim_to_tokens_bound_every_script():
    assert trim_to_tokens("hello", None) == "hello"
    assert trim_to_tokens("", 5) == ""
    assert trim_to_tokens("hello", 100) == "hello"
    assert trim_to_tokens("hello world foo", 2) == "hello world"
    assert trim_to_tokens("abcdefghij", 1) == "abcdefg"
    assert trim_to_tokens(" abcdefghijklmnop", 1) == "abcdefg"
    assert trim_to_tokens("   ", 0) == ""
    assert trim_to_tokens("hello world", 0) == ""
    assert trim_to_tokens("hello world", -1) == ""
    assert trim_to_tokens("你好世界", 2) == "你好"
    assert trim_to_tokens("a你b好", 1) == "a"
    assert tokens_mod._head_length("hello", 0) == 0
    assert tokens_mod._head_length("hello", 2) == 5
    assert tokens_mod._head_length("abcdefghij", 1) == 7
    assert tokens_mod._head_length("你好世界", 2) == 2
    assert tokens_mod._head_length("a你b好", 1) == 1


def test_a_budget_that_cannot_trim_marks_itself_done_with_the_whole_text():
    budget = StreamBudget(1, lambda text, limit: text)
    assert budget.feed("Hello") == "Hello"
    assert budget.done is False
    assert budget.feed(" world") == " world"
    assert budget.done is True
    assert budget.text == "Hello world"
    assert budget.feed("!") == ""


def test_a_trimmed_prefix_that_does_not_extend_the_text_is_dropped():
    budget = StreamBudget(1, lambda text, limit: "X")
    assert budget.feed("Hello") == "Hello"
    assert budget.feed(" world") == ""
    assert budget.done is True
    assert budget.text == "X"


def test_a_stream_budget_passes_a_single_chunk_through_unchanged():
    budget = StreamBudget(3, trim_to_tokens)
    assert budget.feed("abcd") == "abcd"
    assert budget.text == "abcd"
    assert budget._count() == 1
    budget = StreamBudget(3, trim_to_tokens)
    assert budget.feed("ab") == "ab"
    assert budget.text == "ab"


def test_message_fields_reads_objects_and_lists_of_content():
    assert count_message_tokens(SimpleNamespace(content=None, tool_calls=None)) == 0
    assert count_message_tokens(SimpleNamespace(content="hello", tool_calls=None)) == 4
    assert count_message_tokens(SimpleNamespace(content=None, tool_calls=[{"function": {"name": "f", "arguments": "{}"}}])) == 5
    assert count_message_tokens({"content": ["hello", "a", {"type": "image_url"}, {"no": "text"}]}) == 3 + 1 + 1 + 85
    assert count_message_tokens({"content": "", "tool_calls": ["not-a-dict", {"no": "function"}]}) == 3
    assert count_message_tokens({"content": "hi", "tool_calls": [{"function": {"name": 5, "arguments": 6}}]}) == 4
    assert count_message_tokens({"content": "x", "tool_calls": [{"function": "not-a-dict"}]}) == 4
    assert estimate_tokens("") == 0
