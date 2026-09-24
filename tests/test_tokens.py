from danyapi.tokens import StreamBudget, count_message_tokens, count_messages_tokens, count_prompt_tokens, estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_latin():
    assert estimate_tokens("Hello world") == 2
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("Hello world, how are you?") == 6


def test_estimate_tokens_cjk():
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("こんにちは") == 5
    assert estimate_tokens("안녕하세요") == 5


def test_estimate_tokens_mixed():
    assert estimate_tokens("Hello 你好") == 3


def test_count_message_tokens_plain():
    msg = {"role": "user", "content": "Hello world"}
    assert count_message_tokens(msg) == 5


def test_count_message_tokens_content_list():
    msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}
    assert count_message_tokens(msg) == 3 + 1 + 85


def test_count_message_tokens_tool_calls():
    msg = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Moscow"}'}}]}
    assert count_message_tokens(msg) > 3


def test_count_message_tokens_invalid():
    assert count_message_tokens(None) == 0
    assert count_message_tokens("not a dict") == 0
    assert count_message_tokens({}) == 3


def test_count_messages_tokens():
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    assert count_messages_tokens(messages) == count_message_tokens(messages[0]) + count_message_tokens(messages[1])


def test_count_prompt_tokens():
    assert count_prompt_tokens("Hello world") == estimate_tokens("Hello world")
    assert count_prompt_tokens("") == 0


def _trim(text, budget):
    if budget is None or estimate_tokens(text) <= budget:
        return text
    words = text.split(" ")
    parts = []
    total = 0
    for word in words:
        candidate = total + len(word) + (1 if parts else 0)
        if max(1, candidate // 4) > budget:
            break
        parts.append(word)
        total = candidate
    return " ".join(parts)


def test_stream_budget_passthrough_without_limit():
    budget = StreamBudget(None, _trim)
    assert budget.feed("Hello ") == "Hello "
    assert budget.feed("world") == "world"
    assert budget.text == "Hello world"
    assert budget.done is False


def test_stream_budget_trims_once_and_marks_done():
    budget = StreamBudget(2, _trim)
    first = budget.feed("Hello world foo")
    assert first == "Hello world"
    assert budget.done is True
    assert budget.feed(" bar") == ""


def test_stream_budget_drops_piece_that_overflows():
    budget = StreamBudget(1, _trim)
    assert budget.feed("Hello") == "Hello"
    assert budget.done is False
    assert budget.feed(" world") == ""
    assert budget.done is True
    assert budget.text == "Hello"
    assert budget.feed(" foo") == ""


def test_stream_budget_ignores_empty_and_none():
    budget = StreamBudget(2, _trim)
    assert budget.feed("") == ""
    assert budget.feed(None) == ""
    assert budget.text == ""
    assert budget.done is False


def test_stream_budget_partial_piece_is_dropped_when_not_prefix():
    budget = StreamBudget(1, lambda text, limit: "Hello")
    assert budget.feed("Hello") == "Hello"
    assert budget.feed(" world") == ""
    assert budget.done is True
