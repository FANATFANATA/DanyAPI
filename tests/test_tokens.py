from danyapi.tokens import count_message_tokens, count_messages_tokens, count_prompt_tokens, estimate_tokens


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
