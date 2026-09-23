import os

os.environ["DANYAPI_HUMAN_DELAY_MIN"] = "0"
os.environ["DANYAPI_HUMAN_DELAY_MAX"] = "0"
os.environ["BYOK"] = "0"
os.environ["BYOK_MODE"] = "0"
os.environ["DANYAPI_BYOK_MODE"] = "0"
for _key in (
    "DEEPSEEK_TOKENS",
    "QWEN_TOKENS",
):
    os.environ.setdefault(_key, "")