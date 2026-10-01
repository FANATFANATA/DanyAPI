import email.message
import importlib.util
import io
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
_SPEC = importlib.util.spec_from_file_location("danyapi_setup", DOCS / "setup.py")
assert _SPEC is not None
assert _SPEC.loader is not None
setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup)

_TU_SPEC = importlib.util.spec_from_file_location("danyapi_token_utility", DOCS / "token_utility.py")
assert _TU_SPEC is not None
assert _TU_SPEC.loader is not None
token_utility = importlib.util.module_from_spec(_TU_SPEC)
_TU_SPEC.loader.exec_module(token_utility)

_COLLECTER_SPEC = importlib.util.spec_from_file_location("danyapi_collecter", REPO / "collecter.py")
assert _COLLECTER_SPEC is not None
assert _COLLECTER_SPEC.loader is not None
collecter = importlib.util.module_from_spec(_COLLECTER_SPEC)
_COLLECTER_SPEC.loader.exec_module(collecter)


class FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(status, body):
    return patch.object(setup.urllib.request, "urlopen", return_value=FakeResp(status, body))


def test_check_qwen_token_success_flag():
    with _patch_urlopen(200, json.dumps({"success": True, "id": "u1"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is True
    assert detail == ""


def test_check_qwen_token_user_object_without_success():
    with _patch_urlopen(200, json.dumps({"id": "u1", "name": "n", "email": "a@b.c"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is True
    assert detail == ""


def test_check_qwen_token_rejects_unknown_payload():
    with _patch_urlopen(200, json.dumps({"foo": "bar"})):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is False
    assert "rejected" in detail


def test_check_qwen_token_http_401():
    error = urllib.error.HTTPError("https://chat.qwen.ai/api/v1/auths/", 401, "Unauthorized", email.message.Message(), io.BytesIO(b"denied"))
    with patch.object(setup.urllib.request, "urlopen", side_effect=error):
        ok, detail = setup.check_qwen_token("bad")
    assert ok is False
    assert "401" in detail


def test_check_qwen_token_network_error():
    with patch.object(setup.urllib.request, "urlopen", side_effect=OSError("no route")):
        ok, detail = setup.check_qwen_token("tok")
    assert ok is False
    assert "network error" in detail


def test_check_deepseek_token_ok():
    with _patch_urlopen(200, json.dumps({"code": 0})):
        ok, _ = setup.check_deepseek_token("tok")
    assert ok is True


def test_check_deepseek_token_rejected():
    with _patch_urlopen(200, json.dumps({"code": 401, "msg": "bad"})):
        ok, detail = setup.check_deepseek_token("tok")
    assert ok is False
    assert "rejected" in detail


def test_split_tokens_variants():
    assert setup.split_tokens("a, b  c,d") == ["a", "b", "c", "d"]
    assert setup.split_tokens(" , ") == []
    assert setup.split_tokens("") == []


def test_check_provider_valid_tokens():
    calls = []

    def checker(token):
        calls.append(token)
        return True, ""

    with patch.object(setup, "check_deepseek_token", side_effect=checker):
        ok, detail = setup.check_provider("DeepSeek", {"DEEPSEEK_TOKENS": "t1, t2"})
    assert ok is True
    assert detail == ""
    assert set(calls) == {"t1", "t2"}


def test_check_provider_invalid_token_reports_first_failure_in_order():
    calls = []

    def checker(token):
        calls.append(token)
        if token == "t1":
            return True, ""
        return False, "http 401: denied"

    with patch.object(setup, "check_deepseek_token", side_effect=checker):
        ok, detail = setup.check_provider("DeepSeek", {"DEEPSEEK_TOKENS": "t1,t2,t3"})
    assert ok is False
    assert "http 401" in detail
    assert "t2" in calls


def test_check_provider_empty_is_ok():
    ok, detail = setup.check_provider("Qwen", {"QWEN_TOKENS": ""})
    assert ok is True
    assert detail == ""


def test_collect_provider_returns_tokens_key_only(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "tok1,tok2")
    creds = setup.collect_provider("Qwen", {}, {})
    assert creds == {"QWEN_TOKENS": "tok1,tok2"}


class _Proc:
    def __init__(self, stdout):
        self.stdout = stdout


def test_rustup_target_reachable_present(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Proc("aarch64-unknown-linux-android\n"))
    assert setup.rustup_target_reachable() is True


def test_rustup_target_reachable_missing(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Proc("x86_64-unknown-linux-gnu\n"))
    assert setup.rustup_target_reachable() is False


def test_rustup_target_reachable_no_rustup(monkeypatch):
    def boom(*a, **k):
        raise OSError("no rustup")

    monkeypatch.setattr(setup.subprocess, "run", boom)
    assert setup.rustup_target_reachable() is True


def test_update_env_removes_duplicates(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=old1\nB=keep\nA=old2\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ENV_FILE", env_file)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.update_env({"A": "new"})
    text = env_file.read_text(encoding="utf-8")
    assert text.count("A=") == 1
    assert "A=new" in text
    assert "B=keep" in text


def test_update_env_appends_new_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ENV_FILE", env_file)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.update_env({"C": "3"})
    text = env_file.read_text(encoding="utf-8")
    assert "A=1" in text
    assert "C=3" in text


def test_parse_env_strips_inline_comment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DANYAPI_CACHE_DIR=/var/cache/danyapi # my cache\n", encoding="utf-8")
    assert setup.parse_env(env_file)["DANYAPI_CACHE_DIR"] == "/var/cache/danyapi"


def test_parse_env_keeps_hash_inside_quotes_and_without_space(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("A='# not a comment'\nB=no#hash\nC=\"has # inside\"\n", encoding="utf-8")
    values = setup.parse_env(env_file)
    assert values == {"A": "# not a comment", "B": "no#hash", "C": "has # inside"}


def test_quote_round_trips_backslashes_and_quotes(tmp_path):
    env_file = tmp_path / ".env"
    for raw in ("back\\slash", "trail\\", "quo'te", "back\\'x", "windows\\path\\to\\file", "a#b"):
        env_file.write_text(f"K={setup.quote(raw)}\n", encoding="utf-8")
        assert setup.parse_env(env_file)["K"] == raw, raw


def test_mask_secrets_never_shows_the_tail():
    secret = "abcdefghijklmnopqrstuvwxyz0123456789"
    shown = setup.mask_secrets(secret)
    assert secret not in shown
    assert not shown.endswith(secret[-2:])
    assert secret[-2:] not in shown
    assert str(len(secret)) in shown
    assert shown.startswith(secret[:4])


def _token_utility_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), token_utility.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _local_get(path, base):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.fixture
def token_utility_server():
    server = _token_utility_server()
    port = server.server_address[1]
    previous = token_utility.Handler.serve_port
    token_utility.Handler.serve_port = port
    base = f"http://127.0.0.1:{port}"
    token_utility.STATE["deepseek"] = None
    token_utility.STATE["qwen"] = None
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        token_utility.Handler.serve_port = previous


def test_status_and_results_require_the_session_id(token_utility_server):
    sid = token_utility.SESSION_ID
    assert _local_get("/status", token_utility_server)[0] == 403
    assert _local_get("/results", token_utility_server)[0] == 403
    assert _local_get("/status?sid=wrong", token_utility_server)[0] == 403
    assert _local_get(f"/status?sid={sid}", token_utility_server)[0] == 200
    assert _local_get(f"/results?sid={sid}", token_utility_server)[0] == 200


def test_setup_page_carries_the_session_id(token_utility_server):
    body = _local_get("/", token_utility_server)[1].decode()
    assert token_utility.SESSION_ID in body
    assert 'location.href = "/results"' not in body


def test_get_collect_is_gone(token_utility_server):
    status, body = _local_get("/collect?p=deepseek&t=" + "t" * 40, token_utility_server)
    assert status == 200
    assert body.decode().lstrip().startswith("<!DOCTYPE html>")
    assert token_utility.STATE["deepseek"] is None


def test_post_collect_still_registers(token_utility_server):
    body = json.dumps({"provider": "deepseek", "token": "k" * 40}).encode()
    request = urllib.request.Request(
        token_utility_server + "/collect",
        data=body,
        headers={"Content-Type": "application/json", "Origin": "https://chat.deepseek.com"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        assert resp.status == 200
    try:
        assert token_utility.STATE["deepseek"] == "k" * 40
        page = _local_get(f"/results?sid={token_utility.SESSION_ID}", token_utility_server)[1].decode()
        assert "k" * 40 in page
    finally:
        token_utility.STATE["deepseek"] = None


def test_collect_files_ignores_the_ancestor_directory_name(tmp_path):
    root = tmp_path / "references" / "DanyAPI"
    root.mkdir(parents=True)
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    names = sorted(p.relative_to(root).as_posix() for p in collecter.collect_files(root))
    assert names == ["app.py"]


def test_gitignore_last_match_wins():
    assert collecter._matches_gitignore("a/keep.log", ["!keep.log", "*.log"]) is True
    assert collecter._matches_gitignore("a/keep.log", ["*.log", "!keep.log"]) is False
    assert collecter._matches_gitignore("a/other.log", ["*.log"]) is True
    assert collecter._matches_gitignore("a/keep.txt", ["*.log"]) is False


def test_sanitize_xml_text_drops_illegal_code_points():
    cleaned = collecter.sanitize_xml_text("a\x00b\x0bc\x0cd\te")
    assert cleaned == "abcd\te"
    assert collecter.xml_escape(cleaned + " <x> & y") == "abcd\te &lt;x&gt; &amp; y"


def test_collected_bundle_stays_parseable(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "weird.py").write_text("x = '\x00\x0c'\n", encoding="utf-8")
    monkeypatch.setattr(collecter, "ROOT", root)
    assert collecter.main() == 0
    import xml.etree.ElementTree as ET

    tree = ET.parse(root / "collected.xml")
    assert len(tree.getroot().findall("file")) == 1


def test_collect_excludes_credential_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    names = [".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", "id_rsa", "id_ed25519", "prod.tfvars", "home.ovpn", "kubeconfig", "app.py"]
    for name in names:
        (root / name).write_text("secret\n", encoding="utf-8")
    found = sorted(p.name for p in collecter.collect_files(root))
    assert found == ["app.py"]


def test_docs_i18n_tables_are_complete_and_symmetric():
    script = (DOCS / "script.js").read_text(encoding="utf-8")
    page = (DOCS / "index.html").read_text(encoding="utf-8")
    blocks = re.findall(r"\n        (?:ru|en): \{(.*?)\n        \}", script, re.DOTALL)
    assert len(blocks) == 2
    keys = [{m.group(1) for m in re.finditer(r"^\s{12}([a-z0-9_]+): ", block, re.MULTILINE)} for block in blocks]
    assert keys[0] == keys[1]
    used = set(re.findall(r'data-i18n(?:-aria)?="([a-z0-9_]+)"', page))
    assert used <= keys[0]
    for key in ("models_duck_keyless", "models_al_keyless", "models_gc_pro", "models_oc_note_list", "models_qw_3", "cta_title_hi"):
        assert key in keys[0]
    assert "data-i18n-src" in script
    assert "t.models_duck_keyless" not in script


def test_docs_nested_titles_keep_their_span():
    page = (DOCS / "index.html").read_text(encoding="utf-8")
    for key in ("hosted_title_hi", "features_title_hi", "models_title_hi", "qs_title_hi", "faq_title_hi", "cta_title_hi"):
        assert f'data-i18n="{key}"' in page
    assert '<h2 data-i18n="cta_title">Free models.<br /><span class="grad-text" data-i18n="cta_title_hi">Your API.</span></h2>' in page
