"""Regression tests for the browser shim in danyapi/duckai/jsa_solver.js.

The attestation script probes the DOM, so the shim has to reproduce real browser
behaviour. The cases below were each found by comparing the shim against a real
Chrome on a live duck.ai script, and the expected values are what Chrome
returned.

Skipped when node is unavailable: the shim is JavaScript by necessity.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SOLVER = Path(__file__).resolve().parents[1] / "danyapi" / "duckai" / "jsa_solver.js"

NODE = shutil.which("node") or ""

pytestmark = pytest.mark.skipif(not NODE, reason="node is not available")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

HARNESS = r"""
const fs = require("fs");
const { evaluateSource } = require(process.argv[2]);
evaluateSource(fs.readFileSync(process.argv[3], "utf8"), process.argv[4])
  .then((r) => process.stdout.write(JSON.stringify({ ok: true, result: r })))
  .catch((e) => process.stdout.write(JSON.stringify({ ok: false, error: String(e && e.message ? e.message : e) })));
"""

PROBE_SCRIPT = r"""
(() => {
  const parse = (html) => {
    const div = document.createElement("div");
    div.innerHTML = html;
    return { children: div.children.length, all: div.querySelectorAll("*").length, markup: div.innerHTML };
  };
  const box = () => {
    const div = document.createElement("div");
    div.style.cssText = "display:inline-block;padding:8px;position:absolute;visibility:hidden;";
    div.textContent = "x";
    document.body.appendChild(div);
    const rect = div.getBoundingClientRect();
    const out = {
      offsetWidth: div.offsetWidth > 0,
      offsetHeight: div.offsetHeight > 0,
      rect: rect.width > 0 && rect.height > 0,
      display: getComputedStyle(div).getPropertyValue("display").length > 0,
      scrollHeight: div.scrollHeight > 0,
    };
    document.body.removeChild(div);
    return out;
  };
  const hidden = () => {
    const div = document.createElement("div");
    div.style.cssText = "display:none;";
    div.textContent = "x";
    document.body.appendChild(div);
    const out = { offsetWidth: div.offsetWidth, offsetHeight: div.offsetHeight };
    document.body.removeChild(div);
    return out;
  };
  return {
    identity: typeof window === "object" && window === self && window === globalThis,
    windowTag: Object.prototype.toString.call(window),
    errorStack: typeof Error.captureStackTrace,
    arraySubclass: (() => { class A extends Array {} return new A(1, 2, 3).map((x) => x * 2) instanceof A; })(),
    cspMeta: !!document.querySelector('meta[http-equiv="Content-Security-Policy"]'),
    nodeList: document.querySelectorAll("*").constructor.name,
    divProto: HTMLDivElement.prototype instanceof HTMLElement,
    elementProto: HTMLElement.prototype instanceof Element,
    webdriver: navigator.webdriver,
    brFragment: parse("<br><div></br><br></div>"),
    pFragment: parse("<p><div></p><p></div>"),
    liFragment: parse("<ul><li>a<li>b</ul>"),
    box: box(),
    hidden: hidden(),
  };
})()
"""


def _evaluate(tmp_path: Path, script: str) -> dict:
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    target = tmp_path / "target.js"
    target.write_text(script, encoding="utf-8")
    proc = subprocess.run(  # nosec B603
        [NODE, str(harness), str(SOLVER), str(target), USER_AGENT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_shim_reproduces_chrome_identity_probes(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    assert out["ok"] is True, out
    probe = out["result"]
    assert probe["identity"] is True
    assert probe["windowTag"] == "[object Window]"
    assert probe["errorStack"] == "function"
    assert probe["arraySubclass"] is True
    assert probe["nodeList"] == "NodeList"
    assert probe["divProto"] is True
    assert probe["elementProto"] is True
    assert probe["webdriver"] is False


def test_shim_has_no_csp_meta_element(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    assert out["result"]["cspMeta"] is False


def test_shim_layout_reports_a_real_box(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    box = out["result"]["box"]
    assert box == {"offsetWidth": True, "offsetHeight": True, "rect": True, "display": True, "scrollHeight": True}


def test_shim_layout_reports_zero_for_display_none(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    assert out["result"]["hidden"] == {"offsetWidth": 0, "offsetHeight": 0}


def test_shim_parses_end_tag_br_as_a_start_tag(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    fragment = out["result"]["brFragment"]
    assert fragment["markup"] == "<br><div><br><br></div>"
    assert fragment["children"] == 2
    assert fragment["all"] == 4


def test_shim_parses_misnested_paragraphs_like_chrome(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    fragment = out["result"]["pFragment"]
    assert fragment["markup"] == "<p></p><div><p></p><p></p></div>"
    assert fragment["all"] == 4


def test_shim_parses_implicit_list_item_closing(tmp_path):
    out = _evaluate(tmp_path, PROBE_SCRIPT)
    fragment = out["result"]["liFragment"]
    assert fragment["markup"] == "<ul><li>a</li><li>b</li></ul>"


def test_solver_exports_evaluate_script_for_reuse():
    assert SOLVER.is_file()
    text = SOLVER.read_text(encoding="utf-8")
    assert "evaluateScript, evaluateSource" in text
    assert "require.main === module" in text
