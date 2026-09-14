#!/usr/bin/env python3
"""
get_tokens.py - Extract DeepSeek & Qwen tokens for DanyAPI.

Uses your DEFAULT browser (no automation, no dependencies).

    python get_tokens.py

Flow:
  1. A tiny local server starts (127.0.0.1:8765) and a page opens in your
     default browser.
  2. Step by step wizard: first drag the "🔍 Run DanyAPI token utility" button to
     your bookmarks bar (one time only), click Next.
  3. The page guides you to DeepSeek: log in, click the grabber bookmark
     there. The token is sent silently, the DeepSeek tab closes itself,
     the wizard shows a success flash and automatically moves on to Qwen.
  4. Same for Qwen - and when both tokens are in, you land on a results
     screen with your tokens ready to copy.

Everything stays local: the server binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Windows consoles often default to cp1252 which can't render emoji/box glyphs.
# getattr() + str() instead of direct attribute access: pylint cannot infer
# members on the sys.stdout TextIO wrapper (E1101 false positive), and a
# missing/None/empty encoding must skip the re-wrap, exactly as before.
stdout_encoding = str(getattr(sys.stdout, "encoding", "") or "")
if stdout_encoding and stdout_encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RESULT_PORT = 8765

DEEPSEEK_URL = "https://chat.deepseek.com/"
QWEN_URL = "https://chat.qwen.ai/auth?action=signin"

# ----------------------------------------------------------------------------
# Visible text (edit here - no need to dig into the HTML below)
#
# Strings used inside the static HTML go through __KEY__ placeholders; strings
# used by the page's JavaScript are injected as JSON and read via t("key").
# ----------------------------------------------------------------------------

TEXTS: dict[str, str] = {
    # --- Wizard page -----------------------------------------------------
    "page_title": "DanyAPI - Token Utils",
    "header_title": "Token Utilities",
    "header_sub": "DanyAPI uses the internal APIs of DeepSeek and Qwen's free web clients, so it needs your auth tokens."
    "<br>This tool simply helps you to retrieve them.",
    # Step 0 - bookmarklet
    "step0_heading": "One-time: add the token utility bookmarklet",
    "step0_intro": "<b>Drag</b> this button onto your browser's <b>bookmarks bar</b>,<br>(press Ctrl+Shift+B if you don't see the bar):",
    "bookmarklet_label": "🔍 Run DanyAPI token utility",
    "bookmarklet_aria": "DanyAPI token utility - drag this button onto your bookmarks bar",
    "step0_fineprint": "Why is this necessary? We know this looks <i>weird and unfamiliar</i>, but while other methods for"
    " extracting tokens exist, they are <b>not browser-agnostic</b>. This bookmarklet will execute JavaScript in the"
    " context of the provider's page (<i>deepseek.com</i> or <i>qwen.ai</i>), extract your token, and send it to this page.",
    "step0_next": "I added the bookmarklet - Next →",
    # Step 1 - DeepSeek
    "step1_heading": "DeepSeek token",
    "step1_p1": "<b>1.</b> Open DeepSeek by clicking the button below.",
    "step1_p2": "<b>2.</b> Sign in to your account if needed.",
    "step1_p3": "<b>3.</b> On the DeepSeek page, click your <b>“🔍 Run DanyAPI token utility”</b> bookmark.",
    "step1_fineprint": "The tab closes automatically after the token is sent. If the token is valid, a green checkmark"
    " will appear on this page and you will be guided to the next step within a few seconds. Overwise a red cross will"
    " appear, and you can click the button below to try again.",
    "step1_button": "Open DeepSeek →",
    "step1_waiting": "Waiting for the DeepSeek token…",
    # Step 2 - Qwen
    "step2_heading": "Qwen token",
    "step2_p1": "<b>1.</b> Open Qwen by clicking the button below.",
    "step2_p2": "<b>2.</b> Sign in to your account if needed.",
    "step2_p3": "<b>3.</b> On the Qwen page, click the <b>“🔍 Run DanyAPI token utility”</b> bookmark.",
    "step2_fineprint": "The tab closes automatically after the token is sent. If the token is valid, a green checkmark"
    " will appear on this page and you will be guided to the next step within a few seconds. Overwise a red cross will"
    " appear, and you can click the button below to try again.",
    "step2_button": "Open Qwen →",
    "step2_waiting": "Waiting for the Qwen token…",
    # Step 3 - done
    "step3_heading": "Tokens successfully extracted!",
    "step3_text": "Redirecting to your tokens…",
    # Footer
    "footer_public_instance": "Public Instance",
    "footer_docs": "Docs",
    "footer_github": "GitHub",
    "footer_public_url": "https://danyapi.cloudpub.ru",
    "footer_docs_url": "https://danyapi.cloudpub.ru/docs/",
    "footer_github_url": "https://github.com/FANATFANATA/DanyAPI",
    # --- Wizard page JS --------------------------------------------------
    "js_title_wizard": "🔑 DanyAPI token utilities",
    "js_title_step1": "🤖 Step 1 of 2 - DeepSeek",
    "js_title_step2": "🤖 Step 2 of 2 - Qwen",
    "js_title_done": "🎉 All done!",
    "js_alert_popup_blocked": "Popup blocked! Please allow popups for this page and try again.",
    "js_ok_both": "✔ {provider} token received! Redirecting to your tokens shortly…",
    "js_ok_deepseek": "✔ DeepSeek token received! Moving on to Qwen shortly…",
    "js_ok_qwen": "✔ Qwen token received! Moving to your tokens shortly…",
    "js_fail": "✖ No token found - you are probably not logged in. Log in on {provider}, then try again.",
    # --- Results page ----------------------------------------------------
    "results_page_title": "DanyAPI - Your tokens",
    "results_title": "Your Tokens",
    "results_sub": "Tokens for DeepSeek &amp; Qwen were successfully extracted. Use them in your <code>.env</code> when"
    " running the API locally.<br><br>To support us, you can also add them to the public API instance:"
    ' <a href="{public_url}" target="_blank" rel="noopener">{public_url}</a>',
    "results_public_instance": "Public instance",
    "results_pane_ds": "DeepSeek token",
    "results_pane_qw": "Qwen token",
    "results_copy": "Copy",
    "results_copied": "✓ Copied",
    "results_no_token": "❌ No token received - complete the setup page first.",
    "results_footer_again": "Run utility again",
    "results_footer_docs": "Docs",
    "results_footer_github": "GitHub",
    # --- Popup page (shown in the provider tab after collection) --------
    "popup_title": "Token received",
    "popup_message": "✔ Token received - you can close this tab and return to the DanyAPI page.",
}

# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------

STATE: dict[str, Any] = {
    "deepseek": None,
    "qwen": None,
    "deepseek_failed": False,
    "qwen_failed": False,
}

# ----------------------------------------------------------------------------
# Shared HTML assets
#
# The setup page and the results page are visually identical shells, so these
# assets are defined once and injected through the same __key__ placeholder
# machinery used for TEXTS (see _apply_texts). All three are plain URL/URI
# strings; the setup/results templates embed them verbatim.
# ----------------------------------------------------------------------------

# Inline SVG favicon (DanyAPI hexagon logo) as a data: URI, URL-encoded.
FAVICON_DATA_URI = (
    "data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2040%2040'%3E"
    "%3Cdefs%3E%3ClinearGradient%20id='g'%20x1='0'%20y1='0'%20x2='1'%20y2='1'%3E"
    "%3Cstop%20offset='0'%20stop-color='%236c7bff'/%3E%3Cstop%20offset='1'%20stop-color='%2322d3ee'/%3E"
    "%3C/linearGradient%3E%3C/defs%3E"
    "%3Cpath%20d='M20%202.5L35%2011V29L20%2037.5L5%2029V11Z'%20fill='none'%20stroke='url(%23g)'"
    "%20stroke-width='2.6'%20stroke-linejoin='round'/%3E"
    "%3Cpath%20d='M20%2013.5L12.5%2026.5H27.5Z'%20fill='none'%20stroke='url(%23g)'"
    "%20stroke-width='1.8'%20stroke-linejoin='round'/%3E"
    "%3Ccircle%20cx='20'%20cy='13.5'%20r='3'%20fill='url(%23g)'/%3E"
    "%3Ccircle%20cx='12.5'%20cy='26.5'%20r='3'%20fill='url(%23g)'/%3E"
    "%3Ccircle%20cx='27.5'%20cy='26.5'%20r='3'%20fill='url(%23g)'/%3E%3C/svg%3E"
)

# Single Google Fonts request covering all families/weights used by both pages.
FONTS_CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Unbounded:wght@500;700;900"
    "&family=Manrope:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;600;700"
    "&display=swap"
)

# Inline SVG feTurbulence noise overlay as a data: URI, URL-encoded.
NOISE_DATA_URI = (
    "url(\"data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='120'%20height='120'%3E"
    "%3Cfilter%20id='n'%3E%3CfeTurbulence%20type='fractalNoise'%20baseFrequency='0.9'%20numOctaves='2'/%3E"
    "%3C/filter%3E%3Crect%20width='120'%20height='120'%20filter='url(%23n)'/%3E%3C/svg%3E"
    '")'
)

# ----------------------------------------------------------------------------
# Bookmarklet (runs on chat.deepseek.com / chat.qwen.ai, sends token to us)
#
# IMPORTANT: this source gets collapsed into ONE line by build_bookmarklet(),
# so it must contain NO '//' comments and every statement must end with ';'
# (automatic-semicolon-insertion disappears when newlines are removed).
# ----------------------------------------------------------------------------

BOOKMARKLET_SOURCE = r"""
(function () {
  var host = location.hostname.toLowerCase();
  var isDeepSeek = host === "chat.deepseek.com" || host.endsWith(".deepseek.com");
  var isQwen = host === "chat.qwen.ai" || host.endsWith(".qwen.ai");
  if (!isDeepSeek && !isQwen) {
    alert("Please run the DanyAPI token utility on DeepSeek or Qwen.");
    return;
  }
  var p = isDeepSeek ? "deepseek" : "qwen";
  function pt(raw, depth) {
    if (typeof raw !== "string") return null;
    var v = raw.trim();
    if (!v) return null;
    // Only accept values that actually look like auth tokens (JWT or hex),
    // never random storage junk: a logged-out page must NOT produce a
    // "success". Same rules are enforced server-side in register_token().
    if (/^eyJ[A-Za-z0-9._-]{20,}/.test(v) || /^[a-f0-9]{32,}$/i.test(v)) return v;
    if (depth >= 4) return null;
    try {
      var j = JSON.parse(v);
      if (typeof j === "string") return pt(j, depth + 1);
      if (typeof j === "object" && j !== null) {
        // Recursively unwrap ANY JSON shape ({"value":"..."},
        // {"token":{"value":"..."}}, ...) and return the first
        // token-shaped string found.
        for (var key in j) {
          var t = pt(j[key], depth + 1);
          if (t) return t;
        }
      }
    } catch (e) {}
    return null;
  }
  function scan(st) {
    var ks = p === "deepseek" ? ["userToken", "token"] : ["token", "userToken"];
    for (var a = 0; a < ks.length; a++) {
      var t = pt(st.getItem(ks[a]), 0);
      if (t) return t;
    }
    // Fallback scan: only entries whose KEY name mentions "token". Without
    // this guard, unrelated IDs (device ids, analytics ids...) that happen to
    // be 32+ hex chars get grabbed on logged-out pages and fake a success.
    for (var b = 0; b < st.length; b++) {
      var k = st.key(b);
      if (!/token/i.test(k)) continue;
      var t2 = pt(st.getItem(k), 0);
      if (t2) return t2;
    }
    return null;
  }
  // The providers store auth as a known key, but may wrap refreshed values
  // several times (for example {"value":"{\\"token\\":\\"...\\"}"}).
  // Unwrap only those known auth records; do not inspect unrelated storage.
  function opaqueValue(raw, depth) {
    if (typeof raw !== "string") return null;
    var value = raw.trim();
    if (!value || depth > 5) return null;
    if (value.length >= 16 && !/\s/.test(value) && value.charAt(0) !== "{" && value.charAt(0) !== "[" && value.charAt(0) !== "\"") return value;
    try {
      var obj = JSON.parse(value);
      if (typeof obj === "string") return opaqueValue(obj, depth + 1);
      if (!obj || typeof obj !== "object") return null;
      var preferred = ["value", "token", "accessToken", "access_token", "userToken"];
      for (var i = 0; i < preferred.length; i++) {
        var found = opaqueValue(obj[preferred[i]], depth + 1);
        if (found) return found;
      }
    } catch (e) {}
    return null;
  }
  function find() {
    var t = scan(localStorage);
    if (t) return t;
    t = scan(sessionStorage);
    if (t) return t;
    var keys = p === "deepseek" ? ["userToken", "token"] : ["token", "userToken"];
    for (var i = 0; i < keys.length; i++) {
      t = opaqueValue(localStorage.getItem(keys[i]), 0) || opaqueValue(sessionStorage.getItem(keys[i]), 0);
      if (t) return t;
    }
    return null;
  }
  var token = find();
  // Always report back, even with no token: the local server marks the
  // attempt as failed and the wizard shows a red hint instead of a green
  // checkmark, so a logged-out click can never look like a success.
  // Deliver via top-level navigation, NOT fetch/sendBeacon: browsers gate
  // cross-site requests to 127.0.0.1 behind a "local network / device
  // services" permission prompt, but plain navigations are always allowed.
  location.href = "http://127.0.0.1:__PORT__/collect?p=" + encodeURIComponent(p) +
    "&t=" + encodeURIComponent(token || "");
})()
"""


def build_bookmarklet(port: int) -> str:
    """Collapse the readable source into a one-line javascript: URL."""
    src = BOOKMARKLET_SOURCE.replace("__PORT__", str(port))
    lines = [ln.strip() for ln in src.strip().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("//")]
    one_line = "javascript:" + " ".join(lines)
    # Safety net: a stray '//' would comment out everything after it once the
    # code is on a single line ("http://" is the only legitimate use).
    assert "http://" in one_line
    assert "//" not in one_line.replace("http://", ""), "bookmarklet source contains a // comment - it would break on one line"
    return one_line


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ----------------------------------------------------------------------------
# Guided wizard page (step by step)
# ----------------------------------------------------------------------------

SETUP_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__page_title__</title>
<link rel="icon" href="__FAVICON_DATA_URI__" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="__FONTS_CSS_URL__" rel="stylesheet" />
<style>
  :root {
    --font-disp: "Unbounded", sans-serif;
    --font-body: "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "SF Mono", "Fira Code", "Cascadia Code", "JetBrains Mono", Consolas, monospace;
    --accent-soft: rgba(99, 102, 241, 0.14);
    --l-surface: #0d1119;
    --l-border: rgba(255, 255, 255, 0.08);
    --l-border-strong: rgba(255, 255, 255, 0.16);
    --l-text: #e9edf5;
    --l-muted: #9aa3b5;
    --l-faint: #5d6577;
    --l-accent: #6c7bff;
    --l-accent-2: #22d3ee;
    --l-grad: linear-gradient(120deg, #6c7bff 0%, #22d3ee 100%);
    --l-green: #34d399;
    --l-red: #ff5f56;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 15px; scroll-behavior: smooth; }
  body {
    font-family: var(--font-body);
    background: #06070c;
    color: var(--l-text);
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    padding: 2rem 1rem;
  }
  body::before {
    content: "";
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
      radial-gradient(900px 520px at 85% -10%, var(--accent-soft), transparent 60%),
      radial-gradient(760px 480px at -10% 105%, var(--accent-soft), transparent 60%);
  }
  .bg-grid {
    position: fixed; inset: 0; z-index: -3; pointer-events: none;
    background-image:
      linear-gradient(var(--l-border) 1px, transparent 1px),
      linear-gradient(90deg, var(--l-border) 1px, transparent 1px);
    background-size: 56px 56px;
    opacity: 0.5;
    -webkit-mask-image: radial-gradient(ellipse 100% 62% at 50% 0%, #000 0%, transparent 78%);
    mask-image: radial-gradient(ellipse 100% 62% at 50% 0%, #000 0%, transparent 78%);
  }
  .bg-glow { position: fixed; z-index: -2; pointer-events: none; border-radius: 50%; filter: blur(110px); opacity: 0.5; }
  .bg-glow-a { width: 640px; height: 640px; top: -220px; left: 50%; transform: translateX(-50%);
    background: radial-gradient(circle, rgba(108, 123, 255, 0.55), transparent 65%); }
  .bg-glow-b { width: 520px; height: 520px; bottom: -180px; right: -140px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.35), transparent 65%); }
  .noise {
    position: fixed; inset: 0; z-index: 60; pointer-events: none; opacity: 0.035;
    background-image: __NOISE_DATA_URI__;
  }
  .wrap {
    position: relative; z-index: 1;
    background: var(--l-surface);
    border: 1px solid var(--l-border-strong);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    max-width: 520px; width: 100%;
    box-shadow: 0 30px 80px -30px rgba(0, 0, 0, 0.7);
  }
  .brand { display: inline-flex; align-items: center; gap: 0.6rem; text-decoration: none; transition: transform 0.2s ease; }
  .brand:hover { transform: translateY(-1px); }
  .brand-logo { width: 36px; height: 36px; display: block;
    filter: drop-shadow(0 2px 10px rgba(108, 123, 255, 0.5));
    transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
  .brand:hover .brand-logo { animation: logo-spin 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
  @keyframes logo-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .brand-name { font-family: var(--font-disp); font-weight: 700; font-size: 1.15rem; letter-spacing: -0.02em; color: var(--l-text); }
  .brand-accent { background: var(--l-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
  .header { text-align: center; margin-bottom: 2.2rem; }
  h1 { font-family: var(--font-disp); font-weight: 700; font-size: clamp(1.4rem, 3vw, 1.8rem);
    line-height: 1.25; letter-spacing: -0.02em; margin-bottom: 0.6rem; color: var(--l-text); }
  .grad-text { background: var(--l-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
  .sub { color: var(--l-muted); font-size: 0.95rem; }
  .progress { height: 4px; border-radius: 2px; background: var(--l-border); margin-bottom: 2.2rem; overflow: hidden; }
  .progress .bar { height: 100%; width: 0%; background: var(--l-grad); transition: width 0.4s ease; }
  .step {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--l-border); border-radius: 14px;
    padding: 1.2rem 1.35rem; margin-bottom: 0.9rem;
  }
  .step h2 { font-size: 0.95rem; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.6rem; font-weight: 700; color: var(--l-text); }
  .badge {
    width: 24px; height: 24px; border-radius: 50%; background: var(--l-grad);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: #fff; flex-shrink: 0;
  }
  .badge.done { background: var(--l-green); }
  .badge.dim { background: rgba(255, 255, 255, 0.12); }
  .step p { color: var(--l-muted); font-size: 0.88rem; line-height: 1.65; margin-bottom: 0.6rem; }
  .step p b { color: var(--l-text); }
  .fineprint { font-size: 0.8rem; color: var(--l-faint); }
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem;
    background: var(--l-grad); color: #fff; text-decoration: none;
    padding: 0.85rem 1.5rem; border-radius: 11px; font-size: 0.95rem; font-weight: 700;
    border: none; cursor: pointer; font-family: var(--font-body);
    box-shadow: 0 10px 30px -12px rgba(108, 123, 255, 0.7);
    transition: transform 0.18s ease, box-shadow 0.25s ease;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 16px 40px -12px rgba(108, 123, 255, 0.85); }
  .btn:active { transform: translateY(0); }
  .btn.disabled, .btn.disabled:hover {
    opacity: 0.45; pointer-events: none; box-shadow: none;
    transform: none; cursor: default; filter: grayscale(0.6);
  }
  .btn.ghost { background: transparent; border: 1px solid var(--l-border-strong); color: var(--l-muted); box-shadow: none; }
  .bm {
    position: relative; display: inline-block; background: rgba(255, 255, 255, 0.04); color: #ffd479; text-decoration: none;
    padding: 0.85rem 1.5rem; border-radius: 11px; font-size: 0.9rem; font-weight: 700;
    border: 1px dashed #ffd479; cursor: grab;
    -webkit-user-select: none; user-select: none;
    transition: transform 0.18s ease;
  }
  .bm .bm-label { display: block; pointer-events: none; -webkit-user-select: none; user-select: none; }
  .bm .bm-title {
    /* Real text used by browsers as the bookmark name when the link is
       dragged to the bookmarks bar. Visually hidden (the canvas paints the
       label) and unreachable, since the shield covers the whole button. */
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip-path: inset(50%); white-space: nowrap;
    -webkit-user-select: none; user-select: none; pointer-events: none;
  }
  .bm .bm-shield {
    position: absolute; inset: 0; z-index: 1; cursor: grab;
    background: transparent; border: none; padding: 0; margin: 0;
    -webkit-user-select: none; user-select: none;
  }
  .bm:hover { transform: translateY(-2px); }
  code { font-family: var(--font-mono); font-size: 0.8rem; background: rgba(255, 255, 255, 0.04); padding: 0.15rem 0.45rem; border-radius: 6px; }
  .next { margin-top: 1.2rem; }
  .waiting { display: inline-flex; align-items: center; gap: 0.6rem; color: var(--l-muted); font-size: 0.88rem; }
  .waiting .spin {
    width: 16px; height: 16px; border: 2px solid var(--l-border-strong); border-top-color: var(--l-accent-2);
    border-radius: 50%; animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .okmsg { color: var(--l-green); font-weight: 700; font-size: 0.88rem; }
  .errmsg { color: var(--l-red); font-weight: 700; font-size: 0.88rem; }
  .hidden { display: none; }
  .footer-link { margin-top: 1.4rem; text-align: center; color: var(--l-faint); font-size: 0.82rem; }
  .footer-link a { color: var(--l-muted); text-decoration: none; border-bottom: 1px solid transparent; transition: color 0.2s, border-color 0.2s; }
  .footer-link a:hover { color: var(--l-accent-2); border-bottom-color: var(--l-accent-2); }
  @media (max-width: 480px) {
    body { padding: 1rem 0.6rem; align-items: flex-start; padding-top: 2rem; }
    .wrap { padding: 2rem 1.3rem; border-radius: 16px; }
    h1 { font-size: 1.3rem; }
  }
  @media (prefers-reduced-motion: reduce) {
    .brand-logo { animation: none !important; }
    * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
  }
  :focus-visible { outline: 2px solid var(--l-accent); outline-offset: 2px; }
  ::selection { background: var(--accent-soft); color: var(--l-text); }
</style>
</head>
<body>
<div class="bg-grid" aria-hidden="true"></div>
<div class="bg-glow bg-glow-a" aria-hidden="true"></div>
<div class="bg-glow bg-glow-b" aria-hidden="true"></div>
<div class="noise" aria-hidden="true"></div>
<div class="wrap">
  <div class="header">
    <a class="brand" href="/" style="margin-bottom:1.4rem">
      <svg class="brand-logo" viewBox="0 0 40 40" aria-hidden="true">
        <defs>
          <linearGradient id="brand-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#6c7bff"></stop>
            <stop offset="1" stop-color="#22d3ee"></stop>
          </linearGradient>
        </defs>
        <path d="M20 2.5 L35 11 V29 L20 37.5 L5 29 V11 Z" fill="none" stroke="url(#brand-grad)" stroke-width="2.6" stroke-linejoin="round"></path>
        <path d="M20 13.5 L12.5 26.5 H27.5 Z" fill="none" stroke="url(#brand-grad)" stroke-width="1.8" stroke-linejoin="round"></path>
        <circle cx="20" cy="13.5" r="3" fill="url(#brand-grad)"></circle>
        <circle cx="12.5" cy="26.5" r="3" fill="url(#brand-grad)"></circle>
        <circle cx="27.5" cy="26.5" r="3" fill="url(#brand-grad)"></circle>
      </svg>
      <span class="brand-name">Dany<span class="brand-accent">API</span></span>
    </a>
    <h1 id="title"><span class="grad-text">__header_title__</span></h1>
    <p class="sub" id="subtitle">__header_sub__</p>
  </div>
  <div class="progress"><div class="bar" id="bar"></div></div>

  <!-- STEP 0: install bookmarklet -->
  <div class="step" id="step0">
    <h2><span class="badge">0</span> __step0_heading__</h2>
    <p>__step0_intro__</p>
    <p>
      <a class="bm" href="__BOOKMARKLET__" aria-label="__bookmarklet_aria__"><span class="bm-title">__bookmarklet_label__</span><canvas
        class="bm-label" width="214" height="24" aria-hidden="true"></canvas><!--
      --><span class="bm-shield" aria-hidden="true"></span></a>
    </p>
    <p class="fineprint">__step0_fineprint__</p>
    <div class="next">
      <button class="btn" id="btn-next0">__step0_next__</button>
    </div>
  </div>

  <!-- STEP 1: DeepSeek -->
  <div class="step hidden" id="step1">
    <h2><span class="badge">1</span> __step1_heading__</h2>
    <p>__step1_p1__</p>
    <p>__step1_p2__</p>
    <p>__step1_p3__</p>
    <p class="fineprint">__step1_fineprint__</p>
    <a class="btn btn-primary" id="btn-open-deepseek" style="width:100%" href="javascript:void(0)"
      onclick="openProvider('__DEEPSEEK_URL__', 'deepseek'); return false;">__step1_button__</a>
    <div class="next">
      <span class="waiting" id="wait1"><span class="spin"></span>__step1_waiting__</span>
    </div>
  </div>

  <!-- STEP 2: Qwen -->
  <div class="step hidden" id="step2">
    <h2><span class="badge">2</span> __step2_heading__</h2>
    <p>__step2_p1__</p>
    <p>__step2_p2__</p>
    <p>__step2_p3__</p>
    <p class="fineprint">__step2_fineprint__</p>
    <a class="btn btn-primary" id="btn-open-qwen" style="width:100%" href="javascript:void(0)"
      onclick="openProvider('__QWEN_URL__', 'qwen'); return false;">__step2_button__</a>
    <div class="next">
      <span class="waiting" id="wait2"><span class="spin"></span>__step2_waiting__</span>
    </div>
  </div>

  <!-- STEP 3: all done -->
  <div class="step hidden" id="step3">
    <h2><span class="badge done">✓</span> __step3_heading__</h2>
    <p>__step3_text__</p>
  </div>

  <div class="footer-link">
    <a href="__footer_public_url__" target="_blank" rel="noopener">__footer_public_instance__</a> &middot;
    <a href="__footer_docs_url__" target="_blank" rel="noopener">__footer_docs__</a> &middot;
    <a href="__footer_github_url__" target="_blank" rel="noopener">__footer_github__</a>
  </div>
</div>

<script>
// Visible strings, injected from the TEXTS config at the top of get_tokens.py.
const T = __TEXTS_JSON__;
function tfmt(key, provider) {
  return T[key].replace("{provider}", provider.charAt(0).toUpperCase() + provider.slice(1));
}

const state = { step: 0, deepseek: __HAS_DEEPSEEK__, qwen: __HAS_QWEN__ };

// Paint the grabber label as pixels on a canvas: there is no text node in the
// button at all, so no browser can ever turn a drag into a text selection.
(function () {
  var c = document.querySelector("canvas.bm-label");
  if (!c) return;
  var dpr = window.devicePixelRatio || 1;
  c.width = 214 * dpr; c.height = 24 * dpr;
  c.style.width = "214px"; c.style.height = "24px";
  var ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  var full = T.bookmarklet_label;
  var emoji = full.split(" ")[0], text = " " + full.split(" ").slice(1).join(" ");
  ctx.textBaseline = "middle";
  ctx.font = "17px sans-serif";
  ctx.fillText(emoji, 2, 13);
  ctx.font = "700 0.9rem Manrope, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.fillStyle = "#ffd479";
  ctx.fillText(text, 26, 12.5);
})();

function syncFromServer() {
  if (state.deepseek) showToken("1", "deepseek");
  if (state.qwen) showToken("2", "qwen");
  if (state.deepseek && state.qwen) { if (state.step < 3) successPause(finish); return true; }
  // Auto-advance: DeepSeek already in (e.g. page refreshed) → jump straight to Qwen.
  if (state.step === 1 && state.deepseek) { goTo(2); }
  return false;
}

let providerTabs = {};

function openProvider(url, p) {
  const existing = providerTabs[p];
  // Reuse the tab we opened last time if the browser gave us a handle, so we
  // can close it ourselves once the token arrives.
  if (existing && !existing.closed) { existing.focus(); return; }
  const w = window.open(url, "_blank");
  if (w) { providerTabs[p] = w; w.focus(); }
  else {
    alert(T.js_alert_popup_blocked);
  }
}

function tfmt(key, provider) {
  return T[key].replace("{provider}", provider.charAt(0).toUpperCase() + provider.slice(1));
}

function showToken(n, provider) {
  const wait = document.getElementById("wait" + n);
  if (!wait) return;
  // Grey out the provider button - no need to open the tab again.
  const btn = document.getElementById("btn-open-" + provider);
  if (btn) { btn.classList.add("disabled"); btn.removeAttribute("onclick"); }
  // Green validation message also tells the user what happens next.
  const both = state.deepseek && state.qwen;
  let msg;
  if (both) {
    msg = tfmt("js_ok_both", provider);
  } else if (provider === "deepseek") {
    msg = T.js_ok_deepseek;
  } else {
    msg = T.js_ok_qwen;
  }
  wait.innerHTML = '<span class="okmsg">' + msg + '</span>';
  const tab = providerTabs[provider];
  if (tab && !tab.closed) { try { tab.close(); } catch (e) {} }  // close the provider tab from the wizard side
}

function showFail(provider) {
  const n = provider === "deepseek" ? "1" : "2";
  const wait = document.getElementById("wait" + n);
  if (!wait) return;
  // Button stays active so the user can go back, log in, and retry.
  wait.innerHTML = '<span class="errmsg">' + tfmt("js_fail", provider) + '</span>';
}

function goTo(n) {
  state.step = n;
  for (let i = 0; i <= 3; i++) {
    const el = document.getElementById("step" + i);
    if (el) el.classList.toggle("hidden", i !== n);
  }
  document.getElementById("bar").style.width = (n / 3 * 100) + "%";
  if (n === 0) document.getElementById("title").textContent = T.js_title_wizard;
  if (n === 1) document.getElementById("title").textContent = T.js_title_step1;
  if (n === 2) document.getElementById("title").textContent = T.js_title_step2;
  if (n === 3) document.getElementById("title").textContent = T.js_title_done;
  window.scrollTo(0, 0);
}

function finish() {
  if (finishing) return;
  finishing = true;
  goTo(3);
  setTimeout(() => { location.href = "/results"; }, 1200);
}
let finishing = false;

function successPause(next) {
  // Let the user actually see the green checkmark for ~2s before moving on.
  setTimeout(next, 2000);
}

document.getElementById("btn-next0").addEventListener("click", () => {
  if (state.deepseek && state.qwen) { finish(); return; }
  goTo(state.deepseek ? 2 : 1);
  syncFromServer();
});

// Poll for incoming tokens
async function poll() {
  try {
    const r = await fetch("/status");
    const s = await r.json();
    let changed = false;
    for (const p of ["deepseek", "qwen"]) {
      if (s[p] && !state[p]) { state[p] = true; changed = true; showToken(p === "deepseek" ? "1" : "2", p); }
      else if (!s[p] && s[p + "_failed"] && !state[p + "_seen_fail"]) {
        state[p + "_seen_fail"] = true;
        showFail(p);
      }
    }
    if (changed && state.deepseek && state.qwen) { successPause(finish); return; }
    // Auto-advance: as soon as the DeepSeek token lands, linger on the
    // success message for a moment, then move on to Qwen.
    if (changed && state.step === 1 && state.deepseek && !state.qwen) {
      successPause(() => { if (state.step === 1 && !state.qwen) goTo(2); });
    }
  } catch (e) {}
  setTimeout(poll, 1200);
}

if (!syncFromServer()) poll();
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------------
# Results page
# ----------------------------------------------------------------------------

RESULTS_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__results_page_title__</title>
<link rel="icon" href="__FAVICON_DATA_URI__" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="__FONTS_CSS_URL__" rel="stylesheet" />
<style>
  :root {
    --font-disp: "Unbounded", sans-serif;
    --font-body: "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "SF Mono", "Fira Code", "Cascadia Code", "JetBrains Mono", Consolas, monospace;
    --accent-soft: rgba(99, 102, 241, 0.14);
    --l-surface: #0d1119;
    --l-border: rgba(255, 255, 255, 0.08);
    --l-border-strong: rgba(255, 255, 255, 0.16);
    --l-text: #e9edf5;
    --l-muted: #9aa3b5;
    --l-faint: #5d6577;
    --l-accent: #6c7bff;
    --l-accent-2: #22d3ee;
    --l-grad: linear-gradient(120deg, #6c7bff 0%, #22d3ee 100%);
    --l-green: #34d399;
    --l-red: #ff5f56;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 15px; scroll-behavior: smooth; }
  body {
    font-family: var(--font-body);
    background: #06070c;
    color: var(--l-text);
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    min-height: 100vh;
    display: flex; align-items: flex-start; justify-content: center;
    padding: 3rem 1rem;
  }
  body::before {
    content: "";
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
      radial-gradient(900px 520px at 85% -10%, var(--accent-soft), transparent 60%),
      radial-gradient(760px 480px at -10% 105%, var(--accent-soft), transparent 60%);
  }
  .bg-grid {
    position: fixed; inset: 0; z-index: -3; pointer-events: none;
    background-image:
      linear-gradient(var(--l-border) 1px, transparent 1px),
      linear-gradient(90deg, var(--l-border) 1px, transparent 1px);
    background-size: 56px 56px;
    opacity: 0.5;
    -webkit-mask-image: radial-gradient(ellipse 100% 62% at 50% 0%, #000 0%, transparent 78%);
    mask-image: radial-gradient(ellipse 100% 62% at 50% 0%, #000 0%, transparent 78%);
  }
  .bg-glow { position: fixed; z-index: -2; pointer-events: none; border-radius: 50%; filter: blur(110px); opacity: 0.5; }
  .bg-glow-a { width: 640px; height: 640px; top: -220px; left: 50%; transform: translateX(-50%);
    background: radial-gradient(circle, rgba(108, 123, 255, 0.55), transparent 65%); }
  .bg-glow-b { width: 520px; height: 520px; bottom: -180px; right: -140px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.35), transparent 65%); }
  .noise {
    position: fixed; inset: 0; z-index: 60; pointer-events: none; opacity: 0.035;
    background-image: __NOISE_DATA_URI__;
  }
  .wrap {
    position: relative; z-index: 1;
    background: var(--l-surface);
    border: 1px solid var(--l-border-strong);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    max-width: 520px; width: 100%;
    box-shadow: 0 30px 80px -30px rgba(0, 0, 0, 0.7);
  }
  .brand { display: inline-flex; align-items: center; gap: 0.6rem; text-decoration: none; transition: transform 0.2s ease; }
  .brand:hover { transform: translateY(-1px); }
  .brand-logo { width: 36px; height: 36px; display: block;
    filter: drop-shadow(0 2px 10px rgba(108, 123, 255, 0.5));
    transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
  .brand:hover .brand-logo { animation: logo-spin 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
  @keyframes logo-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .brand-name { font-family: var(--font-disp); font-weight: 700; font-size: 1.15rem; letter-spacing: -0.02em; color: var(--l-text); }
  .brand-accent { background: var(--l-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
  .header { text-align: center; margin-bottom: 2.2rem; }
  h1 { font-family: var(--font-disp); font-weight: 700; font-size: clamp(1.4rem, 3vw, 1.8rem);
    line-height: 1.25; letter-spacing: -0.02em; margin-bottom: 0.6rem; color: var(--l-text); }
  .grad-text { background: var(--l-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
  .sub { color: var(--l-muted); font-size: 0.95rem; margin-bottom: 2rem; }
  .sub a { color: var(--l-accent-2); text-decoration: none; border-bottom: 1px solid transparent; transition: color 0.2s, border-color 0.2s; }
  .sub a:hover { color: #fff; border-bottom-color: var(--l-accent-2); }
  .card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--l-border); border-radius: 11px;
    padding: 1rem 1.1rem; margin-bottom: 0.7rem;
  }
  .card-header { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.75rem; }
  .logo {
    width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-weight: 800; font-size: 0.85rem; color: #fff;
    flex-shrink: 0;
  }
  .ds { background: linear-gradient(135deg, #4d6bfe, #2a3f9e); }
  .qw { background: linear-gradient(135deg, #6c4dfe, #9e2af0); }
  .card h2 { font-size: 0.95rem; font-weight: 700; color: var(--l-text); line-height: 1; }
  .token-row { margin-top: 0.75rem; }
  .pane-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    margin-bottom: 0.5rem; font-family: var(--font-mono); font-size: 0.72rem;
    color: var(--l-faint); text-transform: uppercase; letter-spacing: 0.08em;
  }
  .token-box {
    background: #06070c; border: 1px solid var(--l-border-strong); border-radius: 12px;
    padding: 1.1rem 1.2rem; font-family: var(--font-mono); font-size: 0.82rem; color: var(--l-muted);
    word-break: break-all; line-height: 1.7; max-height: 120px; overflow-y: auto; user-select: all;
  }
  .copy-btn {
    appearance: none; border: 1px solid var(--l-border-strong);
    background: rgba(255,255,255,0.05); color: var(--l-muted);
    border-radius: 8px; padding: 0.32rem 0.8rem;
    font-family: var(--font-mono); font-size: 0.7rem; font-weight: 600; cursor: pointer;
    text-transform: none; letter-spacing: 0.02em; white-space: nowrap;
    transition: color 0.2s, border-color 0.2s, background 0.2s, transform 0.15s;
  }
  .copy-btn:hover { color: #fff; border-color: var(--l-accent); background: rgba(108,123,255,0.15); transform: translateY(-1px); }
  .copy-btn:active { transform: translateY(0); }
  .copy-btn.ok { color: var(--l-green); border-color: rgba(52,211,153,0.6); background: rgba(255,255,255,0.05); }
  .footer { margin-top: 2.2rem; text-align: center; color: var(--l-faint); font-size: 0.82rem; }
  .footer a { color: var(--l-muted); text-decoration: none; border-bottom: 1px solid transparent; transition: color 0.2s, border-color 0.2s; }
  .footer a:hover { color: var(--l-accent-2); border-bottom-color: var(--l-accent-2); }
  @media (max-width: 640px) { .token-row { flex-direction: column; } .copy-btn { padding: 0.65rem; } }
  @media (max-width: 480px) {
    body { padding: 1rem 0.6rem; align-items: flex-start; padding-top: 2rem; }
    .wrap { padding: 2rem 1.3rem; border-radius: 16px; }
    h1 { font-size: 1.3rem; }
  }
  @media (prefers-reduced-motion: reduce) {
    .brand-logo { animation: none !important; }
    * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
  }
  :focus-visible { outline: 2px solid var(--l-accent); outline-offset: 2px; }
  ::selection { background: var(--accent-soft); color: var(--l-text); }
</style>
</head>
<body>
<div class="bg-grid" aria-hidden="true"></div>
<div class="bg-glow bg-glow-a" aria-hidden="true"></div>
<div class="bg-glow bg-glow-b" aria-hidden="true"></div>
<div class="noise" aria-hidden="true"></div>
<div class="wrap">
  <div class="header">
    <a class="brand" href="/" style="margin-bottom:1.4rem">
      <svg class="brand-logo" viewBox="0 0 40 40" aria-hidden="true">
        <defs>
          <linearGradient id="brand-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#6c7bff"></stop>
            <stop offset="1" stop-color="#22d3ee"></stop>
          </linearGradient>
        </defs>
        <path d="M20 2.5 L35 11 V29 L20 37.5 L5 29 V11 Z" fill="none" stroke="url(#brand-grad)" stroke-width="2.6" stroke-linejoin="round"></path>
        <path d="M20 13.5 L12.5 26.5 H27.5 Z" fill="none" stroke="url(#brand-grad)" stroke-width="1.8" stroke-linejoin="round"></path>
        <circle cx="20" cy="13.5" r="3" fill="url(#brand-grad)"></circle>
        <circle cx="12.5" cy="26.5" r="3" fill="url(#brand-grad)"></circle>
        <circle cx="27.5" cy="26.5" r="3" fill="url(#brand-grad)"></circle>
      </svg>
      <span class="brand-name">Dany<span class="brand-accent">API</span></span>
    </a>
    <h1><span class="grad-text">__results_title__</span></h1>
    <p class="sub">__RESULTS_SUB__</p>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="logo ds">DS</div>
      <h2>DeepSeek</h2>
    </div>
    <div class="token-row">
      <div class="pane-head"><span>__results_pane_ds__</span><button class="copy-btn" data-target="ds-token">__results_copy__</button></div>
      <div class="token-box" id="ds-token">__DEEPSEEK_TOKEN__</div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="logo qw">Q</div>
      <h2>Qwen</h2>
    </div>
    <div class="token-row">
      <div class="pane-head"><span>__results_pane_qw__</span><button class="copy-btn" data-target="qw-token">__results_copy__</button></div>
      <div class="token-box" id="qw-token">__QWEN_TOKEN__</div>
    </div>
  </div>

  <p class="footer">
    <a href="/">__results_footer_again__</a> &middot;
    <a href="__footer_docs_url__" target="_blank" rel="noopener">__results_footer_docs__</a> &middot;
    <a href="__footer_github_url__" target="_blank" rel="noopener">__results_footer_github__</a>
  </p>
</div>

<script>
document.querySelectorAll("button.copy-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const target = document.getElementById(btn.dataset.target);
    const text = target.textContent.trim();
    try { await navigator.clipboard.writeText(text); }
    catch (e) {
      const range = document.createRange(); range.selectNodeContents(target);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      document.execCommand("copy");
    }
    btn.textContent = "__results_copied__"; btn.classList.add("ok");
    setTimeout(() => { btn.textContent = "__results_copy__"; btn.classList.remove("ok"); }, 1600);
  });
});
</script>
</body>
</html>
"""


def _apply_texts(template: str, extra: dict[str, str] | None = None) -> str:
    """Fill __key__ placeholders with the TEXTS config (plus any extra values)."""
    out = template
    for key, value in TEXTS.items():
        out = out.replace(f"__{key}__", value)
    # Shared HTML assets (identical markup on both pages).
    out = out.replace("__FAVICON_DATA_URI__", FAVICON_DATA_URI)
    out = out.replace("__FONTS_CSS_URL__", FONTS_CSS_URL)
    out = out.replace("__NOISE_DATA_URI__", NOISE_DATA_URI)
    if extra:
        for key, value in extra.items():
            out = out.replace(f"__{key}__", value)
    return out


def render_setup_page(port: int = RESULT_PORT) -> str:
    texts_json = json.dumps(TEXTS, ensure_ascii=False)
    return _apply_texts(
        SETUP_PAGE,
        {
            "BOOKMARKLET": html_escape(build_bookmarklet(port)),
            "DEEPSEEK_URL": DEEPSEEK_URL,
            "QWEN_URL": QWEN_URL,
            "HAS_DEEPSEEK": "true" if STATE["deepseek"] else "false",
            "HAS_QWEN": "true" if STATE["qwen"] else "false",
            "TEXTS_JSON": texts_json,
        },
    )


def render_results_page() -> str:
    def show(tok: str | None) -> str:
        return html_escape(tok) if tok else TEXTS["results_no_token"]

    public_url = TEXTS["footer_public_url"]
    return _apply_texts(
        RESULTS_PAGE,
        {
            "DEEPSEEK_TOKEN": show(STATE["deepseek"]),
            "QWEN_TOKEN": show(STATE["qwen"]),
            "RESULTS_SUB": TEXTS["results_sub"].replace("{public_url}", public_url),
        },
    )


# ----------------------------------------------------------------------------
# Local HTTP server
# ----------------------------------------------------------------------------


SUCCESS_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__popup_title__</title></head>
<body style="background:#06070c;color:#9aa3b5;font-family:-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;padding:24px;font-size:13px">
<!-- Close instantly - this tab was opened by the wizard, so window.close()
     is allowed. The wizard's poll shows the green checkmark instead. -->
<script>try { window.close(); } catch (e) {}</script>
<span id="fb" style="display:none">__popup_message__</span>
<script>
 setTimeout(function () {
   try { window.close(); } catch (e) {}
   document.getElementById("fb").style.display = "inline";
 }, 250);
</script>
</body></html>
"""


def register_token(provider: str, token: str) -> bool:
    """Validate and store a token. Returns True on success.

    Must stay in sync with the bookmarklet's client-side checks: only
    token-shaped values from the provider's auth storage count, so a logged-out
    provider page (or random localStorage junk) cannot be registered. An
    empty/invalid token records a failed attempt instead (visible in /status).
    """
    if provider not in ("deepseek", "qwen"):
        return False
    # The provider can rotate from JWT/hex to an opaque bearer value and may
    # use characters outside the URL-safe subset. The bookmarklet only sends
    # values from the provider's known auth record, so validate shape here
    # without imposing a token alphabet.
    if token and re.fullmatch(r"\S{16,4096}", token):
        STATE[provider] = token
        STATE[provider + "_failed"] = False
        print(f"    ✔ {provider} token received ({len(token)} chars)")
        return True
    STATE[provider + "_failed"] = True
    print(f"    ✖ {provider}: no valid token found (user probably not logged in)")
    return False


class Handler(BaseHTTPRequestHandler):
    # Port the setup page's bookmarklet should call back to. Kept as a class
    # attribute (set by serve()) so the handler needs no module-global state.
    serve_port: int = RESULT_PORT

    def _send(self, body: bytes, status: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # CORS + Private Network Access: requests arriving here come from
        # https://chat.deepseek.com etc. Chrome requires these headers or it
        # silently drops the request (PNA preflight).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlsplit

        parts = urlsplit(self.path)
        path = parts.path
        if path == "/status":
            self._send(json.dumps({k: v for k, v in STATE.items() if not k.endswith("_failed") or v}).encode(), ctype="application/json")
        elif path == "/results":
            self._send(render_results_page().encode())
        elif path == "/collect":
            # Top-level navigation fallback: /collect?p=deepseek&t=TOKEN
            qs = parse_qs(parts.query)
            provider = (qs.get("p") or [""])[0]
            token = (qs.get("t") or [""])[0].strip()
            if register_token(provider, token):
                self._send(SUCCESS_PAGE.encode())
            else:
                # Invalid/missing token: still close the tab like a success -
                # the wizard itself shows the red "no token" hint via /status.
                self._send(SUCCESS_PAGE.encode())
        else:  # "/" and anything else -> setup page
            self._send(render_setup_page(port=self.serve_port).encode())

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/collect":
            self._send(b"not found", 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            # sendBeacon may send text/plain; strip junk before parsing
            data = json.loads(raw)
            if register_token(data.get("provider") or "", (data.get("token") or "").strip()):
                self._send(b'{"ok":true}', ctype="application/json")
            else:
                self._send(b'{"ok":false}', 400, ctype="application/json")
        except Exception:
            self._send(b'{"ok":false}', 400, ctype="application/json")

    def log_message(self, *args: Any) -> None:
        pass


def serve(port: int = RESULT_PORT, open_browser: bool = True) -> None:
    Handler.serve_port = port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"  Local page: {url}")
    print("  (Ctrl+C in this window to stop when you're done.)\n")
    if open_browser:
        print("  Opening your default browser…")
        webbrowser.open_new_tab(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DeepSeek & Qwen tokens for DanyAPI (uses your default browser)")
    parser.add_argument("--port", type=int, default=RESULT_PORT, help=f"local server port (default {RESULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = parser.parse_args()

    print("=" * 60)
    print(" DanyAPI token extractor")
    print(" DeepSeek + Qwen → DEEPSEEK_TOKENS / QWEN_TOKENS")
    print("=" * 60)
    print(" 1. Drag the grabber button to your bookmarks bar (once), click Next")
    print(" 2. DeepSeek: log in, click the grabber bookmark - page auto-advances")
    print(" 3. Qwen: same again - then both tokens are shown automatically\n")

    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
