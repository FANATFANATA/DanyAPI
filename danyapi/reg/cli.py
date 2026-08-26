from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import string
import sys
from pathlib import Path

from .captcha import CapSolverSolver, CaptchaError, HcaptchaSolver, ManualCaptchaSolver, StaticSolver, TwoCaptchaSolver
from .deepseek import DeepSeekRegistrar, RegError
from .email import CodeSource, EmailCodeError, ImapCodeSource, ManualCodeSource, StaticCodeSource

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

PASSWORD_ALPHABET = string.ascii_letters + string.digits


def generate_password(length: int = 16) -> str:
    while True:
        password = "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))
        if any(c.islower() for c in password) and any(c.isupper() for c in password) and any(c.isdigit() for c in password):
            return password


def build_solver(args: argparse.Namespace) -> HcaptchaSolver:
    if args.captcha_token:
        return StaticSolver(args.captcha_token)
    if args.solver == "2captcha":
        if not args.api_key:
            raise SystemExit("--api-key is required for --solver 2captcha")
        return TwoCaptchaSolver(args.api_key)
    if args.solver == "capsolver":
        if not args.api_key:
            raise SystemExit("--api-key is required for --solver capsolver")
        return CapSolverSolver(args.api_key)
    return ManualCaptchaSolver()


def build_code_source(args: argparse.Namespace) -> CodeSource:
    if args.code:
        return StaticCodeSource(args.code)
    if args.imap_host:
        username = args.imap_user
        password = args.imap_pass
        if not username:
            raise SystemExit("--imap-user is required with --imap-host")
        if not password:
            import getpass

            password = getpass.getpass(f"IMAP password for {username}: ")
        return ImapCodeSource(
            host=args.imap_host,
            username=username,
            password=password,
            port=args.imap_port,
            folder=args.imap_folder,
            wait_seconds=args.imap_wait,
        )
    return ManualCodeSource()


def append_token_to_env(token: str) -> Path:
    existing = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    match = re.search(r"^DEEPSEEK_TOKENS=(.*)$", existing, flags=re.MULTILINE)
    tokens: list[str] = []
    if match:
        raw = match.group(1).strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1]
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if token not in tokens:
        tokens.append(token)
    line = "DEEPSEEK_TOKENS=" + ",".join(tokens)
    if match:
        updated = existing[: match.start()] + line + existing[match.end() :]
    else:
        prefix = existing.rstrip() + "\n" if existing.strip() else ""
        updated = prefix + line + "\n"
    ENV_PATH.write_text(updated, encoding="utf-8")
    return ENV_PATH


async def cmd_register(args: argparse.Namespace) -> int:
    email = args.email.strip()
    password = args.password
    if password is None:
        password = generate_password()
        print(f"Generated password: {password}")
    registrar = DeepSeekRegistrar(device_id=args.device_id, timeout=args.timeout)
    try:
        print(f"Device ID: {registrar.device_id}")
        code = args.code
        if not code:
            solver = build_solver(args)
            print("Solving hCaptcha...")
            captcha_token = await solver.solve()
            print("Sending verification email...")
            await registrar.send_email_code(email, captcha_token)
            source = build_code_source(args)
            code = await source.wait_for_code(email)
            print(f"Got code: {code}")
        else:
            print(f"Using provided code: {code}")
        print("Registering...")
        token = await registrar.register(email, password, code)
        user = await registrar.current_user(token)
        print(f"Registered: {email}")
        print(f"User ID: {user.get('id')}")
        print(f"Token: {token}")
        if args.append_env:
            path = append_token_to_env(token)
            print(f"Token appended to {path}")
        if args.json:
            print(json.dumps({"email": email, "password": password, "token": token, "user_id": user.get("id")}))
        return 0
    finally:
        await registrar.aclose()


async def cmd_login(args: argparse.Namespace) -> int:
    registrar = DeepSeekRegistrar(device_id=args.device_id, timeout=args.timeout, waf_token=args.waf_token)
    try:
        print(f"Device ID: {registrar.device_id}")
        token = await registrar.login(args.email.strip(), args.password)
        user = await registrar.current_user(token)
        print(f"Logged in: {args.email.strip()}")
        print(f"User ID: {user.get('id')}")
        print(f"Token: {token}")
        if args.append_env:
            path = append_token_to_env(token)
            print(f"Token appended to {path}")
        if args.json:
            print(json.dumps({"email": args.email.strip(), "token": token, "user_id": user.get("id")}))
        return 0
    finally:
        await registrar.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m danyapi.reg", description="DanyAPI account registration tool")
    sub = parser.add_subparsers(dest="provider", required=True)

    ds = sub.add_parser("deepseek", help="DeepSeek account operations")
    ds_sub = ds.add_subparsers(dest="action", required=True)

    reg = ds_sub.add_parser("register", help="register a new account and get a token")
    reg.add_argument("--email", required=True)
    reg.add_argument("--password", help="account password (generated and printed when omitted)")
    reg.add_argument("--code", help="email verification code (skips captcha and code sending)")
    reg.add_argument("--captcha-token", help="ready hCaptcha token")
    reg.add_argument("--solver", choices=["manual", "2captcha", "capsolver"], default="manual")
    reg.add_argument("--api-key", help="API key for the chosen captcha solver")
    reg.add_argument("--imap-host", help="IMAP host to fetch the verification code from")
    reg.add_argument("--imap-port", type=int, default=993)
    reg.add_argument("--imap-user", help="IMAP login (defaults to the registration email)")
    reg.add_argument("--imap-pass", help="IMAP password (prompted when omitted)")
    reg.add_argument("--imap-folder", default="INBOX")
    reg.add_argument("--imap-wait", type=float, default=120.0, help="seconds to wait for the email")
    reg.add_argument("--device-id")
    reg.add_argument("--timeout", type=float, default=60.0)
    reg.add_argument("--append-env", action="store_true", help="append the token to .env DEEPSEEK_TOKENS")
    reg.add_argument("--json", action="store_true", help="print a JSON summary line")
    reg.set_defaults(func=cmd_register)

    login = ds_sub.add_parser("login", help="log in to an existing account and get a token")
    login.add_argument("--email", required=True)
    login.add_argument("--password", required=True)
    login.add_argument("--waf-token", help="aws-waf-token cookie from a browser on chat.deepseek.com (required when the login endpoint is WAF-challenged)")
    login.add_argument("--device-id")
    login.add_argument("--timeout", type=float, default=60.0)
    login.add_argument("--append-env", action="store_true")
    login.add_argument("--json", action="store_true")
    login.set_defaults(func=cmd_login)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except (RegError, CaptchaError, EmailCodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
