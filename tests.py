from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeAlias

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parent
PYTHON_DIRS = ["danyapi", "tests", "docs"]
BANDIT_SKIPS = "B101,B104,B112,B311,B404,B603"
PYLINT_DISABLES = "import-error,unsubscriptable-object,not-an-iterable"
C_SOURCES = ["danyapi/deepseek/pow_solver.c"]
CLANG_TIDY_CHECKS = (
    "clang-diagnostic-*,clang-analyzer-*,-clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling,-clang-analyzer-optin.taint.TaintedAlloc"
)

BANNED_PATTERNS = ["\u2014", "\u2013", "Zero keys.", "Ноль ключей.", "live demo"]
GUARD_TEXT_DIRS = ["docs"]
GUARD_TEXT_FILES = ["README.md"]

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

StepRunner: TypeAlias = list[str] | Callable[[], tuple[bool, str]]
StepResult: TypeAlias = tuple[str, bool, float, str, str]


def paint(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{RESET}" if enabled else text


def parse_args(argv: list[str]) -> tuple[int, list[str]]:
    jobs = 1
    pytest_args: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-j", "--jobs"):
            if i + 1 >= len(argv):
                print(f"tests: {arg} requires a value", file=sys.stderr)
                raise SystemExit(2)
            try:
                jobs = int(argv[i + 1])
            except ValueError:
                print(f"tests: {arg} expects an integer, got {argv[i + 1]!r}", file=sys.stderr)
                raise SystemExit(2) from None
            i += 2
            continue
        if arg.startswith("--jobs="):
            value = arg.split("=", 1)[1]
            try:
                jobs = int(value)
            except ValueError:
                print(f"tests: --jobs expects an integer, got {value!r}", file=sys.stderr)
                raise SystemExit(2) from None
            i += 1
            continue
        if arg.startswith("-j") and len(arg) > 2:
            value = arg[2:]
            try:
                jobs = int(value)
            except ValueError:
                print(f"tests: -j expects an integer, got {value!r}", file=sys.stderr)
                raise SystemExit(2) from None
            i += 1
            continue
        pytest_args.append(arg)
        i += 1
    if jobs < 1:
        jobs = max(os.cpu_count() or 1, 1)
    return jobs, pytest_args


def iter_guard_targets() -> list[Path]:
    targets = [ROOT / f for f in GUARD_TEXT_FILES]
    for d in GUARD_TEXT_DIRS:
        targets.extend(p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in {".html", ".js", ".css", ".py", ".md", ".sh", ".bat"})
    return sorted(targets)


def check_repo_guards() -> tuple[bool, str]:
    problems: list[str] = []
    for path in iter_guard_targets():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{path}: not utf-8 ({exc})")
            continue
        for pattern in BANNED_PATTERNS:
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    problems.append(f"{path}:{lineno}: banned pattern {pattern!r}")
    if tomllib is not None:
        try:
            tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"pyproject.toml: invalid TOML ({exc})")
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{path}: invalid JSON ({exc})")
    if yaml is not None:
        for path in (ROOT / ".github").rglob("*.yml"):
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"{path}: invalid YAML ({exc})")
    if not problems:
        return True, ""
    shown = problems[:40]
    if len(problems) > 40:
        shown.append(f"... and {len(problems) - 40} more")
    return False, f"{len(problems)} problem(s):\n" + "\n".join(shown)


def run_step(name: str, runner: StepRunner) -> StepResult:
    started = time.monotonic()
    if isinstance(runner, list):
        cmd = runner
        detail = " ".join(cmd)
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError:
            elapsed = time.monotonic() - started
            return name, False, elapsed, f"tool not found: {cmd[0]}", detail
        elapsed = time.monotonic() - started
        ok = proc.returncode == 0
        parts: list[str] = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip())
        if proc.stderr:
            parts.append(proc.stderr.rstrip())
        return name, ok, elapsed, "\n".join(parts), detail
    ok, output = runner()
    elapsed = time.monotonic() - started
    return name, ok, elapsed, output, ""


def print_step(result: StepResult, use_color: bool) -> None:
    name, ok, elapsed, output, detail = result
    print()
    print(f"=== [{name}] {detail}".rstrip())
    if output:
        print(output)
    status = paint("OK", GREEN, use_color) if ok else paint("FAIL", RED, use_color)
    print(f"=== [{name}] {status} ({elapsed:.1f}s)")


def build_steps(pytest_args: list[str]) -> list[tuple[str, StepRunner]]:
    return [
        ("repo guards", check_repo_guards),
        ("py_compile", [sys.executable, "-m", "compileall", "-q", "-f", *PYTHON_DIRS]),
        ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
        ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("isort --check", [sys.executable, "-m", "isort", "--check-only", "."]),
        ("flake8", [sys.executable, "-m", "flake8"]),
        ("pyflakes", [sys.executable, "-m", "pyflakes", *PYTHON_DIRS]),
        ("mypy", [sys.executable, "-m", "mypy", *PYTHON_DIRS]),
        ("pyright", [sys.executable, "-m", "pyright", *PYTHON_DIRS]),
        ("pylint", [sys.executable, "-m", "pylint", "--errors-only", "--recursive=y", f"--disable={PYLINT_DISABLES}", *PYTHON_DIRS]),
        ("bandit", [sys.executable, "-m", "bandit", "-q", "-r", "danyapi", "-s", BANDIT_SKIPS]),
        ("vulture", [sys.executable, "-m", "vulture"]),
        ("clang-format", ["clang-format", "--dry-run", "-Werror", *C_SOURCES]),
        (
            "clang-tidy",
            [
                "clang-tidy",
                f"-checks={CLANG_TIDY_CHECKS}",
                "--warnings-as-errors=*",
                *C_SOURCES,
                "--",
                "-std=c11",
                "-D_CRT_SECURE_NO_WARNINGS",
            ],
        ),
        ("xenon", [sys.executable, "-m", "xenon", "--max-absolute", "F", "--max-modules", "D", "--max-average", "D", "danyapi"]),
        ("pip-audit", [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", "--timeout", "90"]),
        ("pip check", [sys.executable, "-m", "pip", "check"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q", "--cov=danyapi", "--cov-report=term-missing", *pytest_args]),
    ]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    jobs, pytest_args = parse_args(sys.argv[1:])
    steps = build_steps(pytest_args)
    use_color = sys.stdout.isatty()
    results: list[StepResult] = []
    if jobs == 1:
        for name, runner in steps:
            result = run_step(name, runner)
            results.append(result)
            print_step(result, use_color)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda step: run_step(step[0], step[1]), steps))
        for result in results:
            print_step(result, use_color)
    print()
    print("======== SUMMARY ========")
    width = max(len(name) for name, *_rest in results)
    for name, ok, _elapsed, _output, _detail in results:
        raw = "PASS" if ok else "FAIL"
        mark = paint(raw.ljust(4), GREEN if ok else RED, use_color)
        print(f"{mark}  {name.ljust(width)}")
    passed = sum(1 for _name, ok, *_rest in results if ok)
    total = len(results)
    total_time = sum(elapsed for _name, _ok, elapsed, _out, _det in results)
    print()
    if passed != total:
        print(f"RESULT: FAILED ({total - passed} of {total} checks failed)")
        print(f"total time: {total_time:.1f}s")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    print(f"total time: {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
