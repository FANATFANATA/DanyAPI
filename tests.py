from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

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
EXTRA_PYTEST_ARGS = sys.argv[1:]

BANNED_PATTERNS = ["\u2014", "\u2013", "Zero keys.", "Ноль ключей.", "live demo"]
GUARD_TEXT_DIRS = ["docs"]
GUARD_TEXT_FILES = ["README.md"]


def run_cmd(name: str, cmd: list[str]) -> bool:
    print()
    print(f"=== [{name}] {' '.join(cmd)}")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
    except FileNotFoundError:
        print(f"[{name}] SKIPPED: tool not found")
        return True
    elapsed = time.monotonic() - started
    status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"=== [{name}] {status} ({elapsed:.1f}s)")
    return proc.returncode == 0


def iter_guard_targets() -> list[Path]:
    targets = [ROOT / f for f in GUARD_TEXT_FILES]
    for d in GUARD_TEXT_DIRS:
        targets.extend(p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in {".html", ".js", ".css", ".py", ".md", ".sh", ".bat"})
    return sorted(targets)


def check_repo_guards() -> bool:
    print()
    print("=== [repo guards] dashes/banned text")
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
    for problem in problems[:40]:
        print(problem)
    status = "OK" if not problems else f"FAIL ({len(problems)} problems)"
    print(f"=== [repo guards] {status}")
    return not problems


StepRunner = list[str] | Callable[[str], bool]

STEPS: list[tuple[str, StepRunner]] = [
    ("repo guards", lambda _name: check_repo_guards()),
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
    ("pip-audit", [sys.executable, "-m", "pip_audit", "--progress-spinner", "off"]),
    ("pip check", [sys.executable, "-m", "pip", "check"]),
    ("pytest", [sys.executable, "-m", "pytest", "-q", "--cov=danyapi", "--cov-report=term-missing", *EXTRA_PYTEST_ARGS]),
]


def main() -> int:
    results: list[tuple[str, bool]] = []
    for name, runner in STEPS:
        ok = runner(name) if callable(runner) else run_cmd(name, runner)
        results.append((name, ok))
    width = max(len(n) for n, _ in results)
    print()
    print("======== SUMMARY ========")
    failed = False
    for name, ok in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed = True
        print(f"{mark:4}  {name.ljust(width)}")
    if failed:
        print("RESULT: FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
