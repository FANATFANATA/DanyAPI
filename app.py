from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parent


def ensure_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        return
    example = ROOT / ".env.example"
    if example.exists():
        shutil.copyfile(example, env_file)
        print("Created .env from .env.example")
        print("Fill in DEEPSEEK_TOKENS / QWEN_TOKENS and run again.")
        sys.exit(1)
    print("No .env or .env.example found.", file=sys.stderr)
    sys.exit(1)


def build_solver() -> None:
    src_candidates = [
        ROOT / "pow_solver.c",
        ROOT / "danyapi" / "pow_solver.c",
        ROOT / "danyapi" / "solver" / "pow_solver.c",
    ]
    src_path: Path | None = None
    for candidate in src_candidates:
        if candidate.exists():
            src_path = candidate
            break

    if src_path is None:
        matches = list(ROOT.glob("**/pow_solver.c"))
        if matches:
            src_path = matches[0]

    if src_path is None:
        return

    is_win = sys.platform == "win32"
    bin_name = "pow_solver.exe" if is_win else "pow_solver"
    bin_path = src_path.parent / bin_name

    compilers = ["clang", "gcc", "cl"] if is_win else ["clang", "gcc", "cc"]
    chosen_compiler: str | None = None
    for comp in compilers:
        if shutil.which(comp):
            chosen_compiler = comp
            break

    if not chosen_compiler:
        if not bin_path.exists() and not (ROOT / bin_name).exists():
            print(
                "Warning: no C compiler found in PATH to build pow_solver.",
                file=sys.stderr,
            )
        return

    success = False
    if chosen_compiler == "cl":
        cmd = [
            "cl",
            "/nologo",
            "/O2",
            "/Oi",
            "/Ot",
            "/GL",
            str(src_path),
            f"/Fe:{bin_path}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        success = res.returncode == 0
    else:
        cmd_fast = [
            chosen_compiler,
            "-O3",
            "-march=native",
            *(["-pthread"] if not is_win else []),
            str(src_path),
            "-o",
            str(bin_path),
        ]
        res = subprocess.run(cmd_fast, capture_output=True, text=True)
        if res.returncode == 0:
            success = True
        else:
            cmd_compat = [
                chosen_compiler,
                "-O3",
                *(["-pthread"] if not is_win else []),
                str(src_path),
                "-o",
                str(bin_path),
            ]
            res2 = subprocess.run(cmd_compat, capture_output=True, text=True)
            success = res2.returncode == 0

    if success:
        if not is_win:
            os.chmod(bin_path, 0o755)
        root_bin = ROOT / bin_name
        if bin_path.resolve() != root_bin.resolve():
            shutil.copy2(bin_path, root_bin)
            if not is_win:
                os.chmod(root_bin, 0o755)
        print(
            f"Native pow_solver successfully compiled via {chosen_compiler} ({bin_name})"
        )
    elif not bin_path.exists() and not (ROOT / bin_name).exists():
        print("Warning: failed to compile native pow_solver.", file=sys.stderr)


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        print(
            f"DanyAPI requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    ensure_env()
    build_solver()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)

    import uvicorn

    from danyapi.config import settings
    from danyapi.logging import uvicorn_log_config

    print(f"DanyAPI starting on {settings.host}:{settings.port}")
    uvicorn.run(
        "danyapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
