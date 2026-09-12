import pathlib

IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".coverage",
    "egg-info",
}

EXCLUDE_EXTS = {".pyc", ".db", ".cache", ".wasm", ".exe", ".dll", ".so"}

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".md": "markdown",
    ".json": "json",
    ".xml": "xml",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c-header",
    ".hpp": "c++-header",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".lua": "lua",
    ".sql": "sql",
    ".svg": "svg",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".dockerfile": "dockerfile",
    "Dockerfile": "dockerfile",
    "makefile": "makefile",
    ".cmake": "cmake",
    ".log": "text",
    ".csv": "csv",
    ".graphql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",
}

EXT_TO_LANG = {}
for ext, lang in LANG_MAP.items():
    if ext.startswith("."):
        EXT_TO_LANG[ext] = lang
    else:
        EXT_TO_LANG[ext] = lang


def get_lang(filepath: pathlib.Path) -> str:
    name_lower = filepath.name.lower()
    if name_lower in EXT_TO_LANG:
        return EXT_TO_LANG[name_lower]
    ext = filepath.suffix.lower()
    return EXT_TO_LANG.get(ext, "unknown")


def is_ignore(path: pathlib.Path) -> bool:
    return any(p in IGNORE_PATTERNS for p in path.parts)


EXCLUDE_NAMES = {"collected.xml", "collecter.py"}


def should_include(path: pathlib.Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in EXCLUDE_EXTS:
        return False
    return True


def collect_files(root: pathlib.Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and not is_ignore(p) and should_include(p):
            files.append(p)
    files.sort(key=lambda p: str(p.relative_to(root)).lower())
    return files


def read_file_content(fp: pathlib.Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e}"


def main():
    root = pathlib.Path(__file__).resolve().parent
    files = collect_files(root)
    out_path = root / "collected.xml"

    total_size = sum(fp.stat().st_size for fp in files)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<collection source="{root}" files="{len(files)}" totalSize="{total_size}">',
    ]

    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        size = fp.stat().st_size
        lang = get_lang(fp)
        content = read_file_content(fp)

        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines.append(f'  <file name="{rel}" lang="{lang}" size="{size}">{escaped}</file>')

    lines.append("</collection>")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected {len(files)} files ({total_size} bytes) -> {out_path}")


if __name__ == "__main__":
    main()
