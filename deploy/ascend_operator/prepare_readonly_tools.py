#!/usr/bin/env python3
"""Publish canonical operator tools into a read-only Docker bind source."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "fixtures",
    "test_fixtures",
    "tests",
    "self_check",
    "self-check",
}


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in EXCLUDED_NAMES or name.endswith((".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def publish_readonly_tools(source: Path, dest: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing canonical tools source: {source}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, symlinks=True, ignore=_ignore)

    required = ("triton_eval_pipeline.sh", "env.sh", "npu_lease_exec.py")
    missing = [name for name in required if not (dest / name).is_file()]
    if missing:
        raise FileNotFoundError(f"readonly tools missing required file(s): {missing}")

    for path in dest.rglob("*"):
        if path.is_dir() and path.name in EXCLUDED_NAMES:
            raise RuntimeError(f"excluded directory leaked into readonly tools: {path}")

    for path in dest.rglob("*.sh"):
        path.chmod(path.stat().st_mode | 0o111)
    for path in dest.rglob("*.py"):
        if path.read_text(encoding="utf-8", errors="ignore").startswith("#!"):
            path.chmod(path.stat().st_mode | 0o111)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)

    publish_readonly_tools(args.source, args.dest)
    print(f"readonly_tools={args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
