#!/usr/bin/env python3
"""Publish Polar operator-runtime assets into a deployable directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "operator_runtime"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "ascend_operator" / "operator_runtime"

PUBLISHED_ENTRIES = ("CLAUDE.md", ".agents", "skills", "tools", "runtime")
IGNORE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}

REQUIRED_PATHS = (
    "CLAUDE.md",
    "skills/triton-op-designer/SKILL.md",
    "skills/triton-op-coding/SKILL.md",
    "skills/npu-arch/references/npu-arch-guide-triton.md",
    "skills/npu-arch/references/npu-hardware-params.md",
    "tools/triton_eval_pipeline.sh",
    "runtime/prepare_operator_workdir.py",
    ".agents/skills/triton-op-verifier/scripts/validate_triton_impl.py",
    ".agents/skills/triton-op-verifier/scripts/verify.py",
    ".agents/skills/triton-op-verifier/scripts/benchmark.py",
    ".agents/skills/triton-op-verifier/scripts/_common_utils.py",
    ".agents/skills/triton-op-verifier/scripts/_log_utils.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _ignore_names(_dir: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORE_NAMES or any(name.endswith(suffix) for suffix in IGNORE_SUFFIXES)
    }


def _copy_entry(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True, ignore=_ignore_names)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)


def _iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _manifest(source: Path, output: Path) -> dict[str, Any]:
    files = []
    for path in _iter_files(output):
        rel = path.relative_to(output).as_posix()
        if rel == "MANIFEST.json":
            continue
        stat = path.stat()
        files.append(
            {
                "path": rel,
                "sha256": _sha256(path),
                "size": stat.st_size,
                "mode": oct(stat.st_mode & 0o777),
            }
        )
    return {
        "schema_version": 1,
        "published_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": str(source),
        "source_git_commit": _git_commit(source),
        "output": str(output),
        "file_count": len(files),
        "files": files,
    }


def publish(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory missing: {source}")

    missing = [rel for rel in REQUIRED_PATHS if not (source / rel).is_file()]
    if missing:
        raise FileNotFoundError("missing required operator-runtime asset(s): " + ", ".join(missing))

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for rel in PUBLISHED_ENTRIES:
        _copy_entry(source / rel, output / rel)

    manifest = _manifest(source, output)
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o644)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest = publish(args.source, args.output)
    print(
        "published operator runtime: "
        f"source={manifest['source']} output={manifest['output']} files={manifest['file_count']}"
    )
    print(Path(manifest["output"]) / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
