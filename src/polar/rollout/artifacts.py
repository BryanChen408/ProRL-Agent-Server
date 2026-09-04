"""Persist a bounded manifest of session profiling artifacts."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

_PROFILE_SUFFIXES = {".json", ".jsonl", ".csv", ".db", ".sqlite", ".log", ".txt", ".gz"}


def persist_profiling_artifacts(
    source_dir: Path,
    destination_root: Path,
    *,
    session_id: str,
    max_total_bytes: int,
    max_files: int = 1000,
) -> dict[str, Any]:
    """Copy profiling outputs and write a content-addressed manifest.

    Symlinks, source code, binaries, and files outside the byte budget are not
    persisted. This keeps the artifact surface useful for RL-Insight offline
    analysis without turning it into an unrestricted session-directory archive.
    """
    destination_dir = destination_root / f"{session_id}.artifacts"
    destination_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    skipped_bytes = 0
    skipped_files = 0
    if source_dir.is_dir():
        for source in sorted(source_dir.rglob("*")):
            if source.is_symlink() or not source.is_file() or source.suffix.lower() not in _PROFILE_SUFFIXES:
                continue
            size = source.stat().st_size
            if len(entries) >= max_files or size < 0 or total_bytes + size > max_total_bytes:
                skipped_bytes += max(0, size)
                skipped_files += 1
                continue
            relative = source.relative_to(source_dir)
            target = destination_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = _sha256(target)
            relative_text = relative.as_posix()
            entries.append({
                "id": hashlib.sha256(relative_text.encode()).hexdigest()[:16],
                "name": source.name,
                "relative_path": relative_text,
                "kind": _artifact_kind(relative_text),
                "size_bytes": size,
                "sha256": digest,
                "media_type": mimetypes.guess_type(source.name)[0] or "application/octet-stream",
            })
            total_bytes += size
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "artifact_count": len(entries),
        "total_bytes": total_bytes,
        "skipped_bytes": skipped_bytes,
        "skipped_files": skipped_files,
        "artifacts": entries,
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / f"{session_id}.artifacts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _artifact_kind(relative_path: str) -> str:
    value = relative_path.lower()
    if "msprof" in value or "profiler" in value or value.endswith((".db", ".sqlite")):
        return "profiler"
    if "verify" in value:
        return "verification"
    if "perf" in value or "benchmark" in value:
        return "benchmark"
    if "metric" in value:
        return "metrics"
    if value.endswith((".log", ".txt")):
        return "log"
    return "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["persist_profiling_artifacts"]
