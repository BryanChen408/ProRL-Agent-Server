from __future__ import annotations

import re
from typing import Any


LEGACY_RUN_ID = "legacy"


def safe_run_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return text or LEGACY_RUN_ID


def run_id_from_task(task_id: str | None) -> str:
    text = str(task_id or "").strip()
    if not text:
        return LEGACY_RUN_ID
    if text.startswith("polar-op-") or text.startswith("polar-slime-"):
        return LEGACY_RUN_ID
    for marker in ("-polar-op-", "-polar-slime-"):
        if marker in text:
            return safe_run_id(text.split(marker, 1)[0])
    return LEGACY_RUN_ID


def run_id_from_metadata(task_id: str | None, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ("run_id", "polar_run_id", "slime_run_id"):
        if metadata.get(key):
            return safe_run_id(metadata[key])
    return run_id_from_task(task_id)


def run_dir_name(run_id: str | None) -> str | None:
    safe = safe_run_id(run_id)
    if safe == LEGACY_RUN_ID:
        return None
    return f"run_{safe}"


__all__ = ["LEGACY_RUN_ID", "run_dir_name", "run_id_from_metadata", "run_id_from_task", "safe_run_id"]
