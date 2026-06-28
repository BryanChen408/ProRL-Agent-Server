#!/usr/bin/env python3
"""Refresh generated operator task prompts in an existing jsonl file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GEN_OP_ASSETS = ROOT / "gen_op_assets.py"


def _load_gen_module() -> Any:
    spec = importlib.util.spec_from_file_location("polar_gen_op_assets", GEN_OP_ASSETS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEN_OP_ASSETS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_prompts(path: Path, *, backup: bool = True) -> dict[str, Any]:
    module = _load_gen_module()
    old_hash = _sha256(path)
    backup_path = None
    if backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_before_prompt_refresh_{stamp}_{old_hash[:12]}")
        shutil.copy2(path, backup_path)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    rows = 0
    changed = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, path.open("r", encoding="utf-8") as inp:
            for lineno, line in enumerate(inp, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                op = meta.get("op_name") or rec.get("label")
                if not isinstance(op, str) or not op:
                    raise RuntimeError(f"line {lineno}: missing metadata.op_name/label")
                prompt = rec.get("prompt")
                if not isinstance(prompt, list) or not prompt:
                    prompt = [{"role": "user"}]
                    rec["prompt"] = prompt
                if not isinstance(prompt[0], dict):
                    prompt[0] = {"role": "user"}
                prompt[0]["role"] = prompt[0].get("role") or "user"
                before = prompt[0].get("content")
                after = module._instruction(op)
                if before != after:
                    changed += 1
                prompt[0]["content"] = after
                out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "path": str(path),
        "backup": str(backup_path) if backup_path else None,
        "rows": rows,
        "changed": changed,
        "old_sha256": old_hash,
        "new_sha256": _sha256(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Existing operator_tasks.jsonl to rewrite in place.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    result = refresh_prompts(args.jsonl, backup=not args.no_backup)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
