#!/usr/bin/env python3
"""Run one command under a host-level Ascend NPU flock lease."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_pool(spec: str) -> list[str]:
    value = str(spec or "").strip()
    if not value:
        return []
    if "-" in value and "," not in value:
        lo, hi = value.split("-", 1)
        return [str(i) for i in range(int(lo), int(hi) + 1)]
    return [part.strip() for part in value.split(",") if part.strip()]


class Lease:
    def __init__(self, fd: int, device_id: str, lock_path: Path) -> None:
        self.fd = fd
        self.device_id = device_id
        self.lock_path = lock_path

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass


def _write_status(status_file: str | None, payload: dict[str, Any]) -> None:
    if not status_file:
        return
    path = Path(status_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at_unix": time.time()}
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _try_acquire(pool: list[str], lock_dir: Path) -> Lease | None:
    for device_id in pool:
        lock_path = lock_dir / f"npu{device_id}.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return Lease(fd, device_id, lock_path)
        except (BlockingIOError, OSError):
            os.close(fd)
    return None


def acquire_loop(
    pool: list[str],
    lock_dir: Path,
    *,
    poll_interval: float,
) -> Lease:
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        lease = _try_acquire(pool, lock_dir)
        if lease is not None:
            return lease
        time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="Comma/range NPU pool, e.g. 8,9 or 8-11.")
    parser.add_argument("--lock-dir", required=True, help="Directory containing npu{card}.lock files.")
    parser.add_argument("--status-file", help="Optional JSON status file for observer/debug.")
    parser.add_argument("--poll-interval", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command required after --")

    pool = parse_pool(args.pool)
    if not pool:
        parser.error("--pool must contain at least one NPU id")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")

    lock_dir = Path(args.lock_dir)
    status_base: dict[str, Any] = {
        "schema_version": 1,
        "pool": pool,
        "lock_dir": str(lock_dir),
        "pid": os.getpid(),
        "command": command,
    }
    _write_status(args.status_file, {**status_base, "state": "waiting"})

    lease = acquire_loop(pool, lock_dir, poll_interval=args.poll_interval)
    started_at = time.time()
    _write_status(
        args.status_file,
        {
            **status_base,
            "state": "running",
            "device_id": lease.device_id,
            "lock_path": str(lease.lock_path),
            "started_at_unix": started_at,
        },
    )

    child_env = os.environ.copy()
    child_env["ASCEND_RT_VISIBLE_DEVICES"] = lease.device_id
    return_code = 127
    error: str | None = None
    try:
        result = subprocess.run(command, env=child_env)
        return_code = int(result.returncode)
    except FileNotFoundError as exc:
        error = str(exc)
        print(f"npu_lease_exec: command not found: {command[0]}", file=sys.stderr)
    finally:
        ended_at = time.time()
        lease.release()
        _write_status(
            args.status_file,
            {
                **status_base,
                "state": "completed",
                "device_id": lease.device_id,
                "lock_path": str(lease.lock_path),
                "started_at_unix": started_at,
                "ended_at_unix": ended_at,
                "return_code": return_code,
                "error": error,
            },
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
