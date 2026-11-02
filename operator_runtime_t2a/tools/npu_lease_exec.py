#!/usr/bin/env python3
"""Run one command under a host-level Ascend NPU flock lease."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
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
        # A surviving child shares this flock; only close our reference.
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
        quarantine = lock_dir / f"npu{device_id}.quarantine.json"
        if quarantine.exists():
            continue
        lock_path = lock_dir / f"npu{device_id}.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if quarantine.exists():
                os.close(fd)
                continue
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


_NPU_ERROR = re.compile(rb"(?:ErrCode\s*=|error\s+code\s*(?:is\s*)?[:=]?)\s*(507014|507015|507034|507035)\b", re.I)
_CANN_ERROR = re.compile(
    rb"^\[ERROR\]\s+(?:RUNTIME|DRV|ASCENDCL)\((\d+),[^)]+\):"
    rb"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\.(\d{3})\.(\d{3})\s"
)
_DEVICE = re.compile(rb"\b(?:dev_id|device_id|devId|devid)\s*=\s*(\d+)\b", re.I)
_CLEANUP_ERROR = re.compile(
    rb"(?:halEschedSubmitEventSync[^\n]*[Ff]ailed[^\n]*ret=16\b"
    rb"|FreeDcacheAddr:Free dcache stack phy failed[^\n]*drvRet=0x10\b)", re.I
)


def _task_identity(pid, group):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if int(fields[2]) != group or fields[0] in {"Z", "X", "x"}:
            return None
        ticks = int(fields[19])
        born = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME) + ticks / os.sysconf("SC_CLK_TCK")
        return ticks, born
    except (OSError, ValueError, IndexError):
        return None


class _NpuFailure:
    def __init__(self, group, device, started):
        self.group, self.device, self.started = group, str(device).encode(), started
        self.processes = {}

    def feed(self, line):
        header = _CANN_ERROR.match(line)
        if not header:
            return None
        pid = int(header[1])
        identity = _task_identity(pid, self.group)
        if identity is None:
            return None
        try:
            stamp = datetime.strptime(
                (header[2] + b"." + header[3] + header[4]).decode(), "%Y-%m-%d-%H:%M:%S.%f"
            ).timestamp()
        except ValueError:
            return None
        if stamp < max(self.started, identity[1]) or stamp > time.time() + 1:
            return None
        devices = _DEVICE.findall(line)
        if any(device != self.device for device in devices):
            return None
        state = self.processes.setdefault((pid, identity[0]), {"error": None, "responses": {}, "cleanup": None})
        error = _NPU_ERROR.search(line)
        if error and state["error"] is None:
            state["error"] = (error[1].decode(), stamp)
        if state["error"] is None or stamp < state["error"][1]:
            return None
        if (devices and b"Message send finish, no resp" in line
                and re.search(rb'msg_type="msg_client_(?:tsdrv|devmm)"', line)
                and b"ret=-38" in line):
            state["responses"][stamp] = line.decode(errors="replace")[:4096]
            # Two distinct records suffice; repeated copies of one record do not.
            if len(state["responses"]) > 2:
                del state["responses"][min(state["responses"])]
        if _CLEANUP_ERROR.search(line):
            state["cleanup"] = line.decode(errors="replace")[:4096]
        if len(state["responses"]) < 2 or state["cleanup"] is None:
            return None
        current = _task_identity(pid, self.group)
        if current is None or current[0] != identity[0]:
            return None
        return {"error_code": state["error"][0], "fault_pid": pid,
                "reason": "device_unresponsive_and_cleanup_failed",
                "evidence": [*state["responses"].values(), state["cleanup"]]}


def _run_guarded(command, env, lease, failure_path):
    # Read live CANN diagnostics, including children whose ordinary output is captured by msprof.
    env = {**env, "ASCEND_SLOG_PRINT_TO_STDOUT": "1", "ASCEND_GLOBAL_LOG_LEVEL": "3"}
    started = time.time()
    proc = subprocess.Popen(command, env=env, pass_fds=(lease.fd,), start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    detector = _NpuFailure(proc.pid, lease.device_id, started)
    pending = b""
    failure = None
    try:
        while data := os.read(proc.stdout.fileno(), 65536):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            lines = (pending + data).split(b"\n")
            pending = lines.pop()[-65536:]
            for line in lines:
                failure = detector.feed(line)
                if failure:
                    break
            if failure:
                break
        if failure is None:
            return proc.wait()
        # Publish before releasing the lease, so other sessions cannot reuse a
        # device whose driver stopped responding. Recovery requires explicit clearance.
        _write_status(str(lease.lock_path.with_name(f"npu{lease.device_id}.quarantine.json")), {
            "schema_version": 1, "error_type": "npu_execution_failed", **failure,
            "device_id": lease.device_id, "source": "npu_lease_exec",
            "session_id": env.get("SESSION_ID"), "task_id": env.get("TASK_ID"),
        })
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return proc.wait()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass  # A driver-blocked child still owns the inherited flock.
        _write_status(str(failure_path), {
            "error_type": "npu_execution_failed", **failure,
            "device_id": lease.device_id, "child_pid": proc.pid,
            "child_reaped": proc.poll() is not None,
        })
        return 125
    finally:
        proc.stdout.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="Comma/range NPU pool, e.g. 8,9 or 8-11.")
    parser.add_argument("--lock-dir", required=True, help="Directory containing npu{card}.lock files.")
    parser.add_argument("--status-file", help="Optional JSON status file for observer/debug.")
    parser.add_argument("--poll-interval", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--stop-on-npu-error", action="store_true")
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
    failure_path = Path(os.environ.get("ARTIFACTS_DIR") or
                        (Path(args.status_file).parent if args.status_file else lock_dir)) / "npu_failure.json"
    if args.stop_on_npu_error and failure_path.exists():
        print("npu_lease_exec: this session already has an NPU failure", file=sys.stderr)
        return 125
    requested_at = time.time()
    status_base: dict[str, Any] = {
        "schema_version": 2,
        "pool": pool,
        "lock_dir": str(lock_dir),
        "pid": os.getpid(),
        "command": command,
        # requested_at → lease_wait(4 卡验证池饱和)+ exec_seconds(上卡时长)
        "requested_at_unix": requested_at,
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
            "wait_seconds": round(started_at - requested_at, 4),
        },
    )

    child_env = os.environ.copy()
    child_env["ASCEND_RT_VISIBLE_DEVICES"] = lease.device_id
    return_code = 127
    error: str | None = None
    try:
        if args.stop_on_npu_error:
            return_code = _run_guarded(command, child_env, lease, failure_path)
        else:
            result = subprocess.run(command, env=child_env, pass_fds=(lease.fd,))
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
                "wait_seconds": round(started_at - requested_at, 4),
                "exec_seconds": round(ended_at - started_at, 4),
                "return_code": return_code,
                "error": error,
            },
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
