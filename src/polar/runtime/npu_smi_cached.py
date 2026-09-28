#!/usr/bin/env python3
"""Route read-only npu-smi commands to the host; never open NPU devices here."""

import json
import os
import socket
import sys
import time
from pathlib import Path

CACHE_DIR = Path(os.environ.get(
    "POLAR_NPU_SMI_CACHE_DIR",
    str(Path(os.environ.get("POLAR_NPU_LOCK_DIR", "/dev/shm/npu-locks")) / "npu-smi-snapshot"),
))
MAX_MESSAGE = 4 * 1024 * 1024


def validate_args(args):
    """The info namespace is read-only; never expose management verbs or a shell."""
    if (not isinstance(args, list) or len(args) > 64
            or any(not isinstance(a, str) or not a or len(a) > 256
                   or any(ord(c) < 32 for c in a) for a in args)):
        raise ValueError("invalid query arguments")
    if args and args[0] != "info" and args not in [["-h"], ["--help"], ["-v"], ["--version"]]:
        raise ValueError("only npu-smi info/help/version queries are allowed")
    return args


def query(args, directory=None, timeout=15):
    """Shared client for containers and host telemetry. Fail closed, never probe locally."""
    validate_args(args)
    directory = Path(directory) if directory is not None else CACHE_DIR
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(directory / "query.sock"))
        client.sendall(json.dumps(args).encode() + b"\n")
        with client.makefile("rb") as reply:
            data = reply.readline(MAX_MESSAGE + 1)
        if len(data) > MAX_MESSAGE or not data.endswith(b"\n"):
            raise ValueError("invalid host query response")
        result = json.loads(data)
    if (not isinstance(result, dict) or not isinstance(result.get("returncode"), int)
            or not isinstance(result.get("stdout"), str) or not isinstance(result.get("stderr"), str)):
        raise ValueError("invalid host query response")
    return result


def main() -> int:
    args = sys.argv[1:]
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"npu-smi: {exc}", file=sys.stderr)
        return 64
    if args == ["info"]:
        try:
            with (CACHE_DIR / "npu-smi-info.out").open("rb") as snapshot:
                age = max(0, time.time() - os.fstat(snapshot.fileno()).st_mtime)
                data = snapshot.read()
            if data:
                print(f"npu-smi: host snapshot, age={age:.0f}s" + (" (stale)" if age > 60 else ""),
                      file=sys.stderr)
                sys.stdout.buffer.write(data)
                return 0
        except OSError:
            pass
    try:
        result = query(args)
    except (OSError, ValueError) as exc:
        print(f"npu-smi: host query unavailable ({exc}); local probes are disabled", file=sys.stderr)
        return 124
    sys.stdout.write(result["stdout"])
    sys.stderr.write(result["stderr"])
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
