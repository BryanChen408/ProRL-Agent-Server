#!/usr/bin/env python3
"""Share short-lived `npu-smi info` output across Ascend rollout containers."""

import fcntl
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REAL = os.environ.get("POLAR_NPU_SMI_REAL", "/usr/local/bin/npu-smi.real")
CACHE_DIR = Path(os.environ.get("POLAR_NPU_SMI_CACHE_DIR", os.environ.get("POLAR_NPU_LOCK_DIR", "/dev/shm/npu-locks")))
TTL = 30
TIMEOUT = 5


def _cached(path: Path, max_age: int) -> bytes | None:
    try:
        if time.time() - path.stat().st_mtime <= max_age:
            return path.read_bytes()
    except OSError:
        pass
    return None


def main() -> int:
    args = sys.argv[1:]
    if args != ["info"]:
        return subprocess.call([REAL, *args])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "npu-smi-info.out"
    failed = CACHE_DIR / "npu-smi-info.failed"
    with (CACHE_DIR / "npu-smi-info.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _cached(cache, TTL)
        if data is not None:
            sys.stdout.buffer.write(data)
            return 0
        if _cached(failed, TTL) is not None:
            data = _cached(cache, 300)
            if data is not None:
                print("npu-smi: serving stale cached info after probe failure", file=sys.stderr)
                sys.stdout.buffer.write(data)
                return 0
            return 124

        try:
            proc = subprocess.Popen([REAL, "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            failed.touch()
            print(f"npu-smi: {exc}", file=sys.stderr)
            return 127
        try:
            stdout, stderr = proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            # A driver-blocked process may not reap promptly; let the caller continue.
            failed.touch()
            data = _cached(cache, 300)
            if data is not None:
                print("npu-smi: serving stale cached info after probe timeout", file=sys.stderr)
                sys.stdout.buffer.write(data)
                return 0
            print("npu-smi: info timed out after 5s", file=sys.stderr)
            return 124
        if proc.returncode == 0:
            fd, name = tempfile.mkstemp(dir=CACHE_DIR, prefix=".npu-smi-info-")
            with os.fdopen(fd, "wb") as output:
                output.write(stdout)
            os.replace(name, cache)
            sys.stdout.buffer.write(stdout)
            return 0
        failed.touch()
        data = _cached(cache, 300)
        if data is not None:
            print("npu-smi: serving stale cached info after probe failure", file=sys.stderr)
            sys.stdout.buffer.write(data)
            return 0
        sys.stderr.buffer.write(stderr)
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
