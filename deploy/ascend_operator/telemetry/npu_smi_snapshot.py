#!/usr/bin/env python3
"""One host collector publishes plain 'npu-smi info' for agent/judge readers."""

import argparse
import fcntl
import os
import subprocess
import tempfile
import time
from pathlib import Path


def collect(real: str, directory: Path, lock_fd: int, timeout: float) -> bool:
    try:
        # A child outliving its collector keeps the singleton lock, preventing overlap.
        proc = subprocess.Popen([real, "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                pass_fds=(lock_fd,))
    except OSError as exc:
        print(f"npu-smi collector: {exc}", flush=True)
        return False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("npu-smi collector: probe timed out; waiting for exit before any new probe",
              flush=True)
        # A driver-blocked child stops collection; snapshot readers remain nonblocking.
        proc.communicate()
        return False
    if proc.returncode != 0 or not stdout.strip():
        print(f"npu-smi collector: probe failed rc={proc.returncode}: "
              f"{stderr.decode(errors='replace')[:500]}", flush=True)
        return False
    fd, name = tempfile.mkstemp(dir=directory, prefix=".npu-smi-info-")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(stdout)
            os.fchmod(output.fileno(), 0o644)
        os.replace(name, directory / "npu-smi-info.out")
    finally:
        Path(name).unlink(missing_ok=True)
    print("npu-smi collector: snapshot updated", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/dev/shm/npu-locks/npu-smi-snapshot"))
    parser.add_argument("--real", default="/usr/local/bin/npu-smi")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--failure-interval", type=float, default=300)
    args = parser.parse_args()
    if min(args.interval, args.timeout, args.failure_interval) <= 0:
        parser.error("intervals and timeout must be positive")
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "collector.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("npu-smi collector: already running (or prior probe still alive)", flush=True)
            return 0
        while True:
            ok = collect(args.real, args.directory, lock.fileno(), args.timeout)
            time.sleep(args.interval if ok else args.failure_interval)


if __name__ == "__main__":
    raise SystemExit(main())
