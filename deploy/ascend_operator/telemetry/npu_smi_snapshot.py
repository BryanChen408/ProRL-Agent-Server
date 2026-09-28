#!/usr/bin/env python3
"""One host owner serializes/caches all read-only npu-smi queries and publishes info."""

import argparse
import fcntl
import json
import os
import runpy
import socket
import socketserver
import subprocess
import tempfile
import time
from pathlib import Path

_CLIENT = runpy.run_path(str(Path(__file__).resolve().parents[3] / "src/polar/runtime/npu_smi_cached.py"))
validate_args = _CLIENT["validate_args"]
MAX_MESSAGE = _CLIENT["MAX_MESSAGE"]


def socket_ready(directory):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(str(directory / "query.sock"))
        return True
    except OSError:
        return False


def probe(real, args, lock_fd, timeout):
    validate_args(args)
    try:
        # A driver-blocked child retains the singleton lock even if the owner exits.
        proc = subprocess.Popen([real, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                pass_fds=(lock_fd,))
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc) + "\n"}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        print("npu-smi collector: probe timed out; waiting for exit before any new probe", flush=True)
        # Includes continuous info watch: preserve partial output, bound driver occupation.
        stdout, stderr = proc.communicate()
        stderr += b"npu-smi: host query timed out; partial output follows if available\n"
        rc = 124
    return {"returncode": rc, "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace")}


def publish(directory, stdout):
    fd, name = tempfile.mkstemp(dir=directory, prefix=".npu-smi-info-")
    try:
        with os.fdopen(fd, "w") as output:
            output.write(stdout)
            os.fchmod(output.fileno(), 0o644)
        os.replace(name, directory / "npu-smi-info.out")
    finally:
        Path(name).unlink(missing_ok=True)


def collect(real: str, directory: Path, lock_fd: int, timeout: float) -> bool:
    result = probe(real, ["info"], lock_fd, timeout)
    if result["returncode"] or not result["stdout"].strip():
        print(f"npu-smi collector: probe failed rc={result['returncode']}: {result['stderr'][:500]}", flush=True)
        return False
    publish(directory, result["stdout"])
    print("npu-smi collector: snapshot updated", flush=True)
    return True


class QueryServer(socketserver.UnixStreamServer):
    """Single-threaded: periodic collection and client probes cannot overlap."""
    request_queue_size = 128

    def __init__(self, directory, real, lock_fd, timeout=5, ttl=30, failure_ttl=300):
        self.directory, self.real, self.lock_fd = directory, real, lock_fd
        self.probe_timeout, self.ttl, self.failure_ttl = timeout, ttl, failure_ttl
        self.cache = {}
        super().__init__(str(directory / "query.sock"), QueryHandler)
        os.chmod(self.server_address, 0o666)  # Readers also run as non-root in containers.
        self.timeout = 0.2

    def query(self, args):
        validate_args(args)
        key = tuple(args)
        cached = self.cache.get(key)
        if cached and time.monotonic() < cached[0]:
            return cached[1]
        result = probe(self.real, args, self.lock_fd, self.probe_timeout)
        # ponytail: bounded FIFO cache; use LRU if >256 distinct queries recur per TTL.
        if len(self.cache) >= 256:
            self.cache.pop(next(iter(self.cache)))
        ttl = self.failure_ttl if result["returncode"] else self.ttl
        self.cache[key] = (time.monotonic() + ttl, result)
        return result


class QueryHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(1)
        try:
            data = self.rfile.readline(65537)
            if len(data) > 65536 or not data.endswith(b"\n"):
                raise ValueError("invalid query request")
            result = self.server.query(json.loads(data))
        except (ValueError, OSError) as exc:
            result = {"returncode": 64, "stdout": "", "stderr": f"npu-smi: {exc}\n"}
        encoded = json.dumps(result).encode() + b"\n"
        if len(encoded) > MAX_MESSAGE:
            encoded = json.dumps({"returncode": 124, "stdout": "", "stderr": "npu-smi: response too large\n"}).encode() + b"\n"
        try:
            self.wfile.write(encoded)
        except OSError:
            pass  # A timed-out reader must not kill the only driver-query owner.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(os.environ.get("POLAR_NPU_SMI_CACHE_DIR", "/dev/shm/npu-locks/npu-smi-snapshot")))
    parser.add_argument("--real", default="/usr/local/bin/npu-smi")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--failure-interval", type=float, default=300)
    parser.add_argument("--check", action="store_true", help="Check the reader socket without probing NPU devices.")
    args = parser.parse_args()
    if min(args.interval, args.timeout, args.failure_interval) <= 0:
        parser.error("intervals and timeout must be positive")
    if args.check:
        return 0 if socket_ready(args.directory) else 1
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "collector.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ready = socket_ready(args.directory)
            print("npu-smi collector: already running" if ready else
                  "npu-smi collector: prior owner has no query socket; replace the old collector first", flush=True)
            return 0 if ready else 1
        socket_path = args.directory / "query.sock"
        socket_path.unlink(missing_ok=True)
        try:
            with QueryServer(args.directory, args.real, lock.fileno(), args.timeout,
                             args.interval, args.failure_interval) as server:
                next_collect = 0
                while True:
                    if time.monotonic() >= next_collect:
                        ok = collect(args.real, args.directory, lock.fileno(), args.timeout)
                        next_collect = time.monotonic() + (args.interval if ok else args.failure_interval)
                    server.handle_request()
        finally:
            socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
