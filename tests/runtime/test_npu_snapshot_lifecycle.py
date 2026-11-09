"""Exercise collector shutdown with disposable processes, never NPU queries."""

import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


COLLECTOR = Path(__file__).resolve().parents[2] / "deploy/ascend_operator/telemetry/npu_smi_snapshot.py"


def test_stop_only_after_last_service_and_only_matching_directory(tmp_path, monkeypatch):
    stop = runpy.run_path(str(COLLECTOR))["stop_if_unused"]
    directory = tmp_path / "cache"
    other_directory = tmp_path / "other"
    processes = [subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(60)", str(COLLECTOR),
        "--directory", str(d),
    ]) for d in (directory, other_directory)]
    # Simulate two main services without importing or starting the RL framework.
    services = []
    for i, role in enumerate(("serve_gateway", "serve_rollout")):
        proc = tmp_path / str(9100001 + i)
        proc.mkdir()
        (proc / "cmdline").write_bytes(
            b"python\0-c\0from polar.cli import main; main()\0" + role.encode() + b"\0"
        )
        services.append(proc)
    visible = [Path(f"/proc/{p.pid}") for p in processes] + services
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter(visible) if p == Path("/proc") else original(p))
    try:
        assert stop(directory) == 0
        assert processes[0].poll() is None
        visible.remove(services[0])
        assert stop(directory) == 0
        assert processes[0].poll() is None  # Another main service still needs queries.
        visible.remove(services[1])
        os.kill(processes[0].pid, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while Path(f"/proc/{processes[0].pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "T":
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert stop(directory) == 1  # Do not disturb maintenance-paused collectors.
        os.kill(processes[0].pid, signal.SIGCONT)
        assert stop(directory) == 0
        assert processes[0].wait(timeout=3) == -signal.SIGTERM
        assert processes[1].poll() is None  # Another cache directory is untouched.
        assert stop(directory) == 0  # Idempotent after exit.
    finally:
        for process in processes:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
                process.terminate()
            process.wait(timeout=3)
