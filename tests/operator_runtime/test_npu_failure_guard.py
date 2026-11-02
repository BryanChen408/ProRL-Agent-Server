"""CPU-only subprocess tests for fatal NPU errors; no Ascend imports or probes."""
import fcntl
import json
import os
import runpy
import shlex
from pathlib import Path
from datetime import datetime
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from polar.runtime.ascend import acquire_card


GUARD = Path(__file__).resolve().parents[2] / "operator_runtime_t2a/tools/npu_lease_exec.py"


def run_guard(tmp_path, code, *, artifacts="artifacts"):
    return subprocess.run(
        [sys.executable, str(GUARD), "--pool", "0,1", "--lock-dir", str(tmp_path / "locks"),
         "--status-file", str(tmp_path / artifacts / "lease.json"), "--stop-on-npu-error",
         "--", sys.executable, "-c", code],
        env={**os.environ, "ARTIFACTS_DIR": str(tmp_path / artifacts)},
        capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("exit_code", [0, 1])
def test_ordinary_outcomes_do_not_stop_session(tmp_path, exit_code):
    result = run_guard(tmp_path, f"import sys; print('accuracy mismatch'); sys.exit({exit_code})")
    assert result.returncode == exit_code
    assert result.stdout == "accuracy mismatch\n"
    assert not (tmp_path / "artifacts/npu_failure.json").exists()
    assert not list((tmp_path / "locks").glob("*.quarantine.json"))
    lease = acquire_card(["0"], str(tmp_path / "locks"))
    lease.release()


DRIVER_SCRIPT = r"""
import os,sys,time
from datetime import datetime

def emit(body, module="DRV", pid=None):
    stamp=datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    stamp=stamp[:-3]+"."+stamp[-3:]
    line=f"[ERROR] {module}({pid or os.getpid()},python):{stamp} {body}\n"
    sys.stderr.write(line[:12]);sys.stderr.flush();time.sleep(.01)
    sys.stderr.write(line[12:]);sys.stderr.flush()

def driver_failure():
    emit("rtStreamSynchronize:ErrCode=507035", "RUNTIME")
    for _ in range(2):
        emit('Message send finish, no resp. (dev_id='+os.environ['ASCEND_RT_VISIBLE_DEVICES']+'; msg_type="msg_client_tsdrv"; status=1; ret=-38)')
    emit("halEschedSubmitEventSync]Failed. (ret=16; timeout=30000ms)")
"""


@pytest.mark.parametrize("diagnostic", ["error code:507034", "ErrCode=507035", "error code is 507015"])
def test_error_code_alone_allows_failed_attempt_to_exit_and_retry(tmp_path, diagnostic):
    code = f"import sys,time; print({diagnostic!r}, flush=True); time.sleep(.3); sys.exit(1)"
    result = run_guard(tmp_path, code)
    assert result.returncode == 1
    assert not (tmp_path / "artifacts/npu_failure.json").exists()
    assert not list((tmp_path / "locks").glob("*.quarantine.json"))
    assert run_guard(tmp_path, "print('next attempt succeeds')").returncode == 0


def test_driver_failure_stops_only_its_session(tmp_path):
    result = run_guard(tmp_path, DRIVER_SCRIPT + "\ndriver_failure();time.sleep(60)")
    assert result.returncode == 125, result.stderr
    failure = json.loads((tmp_path / "artifacts/npu_failure.json").read_text())
    assert failure["reason"] == "device_unresponsive_and_cleanup_failed"
    assert failure["error_code"] == "507035" and failure["device_id"] == "0"
    assert failure["fault_pid"] == failure["child_pid"] and failure["child_reaped"]
    assert len(failure["evidence"]) == 3
    quarantine = tmp_path / "locks/npu0.quarantine.json"
    record = json.loads(quarantine.read_text())
    assert record["device_id"] == "0" and record["evidence"] == failure["evidence"]
    assert run_guard(tmp_path, "raise AssertionError('must not run')").returncode == 125
    other = run_guard(tmp_path, "import os; print(os.environ['ASCEND_RT_VISIBLE_DEVICES'])", artifacts="other")
    assert other.returncode == 0 and other.stdout == "1\n"
    lease = acquire_card(["0", "1"], str(tmp_path / "locks"))
    assert lease.device_id == "1"
    lease.release()
    quarantine.unlink()  # Explicit operator clearance restores eligibility.
    lease = acquire_card(["0"], str(tmp_path / "locks"))
    lease.release()


def test_live_foreign_process_diagnostics_do_not_stop_this_session(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    try:
        code = DRIVER_SCRIPT + f'''
emit("rtStreamSynchronize:ErrCode=507035", "RUNTIME", {other.pid})
for _ in range(2):
    emit('Message send finish, no resp. (dev_id=0; msg_type="msg_client_tsdrv"; ret=-38)', pid={other.pid})
emit("halEschedSubmitEventSync]Failed. (ret=16)", pid={other.pid})
time.sleep(.2)
sys.exit(1)
'''
        assert run_guard(tmp_path, code).returncode == 1
        assert not (tmp_path / "artifacts/npu_failure.json").exists()
        assert other.poll() is None
    finally:
        other.kill()
        other.wait(timeout=3)


def test_quarantine_does_not_revoke_existing_lease(tmp_path):
    module = runpy.run_path(str(GUARD))
    held = acquire_card(["0"], str(tmp_path))
    marker = tmp_path / "npu0.quarantine.json"
    marker.write_text("{}")
    try:
        # Even a malformed marker is fail-closed; the live owner keeps its lock.
        marker.write_text("not json")
        with (tmp_path / "npu0.lock").open("a+") as f:
            with pytest.raises(BlockingIOError):
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module["_try_acquire"](["0"], tmp_path) is None
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire_card(["0"], str(tmp_path))
        for acquire in [module["_try_acquire"], acquire_card]:
            lease = acquire(["0", "1"], tmp_path)
            assert lease.device_id == "1"
            lease.release()
    finally:
        held.release()
    assert module["_try_acquire"](["0"], tmp_path) is None
    marker.unlink()
    lease = module["_try_acquire"](["0"], tmp_path)
    assert lease.device_id == "0"
    lease.release()


@pytest.mark.parametrize("runtime_allocator", [False, True])
def test_quarantine_published_during_acquisition_is_rechecked(tmp_path, monkeypatch, runtime_allocator):
    module = runpy.run_path(str(GUARD))
    original = fcntl.flock
    calls = 0

    def publish_after_lock(fd, operation):
        nonlocal calls
        result = original(fd, operation)
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            calls += 1
            if calls == 1:
                (tmp_path / "npu0.quarantine.json").write_text("{}")
        return result

    monkeypatch.setattr(fcntl, "flock", publish_after_lock)
    acquire = acquire_card if runtime_allocator else module["_try_acquire"]
    lease = acquire(["0", "1"], tmp_path)
    assert lease.device_id == "1"
    lease.release()
    # Skipping a quarantined card must not leak the just-acquired lock.
    with (tmp_path / "npu0.lock").open("a+") as f:
        original(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("mode,stage", [
    ("quick", "profile"), ("standard", "profile"), ("standard", "sample"), ("standard", "warmup"),
])
def test_nested_profiler_driver_failure_exits_without_killing_neighbor(tmp_path, mode, stage):
    profiler = GUARD.parents[1] / "skills/ops-profiling/scripts/msprof_perf_summary.py"
    phase_pid = tmp_path / "phase.pid"
    inner_pid = tmp_path / "inner.pid"
    inner = DRIVER_SCRIPT + f"\nfrom pathlib import Path\nPath({str(inner_pid)!r}).write_text(str(os.getpid()))\ndriver_failure()\ntime.sleep(60)\n"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_msprof = bin_dir / "msprof"
    prelude = "import sys\n"
    if stage == "sample":
        prelude += "if '--aic-mode=sample-based' not in sys.argv: sys.exit(0)\n"
    fake_msprof.write_text(f"#!{sys.executable}\n{prelude}{inner}\n")
    fake_msprof.chmod(0o755)
    extra = f", warmup={int(stage == 'warmup')}" if mode == "standard" else ""
    code = f"""
import os, pathlib, runpy
pathlib.Path({str(phase_pid)!r}).write_text(str(os.getpid()))
os.environ['PATH'] = {str(bin_dir)!r} + os.pathsep + os.environ['PATH']
module = runpy.run_path({str(profiler)!r})
module['_run_msprof_{mode}']({inner!r}, {str(tmp_path / 'profile')!r}, 0{extra})
"""
    neighbor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        result = run_guard(tmp_path, code)
        assert result.returncode == 125, result.stderr
        failure = json.loads((tmp_path / "artifacts/npu_failure.json").read_text())
        assert failure["fault_pid"] == int(inner_pid.read_text())
        assert failure["child_reaped"] is True
        assert neighbor.poll() is None
        assert run_guard(tmp_path, "print('another task')", artifacts="other").returncode == 0
    finally:
        neighbor.kill()
        neighbor.wait(timeout=3)
        if phase_pid.exists():
            try:
                os.killpg(int(phase_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def log_line(body, pid=123, stamp=None):
    stamp = stamp or datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    stamp = stamp[:-3] + "." + stamp[-3:]
    return f"[ERROR] DRV({pid},python):{stamp} {body}".encode()


ERROR = "rtStreamSynchronize:ErrCode=507035"
NO_RESPONSE = 'Message send finish, no resp. (dev_id=1; msg_type="msg_client_tsdrv"; ret=-38)'
CLEANUP = "halEschedSubmitEventSync]Failed. (ret=16; timeout=30000ms)"


@pytest.mark.parametrize("excluded", ["old", "future", "other_card", "other_pid", "mixed_pids", "pid_reused", "duplicate", "ordinary_cleanup", "exited", "quoted", "before_birth"])
def test_incomplete_or_unrelated_evidence_does_not_kill(monkeypatch, excluded):
    m = runpy.run_path(str(GUARD))
    now = time.time()
    live={123:(1,now-1),124:(2,now-1)}
    monkeypatch.setitem(m['_NpuFailure'].feed.__globals__, '_task_identity', lambda pid,group: live.get(pid))
    detector=m['_NpuFailure'](999,1,now-1)
    pid=555 if excluded=='other_pid' else 123
    stamp = datetime.fromtimestamp(now-10 if excluded=='old' else now+10 if excluded=='future' else now).strftime("%Y-%m-%d-%H:%M:%S.%f")
    if excluded == 'before_birth':
        live[123] = (1, now + 1)
    first=log_line(ERROR,pid,stamp)
    assert detector.feed(b'quoted '+first if excluded=='quoted' else first) is None
    if excluded == 'pid_reused':
        live[123] = (3, now - 1)
    response=NO_RESPONSE.replace('dev_id=1','dev_id=2') if excluded=='other_card' else NO_RESPONSE
    one=log_line(response,pid,stamp)
    two=one if excluded=='duplicate' else log_line(response,pid,datetime.fromtimestamp((now-10 if excluded=='old' else now+10 if excluded=='future' else now)+.0001).strftime("%Y-%m-%d-%H:%M:%S.%f"))
    assert detector.feed(one) is None
    assert detector.feed(two) is None
    if excluded == 'exited':
        live.pop(123)
    body="NPU shutdown synchronize device failed; Va can't free; Memory is being used" if excluded=='ordinary_cleanup' else CLEANUP
    assert detector.feed(log_line(body,124 if excluded=='mixed_pids' else pid,stamp)) is None


def test_complete_current_driver_evidence_is_required(monkeypatch):
    m = runpy.run_path(str(GUARD))
    now = time.time()
    monkeypatch.setitem(m['_NpuFailure'].feed.__globals__, '_task_identity', lambda pid,group: (1,now-1) if pid==123 else None)
    detector=m['_NpuFailure'](999,1,now-1)
    for body in (ERROR,NO_RESPONSE,NO_RESPONSE):
        assert detector.feed(log_line(body)) is None
        time.sleep(.001)
    result=detector.feed(log_line(CLEANUP))
    assert result['error_code']=='507035' and result['fault_pid']==123


def test_killed_lease_parent_does_not_unlock_live_child(tmp_path):
    child_file = tmp_path / "child"
    proc = subprocess.Popen(
        [sys.executable, str(GUARD), "--pool", "0", "--lock-dir", str(tmp_path), "--",
         sys.executable, "-c", f"import os,time,pathlib; pathlib.Path({str(child_file)!r}).write_text(str(os.getpid())); time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child = None
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        child = int(child_file.read_text())
        proc.kill()
        proc.wait(timeout=3)
        with (tmp_path / "npu0.lock").open("a+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_driver_blocked_child_has_bounded_cleanup(tmp_path, monkeypatch):
    module = runpy.run_path(str(GUARD))
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"driver evidence\n")
    os.close(write_fd)
    monkeypatch.setattr(module["_NpuFailure"], "feed", lambda self,line: {"error_code":"507035", "reason":"device_unresponsive_and_cleanup_failed"})
    child = Mock(pid=987654, stdout=os.fdopen(read_fd, "rb"))
    child.poll.return_value = None
    actions = []

    def blocked_wait(*args, **kwargs):
        actions.append("wait")
        raise subprocess.TimeoutExpired("fake-driver-child", 2)

    child.wait.side_effect = blocked_wait
    monkeypatch.setattr(module["subprocess"], "Popen", lambda *a, **kw: child)
    kill = Mock(side_effect=lambda *args: actions.append("kill"))
    monkeypatch.setattr(module["os"], "killpg", kill)
    lease = module["_try_acquire"](["0"], tmp_path)
    try:
        assert module["_run_guarded"]([], {}, lease, tmp_path / "npu_failure.json") == 125
        kill.assert_called_once_with(child.pid, signal.SIGKILL)
        child.wait.assert_called_once_with(timeout=2)
        assert actions == ["kill", "wait"]
        failure = json.loads((tmp_path / "npu_failure.json").read_text())
        assert failure["child_reaped"] is False
    finally:
        lease.release()


def test_already_exited_native_crash_does_not_mark_session_terminal(tmp_path):
    result = run_guard(tmp_path, "import os,signal,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.kill(os.getpid(),signal.SIGSEGV)")
    assert result.returncode == 256 - signal.SIGSEGV
    assert not (tmp_path / "artifacts/npu_failure.json").exists()


def test_pipeline_guards_only_agent_and_keeps_previous_best(tmp_path):
    source = GUARD.with_name("ascendc_eval_pipeline.sh").read_text()
    run_phase = "run_npu_phase() {" + source.split("run_npu_phase() {", 1)[1].split("\n# Agent 默认", 1)[0]
    on_exit = "_on_exit() {" + source.split("_on_exit() {", 1)[1].split("\ntrap _on_exit EXIT", 1)[0]
    fake = tmp_path / "lease.py"
    fake.write_text("import json,sys; print(json.dumps(sys.argv[1:]))")
    common = "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in {
        "OUT_DIR": tmp_path, "ARTIFACTS_DIR": tmp_path, "POLAR_NPU_LEASE_POOL": "0",
        "POLAR_NPU_LOCK_DIR": tmp_path, "AST_CHECK_PYTHON": sys.executable,
        "NPU_LEASE_EXEC": fake, "PIPELINE_PHASE": "generation", "PIPELINE_ATTEMPT": "1",
        "PIPELINE_LIMIT": "40", "CUR_HASH": "",
    }.items())
    for agent_side in (0, 1):
        script = f"{common}\nAGENT_SIDE={agent_side}\n{run_phase}\nrun_npu_phase verify unused"
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        assert ("--stop-on-npu-error" in json.loads(result.stdout)) == bool(agent_side)
    script = common + "\nAGENT_SIDE=1\n" + on_exit + """
pipeline_over_limit() { return 1; }
pipeline_at_limit() { return 1; }
promote_candidate() { echo promoted; }
_on_exit
"""
    assert subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout == "promoted\n"
    (tmp_path / "npu_failure.json").write_text('{}')
    assert subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout == ""
