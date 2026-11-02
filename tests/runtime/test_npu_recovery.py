import asyncio
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polar.config.topology import NpuRecoveryConfig
from polar.runtime import npu_recovery as recovery


RECORD = {"kind": "update", "phase": "ready_for_training",
          "transition_id": "t-0-1", "policy_namespace": "test"}


class Runtime:
    def __init__(self):
        self.reset_calls = 0
        self.probed = []
        self.stopped = False
        self.bad = set()
        self.reply = {"confirmed": True, "timed_out": False, "child_reaped": True,
                      "success": True, "return_code": 0}
        self.during_probe = None
        self.hold = None

    async def start(self):
        pass

    async def stop(self):
        self.stopped = True

    async def exec(self, command, *, env=None, timeout_sec=None):
        if env is None:
            self.reset_calls += 1
            if self.hold is not None:
                await self.hold.wait()
            return SimpleNamespace(return_code=0, stdout="POLAR_RESET_RESULT " + json.dumps(self.reply), stderr="")
        card = int(env["ASCEND_RT_VISIBLE_DEVICES"])
        self.probed.append(card)
        if self.during_probe:
            self.during_probe(card)
        return SimpleNamespace(return_code=int(card in self.bad),
            stdout="POLAR_NPU_HEALTH " + json.dumps({"card": card, "ok": True}), stderr="")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    runtime = Runtime()
    config = NpuRecoveryConfig(image="test", lock_dir=str(tmp_path / "locks"))
    service = recovery.NpuRecovery(config, tmp_path / "control", "http://gateway")
    service.locks.mkdir()
    for card in (4, 6):
        recovery.write_json(service._marker(card), {"device_id": str(card), "source": "original_fault"})
    read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k:
        "Atlas 200T A2 Box16" if str(p) == "/sys/class/dmi/id/product_name" else read(p, *a, **k))
    monkeypatch.setattr(service, "_window_matches", lambda state: True)
    monkeypatch.setattr(service, "_pause_collector", AsyncMock(return_value=[]))
    monkeypatch.setattr(recovery, "create_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(recovery, "SETTLE_SECONDS", 0)
    monkeypatch.setattr(recovery, "BUSY_WAIT_SECONDS", 0)
    return service, runtime


def test_normal_and_non_update_paths_never_start(setup):
    service, runtime = setup
    for changes in ({"phase": "serving"}, {"kind": "bootstrap"}, {"kind": "initialize"}):
        assert service.start({**RECORD, **changes}) is None
    for card in (4, 6):
        service._marker(card).unlink()
    assert service.start(RECORD) is None
    assert runtime.reset_calls == 0
    assert service.resume_error() is None


def test_recovery_is_singleton_and_idempotent_with_live_monitor(setup):
    service, runtime = setup

    async def run():
        runtime.hold = asyncio.Event()
        task = service.start(RECORD)
        while runtime.reset_calls == 0:
            await asyncio.sleep(.001)
        assert service.resume_error()
        assert all(service._marker(c).exists() for c in range(16))
        other = recovery.NpuRecovery(service.config, service.control_dir, service.gateway_url)
        assert other.start(RECORD) is None
        runtime.hold.set()
        await task
        assert other.start(RECORD) is None
        assert other.resume_error() is None

    asyncio.run(run())
    assert runtime.reset_calls == 1 and runtime.probed == list(range(16))
    assert runtime.stopped
    assert not list(service.locks.glob("*.quarantine.json"))
    allocator = runpy.run_path("operator_runtime_t2a/tools/npu_lease_exec.py")
    for c in ("4", "6"):
        lease = allocator["_try_acquire"]([c], service.locks)
        assert lease is not None
        assert allocator["_try_acquire"]([c], service.locks) is None
        lease.release()


def test_failed_card_remains_isolated_but_healthy_cards_resume(setup):
    service, runtime = setup
    runtime.bad = {6}
    asyncio.run(_complete(service))
    assert service._marker(6).exists()
    assert not service._marker(4).exists()
    assert service.snapshot()["available_cards"] == [c for c in range(16) if c != 6]
    assert service.resume_error() is None


async def _complete(service):
    task = service.start(RECORD)
    assert task is not None
    await task


def test_fault_card_old_lease_does_not_block_reset_and_is_not_forced_free(setup):
    service, runtime = setup
    fd = recovery.try_lock(service.locks / "npu6.lock")
    try:
        inode = (service.locks / "npu6.lock").stat().st_ino
        asyncio.run(_complete(service))
        assert runtime.reset_calls == 1 and 6 not in runtime.probed
        assert service._marker(6).exists()
        assert (service.locks / "npu6.lock").stat().st_ino == inode
        assert recovery.try_lock(service.locks / "npu6.lock") is None
    finally:
        os.close(fd)


def test_healthy_card_cleanup_is_not_interrupted(setup):
    service, runtime = setup
    fd = recovery.try_lock(service.locks / "npu1.lock")
    try:
        asyncio.run(_complete(service))
        assert runtime.reset_calls == 0
        assert service.snapshot()["state"] == "not_started"
        assert service.resume_error() is None
        assert sorted(p.name for p in service.locks.glob("*.quarantine.json")) == ["npu4.quarantine.json", "npu6.quarantine.json"]
        assert recovery.try_lock(service.locks / "npu1.lock") is None
    finally:
        os.close(fd)


def test_initialization_timeout_restores_original_pool_without_reset(setup):
    service, runtime = setup
    runtime.reply.update(confirmed=False, timed_out=True, return_code=-9, success=False)
    asyncio.run(_complete(service))
    assert runtime.probed == [] and runtime.stopped
    assert service.snapshot()["state"] == "not_started"
    assert service.resume_error() is None
    assert len(list(service.locks.glob("*.quarantine.json"))) == 2


def test_ambiguous_reset_blocks_resume_and_never_retries(setup):
    service, runtime = setup
    runtime.reply.update(timed_out=True, child_reaped=False, return_code=None, success=False)
    asyncio.run(_complete(service))
    assert service.snapshot()["state"] == "unknown"
    assert service.resume_error()
    assert len(list(service.locks.glob("*.quarantine.json"))) == 16
    assert not runtime.stopped
    assert service.start({**RECORD, "transition_id": "next"}) is None
    assert runtime.reset_calls == 1


def test_newer_quarantine_is_not_erased_by_successful_probe(setup):
    service, runtime = setup
    def update(card):
        if card == 6:
            recovery.write_json(service._marker(6), {"source": "new_failure"})
    runtime.during_probe = update
    asyncio.run(_complete(service))
    assert json.loads(service._marker(6).read_text()) == {"source": "new_failure"}
    assert 6 not in service.snapshot()["available_cards"]


def test_unreadable_state_and_partial_reset_domain_fail_closed(setup):
    service, _ = setup
    recovery.write_json(service.path, {})
    assert service.resume_error()
    assert service.start(RECORD) is None
    service.path.write_text("[]")
    assert service.resume_error()
    with pytest.raises(ValueError, match="full 0..15"):
        NpuRecoveryConfig(image="test", pool=[4, 6])


@pytest.mark.parametrize("initial_wait", [0, 45])
def test_reset_child_answers_real_pty_prompts(tmp_path, initial_wait):
    fake = tmp_path / "fake-smi"
    fake.write_text("#!/usr/bin/env python3\n"
        "print('Are you sure you want to continue resetting?(Y/N)',flush=True)\n"
        "assert input()=='y'\n"
        "print('it will reboot all devices, do you want to continue reboot? [y/n]',flush=True)\n"
        "assert input()=='y'\nprint('Reset server successfully.',flush=True)\n")
    fake.chmod(0o755)
    code = recovery.RESET_PROGRAM.replace("'/usr/local/bin/npu-smi'", repr(str(fake)))
    code = code.replace("started=time.monotonic();", f"started=time.monotonic()-{initial_wait};")
    result = subprocess.run([sys.executable, "-c", code, "4"], capture_output=True, text=True, timeout=5)
    records = [json.loads(l.split("POLAR_RESET_RESULT ", 1)[1]) for l in result.stdout.splitlines() if l.startswith("POLAR_RESET_RESULT ")]
    assert result.returncode == 0 and records == [{"return_code": 0, "confirmed": True,
        "timed_out": False, "child_reaped": True, "success": True}]


def test_probe_program_compiles_without_importing_torch():
    compile(recovery.PROBE_PROGRAM, "npu-health-child", "exec")


def test_training_window_ends_during_container_start(setup, monkeypatch):
    service, runtime = setup
    windows = iter([True, True, False])
    monkeypatch.setattr(service, "_window_matches", lambda state: next(windows))
    asyncio.run(_complete(service))
    assert runtime.reset_calls == 0 and runtime.stopped
    assert service.resume_error() is None
    assert len(list(service.locks.glob("*.quarantine.json"))) == 2


def test_invisible_active_query_collector_prevents_reset(setup):
    service, runtime = setup
    directory = service.locks / "npu-smi-snapshot"
    directory.mkdir()
    fd = recovery.try_lock(directory / "collector.lock")
    try:
        asyncio.run(_complete(service))
        assert runtime.reset_calls == 0
        assert service.resume_error() is None
    finally:
        os.close(fd)


@pytest.mark.parametrize('end_window', [False, True])
def test_gateway_read_timeout_retries_but_rechecks_training_window(setup, monkeypatch, end_window):
    import io
    from unittest.mock import Mock
    service, _ = setup
    recovery.write_json(service.control_dir / 'policy-transition.json', {'current': RECORD})
    gate = {**RECORD, 'paused': True}
    opener = Mock()
    opener.open.side_effect = [TimeoutError('busy gateway'), io.StringIO(json.dumps(gate))]
    monkeypatch.setattr(recovery.urllib.request, 'build_opener', lambda *a: opener)
    def pause(_):
        if end_window:
            recovery.write_json(service.control_dir / 'policy-transition.json', {'current': {**RECORD, 'phase': 'serving'}})
    monkeypatch.setattr(recovery.time, 'sleep', pause)
    assert recovery.NpuRecovery._window_matches(service, RECORD) is not end_window
    assert opener.open.call_count == (1 if end_window else 2)


def test_gateway_read_retries_are_bounded(setup, monkeypatch):
    from unittest.mock import Mock
    service, _ = setup
    recovery.write_json(service.control_dir / 'policy-transition.json', {'current': RECORD})
    opener = Mock()
    opener.open.side_effect = TimeoutError('still busy')
    monkeypatch.setattr(recovery.urllib.request, 'build_opener', lambda *a: opener)
    ticks = iter([0, 61])
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        recovery.NpuRecovery._window_matches(service, RECORD)
    assert opener.open.call_count == 1
