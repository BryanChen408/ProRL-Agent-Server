"""Unit tests for polar.runtime.ascend — per-card remap recipe + host flock allocator (no Docker).

Standalone: `python tests/runtime/test_ascend.py`  | or via pytest.
"""

from __future__ import annotations

import fcntl
import json
import os
import runpy
import time
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import pytest
except ModuleNotFoundError:  # standalone without pytest
    import contextlib

    class _Pytest:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            else:
                raise AssertionError(f"DID NOT RAISE {exc.__name__}")

    pytest = _Pytest()  # type: ignore[assignment]

from polar.runtime.ascend import (  # noqa: E402
    acquire_card,
    ascend_create_args,
    ascend_mount_create_args,
    parse_pool,
)


def _vals(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag and i + 1 < len(args)]


def test_parse_pool():
    assert parse_pool("8,9,10,11") == ["8", "9", "10", "11"]
    assert parse_pool([8, 9]) == ["8", "9"]
    assert parse_pool("8-11") == ["8", "9", "10", "11"]
    assert parse_pool("") == []


def test_recipe_scopes_to_one_card():
    # B recipe (validated on Node-5-88 + OpenHands): privileged + /dev:/dev (enumeration) +
    # RT=<physical card> (scope -> concurrency-safe).
    args = ascend_create_args({"device_id": 9})
    assert "--privileged" in args
    assert "--ipc" in args
    assert "host" in _vals(args, "--ipc")
    assert "--shm-size" in args
    assert "500g" in _vals(args, "--shm-size")
    vols = _vals(args, "-v")
    assert "/dev:/dev" in vols
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in vols
    assert "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro" in vols
    assert "/usr/local/dcmi:/usr/local/dcmi:ro" in vols
    assert "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" in vols
    assert "/etc/ascend_install.info:/etc/ascend_install.info:ro" in vols
    assert "/usr/local/sbin:/usr/local/sbin:ro" in vols
    env = dict(p.split("=", 1) for p in _vals(args, "-e"))
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "9"         # scope to the leased physical card


def test_recipe_env_cannot_override_leased_device():
    args = ascend_create_args(
        {
            "device_id": 9,
            "env": {"ASCEND_RT_VISIBLE_DEVICES": "0", "CUSTOM_FLAG": "1"},
            "mounts": ["/host/tools:/tools:ro"],
        }
    )

    env = dict(p.split("=", 1) for p in _vals(args, "-e"))
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "9"
    assert env["CUSTOM_FLAG"] == "1"
    assert "/host/tools:/tools:ro" in _vals(args, "-v")


def test_mount_recipe_does_not_require_or_inject_device_id():
    args = ascend_mount_create_args(
        {
            "env": {"CUSTOM_FLAG": "1"},
            "mounts": ["/host/tools:/tools:ro"],
        }
    )

    assert "--privileged" in args
    assert "/dev:/dev" in _vals(args, "-v")
    assert "/host/tools:/tools:ro" in _vals(args, "-v")
    env = dict(p.split("=", 1) for p in _vals(args, "-e"))
    assert env == {"CUSTOM_FLAG": "1"}
    assert "ASCEND_RT_VISIBLE_DEVICES" not in env


def test_npu_smi_snapshot_collector_and_readers():
    snapshot_mount = "/dev/shm/npu-locks/npu-smi-snapshot:/dev/shm/npu-locks/npu-smi-snapshot:ro"
    vols = _vals(ascend_mount_create_args({"cache_npu_smi_info": True}), "-v")
    assert snapshot_mount in vols
    assert not any("npu-smi.real" in v for v in vols)
    assert any(v.endswith(":/usr/local/bin/npu-smi:ro") for v in vols)
    assert any(v.endswith(":/usr/local/sbin/npu-smi:ro") for v in vols)
    assert "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" not in vols
    custom = _vals(ascend_mount_create_args({"cache_npu_smi_info": True, "lock_dir": "/custom"}), "-v")
    assert "/custom/npu-smi-snapshot:/custom/npu-smi-snapshot:ro" in custom

    wrapper = Path(_SRC) / "polar/runtime/npu_smi_cached.py"
    collector = wrapper.parents[3] / "deploy/ascend_operator/telemetry/npu_smi_snapshot.py"
    collect = runpy.run_path(str(collector))["collect"]
    with tempfile.TemporaryDirectory() as d:
        directory = Path(d)
        real = directory / "real.py"
        real.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys, time\n"
            "p = pathlib.Path(__file__).with_name('calls')\n"
            "p.write_text((p.read_text() if p.exists() else '') + 'x')\n"
            "mode = pathlib.Path(__file__).with_name('mode')\n"
            "if mode.exists():\n"
            "    if mode.read_text() == 'slow': time.sleep(10)\n"
            "    else: sys.exit(1)\n"
            "assert sys.argv[1:] == ['info']\n"
            "print('Ascend 910B1')\n"
        )
        real.chmod(0o755)
        env = {**os.environ, "POLAR_NPU_SMI_REAL": str(real), "POLAR_NPU_SMI_CACHE_DIR": d}

        def read(*args):
            return subprocess.run([sys.executable, str(wrapper), *args], env=env,
                                  capture_output=True, timeout=5)

        assert read("info").returncode == 124
        for args in [("info", "-l"), ("info", "-t", "memory", "-i", "4"), ()]:
            assert read(*args).returncode == 124  # No host query service yet.
        for args in [("--collect",), ("set", "-t", "power"), ("reset",)]:
            assert read(*args).returncode == 64
        assert not (directory / "calls").exists()  # No reader path starts a real query.

        with (directory / "collector.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # A second collector must exit without issuing a query.
            second = subprocess.run([sys.executable, str(collector), "--directory", d,
                                     "--real", str(real)], capture_output=True, timeout=5)
            assert second.returncode == 1 and b"prior owner" in second.stdout
            assert not (directory / "calls").exists()
            assert collect(str(real), directory, lock.fileno(), 2)
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda _: read("info"), range(6)))
            assert all(r.returncode == 0 and r.stdout == b"Ascend 910B1\n" for r in results)
            assert (directory / "calls").read_text() == "x"
            snapshot = directory / "npu-smi-info.out"
            old = time.time() - 3600
            os.utime(snapshot, (old, old))
            stale = read("info")
            assert stale.returncode == 0 and b"stale" in stale.stderr
            (directory / "mode").write_text("fail")
            assert not collect(str(real), directory, lock.fileno(), 2)
            (directory / "mode").write_text("slow")
            assert not collect(str(real), directory, lock.fileno(), 0.1)
            assert snapshot.read_bytes() == b"Ascend 910B1\n"
            assert snapshot.stat().st_mtime == old
            assert (directory / "calls").read_text() == "xxx"


def test_npu_smi_parameter_queries_share_host_owner():
    wrapper = Path(_SRC).resolve() / "polar/runtime/npu_smi_cached.py"
    collector = wrapper.parents[3] / "deploy/ascend_operator/telemetry/npu_smi_snapshot.py"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        real = root / "real.py"
        real.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, json, sys, time, os\n"
            "root = pathlib.Path(__file__).parent\n"
            "fd = os.open(root/'active', os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n"
            "os.close(fd)\n"
            "with (root/'calls').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "print('answer: '+json.dumps(sys.argv[1:]), flush=True)\n"
            "try:\n"
            "    if 'slow' in sys.argv: time.sleep(5)\n"
            "    else: time.sleep(0.03)\n"
            "finally: (root/'active').unlink()\n"
            "if 'error' in sys.argv:\n"
            "    print('native query error', file=sys.stderr); sys.exit(7)\n"
        )
        real.chmod(0o755)
        env = {**os.environ, "POLAR_NPU_SMI_CACHE_DIR": d}
        command = [sys.executable, str(collector), "--directory", d, "--real", str(real),
                   "--interval", "0.4", "--timeout", "0.15", "--failure-interval", "0.6"]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not (root / "npu-smi-info.out").exists():
                assert proc.poll() is None, proc.communicate()
                assert time.monotonic() < deadline
                time.sleep(0.01)

            def read(args):
                return subprocess.run([sys.executable, str(wrapper), *args], env=env,
                                      capture_output=True, timeout=5)

            board = ["info", "-t", "board", "-i", "4", "-c", "0"]
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(read, [board] * 6))
            assert all(r.returncode == 0 and b'"board"' in r.stdout for r in results)
            calls = lambda: [json.loads(l) for l in (root / "calls").read_text().splitlines()]
            assert calls().count(board) == 1
            for args in [["info", "-l"], ["info", "-m"], ["info", "proc", "-i", "0"],
                         ["info", "-t", "memory", "-i", "4"], ["info", "--help"], ["--help"], ["-v"], []]:
                assert read(args).returncode == 0
            time.sleep(0.45)
            assert read(board).returncode == 0
            assert calls().count(board) == 2  # Expired results are refreshed centrally.
            error = ["info", "-t", "error"]
            for _ in range(2):
                r = read(error)
                assert r.returncode == 7 and b"native query error" in r.stderr
            assert calls().count(error) == 1
            exporter = runpy.run_path(str(collector.with_name("npu_smi_exporter.py")))
            previous = os.environ.get("POLAR_NPU_SMI_CACHE_DIR")
            os.environ["POLAR_NPU_SMI_CACHE_DIR"] = d
            try:
                assert '"board"' in exporter["_run_smi"](["-t", "board", "-i", "4", "-c", "0"])
                assert exporter["_run_smi"](["-t", "error"]).startswith("__ERROR__ rc=7")
            finally:
                if previous is None:
                    os.environ.pop("POLAR_NPU_SMI_CACHE_DIR", None)
                else:
                    os.environ["POLAR_NPU_SMI_CACHE_DIR"] = previous
            # Validate on the server too: bypassing the wrapper cannot expose writes.
            import socket
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(root / "query.sock"))
                client.sendall(b'["reset", "-i", "0"]\n')
                assert json.loads(client.makefile("rb").readline())["returncode"] == 64
            assert not any(a and a[0] == "reset" for a in calls())
            second = subprocess.run(command, capture_output=True, timeout=5)
            assert second.returncode == 0 and b"already running" in second.stdout
            r = read(["info", "watch", "-i", "0", "-s", "slow"])
            assert r.returncode == 124 and b"answer:" in r.stdout and b"timed out" in r.stderr
            # The killed fake has no finally cleanup; real drivers do not use this test marker.
            (root / "active").unlink(missing_ok=True)
            assert read(["info", "-t", "health"]).returncode == 0
            assert (root / "npu-smi-info.out").read_text().startswith("answer:")
        finally:
            proc.terminate()
            proc.communicate(timeout=5)
        # A dead socket/owner never causes a local probe fallback.
        assert read(board).returncode == 124


def test_no_per_card_device_remap():
    # we scope via RT + /dev:/dev, NOT the per-card `--device=davinciN:davinci0` (507899 on this host)
    args = ascend_create_args({"device_id": 9})
    assert "--device" not in args


def test_requires_device_id():
    with pytest.raises(ValueError):
        ascend_create_args({})
    with pytest.raises(TypeError):
        ascend_create_args("8")  # type: ignore[arg-type]


def test_acquire_is_fair_exclusive_and_releasable():
    with tempfile.TemporaryDirectory() as d:
        pool = ["8", "9"]
        a = acquire_card(pool, d)
        b = acquire_card(pool, d)
        assert {a.device_id, b.device_id} == {"8", "9"}    # two acquires -> two distinct cards
        with pytest.raises(RuntimeError):                  # pool exhausted
            acquire_card(pool, d)
        a.release()
        c = acquire_card(pool, d)                          # freed card is re-acquirable
        assert c.device_id == a.device_id
        b.release()
        c.release()


def test_acquire_empty_pool_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            acquire_card([], d)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
