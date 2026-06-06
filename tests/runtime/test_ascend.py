"""Unit tests for polar.runtime.ascend.ascend_create_args (no Docker needed).

Standalone: `python tests/runtime/test_ascend.py`  | or via pytest.
"""

from __future__ import annotations

import os
import sys

# Allow running without `pip install -e .` (bootstrap src/ onto the path).
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import pytest  # noqa: E402
except ModuleNotFoundError:  # standalone run without pytest installed
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

from polar.runtime.ascend import LOCK_MOUNT_TARGET, ascend_create_args  # noqa: E402

_OK = {"device_ids": "8,9,10,11", "lock_dir": "/host/npu-locks"}


def _pairs(args, flag):
    """All values following each occurrence of `flag` (e.g. every -v / -e)."""
    return [args[i + 1] for i, a in enumerate(args) if a == flag and i + 1 < len(args)]


def test_requires_device_ids_and_lock_dir():
    with pytest.raises(ValueError):
        ascend_create_args({"lock_dir": "/x"})
    with pytest.raises(ValueError):
        ascend_create_args({"device_ids": "8"})
    with pytest.raises(TypeError):
        ascend_create_args("8,9")  # type: ignore[arg-type]


def test_core_recipe_present():
    args = ascend_create_args(_OK)
    assert "--privileged" in args
    assert _pairs(args, "--ipc") == ["host"]
    assert _pairs(args, "--shm-size") == ["500g"]
    vols = _pairs(args, "-v")
    assert "/dev:/dev" in vols
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in vols
    assert f"/host/npu-locks:{LOCK_MOUNT_TARGET}" in vols  # flock dir mounted


def test_eval_env_emitted():
    env = dict(p.split("=", 1) for p in _pairs(ascend_create_args(_OK), "-e"))
    assert env["EVAL_DEVICE_IDS"] == "8,9,10,11"
    assert env["EVAL_LOCK_DIR"] == LOCK_MOUNT_TARGET
    assert env["EVAL_ENV_NAME"] == "ASCEND_RT_VISIBLE_DEVICES"
    assert env["EVAL_DEVICE_PREFIX"] == "npu"


def test_overrides_and_extra_mounts_env():
    args = ascend_create_args({
        **_OK, "shm_size": "200g", "ipc": "",
        "mounts": ["/data:/data:ro"],
        "env": {"EVAL_DEVICE_IDS": "12,13", "EXTRA": "1"},  # env overrides default
    })
    assert _pairs(args, "--shm-size") == ["200g"]
    assert "--ipc" not in args  # disabled
    assert "/data:/data:ro" in _pairs(args, "-v")
    env = dict(p.split("=", 1) for p in _pairs(args, "-e"))
    assert env["EVAL_DEVICE_IDS"] == "12,13"  # overridden
    assert env["EXTRA"] == "1"


def test_no_lifecycle_or_network_flags():
    # Polar's DockerRuntime owns name/network/image/sleep — ascend must not touch them.
    args = ascend_create_args(_OK)
    for forbidden in ("--name", "--network", "--rm", "--entrypoint", "sleep"):
        assert forbidden not in args


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
