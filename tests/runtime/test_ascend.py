"""Unit tests for polar.runtime.ascend — per-card remap recipe + host flock allocator (no Docker).

Standalone: `python tests/runtime/test_ascend.py`  | or via pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile

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

from polar.runtime.ascend import acquire_card, ascend_create_args, parse_pool  # noqa: E402


def _vals(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag and i + 1 < len(args)]


def test_parse_pool():
    assert parse_pool("8,9,10,11") == ["8", "9", "10", "11"]
    assert parse_pool([8, 9]) == ["8", "9"]
    assert parse_pool("8-11") == ["8", "9", "10", "11"]
    assert parse_pool("") == []


def test_recipe_remaps_card_to_davinci0():
    args = ascend_create_args({"device_id": 8})
    devs = _vals(args, "--device")
    assert "/dev/davinci8:/dev/davinci0" in devs           # physical 8 -> container davinci0
    assert "/dev/davinci_manager" in devs and "/dev/devmm_svm" in devs and "/dev/hisi_hdc" in devs
    env = dict(p.split("=", 1) for p in _vals(args, "-e"))
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0"         # torch_npu default-device-0 aligns
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in _vals(args, "-v")


def test_no_privileged_no_full_dev():
    # the whole point: avoid the DCMI exclusive lock (-8005) on share-disabled hosts
    args = ascend_create_args({"device_id": 8})
    assert "--privileged" not in args
    assert "/dev:/dev" not in _vals(args, "-v")


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
