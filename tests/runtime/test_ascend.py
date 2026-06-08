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


def test_recipe_scopes_to_one_card():
    # B recipe (validated on Node-5-88 + OpenHands): privileged + /dev:/dev (enumeration) +
    # RT=<physical card> (scope -> concurrency-safe).
    args = ascend_create_args({"device_id": 9})
    assert "--privileged" in args
    vols = _vals(args, "-v")
    assert "/dev:/dev" in vols
    assert "/usr/local/dcmi:/usr/local/dcmi:ro" in vols
    assert "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" in vols
    env = dict(p.split("=", 1) for p in _vals(args, "-e"))
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "9"         # scope to the leased physical card


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
