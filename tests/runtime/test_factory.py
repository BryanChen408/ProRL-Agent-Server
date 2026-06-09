"""Unit tests for polar.runtime.factory — backend capability gates (no Docker/Apptainer binary).

Standalone: `python tests/runtime/test_factory.py`  | or via pytest.

Covers the Ascend-passthrough guard (audit 2026-06-09 🟠5): kwargs.ascend on a backend that does
NOT implement the passthrough recipe (only docker does) must be REJECTED at create time — otherwise
an operator rollout silently runs with no leased card (davinci0 collision / aclInit failure).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ApptainerRuntime.__init__ resolves a binary; point it at a dummy so we can construct the runtime
# on a host without apptainer installed (the binary is only used at start(), not at validation).
os.environ.setdefault("POLAR_APPTAINER_BIN", "/bin/false")

try:
    import pytest
except ModuleNotFoundError:  # standalone without pytest
    import contextlib

    class _Pytest:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:  # noqa: BLE001
                if match is not None and match not in str(e):
                    raise AssertionError(f"{match!r} not in {e!r}")
                return
            else:
                raise AssertionError(f"DID NOT RAISE {exc.__name__}")

    pytest = _Pytest()  # type: ignore[assignment]

from polar.runtime.factory import create_runtime  # noqa: E402
from polar.runtime.models import RuntimeSpec  # noqa: E402

_ASCEND = {"pool": "0,1", "lock_dir": "/tmp/npu-locks"}


def _spec(backend: str, **kw) -> RuntimeSpec:
    return RuntimeSpec(backend=backend, image="img:latest", **kw)


def test_docker_accepts_ascend():
    with tempfile.TemporaryDirectory() as d:
        rt = create_runtime(_spec("docker", kwargs={"ascend": _ASCEND}), "s", Path(d))
        assert rt.supports_ascend is True


def test_apptainer_rejects_ascend():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="Ascend"):
            create_runtime(_spec("apptainer", kwargs={"ascend": _ASCEND}), "s", Path(d))


def test_apptainer_without_ascend_ok():
    # the guard fires ONLY on kwargs.ascend — a plain apptainer runtime is still valid
    with tempfile.TemporaryDirectory() as d:
        rt = create_runtime(_spec("apptainer", kwargs={"volumes": []}), "s", Path(d))
        assert rt.supports_ascend is False


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
