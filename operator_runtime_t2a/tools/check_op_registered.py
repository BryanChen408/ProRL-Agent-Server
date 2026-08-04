#!/usr/bin/env python3
"""Post-compile smoke check: is the operator actually reachable via torch.ops.npu?

Packaging mistakes (nested NpuExtension name, .so built into a path the
submission's own loader does not glob, wheel install silently skipped) do not
fail the compile step. They surface much later, inside Step2b verify, as

    AttributeError: '_OpNamespace' 'npu' object has no attribute '<op>'

which the judge then reports as correctness_failed -> "D类-精度不匹配". The agent
is told it has a numerical problem when the operator never loaded at all, and
burns its whole pipeline budget tuning numerics.

This check reproduces verification_ascendc.py's import conditions exactly (same
sys.path entries, same module load) but requires no NPU, so it can run in Step2.
Exit 0 = an npu op was registered. Exit 3 = packaging/registration failure.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import traceback
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CALL_RE = re.compile(r"torch\s*\.\s*ops\s*\.\s*npu\s*\.\s*(\w+)")


def _called_op_names(source: str) -> list[str]:
    """The torch.ops.npu.<name> operators the candidate actually calls.

    Checking "is the npu namespace non-empty" is useless: torch_npu itself
    registers thousands of npu:: ops. The only meaningful question is whether the
    specific name this submission calls resolves.
    """
    seen: list[str] = []
    for name in CALL_RE.findall(source):
        if name not in seen:
            seen.append(name)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", help="the <op_name>/ directory inside judge work dir")
    ap.add_argument("--workdir", default=os.environ.get("WORKDIR") or ".")
    args = ap.parse_args()

    task_dir = Path(args.task_dir).resolve()
    cand = task_dir / "model_new_ascendc.py"
    kernel_build = task_dir / "kernel" / "build"

    if not cand.is_file():
        print(f"[op-smoke] FAIL missing candidate: {cand}")
        return 3

    # Mirror verification_ascendc.py::_setup_paths
    for p in ([str(Path(args.workdir).resolve())]
              + ([str(kernel_build)] if kernel_build.is_dir() else [])):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    wanted = _called_op_names(cand.read_text(encoding="utf-8", errors="replace"))

    try:
        module = _load_module(cand, f"{task_dir.name}_smoke")
    except Exception as exc:
        print(f"[op-smoke] FAIL importing model_new_ascendc.py: "
              f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3

    if not hasattr(module, "ModelNew"):
        print("[op-smoke] FAIL model_new_ascendc.py defines no class ModelNew")
        return 3

    if not wanted:
        print("[op-smoke] WARN no literal torch.ops.npu.<op> call found in "
              "model_new_ascendc.py; skipping registration check")
        return 0

    ns = torch.ops.npu
    missing = []
    for name in wanted:
        try:
            getattr(ns, name)
        except Exception:
            missing.append(name)

    if missing:
        so_files = sorted(str(p.relative_to(task_dir))
                          for p in task_dir.rglob("*.so"))
        print(f"[op-smoke] FAIL torch.ops.npu.{missing[0]} is not registered after "
              "importing model_new_ascendc.py — the kernel .so was never loaded.")
        print(f"[op-smoke]   called but unresolved: {missing}")
        print(f"[op-smoke]   .so files present: {so_files or 'NONE'}")
        print("[op-smoke]   This is a packaging/import problem, NOT a numerical one. "
              "Make sure setup.py builds the .so where your loader looks, and that "
              "importing model_new_ascendc.py triggers TORCH_LIBRARY registration.")
        return 3

    print(f"[op-smoke] ok resolved={wanted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
