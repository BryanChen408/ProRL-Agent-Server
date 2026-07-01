#!/usr/bin/env python3
"""Stage files for CANNBot verifier scripts.

This is a mechanical file-name adapter. It does not run verification and does
not define a new submission format.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _require_safe_name(op_name: str) -> None:
    if not SAFE_NAME.fullmatch(op_name):
        raise ValueError(f"unsafe op_name: {op_name!r}")


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_verifier_inputs(
    *,
    op_name: str,
    task_path: str | Path,
    impl_path: str | Path,
    verify_dir: str | Path,
    triton_impl_name: str = "triton_ascend_impl",
) -> dict[str, str]:
    """Stage task and implementation files under CANNBot verifier names."""

    _require_safe_name(op_name)
    _require_safe_name(triton_impl_name)

    task = Path(task_path)
    impl = Path(impl_path)
    target_dir = Path(verify_dir)

    staged_task = target_dir / f"{op_name}_torch.py"
    staged_impl = target_dir / f"{op_name}_{triton_impl_name}.py"
    _copy_file(task, staged_task)
    _copy_file(impl, staged_impl)

    staged: dict[str, str] = {
        "task": str(staged_task),
        "impl": str(staged_impl),
    }

    sidecar = task.with_suffix(".json")
    if sidecar.is_file():
        staged_sidecar = target_dir / f"{op_name}_torch.json"
        _copy_file(sidecar, staged_sidecar)
        staged["sidecar"] = str(staged_sidecar)

    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--verify-dir", required=True)
    parser.add_argument("--triton-impl-name", default="triton_ascend_impl")
    args = parser.parse_args(argv)

    stage_verifier_inputs(
        op_name=args.op_name,
        task_path=args.task,
        impl_path=args.impl,
        verify_dir=args.verify_dir,
        triton_impl_name=args.triton_impl_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
