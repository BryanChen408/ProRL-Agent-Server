#!/usr/bin/env python3
"""Prepare a per-session operator workdir.

This script prepares one session workdir from canonical assets mounted at
/opt/canonical and creates a minimal editable ModelNew starter file.
"""

from __future__ import annotations

import argparse
import ast
import keyword
import re
import shutil
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _forward_arg_names(task_path: Path) -> list[str]:
    try:
        tree = ast.parse(task_path.read_text())
    except Exception:
        return ["x"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Model":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                names = [arg.arg for arg in item.args.args if arg.arg != "self"]
                return names or ["x"]
    return ["x"]


def _write_stub(task_path: Path, op_name: str, submission_path: Path) -> None:
    args = _forward_arg_names(task_path)
    if not all(arg.isidentifier() and not keyword.iskeyword(arg) for arg in args):
        args = ["x"]
    first = args[0]
    signature = ", ".join(args)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        f"""import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _{op_name.replace("-", "_").replace(".", "_")}_starter_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, {signature}):
        out = torch.empty_like({first})
        n_elements = {first}.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _{op_name.replace("-", "_").replace(".", "_")}_starter_kernel[grid]({first}, out, n_elements, BLOCK_SIZE=1024)
        return out
""",
        encoding="utf-8",
    )


def _prepare_tools(canonical: Path, workdir: Path, *, readonly_tools: bool) -> None:
    src = canonical / "tools"
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {src}")
    dst = workdir / "tools"
    if readonly_tools:
        # Docker binds canonical tools to this path as :ro. Do not remove or overwrite it.
        dst.mkdir(parents=True, exist_ok=True)
        return
    _copy_tree(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--workdir", default="/opt/workspace/agent_workdir")
    parser.add_argument("--canonical-root", default="/opt/canonical")
    parser.add_argument("--task-path")
    parser.add_argument("--submission-path")
    parser.add_argument("--no-stub", action="store_true")
    parser.add_argument("--require-claude", action="store_true")
    parser.add_argument(
        "--readonly-tools",
        action="store_true",
        help="Do not copy canonical tools into workdir; expect workdir/tools to be a read-only bind mount.",
    )
    args = parser.parse_args(argv)

    if not SAFE_NAME.fullmatch(args.op_name):
        raise SystemExit(f"unsafe op_name: {args.op_name!r}")

    workdir = Path(args.workdir)
    canonical = Path(args.canonical_root)
    task_path = Path(args.task_path or workdir / "src" / f"{args.op_name}.py")
    submission_path = Path(
        args.submission_path
        or workdir / "output" / "submission" / f"{args.op_name}_impl.py"
    )

    for rel in ("src", "output/submission", "judge_out"):
        (workdir / rel).mkdir(parents=True, exist_ok=True)

    _prepare_tools(canonical, workdir, readonly_tools=args.readonly_tools)
    _copy_tree(canonical / ".agents", workdir / ".agents")
    _copy_file(canonical / "CLAUDE.md", workdir / "CLAUDE.md")
    if not (canonical / "skills").is_dir():
        raise FileNotFoundError(f"required directory missing: {canonical / 'skills'}")
    if not task_path.is_file():
        raise FileNotFoundError(f"task file missing: {task_path}")

    if not args.no_stub:
        _write_stub(task_path, args.op_name, submission_path)

    if args.require_claude and shutil.which("claude") is None:
        raise FileNotFoundError("required executable missing on PATH: claude")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
