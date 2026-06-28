#!/usr/bin/env python3
"""Generate Polar/slime operator assets from a KernelBench-style parquet.

The input parquet is expected to contain an ``extra_info`` column with at least:
``op_name`` and ``task_code``. This emits:

* ``operator_tasks.jsonl`` for slime ``--prompt-data``.
* ``op_tasks/<op_name>.py`` reference files consumed by the Polar runtime and
  the fresh judge runtime.

The generated ``op_name`` is a filename stem, so this script rejects path-like
or shell-sensitive names instead of allowing accidental writes outside the
operator asset directory.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

SAFE_OP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

META_KEYS = (
    "op_name",
    "arch",
    "entry_point",
    "operator_backend",
    "kernelbench_level",
    "kernelbench_problem_id",
    "kernelbench_name",
    "data_source",
    "ability",
    "uid",
)


def _instruction(op: str) -> str:
    return (
        f"Implement a Triton operator for Ascend NPU. The reference task is at src/{op}.py. "
        f"Write your implementation as class ModelNew to output/submission/{op}_impl.py.\n\n"
        "强制执行规则：\n"
        f"- 第一次 Write/Edit/MultiEdit output/submission/{op}_impl.py 之后，下一次工具调用必须运行：\n"
        f"  bash tools/triton_eval_pipeline.sh --op_name {op} "
        f"--impl output/submission/{op}_impl.py --task src/{op}.py --out_dir judge_out\n"
        f"- 在运行上述固定入口之前，禁止第二次 Write/Edit/MultiEdit output/submission/{op}_impl.py。\n"
        "- 禁止总结、点评、改写 reference 文档；reference 只用于实现和修复代码。"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("extra_info is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if not isinstance(parsed, dict):
            raise TypeError(f"extra_info parsed to {type(parsed).__name__}, expected dict")
        return parsed
    raise TypeError(f"extra_info is neither dict nor parseable str: {type(value).__name__}")


def _validate_op_name(value: Any, *, row: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"row{row}: op_name must be a string, got {type(value).__name__}")
    op = value.strip()
    if not SAFE_OP_NAME_RE.fullmatch(op):
        raise ValueError(
            f"row{row}: unsafe op_name {value!r}; expected a basename matching "
            f"{SAFE_OP_NAME_RE.pattern!r}"
        )
    if Path(op).name != op:
        raise ValueError(f"row{row}: op_name must be a basename, got {value!r}")
    return op


def _metadata(extra_info: dict[str, Any], op: str) -> dict[str, Any]:
    meta = {key: extra_info[key] for key in META_KEYS if extra_info.get(key) is not None}
    meta["op_name"] = op
    return meta


def build_assets(*, parquet: Path, out_dir: Path, limit: int = 0) -> int:
    import pandas as pd

    df = pd.read_parquet(parquet)
    if limit:
        df = df.iloc[:limit]

    tasks_dir = out_dir / "op_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "operator_tasks.jsonl"

    rows_out = 0
    skipped: list[str] = []
    code_lens: list[int] = []
    op_names: list[str] = []
    seen_ops: set[str] = set()

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for row in range(len(df)):
            extra_info = _as_dict(df.iloc[row]["extra_info"])
            raw_op = extra_info.get("op_name")
            code = extra_info.get("task_code")
            if not raw_op or not code or not str(code).strip():
                skipped.append(f"row{row}:op={raw_op!r} code_empty={not bool(code)}")
                continue

            op = _validate_op_name(raw_op, row=row)
            if op in seen_ops:
                raise ValueError(f"row{row}: duplicate op_name {op!r}")
            seen_ops.add(op)

            task_path = tasks_dir / f"{op}.py"
            task_path.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")

            payload = {
                "prompt": [{"role": "user", "content": _instruction(op)}],
                "label": op,
                "metadata": _metadata(extra_info, op),
            }
            jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows_out += 1
            code_lens.append(len(code))
            op_names.append(op)

    print(f"[gen] parquet rows={len(df)} -> emitted={rows_out} skipped={len(skipped)}")
    print(f"[gen] jsonl    -> {jsonl_path}")
    print(f"[gen] tasks    -> {tasks_dir}/<op_name>.py  ({rows_out} files)")
    if code_lens:
        sorted_lens = sorted(code_lens)
        print(
            "[gen] task_code chars: "
            f"min={sorted_lens[0]} med={sorted_lens[len(sorted_lens) // 2]} max={sorted_lens[-1]}"
        )
    for chars, op in sorted(zip(code_lens, op_names))[:6]:
        print(f"[gen] simple_op {op} ({chars} chars)")
    if skipped:
        suffix = " ..." if len(skipped) > 5 else ""
        print(f"[gen] skipped: {skipped[:5]}{suffix}")
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, default=Path("/home/docker/kernelbench_openhands.parquet"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/docker/blackbox/Workspace/polar_slime_ascend/op_assets"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()
    build_assets(parquet=args.parquet, out_dir=args.out_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
