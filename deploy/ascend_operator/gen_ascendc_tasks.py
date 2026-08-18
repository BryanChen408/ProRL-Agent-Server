#!/usr/bin/env python3
"""生成 AscendC 算子 RL 的 operator_tasks.jsonl(NPUKernelBench 目录 → jsonl)。

与 triton 的 gen_op_assets.py 并行、独立(不改 triton 那份)。数据源是 NPUKernelBench/level<N>
目录下的 `{id}_{name}.py`(torch 参考 Model + get_input_groups/get_inputs,读同名 .json 用例)。

prompt = ascendc-coder 式(对齐 plugins-community/ascendc-ops-lab-developer 的 run_benchmark 提示 +
polar RL 的 tarball 提交契约)。metadata.operator_backend=ascendc。

用法:
  python gen_ascendc_tasks.py --benchmark-dir /home/docker/NPUKernelBench --level 1 \
      --out /home/docker/datasets/op_tasks/npukernelbench_level1_ascendc/operator_tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_OP_FILE = re.compile(r"^(\d+)_(.+)\.py$")


def _instruction(op_name: str, model_rel: str) -> str:
    """AscendC-coder 指令 —— 结构对齐 triton op_tasks(Task / 固定验证入口 / Rules)。"""
    tar = f"output/submission/{op_name}_impl.tar.gz"
    # 固定入口 = tools/ascendc_eval_pipeline.sh —— **agent 自检与 judge 判分是同一个脚本**
    # (对齐 triton;两侧行为按有无 {op}/ 源目录自适应)。它把
    # 预算计数 / 内容哈希短路 / NPU 抢卡 / 自动打包收在一起,对齐 triton 的
    # tools/triton_eval_pipeline.sh。tools/ 是只读挂载,agent 改不了。
    val = (f"bash tools/ascendc_eval_pipeline.sh --op_name {op_name} "
           f"--impl output/submission/{op_name}_impl.tar.gz --out_dir judge_out")
    return (
        f"Implement an AscendC operator for Ascend NPU. The reference task is at {model_rel} "
        f"(class Model + get_input_groups). Produce a self-contained project directory `{op_name}/`.\n\n"
        f"The `{op_name}/` project must contain:\n"
        "- model_new_ascendc.py (class ModelNew whose forward ONLY calls torch.ops.npu.<op> plus tensor "
        "create/reshape; no plain-torch compute)\n"
        "- kernel/ (op_host/ + op_kernel/ + register.cpp + ops.h + self-contained CMakeLists.txt + setup.py)\n"
        "You do NOT need to ship model.py or the case-spec .json: the judge injects the dataset originals "
        "and overwrites whatever you submit. Do NOT ship build/, dist/, *.so, *.a or *.whl: the judge "
        "rebuilds from source in a fresh container and ignores prebuilt artifacts.\n\n"
        "Follow the ascendc-* skills workflow: simple ops via case-simplifier -> code-gen; complex ops via "
        "ascendc-tilelang-designer -> ascendc-translator. SoC uses SOC_VERSION env (910B2C / A2); "
        "CMakeLists paths use x86_64-linux.\n\n"
        "Use this fixed validation entry as the only executable validation path:\n"
        f"  {val}\n"
        f"It runs the degradation check, compiles, verifies against the reference, benchmarks, AND packs "
        f"{tar} for you (keeping a `.best.tar.gz` of your highest-scoring version so far). Run it after "
        "EVERY repair iteration — a session that is cut off mid-way still gets graded on its best packed "
        "version, so running it early and often strictly dominates saving it for the end.\n\n"
        "Rules:\n"
        f"- Read {model_rel} and relevant skill/reference docs only as needed to implement the candidate.\n"
        f"- The graded submission is the single tarball {tar} (or its .best variant); never packed -> scores 0.\n"
        "- model_new_ascendc.py.forward must call torch.ops.npu.<op> (no plain-torch fallback), else the "
        "degradation check fails.\n"
        # 逐字派生 triton 任务 prompt 的同一条 Rule(见 op_assets_cudallm_filtered189)。
        # NPU 卡是 agent/judge 共享的卡池,固定入口内部会排队抢锁;任何绕过它的探针都会
        # 和别的 session 撞在同一张卡上(性能测量失真 = reward 失真)。
        "- Do not run custom Python tests, manual import/forward checks, torch.allclose, temporary "
        "kernels, npu-smi/environment/API introspection, verifier introspection, or any executable "
        "probe that touches the NPU. The fixed validation entry is the only executable validation "
        "path, and the only thing allowed to acquire an NPU card. Never set ASCEND_RT_VISIBLE_DEVICES "
        "yourself.\n"
        "- Do not read, modify, inspect, or delete anything under tools/, the verifier scripts under "
        ".claude/skills/ascendc-*/scripts/, or pipeline parameters (SOC_VERSION / warmup / repeats / "
        "precision thresholds are fixed by the entry).\n"
        # 目标线与 CLAUDE.md 4-S.4 / ascendc_eval_pipeline.sh 的 PERF_TARGET 是同一个数,三处要同步。
        # 只写"实现算子"时实测中位 speedup 0.859x、58.8% 慢于 torch:agent 精度一过就收工,
        # 而 reward 在 1.0x 才 0.75、更快才涨,等于把分数留在桌上。
        "- Correctness is not the finish line: anything slower than 1.1x the PyTorch reference does "
        "NOT meet the bar. After the entry reports success, keep optimizing the kernel and re-run it; "
        "it keeps a .best.tar.gz of your fastest verified version, so a failed optimization attempt "
        "never costs the score you already banked.\n"
        "- The fixed entry has a call budget; when it prints LIMIT_EXHAUSTED, stop immediately.\n"
        "- Follow ./CLAUDE.md for the full workflow and judging contract."
    )


def build(
    benchmark_dir: Path,
    level: int,
    out_path: Path,
    arch: str = "ascend910b1",
    only: set[str] | None = None,
) -> int:
    level_dir = benchmark_dir / f"level{level}"
    if not level_dir.is_dir():
        raise SystemExit(f"level 目录不存在: {level_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for py in sorted(level_dir.glob("*.py")):
            m = _OP_FILE.match(py.name)
            if not m:
                continue
            op_name = py.stem  # e.g. 1_GELU
            if only and op_name not in only:
                continue
            name = m.group(2)
            payload = {
                "prompt": [{"role": "user", "content": _instruction(op_name, f"input/{op_name}.py")}],
                "label": op_name,
                "metadata": {
                    "op_name": op_name,
                    "entry_point": "Model",
                    "operator_backend": "ascendc",
                    "arch": arch,
                    "ops": [name],
                    "op_display_name": name,
                    "data_source": "npu-kernel-bench",
                    "ability": "code",
                    "level": str(level),
                    "uid": op_name,
                    "solve_rate": 0.5,   # 占位;跑一轮后可用真实通过率回填(triton 侧有此字段做 curriculum)
                    "skill_tags": [],
                },
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows += 1
    print(f"[gen-ascendc] level{level}: {rows} 个算子 → {out_path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arch", default="ascend910b1")
    ap.add_argument(
        "--ops",
        default="",
        help="逗号分隔的算子名白名单(如 3_Add),只生成这些 —— 单算子冒烟用:"
             "把结果指给 vime 的 OPERATOR_TASK_JSONL,OPERATOR_TASKS_DIR 保持不变即可,零代码改动。",
    )
    args = ap.parse_args()
    only = {o.strip() for o in args.ops.split(",") if o.strip()} or None
    n = build(args.benchmark_dir, args.level, args.out, args.arch, only)
    if only and n != len(only):
        raise SystemExit(f"[gen-ascendc] --ops 指定 {sorted(only)} 但只生成了 {n} 行,检查算子名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
