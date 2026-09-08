#!/usr/bin/env python3
"""t3a(replica)workdir prepare:复刻官方 tilelang2ascendc-ops-generator 安装产物 + RL 接线。

与 t2a 的 prepare 是两条独立路径——t3a 不生成骨架、不写 skillOverrides、不装 stop_guard、
不做基准注入(那些是我方 t2a 的机制;判分接线 R5 在 judge 侧,不进 agent workdir)。

CLI 与 t2a 的 prepare 保持同形(topology 只需切 paths.operator_runtime_dir):
  --op-name / --workdir / --canonical-root / --task-path / --backend(忽略,恒 ascendc)
  --no-stub(忽略) / --readonly-tools(只校验已挂载工具) / --only-project-skills(忽略,恒全量) / --require-claude

产出的 workdir 布局(= 官方 init.sh global claude 的安装产物 + R1-R4 补丁):
  input/{op}.py(+{op}.json)     ← 数据集原件(判分时 judge 侧也会各拿一份,互不影响)
  CLAUDE.md                     ← canonical(global 形态,workflows 已重写为 /opt/canonical 绝对路径)
  .claude/skills/               ← canonical/skills(16 族全量,实体拷贝)
  .claude/agents/               ← canonical/agents(子代理注册,复刻必需)
  .claude/hooks/                ← canonical/hooks(已含 R3/R4 补丁)
  .claude/workflows/            ← canonical/workflows(archive_tasks 等)
  .claude/settings.json         ← settings.json.template 渲染:${CLAUDE_PLUGIN_ROOT} → workdir
  tools/npu_lease_exec.py       ← R2 租约执行器
  tools/npu_wrap.sh             ← R2/R3 的包装入口
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

MARKER = "[prepare-t3a]"


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"canonical 缺目录: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"canonical 缺文件: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _synthesize_case_json(workdir: Path, op: str, task_py: Path) -> None:
    """[R7] cudallm 案源合成(官方约定:同名 .json 与 .py 并排,JSONL 案源)。

    参考方(cannbot 基准)的 cudallm 数据集自带同名 .json,官方流程的 Phase 2 精简、
    verify、perf、judge benchmark 全走 json 链路。本仓 cudallm 数据集只有 get_inputs()
    内联输入:verify 有兜底能跑,但 perf/judge benchmark 的 case 枚举只吃文件
    (msprof_perf_summary.py: n_cases=len(cases),零兜底)→ 空表 benchmark_failed。
    这里在 prepare 时从 get_inputs() 合成 input/{op}.json,把 cudallm 接回官方 json 链路。
    已有 json(npukernelbench/加工数据集)或无 get_inputs → 不动;合成失败只告警不阻断。
    只作用在 t3a(用户裁决);t2a prepare 不加此逻辑。
    """
    json_path = workdir / "input" / f"{op}.json"
    if json_path.is_file():
        return
    if "get_inputs" not in task_py.read_text(encoding="utf-8", errors="replace"):
        return
    helper = r"""
import importlib.util, inspect, json, sys
py, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("_op_task", py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
if not isinstance(inputs, (list, tuple)):
    raise RuntimeError(f"get_inputs() 返回类型异常: {type(inputs)}")
try:
    params = [p.name for p in inspect.signature(mod.Model.forward).parameters.values()][1:]
except Exception:
    params = []
recs = []
for i, v in enumerate(inputs):
    name = params[i] if i < len(params) else f"input{i}"
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        recs.append({"name": name, "type": "tensor", "required": True,
                     "dtype": str(v.dtype).replace("torch.", ""),
                     "shape": [int(s) for s in v.shape]})
    elif isinstance(v, (bool, int, float, str)):
        recs.append({"name": name, "type": "attr", "required": True,
                     "dtype": type(v).__name__, "value": v})
    else:
        raise RuntimeError(f"不支持的输入类型[{i}]: {type(v)}")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"inputs": recs}, ensure_ascii=False) + "\n")
"""
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        r = subprocess.run(
            [sys.executable, "-c", helper, str(task_py), str(json_path)],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if r.returncode == 0 and json_path.is_file():
            print(f"{MARKER} case json synthesized from get_inputs(): {json_path}")
        else:
            print(f"{MARKER} WARN: case json 合成跳过: "
                  f"{(r.stderr or r.stdout).strip()[:200]}", file=sys.stderr)
    except Exception as exc:
        print(f"{MARKER} WARN: case json 合成失败(不阻断): {type(exc).__name__}: {exc}",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-name", required=True)
    ap.add_argument("--workdir", default="/opt/workspace/agent_workdir")
    ap.add_argument("--canonical-root", default="/opt/canonical")
    ap.add_argument("--task-path")
    ap.add_argument("--backend", default="ascendc")
    ap.add_argument("--no-stub", action="store_true")
    ap.add_argument("--readonly-tools", action="store_true")
    ap.add_argument("--only-project-skills", action="store_true")
    ap.add_argument("--require-claude", action="store_true")
    args = ap.parse_args()

    context_tokens = os.environ.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS")
    if context_tokens is not None and (
        not context_tokens.isascii() or not context_tokens.isdecimal() or int(context_tokens) <= 0
    ):
        raise ValueError("CLAUDE_CODE_MAX_CONTEXT_TOKENS must be a positive integer")

    workdir = Path(args.workdir)
    canonical = Path(args.canonical_root)
    op = args.op_name
    print(f"{MARKER} op={op} workdir={workdir} canonical={canonical}")

    # 0. 数据集原件(input/ 已由 topology 的 upload_file 就位;get_input_groups 需要同名 json 就 fail fast)
    task_py = workdir / "input" / f"{op}.py"
    if not task_py.is_file():
        src = Path(args.task_path) if args.task_path else task_py
        if src.is_file():
            task_py.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, task_py)
    if not task_py.is_file():
        raise FileNotFoundError(f"task file missing: {task_py}")
    if "get_input_groups" in task_py.read_text(encoding="utf-8", errors="replace") \
            and not (workdir / "input" / f"{op}.json").is_file():
        raise FileNotFoundError(f"required case file missing: input/{op}.json")
    _synthesize_case_json(workdir, op, task_py)

    # 1. 复刻件
    claude_dir = workdir / ".claude"
    _copy_tree(canonical / "skills", claude_dir / "skills")
    _copy_tree(canonical / "agents", claude_dir / "agents")
    _copy_tree(canonical / "hooks", claude_dir / "hooks")
    _copy_tree(canonical / "workflows", claude_dir / "workflows")
    _copy_file(canonical / "CLAUDE.md", workdir / "CLAUDE.md")

    # 2. settings.json:官方 init.sh 的生成语义(hooks.json 里的 ${CLAUDE_PLUGIN_ROOT} → 绝对路径)
    template = (canonical / "settings.json.template").read_text(encoding="utf-8")
    rendered = template.replace("${CLAUDE_PLUGIN_ROOT}", str(claude_dir))
    settings_path = claude_dir / "settings.json"
    settings_path.write_text(rendered, encoding="utf-8")
    json.loads(rendered)  # 渲染结果必须是合法 JSON
    print(f"{MARKER} settings.json rendered (CLAUDE_PLUGIN_ROOT={claude_dir})")

    # 3. R2 卡池接线(tools/ 只放租约两件,不放任何我方评测脚本——t3a 没有固定入口)
    tools = workdir / "tools"
    npu_wrap = tools / "npu_wrap.sh"
    if not args.readonly_tools:
        tools.mkdir(parents=True, exist_ok=True)
        _copy_file(canonical / "tools" / "npu_lease_exec.py", tools / "npu_lease_exec.py")
        _copy_file(canonical / "tools" / "npu_wrap.sh", npu_wrap)
        npu_wrap.chmod(npu_wrap.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # readonly 模式由 topology 挂载 tools;不复制、不 chmod,缺件仍必须失败。
    for name in ("npu_lease_exec.py", "npu_wrap.sh"):
        if not (tools / name).is_file():
            raise FileNotFoundError(f"t3a tools 缺文件: {tools / name}")
        if not os.access(tools / name, os.R_OK):
            raise PermissionError(f"t3a tools 不可读: {tools / name}")

    # 3b. judge 侧接线(只在 eval prepare,即不带 --require-claude 时):
    #   把 canonical/judge/(我方判分链 + 判优驱动)铺进 workdir/judge/,并把租约执行器一并带上。
    #   agent 侧(--require-claude)绝不铺——judge 文件不进 agent 可见路径。
    if not args.require_claude:
        judge_dir = workdir / "judge"
        _copy_tree(canonical / "judge", judge_dir)
        _copy_file(canonical / "tools" / "npu_lease_exec.py", judge_dir / "npu_lease_exec.py")
        jb = judge_dir / "judge_best.sh"
        jb.chmod(jb.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"{MARKER} judge side wired: {judge_dir}")

    # 4. 向上游路径契约 fail fast(防"210 行知识静默消失"类事故;清单即复刻关键路径)
    probes = [
        claude_dir / "agents" / "tilelang2ascendc-kernel-generator.md",
        claude_dir / "hooks" / "skill_script_hook.py",
        claude_dir / "hooks" / "doc_gate.py",
        claude_dir / "skills" / "tilelang2ascend-translator" / "scripts" / "evaluate_ascendc.sh",
        claude_dir / "skills" / "tilelang2ascend-translator" / "scripts" / "verification_ascendc.py",
        claude_dir / "skills" / "tilelang2ascend-translator" / "scripts" / "build_ascendc.py",
        claude_dir / "workflows" / "templates" / "archive_tasks",
        tools / "npu_wrap.sh",
    ]
    missing = [str(p) for p in probes if not p.exists()]
    if missing:
        raise FileNotFoundError(f"t3a workdir 契约断裂: {missing}")
    if not os.access(tools / "npu_wrap.sh", os.X_OK):
        raise PermissionError("tools/npu_wrap.sh 不可执行")

    if args.require_claude and shutil.which("claude") is None:
        raise FileNotFoundError("required executable missing on PATH: claude")

    print(f"{MARKER} done: skills={len(list((claude_dir / 'skills').iterdir()))} "
          f"agents={len(list((claude_dir / 'agents').iterdir()))} "
          f"hooks={len(list((claude_dir / 'hooks').iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
