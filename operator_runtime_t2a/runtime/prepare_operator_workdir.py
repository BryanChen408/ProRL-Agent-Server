#!/usr/bin/env python3
"""Prepare a per-session operator workdir.

This script prepares one session workdir from canonical assets mounted at
/opt/canonical and creates a minimal editable ModelNew starter file.
"""

from __future__ import annotations

import argparse
import ast
import json
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


def _assert_upstream_paths(workdir: Path) -> None:
    """Fail fast when the doc-referenced paths don't resolve in the prepared workdir.

    Layout mirrors what upstream `init.sh` produces — everything flat under `.claude/`:
    `.claude/{skills,workflows}`.  Skill references use `../../../workflows/...`, which from
    `.claude/skills/<skill>/references/` lands on `.claude/` — same as post-install.

    Nothing in the repo cross-checks doc paths against the real tree, so a layout change
    upstream would otherwise only surface when a rollout trips over it.
    """
    probes = [
        # Phase 1.2 的固定动作要 cp 的文件
        workdir / ".claude/skills/tilelang2ascend-operator-project-init"
        / "templates/ascend-kernel/csrc/utils/torch_kernel_helper.h",
    ]
    if (workdir / ".claude" / "workflows").is_dir():
        probes.append(workdir / ".claude/workflows/templates/archive_tasks")
    for probe in probes:
        if not probe.exists():
            raise FileNotFoundError(
                f"upstream path contract broken: {probe} 不可读 —— "
                f"canonical 布局变了,或 CLAUDE.md 引用的路径与实际不符。"
            )


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # prepare 的 upload_file 可能已经把源文件直接放到目标位置(ascendc: input/{op}.py),
    # 此时 src 就是 dst,shutil.copy2 会抛 SameFileError → 视作"已就位"直接返回。
    # triton 路径 src(canonical/…)恒 ≠ dst(workdir/…),行为一字不变。
    try:
        if dst.exists() and src.samefile(dst):
            return
    except OSError:
        pass
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


# Claude Code CLI 自带(bundled)的 skill —— 不来自 canonical/skills,与写算子无关:
# 每个 session 白占 ~4.3k 字符 prompt,还可能被模型误调(WebSearch/WebFetch 已 disallow,调了必失败)。
# 清单实测自镜像里的 claude 2.1.168(session 首条 init 消息 + prompt 里的 skill listing)。
# ⚠️ 这是"名单"不是"规则":CLI 只支持 skillOverrides={精确名: off},无通配符、无 allowlist
#    (sessionSkillAllowlist 只对 Agent SDK 开放,`claude -p` 够不到)。CLI 升级新增 bundled skill
#    时这里要跟着补 —— 巡检办法:读 session 的 logs/agent/claude-code.txt 首行 init 的 slash_commands,
#    出现不在本表也不在 canonical/skills 里的名字即为新增。
# ⭐ 根治:镜像里的 claude 升到 2.1.216+,那版支持 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1
#    (官方描述:bundled skills 整体移除,.claude/skills/ 不受影响)——一条规则、零名单。
#    profile.ascendc.yaml 已经把该 env 配好,升级镜像即自动接管。
_CLI_BUNDLED_SKILLS = (
    "deep-research",
    "update-config",
    "keybindings-help",
    "verify",
    "code-review",
    "simplify",
    "fewer-permission-prompts",
    "loop",
    "claude-api",
    "run",
    "init",
    "review",
    "security-review",
)


def _non_project_skills(canonical: Path) -> list[str]:
    """规则:凡不是本项目(canonical/skills)提供的 CLI 自带 skill,一律关掉。

    以 canonical/skills 的实际内容为准 —— 往 canonical 里加/删 skill 自动跟随,
    同名时(比如哪天我们自己也叫 review)以本项目的为准,不会被误关。
    """
    ours = {p.name for p in (canonical / "skills").iterdir() if p.is_dir()}
    return [name for name in _CLI_BUNDLED_SKILLS if name not in ours]


def _write_skill_overrides(workdir: Path, names: list[str]) -> None:
    """把 CLI 自带的 skill 从"模型可见清单"里摘掉(Claude Code projectSettings.skillOverrides)。

    `{name: "off"}` = 既不列进 prompt 也不允许模型调用(claude 2.1.168 起支持,
    实测 schema: skillOverrides: Record<str, on|name-only|user-invocable-only|off>)。
    更高版本另有 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1 可一把关(2.1.168 不认)。
    **不传 --skill-overrides-off 就一个字节都不写**,triton 路径与旧 topology 行为不变。
    """
    if not names:
        return
    settings_path = workdir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if settings_path.is_file():  # 别覆盖别人写的 hooks/permissions
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict):
            data = loaded
    overrides = data.get("skillOverrides")
    if not isinstance(overrides, dict):
        overrides = {}
    overrides.update({name: "off" for name in names})
    data["skillOverrides"] = overrides
    settings_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _prepare_ascendc_workdir(args) -> int:
    """AscendC 分支(与 triton 路径完全并行,triton 逻辑不受影响)。

    - input/{op}.py(+ 同名 .json;NPUKernelBench 的 get_input_groups 需读同名 .json)
    - canonical 的 tools/ + skills/(→ .claude/skills,对齐 init.sh 约定)+ CLAUDE.md
    - 不写 triton stub(agent 自己按 skill 建 {op}/kernel/)
    canonical(--canonical-root)= operator_runtime_ascendc(含 skills/ CLAUDE.md tools/)。
    """
    workdir = Path(args.workdir)
    canonical = Path(args.canonical_root)
    op = args.op_name
    for rel in ("input", "output/submission", "judge_out"):
        (workdir / rel).mkdir(parents=True, exist_ok=True)

    task_path = Path(args.task_path or (workdir / "input" / f"{op}.py"))
    if not task_path.is_file():
        raise FileNotFoundError(f"task file missing: {task_path}")
    _copy_file(task_path, workdir / "input" / f"{op}.py")
    json_src = task_path.with_suffix(".json")
    json_dst = workdir / "input" / json_src.name
    if json_src.is_file():
        _copy_file(json_src, json_dst)
    # NPUKernelBench 的 model.py 用 get_input_groups() 读同名 .json(用例规格)。
    # 缺了这个文件 agent 能写完整个 kernel、judge 侧 verification 才炸(烧掉整轮预算才暴露)
    # → 在 prepare 就 fail fast,错误明确指向缺件而不是"对拍失败"。
    if not json_dst.is_file():
        task_text = task_path.read_text(encoding="utf-8", errors="replace")
        if "get_input_groups" in task_text:
            raise FileNotFoundError(
                f"required case file missing: {json_dst} "
                f"({task_path.name} 用 get_input_groups() 读同名 .json)"
            )

    _prepare_tools(canonical, workdir, readonly_tools=args.readonly_tools)  # tools/ 可能是只读 bind mount
    if not (canonical / "skills").is_dir():
        raise FileNotFoundError(f"required directory missing: {canonical / 'skills'}")
    _copy_tree(canonical / "skills", workdir / ".claude" / "skills")
    # 与 install 产物同构:.claude/{agents,skills,workflows}。
    # agents/ 放的是 ops-direct-invoke 那 4 个(architect / design-reviewer / developer /
    # reviewer),简单算子路径靠它们产出 DESIGN/PLAN/WALKTHROUGH/REVIEW。
    # **不放我们自己那份** —— CLAUDE.md 已经是它,再注册一遍会被 claude-code 当成可调
    # subagent,平白多一层嵌套。
    for sub in ("agents", "workflows"):
        if (canonical / sub).is_dir():
            _copy_tree(canonical / sub, workdir / ".claude" / sub)
    _assert_upstream_paths(workdir)
    _copy_file(canonical / "CLAUDE.md", workdir / "CLAUDE.md")
    off_names = _non_project_skills(canonical) if args.only_project_skills else []
    off_names += [
        n.strip()
        for n in (args.skill_overrides_off or "").split(",")
        if n.strip() and n.strip() not in off_names
    ]
    if off_names:
        print(f"[prepare] skillOverrides off ({len(off_names)}): {','.join(off_names)}")
    _write_skill_overrides(workdir, off_names)

    if args.require_claude and shutil.which("claude") is None:
        raise FileNotFoundError("required executable missing on PATH: claude")
    return 0


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
        "--backend", default="triton", choices=["triton", "ascendc"],
        help="triton(默认,原样不变)| ascendc(NPUKernelBench input/{op}.py+.json + canonical skills/CLAUDE.md,不写 triton stub)",
    )
    parser.add_argument(
        "--only-project-skills",
        action="store_true",
        help="规则:只保留 canonical/skills 提供的 skill,CLI 自带的一律 skillOverrides=off"
             "(既不列进 prompt 也不许模型调用)。不传=不写 settings.json,行为不变。仅 ascendc 分支。",
    )
    parser.add_argument(
        "--skill-overrides-off",
        default="",
        help="额外要关的 skill 名(逗号分隔),追加在 --only-project-skills 之上;"
             "CLI 升级新增 bundled skill 时可先用它兜住,不必改代码。",
    )
    parser.add_argument(
        "--readonly-tools",
        action="store_true",
        help="Do not copy canonical tools into workdir; expect workdir/tools to be a read-only bind mount.",
    )
    args = parser.parse_args(argv)

    if not SAFE_NAME.fullmatch(args.op_name):
        raise SystemExit(f"unsafe op_name: {args.op_name!r}")

    if args.backend == "ascendc":
        return _prepare_ascendc_workdir(args)

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
