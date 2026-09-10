#!/usr/bin/env bash
# build_t3a_replica.sh — 从 cannbot-skills 复刻官方 tilelang2ascendc-ops-generator 环境,
# 打 R1-R4 接线补丁,产出 REPLICATION_LEDGER.md 对账表(白名单外零 diff 证明无夹带)。
# 可重跑:上游同步时改 SRC 或重新执行即可。
set -euo pipefail

SRC="${CANNBOT_SRC:-/home/docker/cannbot-skills}"
DST="${T3A_DST:-/home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a}"
PLUGIN="$SRC/plugins-community/tilelang2ascendc-ops-generator"
PIN="$(git -C "$SRC" rev-parse HEAD)"

SKILLS_WHITELIST="npu-arch ascendc-crash-debug ascendc-api-best-practices ascendc-docs-search ascendc-tiling-design \
tilelang2ascend-case-simplifier tilelang2ascend-operator-project-init ops-profiling \
ascendc-precision-debug tilelang2ascend-precision-tuning tilelang2ascend-tilelang-designer \
tilelang2ascend-translator tilelang2ascend-trace-recorder \
tilelang-op-design tilelang-op-develop tilelang-perf-optimization ascendc-perf-optimize"

echo "== [1/6] 铺复刻件(pin ${PIN:0:12})"
rm -rf "$DST"
mkdir -p "$DST/skills" "$DST/agents" "$DST/hooks" "$DST/tools" "$DST/runtime"

for name in $SKILLS_WHITELIST; do
  if [ -d "$SRC/ops/$name" ]; then cp -r "$SRC/ops/$name" "$DST/skills/$name";
  elif [ -d "$PLUGIN/skills/$name" ]; then cp -r "$PLUGIN/skills/$name" "$DST/skills/$name";
  else echo "FATAL: skill $name 在官方仓两个位置都不存在" >&2; exit 1; fi
done
# init.sh 的安装动作之一:attention-patterns 从 translator 复制到 designer(官方安装语义)
if [ -d "$DST/skills/tilelang2ascend-translator/references/attention-patterns" ]; then
  mkdir -p "$DST/skills/tilelang2ascend-tilelang-designer/references"
  rm -rf "$DST/skills/tilelang2ascend-tilelang-designer/references/attention-patterns"
  cp -r "$DST/skills/tilelang2ascend-translator/references/attention-patterns" \
        "$DST/skills/tilelang2ascend-tilelang-designer/references/attention-patterns"
fi

cp "$PLUGIN/agents/tilelang2ascendc-kernel-generator.md" "$DST/agents/"
# Match cannbot init.sh's frontmatter parsing; never silently ship a declared
# developer dependency without its Skill installation.
python3 - "$DST" <<'PYDEPS'
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
for agent in (root / "agents").glob("*.md"):
    frontmatter = agent.read_text().split("---", 2)[1]
    block = re.search(r"^skills:\n((?:\s+-\s+.+\n?)*)", frontmatter, re.MULTILINE)
    if block is None:
        raise SystemExit(f"Missing native skill dependency declaration: {agent}")
    for name in re.findall(r"^\s+-\s+(.+)$", block.group(1), re.MULTILINE):
        if not (root / "skills" / name / "SKILL.md").is_file():
            raise SystemExit(f"Missing native agent dependency: {name}")
PYDEPS
cp "$PLUGIN/hooks/"* "$DST/hooks/"
rm -rf "$DST/workflows"; cp -r "$PLUGIN/workflows" "$DST/workflows"
# CLAUDE.md = 官方 AGENTS.md 的 global 安装形态(workflows 引用重写为 canonical 绝对路径)
sed -e "s#bash workflows/scripts/#bash /opt/canonical/workflows/scripts/#g" \
    -e "s#](workflows/#](/opt/canonical/workflows/#g" \
    -e "s#\`workflows/#\`/opt/canonical/workflows/#g" \
    "$PLUGIN/AGENTS.md" > "$DST/CLAUDE.md"
# settings.json 模板:hooks.json 里 ${CLAUDE_PLUGIN_ROOT} 由 prepare 按 session 渲染
cp "$PLUGIN/hooks/hooks.json" "$DST/settings.json.template"
# Claude Code registers the agent by frontmatter name; align all dispatch/reentry references.
sed -i 's/ascend-kernel-developer/tilelang2ascendc-kernel-generator/g' \
  "$DST/CLAUDE.md" "$DST/workflows/development-guide.md" "$DST/workflows/task-prompts.md" \
  "$DST/hooks/session-start-tilelang2ascendc-ops-generator" \
  "$DST/skills/tilelang2ascend-translator/SKILL.md"

# R5 判分/judge 工具链整体复用 t2a 的(只在 judge 侧运行,不进 agent 可见路径时不打包进 workdir)
cp /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/tools/npu_lease_exec.py "$DST/tools/"

echo "== [1.5/6] 铺 RL 接线件(deploy/ascend_operator/t3a/ 为 source of truth)"
ASSETS=/home/docker/polar_can/ProRL-Agent-Server/deploy/ascend_operator/t3a
cp "$ASSETS/hooks/t3a_running_best.py" "$ASSETS/hooks/workflow_hook.py" "$DST/hooks/"
cp "$ASSETS/convert_task_prompts.py" "$DST/hooks/"
# Read-only completion checkpoint, using the existing attempt/candidate store.
python3 - "$DST/settings.json.template" <<'PYSTOP'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text())
s["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command":
    'python3 "${CLAUDE_PLUGIN_ROOT}/hooks/t3a_running_best.py" --stop'}]}]
workflow = {"type": "command", "command":
    'python3 "${CLAUDE_PLUGIN_ROOT}/hooks/workflow_hook.py"'}
s["hooks"]["UserPromptSubmit"] = [{"hooks": [workflow]}]
s["hooks"]["PreToolUse"] = [
    {"matcher": "Agent|Task|Skill|Bash|Write|Edit|MultiEdit", "hooks": [workflow]},
    *[group for group in s["hooks"]["PreToolUse"] if group.get("matcher") != "Bash"],
]
p.write_text(json.dumps(s, indent=2) + "\n")
PYSTOP
mkdir -p "$DST/judge"
# judge 侧 pipeline 在租约模式下找 ${_SCRIPT_DIR}/npu_lease_exec.py(= judge/),
# 缺它时 verify 的 python 报错会被误判成 ascendc_run_crashed(acos 冒烟实证);
# 且本行必须在 mkdir 之后(rm -rf 重建时 judge/ 尚不存在,提前 cp 直接中断构建)。
cp /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/tools/npu_lease_exec.py "$DST/judge/"
cp /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/tools/{ascendc_eval_pipeline.sh,pack_submission.sh,env.sh,detect_stateful_impl.py,check_op_registered.py} "$DST/judge/"
# The shared detector needs T2A's verified input binding helpers. Export their
# exact source for judge only; keep the native T3A agent skills unchanged.
python3 - "$DST/judge/input_contract.py" <<'PYCONTRACT'
import ast
from pathlib import Path
import sys
source = Path('/home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/skills/ops-profiling/scripts/msprof_perf_summary.py').read_text()
names = {'_find_cls', '_clone', '_move', '_seed_model', '_resolve_input_groups', '_forward_signature', '_bind_case'}
functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in functions} == names
Path(sys.argv[1]).write_text('# Generated from T2A ops-profiling; edit its source, not this copy.\nimport inspect\nfrom collections.abc import Mapping\n\n' + '\n\n'.join(ast.get_source_segment(source, n) for n in functions) + '\n')
PYCONTRACT
cp "$ASSETS/judge/judge_best.sh" "$ASSETS/judge/t3a_process_reward.py" "$DST/judge/"
chmod +x "$DST/judge/judge_best.sh"
mkdir -p "$DST/runtime"
cp "$ASSETS/runtime/prepare_operator_workdir.py" "$DST/runtime/"

echo "== [1.6/6] R5: skill_script_hook 接入 running-best 记录(is_eval_ascendc 判决落点)"
python3 - "$DST/hooks/skill_script_hook.py" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
old = '''    if is_eval_ascendc:
        classification = _classify_result(exit_code, stdout, stderr)
        output_dir = _extract_output_dir(command)
        _update_precision_gate(classification, stdout, output_dir)'''
new = '''    if is_eval_ascendc:
        classification = _classify_result(exit_code, stdout, stderr)
        output_dir = _extract_output_dir(command)
        _update_precision_gate(classification, stdout, output_dir)
    try:
        from t3a_running_best import record_attempt
        _t3a_script = next(
            (s for s in ("evaluate_ascendc.sh", "verification_ascendc.py",
                         "evaluate_tilelang.sh", "verification_tilelang.py",
                         "msprof_profile_run.sh") if s in command),
            "script")
        _t3a_cls = classification
        if _t3a_cls is None and any(s in command for s in ("verification_ascendc.py", "evaluate_tilelang.sh", "verification_tilelang.py")):
            _t3a_cls = _classify_result(exit_code, stdout, stderr)
        record_attempt(
            command=command, classification=_t3a_cls, exit_code=exit_code,
            stdout=stdout, duration_ms=duration_ms, cwd=cwd, project_root=PROJECT_ROOT,
            script=_t3a_script,
        )
    except Exception as _t3a_exc:
        print(f"[t3a-running-best] hook wiring failed: {_t3a_exc}", file=sys.stderr)'''
assert old in t, "R5 插入点未匹配(官方源变了?)"
t = t.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== [1.7/6] R-paths: 修 4 个跨 skill import 的路径反推(实体布局必断)"
python3 /home/docker/polar_can/ProRL-Agent-Server/deploy/ascend_operator/t3a/patch_rpaths.py "$DST"

echo "== [1.8/6] R6: skill_script_hook 补内联 -c 形态拦截(agent 自造 wrapper 丢判决,Abs 实证)"
python3 - "$DST/hooks/skill_script_hook.py" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
old = '''        if re.search(pattern, command):
            return True
    return False'''
new = '''        if re.search(pattern, command):
            return True
        # [R6] 内联 -c 形态：agent 遇到 import 故障时会用
        #   python3 -c "import sys; sys.path.insert(...); exec(open('.../verification_ascendc.py').read())"
        # 自造 wrapper 跑评测脚本，绕开上面的"解释器+脚本路径"主匹配 → 判决不入
        # attempt stream、最好的一次验证丢候选（Abs 冒烟实证）。解释器 + -c + 引用
        # 白名单脚本名，同样视为该脚本调用，由 hook 代跑并入流。
        # 读源码形态（cat/grep/sed/Read 工具）不带"解释器 -c"主调，不会误抓。
        inline = (
            _LEADING_PREFIX
            + interp
            + r"\\s+-c\\s+[\\"'][\\s\\S]*?"
            + script
        )
        if re.search(inline, command):
            return True
    return False'''
assert old in t, "R6 插入点未匹配(官方源变了?)"
t = t.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== [1.9/6] R8: 直接形态的 NPU 脚本调用外层包租约执行器(verification/tilelang 裸奔占卡,acos 实证)"
python3 - "$DST/hooks/skill_script_hook.py" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
anchor = "def execute_intercepted(command: str) -> None:"
defs = '''# [R8] 直接形态的 NPU 脚本调用外层包租约执行器:R3 注入的 NPU_WRAP 只有
# evaluate_ascendc.sh 读(R2b 补的);python3 verification_ascendc.py / evaluate_tilelang.sh
# 等直接调用时变量闲置、裸奔占卡(acos 冒烟实证)。hook 代跑时对这类脚本外层包租约;
# evaluate_ascendc.sh 自包含 NPU_WRAP,排除防嵌套租约死锁;validate/build 不碰 NPU 不包。
# 只替换"执行形态",record_attempt/判决块里展示的仍是 agent 原始命令(配对不受影响)。
_LEASE_WRAP_SCRIPTS = ("verification_ascendc.py", "verification_tilelang.py",
                       "evaluate_tilelang.sh", "msprof_profile_run.sh", "performance.py")


def _lease_wrap_command(command: str) -> str:
    import shlex as _shlex
    pool = os.environ.get("POLAR_NPU_LEASE_POOL")
    if not pool:
        return command
    if not any(s in command for s in _LEASE_WRAP_SCRIPTS):
        return command
    for cand in (os.path.join(PROJECT_ROOT or "", "tools", "npu_lease_exec.py"),
                 os.path.join(os.getcwd(), "tools", "npu_lease_exec.py")):
        if os.path.isfile(cand):
            lock_dir = os.environ.get("POLAR_NPU_LOCK_DIR", "/dev/shm/npu-locks")
            return (
                f"python3 {_shlex.quote(cand)} --pool {_shlex.quote(pool)} "
                f"--lock-dir {_shlex.quote(lock_dir)} -- bash -o pipefail -c {_shlex.quote(command)}"
            )
    raise FileNotFoundError("NPU lease configured but tools/npu_lease_exec.py is missing")


'''
assert anchor in t, "R8 定义插入点未匹配(官方源变了?)"
t = t.replace(anchor, defs + anchor, 1)
old_call = "exit_code, stdout, stderr, duration_ms = _run_command(command, cwd)"
new_call = "exit_code, stdout, stderr, duration_ms = _run_command(_lease_wrap_command(command), cwd)"
assert old_call in t, "R8 调用点未匹配(官方源变了?)"
t = t.replace(old_call, new_call, 1)
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== [2/6] R2a: agent md 删 SOC 检测优先级 2/3(npu-smi 在共享卡池是违禁品)"
python3 - "$DST/agents/tilelang2ascendc-kernel-generator.md" <<'PY'
import sys, re
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
t = t.replace('- 设置环境变量 `ASCEND_RT_VISIBLE_DEVICES=${npu}`',
              '- 共享卡池模式下，`npu` 参数不用于绑定设备；设备仅由租约执行器在运行时分配，禁止自行设置 `ASCEND_RT_VISIBLE_DEVICES`。')
start = t.index('#### 优先级 2')
end = t.index('**存储**')
t = t[:start] + '''> 本环境为共享卡池:`SOC_VERSION` 已由环境注入(优先级 1 恒命中),**禁止运行 npu-smi、
> 禁止自设 `ASCEND_RT_VISIBLE_DEVICES`**(多容器共享卡池,自设会抢走别人的卡)。

''' + t[end:]
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== 派发契约: 租约选卡、已精简用例、取消 agent Phase 6"
python3 - "$DST" <<'PROMPTPY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
replacements = {
    "workflows/task-prompts.md": {
        "- NPU 设备: {npu}": "- NPU 设备: 由共享卡池租约执行器分配，不指定卡号",
        "Phase 0: 参数确认 — 解析参数，设置 ASCEND_RT_VISIBLE_DEVICES={npu}":
            "Phase 0: 参数确认 — 解析任务路径和用例范围；NPU 操作走租约执行器，禁止自行设置 ASCEND_RT_VISIBLE_DEVICES 或运行 npu-smi",
        "Phase 2: Case 精简 — 调用 tilelang2ascend-case-simplifier 精简测试用例":
            "Phase 2: Case 检查 — 若任务已指定精简用例集，备份并原样保留全部输入用例；否则调用 tilelang2ascend-case-simplifier",
        "Phase 5: 性能分析 — 调用 ops-profiling（--compare 模式）":
            "Phase 5: 性能分析 — 调用 ops-profiling（--quick 模式）",
    },
    "agents/tilelang2ascendc-kernel-generator.md": {
        "| NPU 设备 | 通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量设置 |":
            "| NPU 设备 | 仅由共享卡池租约执行器分配和注入，禁止 agent 自行设置 |",
        "## Phase 2: 测试用例精简":
            "## Phase 2: 测试用例精简\n\n若任务明确指定已精简的 simple 用例集：只核对并备份工作目录 JSON，保留所提供的全部 5 条，不再执行下面的精简操作；然后进入 Phase 3。simple 不改变算子分类路由。",
    },
    "CLAUDE.md": {},
    "workflows/development-guide.md": {},
}
for relative, changes in replacements.items():
    path = root / relative
    text = path.read_text()
    for old, new in changes.items():
        if new in text:
            continue
        if old not in text:
            raise ValueError(f"T3A prompt patch anchor missing: {relative}: {old}")
        text = text.replace(old, new, 1)
    if relative == "agents/tilelang2ascendc-kernel-generator.md":
        # Remove the old transition rules and the entire final-validation stage.
        start = "**Phase 5 → Phase 6 → Phase 7 流转规则（不可跳过）**"
        if start in text:
            end = text.index("## Phase 7: Trace 记录", text.index(start))
            text = text[:text.index(start)] + (
                "**Phase 5 → Phase 7 流转规则**：\n"
                "无论性能分析成功或失败，都直接进入 Phase 7 生成 trace.md。\n\n---\n\n"
            ) + text[end:]
        text = text.replace("继续 Phase 6", "继续 Phase 7")
    # All four entry points must agree; this also removes obsolete error-table rows.
    text = re.sub(r"^(?!本 RL 流程取消 Phase 6).*Phase 6[^\n]*\n", "", text, flags=re.MULTILINE)
    notice = ("本 RL 流程取消 Phase 6：Phase 5 后直接进入 Phase 7 记录 trace，"
              "agent 不恢复用例或追加最终验证；独立 judge 使用 input/ 中的完整任务用例验收候选。\n")
    if notice not in text:
        text += "\n" + notice
    # Use the installed skill paths in the native developer's Phase 1.2.
    text = text.replace(
        "plugins-community/tilelang2ascendc-ops-generator/skills/",
        "/opt/workspace/agent_workdir/.claude/skills/",
    )
    contract = """## T3A 执行契约（优先于下文通用模板）

- 主 agent 只调度和检查；禁止主链初始化工程或写实现。只派发已注册的
  `tilelang2ascendc-kernel-generator`，其他设计/转译组件用 Skill 调用，不能当 Agent 类型。
- 每次派发/恢复必须原样传递 input 参考路径、用例策略、输出根目录、租约策略和取消末尾验证阶段的策略。
- 开发子 agent 按自身 Phase 1.2 创建 `{output_dir}/kernel/`，复用已安装 project-init
  模板中的固定文件；不要执行 standalone project-init 的 `ascend-kernel/csrc/ops/` 布局。
  model.py、model_new_tilelang.py、model_new_ascendc.py 和 design/ 均位于同一个 output_dir。
- simple 数据集保留输入的全部 5 条用例，不调用 case-simplifier；simple 不是算子分类。
- TileLang 验证是中间步骤，不能代替 Phase 4 AscendC 实现和 evaluate_ascendc.sh。
  子 agent 提前返回时，从未完成阶段恢复同一开发子 agent；禁止把 stub 或 TileLang 产物宣称为完成。
- 完成前读取真实 AscendC 评测结果和 trace.md，并检查 kernel/ 与 model_new_ascendc.py。
  失败按真实失败报告；独立 judge 负责最终验收，不能拿 judge 替代开发阶段。
- 本 RL 场景 pipeline/verify 不按次数终止，统一受外部 pipeline 时间预算约束。
  下文重试数字仅用于诊断分阶段建议，不构成次数上限；不得因达到次数提前转 Phase 7 或请求用户。
  同错反复出现应更换有证据的修复策略；D1/D2 的诊断和文档门禁继续保留。
- 禁止自行设置设备可见性；NPU 调用仅通过现有租约评测脚本。

"""
    if contract not in text:
        # Preserve agent frontmatter registration.
        pos = text.index("\n---", 4) + 4 if text.startswith("---\n") else 0
        text = text[:pos] + "\n\n" + contract + text[pos:]
    path.write_text(text)

# The standalone init skill belongs to a different project layout. Reuse the
# native developer's exact initialization section instead of inventing another.
p = root / "skills/tilelang2ascend-operator-project-init/SKILL.md"
original = p.read_text()
frontmatter = original.split("---", 2)[1]
agent = (root / "agents/tilelang2ascendc-kernel-generator.md").read_text()
initialization = agent.split("### 1.2 初始化 kernel 工程", 1)[1].split("### 1.3", 1)[0]
p.write_text("---" + frontmatter + "---\n\n# T3A 工程初始化\n\n"
             "仅开发子 agent 执行；主 agent 应派发 tilelang2ascendc-kernel-generator。\n"
             "以下逐字复用开发子 agent Phase 1.2；固定模板保留在本 skill 的 templates/。\n"
             + initialization.rstrip() + "\n")
PROMPTPY

echo "== [3/6] R2b: evaluate_ascendc.sh 设备获取改租约注入"
EVAL="$DST/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh"
python3 - "$EVAL" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
# 不写死默认卡号:卡由 npu_lease 注入;裸跑(无租约环境)时与原默认 3 一致但可覆盖
t = t.replace(
  'ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-3}"',
  '# [R2b] 卡由租约注入,不再写死默认值(原默认 3 会让多容器互抢)\n'
  'ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-}"')
# verify 调用包租约执行器(若环境提供);npu_wrap.sh 无租约配置时直通
t = t.replace(
  '  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \\\n'
  '  python .claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py "${TASK_DIR}" ${NON_COMPUTE_FLAG}',
  '  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \\\n'
  '  "${NPU_WRAP:-python}" .claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py "${TASK_DIR}" ${NON_COMPUTE_FLAG}')
open(p, 'w', encoding='utf-8').write(t)
PY
# 租约包装器(无 POLAR_NPU_LEASE_POOL 时直通,有则抢锁执行;由 hook/脚本以 NPU_WRAP 引用)
cat > "$DST/tools/npu_wrap.sh" <<'EOF'
#!/usr/bin/env bash
# [R2] 租约包装:设了 POLAR_NPU_LEASE_POOL 就抢锁执行,否则直通(与官方裸跑语义一致)
set -uo pipefail
if [[ -n "${POLAR_NPU_LEASE_POOL:-}" ]]; then
  exec python3 "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/npu_lease_exec.py" \
    --pool "$POLAR_NPU_LEASE_POOL" \
    --lock-dir "${POLAR_NPU_LOCK_DIR:-/dev/shm/npu-locks}" -- python "$@"
else
  exec python "$@"
fi
EOF
chmod +x "$DST/tools/npu_wrap.sh"
# evaluate_ascendc.sh 顶部注入 NPU_WRAP 默认指向(不改原脚本语义:无租约=python 直通)
python3 - "$EVAL" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
anchor = 'ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-}"'
inject = anchor + '''
# [R2b] NPU 步骤包装器:默认找 workdir/tools/npu_wrap.sh;不存在则退回 python 直通
if [[ -z "${NPU_WRAP:-}" ]]; then
  _wrap="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/npu_wrap.sh"
  [[ -x "$_wrap" ]] || _wrap="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/tools/npu_wrap.sh"
  [[ -x "$_wrap" ]] && NPU_WRAP="$_wrap" || NPU_WRAP="python"
fi'''
t = t.replace(anchor, inject, 1)
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== [4/6] R3: skill_script_hook 代跑注入 NPU_WRAP 环境变量(env 参数,非命令串前缀)"
python3 - "$DST/hooks/skill_script_hook.py" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
old = '''def _run_command(command: str, cwd: str):
    """Execute a shell command and return (exit_code, stdout, stderr, duration_ms)."""
    start_time = time.time()
    try:
        import shlex
        cmd_parts = shlex.split(command)
        proc = subprocess.run(
            cmd_parts, shell=False, cwd=cwd,
            capture_output=True, text=True, timeout=1800,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""'''
new = '''def _npu_wrap_env(command: str) -> dict:
    """[R3] 需要 NPU 的脚本注入 NPU_WRAP 环境变量(指向 workdir/tools/npu_wrap.sh)。
    经 env 参数传递而不是命令串前缀——前缀形式会被当成可执行文件名。
    无租约配置时 npu_wrap 直通,语义与官方裸跑一致。"""
    import os
    env = dict(os.environ)
    npu_scripts = ("evaluate_ascendc.sh", "verification_ascendc.py",
                   "evaluate_tilelang.sh", "verification_tilelang.py",
                   "msprof_profile_run.sh")
    if not any(s in command for s in npu_scripts):
        return env
    for cand in (os.path.join(os.getcwd(), "tools", "npu_wrap.sh"),
                 os.path.join(PROJECT_ROOT or "", "tools", "npu_wrap.sh")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            env["NPU_WRAP"] = cand
            break
    return env


def _run_command(command: str, cwd: str):
    """Execute a shell command and return (exit_code, stdout, stderr, duration_ms)."""
    start_time = time.time()
    try:
        # [R3b] bash -c 执行:export/&&/;/管道等复合命令形式全兼容
        # (原 shlex.split+shell=False 会把 export 当可执行文件 → FileNotFoundError)
        proc = subprocess.run(
            ["bash", "-o", "pipefail", "-c", command], shell=False, cwd=cwd,
            capture_output=True, text=True, timeout=1800,
            env=_npu_wrap_env(command),
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""'''
assert old in t, "R3 旧块未匹配(官方源变了?)"
t = t.replace(old, new, 1)
# A command inside the lease must not override its assigned device.
old_guard = "    if should_intercept(command):\n        execute_intercepted(command)"
new_guard = '''    if (os.environ.get("POLAR_NPU_LEASE_POOL") and should_intercept(command)
            and re.search(r"\\bASCEND_RT_VISIBLE_DEVICES\\s*=", command)):
        _protocol_logger.info(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": "删除命令中的 ASCEND_RT_VISIBLE_DEVICES 赋值；评测设备由租约分配。"
        }}, ensure_ascii=False))
        return
    if should_intercept(command):
        execute_intercepted(command)'''
assert old_guard in t
t = t.replace(old_guard, new_guard, 1)
# A successful sub-step cannot turn a failed shell pipeline into PASS.
assert 'if last_result == "pass":' in t
t = t.replace('if last_result == "pass":', 'if last_result == "pass" and exit_code == 0:', 1)
open(p, 'w', encoding='utf-8').write(t)
PY

echo "== [5/6] R4: doc_gate 文档类别适配这版 devkit(guide 类并入示例,API 类指扁平树)"
python3 - "$DST/hooks/doc_gate.py" <<'R4PY'
import sys, re
p = sys.argv[1]
t = open(p, encoding='utf-8').read()
NEW = '''
# ── foundational doc categories ──────────────────────────────
# [R4-v2] 这版 devkit(9.0.0/a2a3)没有 docs/guide/ 树:
#   - 「矢量编程流水线」「UB/TBuf-TQue」两类官方指南在本版不存在 → 标记 optional 且
#     patterns 与示例重叠(读示例即覆盖),不会演化成「永远覆盖不了 → 全拒」;
#   - 「API参考文档」的 patterns 改到这版实际的扁平树(docs/zh/api/、docs/api/);
#   - 「官方示例代码」路径本版存在,保持不变。
# 语义验证(模拟):0读=拒(预算0) / 只读示例=3/4 / 示例+API=4/4 预算50。
FOUNDATIONAL_CATEGORIES = {
    "矢量编程流水线指南": {
        "optional": True,
        "patterns": [
            "/examples/01_simd_cpp_api/",
        ],
        "desc": "CopyIn→Compute→CopyOut 正确流水线模式(本版 devkit 无独立指南,并入示例)",
        "example": "asc-devkit/examples/01_simd_cpp_api/README.md",
    },
    "UB缓冲区/TBuf-TQue管理指南": {
        "optional": True,
        "patterns": [
            "/examples/01_simd_cpp_api/",
        ],
        "desc": "UB 临时缓冲区与 TQue/TBuf 使用模式(本版 devkit 无独立指南,并入示例)",
        "example": "asc-devkit/examples/01_simd_cpp_api/README.md",
    },
    "API参考文档": {
        "patterns": [
            "/docs/zh/api/",
            "/docs/api/context/",
            "/docs/api/",
        ],
        "desc": "所用 API 的完整签名、dtype 支持矩阵、参数约束",
        "example": "asc-devkit/docs/zh/api/DataCopyPad(ISASI).md",
    },
    "官方示例代码": {
        "patterns": [
            "/asc-devkit/examples/01_simd_cpp_api/",
        ],
        "desc": "官方 CopyIn→Compute→CopyOut 完整 kernel 实现，验证 TQue 管道的正确用法",
        "example": (
            "asc-devkit/examples/01_simd_cpp_api/02_features/00_compilation/"
            "custom_op/op_kernel/add_custom/add_custom_kernel.cpp"
        ),
    },
}
'''
start = t.index('# ── foundational doc categories')
m = re.search(r'\n\}\n', t[start:])
assert m, 'FOUNDATIONAL_CATEGORIES block end not found'
t = t[:start] + NEW.strip() + '\n' + t[start + m.end():]
open(p, 'w', encoding='utf-8').write(t)
R4PY

# T3A is time-bounded. Keep D1/D2 routing and edit gates, without a terminal quota.
python3 - "$DST/hooks/doc_gate.py" <<'PYGATE'
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text()
start = t.index("    if total >= 12:\n")
end = t.index('    if stage == "D1":', start)
t = t[:start] + t[end:]
t = t.replace('stage_info = f"D-2 阶段: ascendc-precision-tuning, 已用 {used}/{max_calls} 次"',
              'stage_info = f"D-2 阶段: ascendc-precision-tuning, 已调用 {used} 次（不按次数终止）"')
p.write_text(t)
PYGATE

echo "== [6/6] R1: CLAUDE.md 尾部追加非交互与卡池禁令(最小覆盖)"
cat >> "$DST/CLAUDE.md" <<'EOF'

---

## 本环境覆盖(RL 接线,仅三条)

- 这是非交互运行:**禁止向用户提问**,没有人会回答;按当前信息继续推进即可。
- **禁止自设 `ASCEND_RT_VISIBLE_DEVICES`、禁止运行 `npu-smi`**(多容器共享卡池,自设会抢走别人的卡;`SOC_VERSION` 已在环境变量里)。
- NPU 使用一律经评测/验证脚本走(内部由租约调度);不要手写占卡探针。
- 文档检索一律**从 `$ASC_DEVKIT_DIR` 根目录用 Glob/Grep 搜**,不要按文档里的字面路径直接 open——本环境 devkit 的 docs/ 是扁平树(docs/zh/api/ 与 docs/api/context/ 两棵),且 API 页编号会变(如 ReduceSum-90 实为 ReduceSum-34);examples/ 在 examples/01_simd_cpp_api/ 下。搜不到就换关键词(去编号、换同义词),不要下「文档不存在」的结论。

EOF

# ---- 对账表 ----
echo "== 生成 REPLICATION_LEDGER.md"
python3 - "$SRC" "$DST" "$PIN" "$SKILLS_WHITELIST" <<'PY'
import subprocess, sys, os
src, dst, pin, wl = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
plugin = f"{src}/plugins-community/tilelang2ascendc-ops-generator"
lines = [f"# REPLICATION_LEDGER — 复刻对账表", "",
         f"- 源: {src} @ `{pin}`",
         f"- 目标: {dst}", "",
         "## 白名单补丁点(允许存在的全部差异)",
         "- `agents/tilelang2ascendc-kernel-generator.md`:R2a 删 SOC 优先级 2/3(npu-smi 违禁)",
         "- `skills/tilelang2ascend-tilelang-designer/references/attention-patterns`:官方 init.sh 的安装动作(从 translator 复制)",
         "- `skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh`:R2b 设备获取改租约",
         "- `hooks/skill_script_hook.py`:R3 代跑外包租约",
         "- `hooks/doc_gate.py`:R4 文档路径重映射+降级",
         "- `CLAUDE.md`:global 形态路径重写(官方 init.sh 语义)+ R1 尾部三条覆盖",
         "- `settings.json.template` ← hooks/hooks.json(按 init.sh 的 settings 生成语义)",
         "- `tools/npu_lease_exec.py`、`tools/npu_wrap.sh`:RL 卡池接线(我方件)",
         "- `hooks/t3a_running_best.py`:R5 attempt stream+running-best(我方件)",
         "- `hooks/skill_script_hook.py` 的 record_attempt 调用点:R5 接线(白名单内)",
         "- `hooks/skill_script_hook.py` 的 should_intercept 内联 -c 匹配:R6(python -c 自造 wrapper 丢判决,Abs 冒烟实证)",
         "- `hooks/skill_script_hook.py` 的 _lease_wrap_command:R8(直接形态 NPU 脚本外层包租约,acos 冒烟实证)",
         "- `judge/`(t2a 判分链脚本 + judge_best.sh):R5 judge 侧接线(我方件,不进 agent workdir)",
         "- `runtime/prepare_operator_workdir.py`:t3a prepare(我方件,CLI 与 t2a 同形)",
         "- T3A 执行契约: 原生 Phase 1.2 工程布局、安装路径、派发策略、时间预算覆盖; doc_gate 仅去除次数终止",
         "- Stop: 复用 t3a_running_best 只读检查已有 AscendC 候选和 trace; 不追加评测",
         "- 4 个脚本的 R-paths 补丁(verification_ascendc/validate_ascendc_impl/verification_tilelang/validate_tilelang_impl):跨 skill import 改向上搜/内联",
         "", "## 逐件对账", ""]
bad = 0
def diff_dir(src_dir, dst_dir, label):
    global bad
    r = subprocess.run(["diff", "-rq", src_dir, dst_dir], capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.strip()]
    status = "✅ 一致" if not out else f"⚠️ {len(out)} 处差异(应全部在白名单内)"
    if out: bad += len(out)
    lines.append(f"- **{label}**: {status}")
    for l in out[:6]:
        lines.append(f"    - `{l}`")

for name in wl:
    src_dir = f"{src}/ops/{name}" if os.path.isdir(f"{src}/ops/{name}") else f"{plugin}/skills/{name}"
    diff_dir(src_dir, f"{dst}/skills/{name}", f"skills/{name}")
diff_dir(f"{plugin}/agents", f"{dst}/agents", "agents/(白名单:R2a)")
diff_dir(f"{plugin}/hooks", f"{dst}/hooks", "hooks/(白名单:R3/R4)")
diff_dir(f"{plugin}/workflows", f"{dst}/workflows", "workflows/")
lines += ["", f"**差异总数 {bad}(每一行都应能对应到白名单某一条;对不上的=夹带或漏拷,必须清零)。**"]
open(f"{dst}/REPLICATION_LEDGER.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[:20]))
print(f"... 完整对账表: {dst}/REPLICATION_LEDGER.md")
PY

echo "== 完成: $DST"
ls "$DST"
