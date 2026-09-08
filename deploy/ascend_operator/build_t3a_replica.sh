#!/usr/bin/env bash
# build_t3a_replica.sh — 从 cannbot-skills 复刻官方 tilelang2ascendc-ops-generator 环境,
# 打 R1-R4 接线补丁,产出 REPLICATION_LEDGER.md 对账表(白名单外零 diff 证明无夹带)。
# 可重跑:上游同步时改 SRC 或重新执行即可。
set -euo pipefail

SRC="${CANNBOT_SRC:-/home/docker/cannbot-skills}"
DST="${T3A_DST:-/home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a}"
PLUGIN="$SRC/plugins-community/tilelang2ascendc-ops-generator"
PIN="$(git -C "$SRC" rev-parse HEAD)"

SKILLS_WHITELIST="npu-arch ascendc-api-best-practices ascendc-docs-search ascendc-tiling-design \
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
cp "$ASSETS/hooks/t3a_running_best.py" "$DST/hooks/"
mkdir -p "$DST/judge"
# judge 侧 pipeline 在租约模式下找 ${_SCRIPT_DIR}/npu_lease_exec.py(= judge/),
# 缺它时 verify 的 python 报错会被误判成 ascendc_run_crashed(acos 冒烟实证);
# 且本行必须在 mkdir 之后(rm -rf 重建时 judge/ 尚不存在,提前 cp 直接中断构建)。
cp /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/tools/npu_lease_exec.py "$DST/judge/"
cp /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t2a/tools/{ascendc_eval_pipeline.sh,pack_submission.sh,env.sh,detect_stateful_impl.py,check_op_registered.py} "$DST/judge/"
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
        if _t3a_cls is None and "verification_ascendc.py" in command:
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
                f"--lock-dir {_shlex.quote(lock_dir)} -- {command}"
            )
    return command


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
            ["bash", "-c", command], shell=False, cwd=cwd,
            capture_output=True, text=True, timeout=1800,
            env=_npu_wrap_env(command),
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""'''
assert old in t, "R3 旧块未匹配(官方源变了?)"
t = t.replace(old, new, 1)
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
