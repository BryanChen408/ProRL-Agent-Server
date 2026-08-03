#!/usr/bin/env bash
# Tier-0 checks for the Polar DockerRuntime slime mainline. This script does
# not start training, rollout servers, gateways, or cleanup scripts.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# 设了 POLAR_PROFILE 就按它选 canonical 树。_paths.sh 的默认值恒为 operator_runtime,
# 不读 profile 会让 t2a run 的 preflight 静默检查 triton 树(flavor 判错,看着过了其实
# 检的是另一棵树)。这里只解析 paths.operator_runtime_dir,不调 load_polar_profile.py ——
# 那个 loader 会 mkdir run 目录并写 effective_topology.yaml,preflight 不该有这种副作用。
# 判定逻辑与 loader 的 _operator_runtime_dir() 一致(含 workflow=cannbot 的特例)。
# 不设 POLAR_PROFILE 时一字不变(triton 老用法照旧)。
if [[ -n "${POLAR_PROFILE:-}" ]]; then
  POLAR_OPERATOR_RUNTIME_DIR="$("${POLAR_PYTHON:-python3}" - \
      "${POLAR_PROFILE}" "${POLAR_REPO_ROOT}" <<'RUNTIMEDIR'
import sys
from pathlib import Path
import yaml

profile, repo = Path(sys.argv[1]), Path(sys.argv[2])
data = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
workflow = ((data.get("operator_runtime") or {}).get("workflow")) or "legacy"
if workflow == "cannbot":
    print((repo / "operator_runtime" / "cannbot").resolve())
else:
    configured = Path(((data.get("paths") or {}).get("operator_runtime_dir")) or "operator_runtime")
    print(configured if configured.is_absolute() else (repo / configured).resolve())
RUNTIMEDIR
  )"
  export POLAR_OPERATOR_RUNTIME_DIR
fi

ROOT="${POLAR_DEPLOY_DIR}"
POLAR_ROOT="${POLAR_ROOT:-${POLAR_REPO_ROOT}}"
SKILLS_DIR="${POLAR_SKILLS_DIR:-${POLAR_OPERATOR_RUNTIME_DIR}}"
DEFAULT_TASK_ASSETS="${ROOT}/fixtures/operator_assets"
TASKS_DIR="${OPERATOR_TASKS_DIR:-${POLAR_TASKS_DIR:-${DEFAULT_TASK_ASSETS}/op_tasks}}"
TASK_JSONL="${OPERATOR_TASK_JSONL:-${POLAR_TASK_JSONL:-${DEFAULT_TASK_ASSETS}/operator_tasks.jsonl}}"
SGLANG_ROOT="${SGLANG_ROOT:-/workspace/sglang/python/sglang}"
RUNTIME_IMAGE="${POLAR_OP_IMAGE:-sandbox:v1}"
DEVICE_POOL="${POLAR_DEVICE_POOL:-0}"
LOCK_DIR="${POLAR_LOCK_DIR:-/dev/shm/npu-locks}"
MODEL_SERVED="${POLAR_MODEL_SERVED:-model-served-placeholder}"
ROUTER_IP="${SGLANG_ROUTER_IP:-127.0.0.1}"
ROUTER_PORT="${SGLANG_ROUTER_PORT:-4077}"
READONLY_TOOLS_DIR="${POLAR_READONLY_TOOLS_DIR:-$(mktemp -d /tmp/polar-readonly-tools.XXXXXX)}"

log() { printf '[preflight] %s\n' "$*"; }

log "bash syntax"
bash -n "$0"
bash -n "${ROOT}/check_polar_runtime_image.sh"
bash -n "${POLAR_ROOT}/scripts/patch/patch_sglang.sh"

# canonical 树按自带的评测入口判型:ascendc 树没有 triton 资产,反之亦然。
if [[ -f "${SKILLS_DIR}/tools/ascendc_eval_pipeline.sh" ]]; then FLAVOR=ascendc; else FLAVOR=triton; fi
log "operator assets (flavor=${FLAVOR})"
python3 - "$SKILLS_DIR" "$TASKS_DIR" "$TASK_JSONL" "$FLAVOR" <<'PY'
import json
import re
import sys
from pathlib import Path

skills = Path(sys.argv[1])
tasks = Path(sys.argv[2])
jsonl = Path(sys.argv[3])
flavor = sys.argv[4]
required = [
    skills / "CLAUDE.md",
    skills / "tools" / "npu_lease_exec.py",
    skills / "runtime" / "prepare_operator_workdir.py",
]
if flavor == "ascendc":
    required += [
        skills / "tools" / "ascendc_eval_pipeline.sh",
        skills / "skills" / "tilelang2ascend-translator" / "SKILL.md",
        skills / "workflows" / "templates" / "archive_tasks",
    ]
else:
    required += [
        skills / "tools" / "triton_eval_pipeline.sh",
        skills / ".agents" / "skills" / "triton-op-verifier" / "scripts" / "verify.py",
        skills / ".agents" / "skills" / "triton-op-verifier" / "scripts" / "benchmark.py",
        skills / "skills" / "triton-op-verifier" / "SKILL.md",
    ]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing canonical asset(s): " + ", ".join(missing))
if not jsonl.is_file():
    raise SystemExit(f"missing task jsonl: {jsonl}")
row = json.loads(jsonl.read_text().splitlines()[0])
op = (row.get("metadata") or {}).get("op_name")
if not isinstance(op, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", op):
    raise SystemExit(f"unsafe/missing op_name in first row: {op!r}")
task = tasks / f"{op}.py"
if not task.is_file():
    raise SystemExit(f"missing task file for first row: {task}")
print(f"first_op={op}")
PY

log "operator tool syntax"
python3 - "$SKILLS_DIR" <<'PY'
import py_compile
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
for path in (
    root / "runtime" / "prepare_operator_workdir.py",
    root / "tools" / "npu_lease_exec.py",
):
    with tempfile.NamedTemporaryFile(suffix=".pyc") as tmp:
        py_compile.compile(str(path), cfile=tmp.name, doraise=True)
PY

if [[ "$FLAVOR" == ascendc ]]; then
log "CLAUDE.md 与上游一致(生成物比对)"
# CLAUDE.md = 上游 agent md + build_claude_md.py 里的声明式 delta + claude_override.md。
# 它是我们自己的文件,任何 skills diff 都覆盖不到 —— 一个月里 210 行上游原话就是这么静默消失的
# (双路径、4.5D 阶梯、解析用户输入…),直到人工读轨迹才发现。改动只能走 DELTA 或覆盖区。
python3 "${ROOT}/build_claude_md.py" --canonical "$SKILLS_DIR" --check

log "skill reference list up to date"
# CLAUDE.md 的「Skill 参考资料」块由 gen_skill_reference_list.py 从目录树生成。
# 上游那份是手写的、粒度不一(有的带 scripts/、多数只给裸文件名),而裸文件名配合
# 标题里的 .../skills/<name>/ 前缀会被读成"在 skill 根目录" —— 实测 agent 至少 10 次
# 因此解析错路径。生成 + 校验,把这类漂移挡在起 run 之前。
python3 "${ROOT}/gen_skill_reference_list.py" --canonical "$SKILLS_DIR" --check
fi

if [[ "$FLAVOR" == ascendc ]]; then
log "self-containment(拷贝后仍可解析)"
# 适配不变量:上游 init.sh 把 .claude/skills/<name> 建成软链指回 clone,脚本 .resolve()
# 落在仓库树里,可以按层数反推仓库根;我们是实拷进每个 session 的 workdir,那个前提不成立。
# 所以 canonical 树里任何东西都不许依赖"我还在 clone 里"。三条机械校验:
#   ① 硬编码 parents[N] 定位兄弟 skill  ② 残留软链(拷贝后必悬空)
# md 之间的相对引用不进闸门:大量是省略 .md 后缀的互链和占位名,误报盖过信号;
# 已知悬空的 6 条散文引用记在 MIGRATION_T2A.md。
python3 - "$SKILLS_DIR" <<'SELFCONTAINED'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
skills = root / "skills"
names = {p.name for p in skills.iterdir() if p.is_dir()}
bad = []

for script in skills.rglob("*.py"):
    if "evals" in script.parts:
        continue
    for m in re.finditer(r"parents\[\d+\][^\n]*", script.read_text(encoding="utf-8", errors="replace")):
        if any(f'"{n}"' in m.group(0) or f"'{n}'" in m.group(0) for n in names):
            bad.append(f"[parents] {script.relative_to(root)}: {m.group(0).strip()}")

for link in root.rglob("*"):
    if link.is_symlink():
        bad.append(f"[symlink] {link.relative_to(root)} -> {link.readlink()}")

if bad:
    sys.exit("canonical 树依赖了拷贝后不存在的东西:\n  " + "\n  ".join(bad))
SELFCONTAINED
fi

if [[ "$FLAVOR" == ascendc ]]; then
log "msprof invocation contract"
# 钉死 --repeats 1。msprof_perf_summary.py quick 模式:wrapper 在同一进程内先跑 repeats-1 次
# 预热再计时,采集到的总 kernel 时间除以 repeats。按输入做 memo 的实现(输出随输入变化,
# detect_stateful_impl.py 会正常放行)在同一输入上只发一次 kernel,却被除以 N → speedup 虚高 N 倍。
# repeats=1 时内部预热=0、除数=1,该放大不存在。这是 detect 覆盖不到、只有本约束挡得住的一类。
# --compare 不禁:它是 standard 模式(8 轮采 7 个 aic-metrics),不做 ÷repeats,与缓存无关;
# 我们只是用不上那些指标、且采集慢 8 倍,所以不用,但没有正确性理由去禁。
python3 - "$SKILLS_DIR" <<'MSPROFCONTRACT'
import re
import sys
from pathlib import Path

pipeline = Path(sys.argv[1]) / "tools" / "ascendc_eval_pipeline.sh"
if not pipeline.is_file():
    sys.exit(f"missing {pipeline}")
code = "\n".join(
    line for line in pipeline.read_text(encoding="utf-8").splitlines()
    if not line.lstrip().startswith("#")
)
bad = [m for m in re.findall(r"--repeats\s+(\S+)", code) if m != "1"]
if bad:
    sys.exit(f"ascendc_eval_pipeline.sh must pin --repeats 1, found: {bad}")
if "--repeats 1" not in code:
    sys.exit("ascendc_eval_pipeline.sh must pass --repeats 1 explicitly "
             "(relying on the upstream default lets it drift silently)")
MSPROFCONTRACT
fi

log "readonly operator tools"
python3 "${ROOT}/prepare_readonly_tools.py" \
  --source "${SKILLS_DIR}/tools" \
  --dest "${READONLY_TOOLS_DIR}"
bash -n "${READONLY_TOOLS_DIR}/$([[ "$FLAVOR" == ascendc ]] && echo ascendc_eval_pipeline.sh || echo triton_eval_pipeline.sh)"
python3 - "${READONLY_TOOLS_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for forbidden in ("fixtures", "test_fixtures", "tests", "self_check", "self-check", "__pycache__"):
    leaked = [path for path in root.rglob(forbidden)]
    if leaked:
        raise SystemExit(f"forbidden readonly tools content leaked: {leaked}")
print("readonly_tools=ok")
PY

log "gen_op_assets contract"
python3 - "$ROOT" <<'PY'
import importlib.util
import py_compile
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1]) / "gen_op_assets.py"
with tempfile.NamedTemporaryFile(suffix=".pyc") as tmp:
    py_compile.compile(str(path), cfile=tmp.name, doraise=True)

spec = importlib.util.spec_from_file_location("polar_gen_op_assets", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module._validate_op_name("kernelbench_l1_19_19_ReLU", row=0)
try:
    module._validate_op_name("../bad", row=0)
except ValueError:
    pass
else:
    raise SystemExit("unsafe op_name accepted")
PY

log "SGLang patch dry-run"
TMP_SGLANG="$(mktemp -d /tmp/polar-sglang-dryrun.XXXXXX)"
cp -a "${SGLANG_ROOT}/." "${TMP_SGLANG}/"
SGLANG_ROOT="${TMP_SGLANG}" bash "${POLAR_ROOT}/scripts/patch/patch_sglang.sh"
python3 - "$TMP_SGLANG" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
protocol = (root / "srt/entrypoints/openai/protocol.py").read_text()
serving = (root / "srt/entrypoints/openai/serving_chat.py").read_text()
if "token_id: int = 0" in protocol:
    raise SystemExit("unsafe token_id default remains in protocol.py")
if "token_id = logprobs.token_ids[token_idx] if token_idx < len(logprobs.token_ids) else 0" in serving:
    raise SystemExit("unsafe token_id fallback remains in serving_chat.py")
for needle in (
    "SGLang logprob token_id contract violated",
    "SGLang logprob token text contract violated",
):
    if needle not in serving:
        raise SystemExit(f"missing strict contract: {needle}")
print("sglang_patch_contract=ok")
PY

log "render TaskRequest and topology"
PYTHONPATH="${POLAR_ROOT}/src:${PYTHONPATH:-}" python3 "${ROOT}/check_render_contract.py" \
  --polar-root "${POLAR_ROOT}" \
  --config "${ROOT}/polar_config.yaml" \
  --topology "${ROOT}/topology.yaml" \
  --skills-dir "${SKILLS_DIR}" \
  --readonly-tools-dir "${READONLY_TOOLS_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --task-jsonl "${TASK_JSONL}" \
  --image "${RUNTIME_IMAGE}" \
  --device-pool "${DEVICE_POOL}" \
  --lock-dir "${LOCK_DIR}" \
  --model-served "${MODEL_SERVED}" \
  --router-ip "${ROUTER_IP}" \
  --router-port "${ROUTER_PORT}"

log "Polar focused tests"
PYTHONPATH="${POLAR_ROOT}/src:${PYTHONPATH:-}" pytest -q \
  -p no:cacheprovider \
  "${POLAR_ROOT}/tests/agent/test_claude_code_preset.py" \
  "${POLAR_ROOT}/tests/test_patch_sglang_contract.py" \
  "${POLAR_ROOT}/tests/runtime/test_ascend.py" \
  "${POLAR_ROOT}/tests/runtime/test_docker_runtime_contract.py" \
  "${POLAR_ROOT}/tests/runtime/test_factory.py" \
  "${POLAR_ROOT}/tests/gateway/test_lazy_eval_runtime.py" \
  "${POLAR_ROOT}/tests/trajectory/test_operator_judge.py" \
  "${POLAR_ROOT}/tests/trajectory/test_prefix_merging_builder.py" \
  "${POLAR_ROOT}/tests/trajectory/test_engine_trajectory_equivalence.py" \
  "${POLAR_ROOT}/tests/slime_bridge/test_reward_post_process.py" \
  "${POLAR_ROOT}/tests/slime_bridge/test_dockerruntime_mainline_contract.py" \
  "${POLAR_ROOT}/tests/slime_bridge/test_config.py"

if [[ -d "${SKILLS_DIR}/tests" ]]; then
  log "operator skills tests"
  pytest -q -p no:cacheprovider \
    "${SKILLS_DIR}/tests/test_prepare_operator_workdir.py" \
    "${SKILLS_DIR}/tests/tools/test_npu_lease_pipeline.py"
else
  log "operator skills tests skipped; ${SKILLS_DIR}/tests not present"
fi

if [[ "${POLAR_RUN_IMAGE_GATE:-0}" == "1" ]]; then
  log "runtime image gate"
  bash "${ROOT}/check_polar_runtime_image.sh" \
    --image "${RUNTIME_IMAGE}" --with-npu --device "${POLAR_IMAGE_GATE_DEVICE:-11}"
else
  log "runtime image gate skipped; set POLAR_RUN_IMAGE_GATE=1 to run it"
fi

log "PASS"
