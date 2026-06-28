#!/usr/bin/env bash
# Tier-0 checks for the Polar DockerRuntime slime mainline. This script does
# not start training, rollout servers, gateways, or cleanup scripts.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

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

log "operator assets"
python3 - "$SKILLS_DIR" "$TASKS_DIR" "$TASK_JSONL" <<'PY'
import json
import re
import sys
from pathlib import Path

skills = Path(sys.argv[1])
tasks = Path(sys.argv[2])
jsonl = Path(sys.argv[3])
required = [
    skills / "CLAUDE.md",
    skills / "tools" / "triton_eval_pipeline.sh",
    skills / "tools" / "npu_lease_exec.py",
    skills / "runtime" / "prepare_operator_workdir.py",
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

log "readonly operator tools"
python3 "${ROOT}/prepare_readonly_tools.py" \
  --source "${SKILLS_DIR}/tools" \
  --dest "${READONLY_TOOLS_DIR}"
bash -n "${READONLY_TOOLS_DIR}/triton_eval_pipeline.sh"
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
