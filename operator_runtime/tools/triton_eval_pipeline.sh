#!/usr/bin/env bash
# =============================================================================
# triton_eval_pipeline.sh — Triton 固定评测入口
#
# 流程：AST 退化检查 → 数值正确性(verify.py) → 性能(benchmark.py) → stdout 判定 + metrics 文件
#
# 可信约束：
#   warmup/repeats/timeout 全部写死；**绝不**向 benchmark.py 传
#   --skip_framework / --framework_latency_ms / --verify_not_required
#   （这三个能任意抬 speedup / 绕过正确性闸门）。framework 基线一律实测。
#
# 用法：
#   bash triton_eval_pipeline.sh --op_name <op> --impl <impl.py> --task <ref_{op}.py> [--out_dir <dir>]
#   不传 --out_dir 时默认写入 judge_out，与 fresh judge 保持同一 artifact 契约。
# =============================================================================
set -euo pipefail

# ----- 固定参数（agent 不可改）-----
readonly WARMUP=5
readonly REPEATS=50
readonly VERIFY_TIMEOUT=900
readonly IMPL_NAME="triton_ascend_impl"   # verify/benchmark 据此推 {op}_{IMPL_NAME}.py + verify_result.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"        # = agent_workdir
VERIFIER_SCRIPTS="${ROOT}/.agents/skills/triton-op-verifier/scripts"

# 解释器：沿用 env.sh（AST_CHECK_PYTHON 走 venv 免 NPU；OPERATOR_PYTHON 走 conda+NPU）
# 缺省回落 python3，便于无 env.sh 的环境。
# shellcheck source=/dev/null
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"
AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"
PIPELINE_GEN_MAX="${POLAR_GEN_PIPELINE_MAX:-6}"
PIPELINE_OPT_MAX="${POLAR_OPT_PIPELINE_MAX:-3}"

# NPU lease：若 POLAR_NPU_LEASE_POOL 已配置，则 verify/benchmark 子进程内部按需抢卡；
# 未配置时保持旧行为，使用容器当前 ASCEND_RT_VISIBLE_DEVICES。
POLAR_NPU_LOCK_DIR="${POLAR_NPU_LOCK_DIR:-/tmp/npu-locks}"
NPU_LEASE_EXEC="${NPU_LEASE_EXEC:-${SCRIPT_DIR}/npu_lease_exec.py}"

run_npu_phase() {
  local phase="$1"; shift
  if [[ -n "${POLAR_NPU_LEASE_POOL:-}" ]]; then
    "${AST_CHECK_PYTHON}" "${NPU_LEASE_EXEC}" \
      --pool "${POLAR_NPU_LEASE_POOL}" \
      --lock-dir "${POLAR_NPU_LOCK_DIR}" \
      --status-file "${OUT_DIR}/npu_lease_status.${phase}.json" \
      -- "$@"
  else
    "$@"
  fi
}

PIPELINE_PHASE="generation"
PIPELINE_ATTEMPT=1
PIPELINE_LIMIT="${PIPELINE_GEN_MAX}"
BUDGET_STATE_FILE=""
PIPELINE_GEN_COUNT=0
PIPELINE_OPT_COUNT=0
PIPELINE_FIRST_SUCCESS=0
PIPELINE_STATUS_FILE=""

pipeline_state_init() {
  BUDGET_STATE_FILE="${OUT_DIR}/.triton_eval_pipeline_budget.json"
  local first_success=0
  [[ -f "${OUT_DIR}/metrics.best.json" ]] && first_success=1
  PIPELINE_FIRST_SUCCESS="${first_success}"
  PIPELINE_PHASE="generation"
  PIPELINE_LIMIT="${PIPELINE_GEN_MAX}"
  if [[ "${first_success}" == "1" ]]; then
    PIPELINE_PHASE="optimization"
    PIPELINE_LIMIT="${PIPELINE_OPT_MAX}"
  fi
  local counts
  counts=$(PIPELINE_STATE_FILE="${BUDGET_STATE_FILE}" PIPELINE_PHASE="${PIPELINE_PHASE}" python3 -c '
import json, os
from pathlib import Path
path = Path(os.environ["PIPELINE_STATE_FILE"])
data = {}
if path.exists():
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}
phase = os.environ["PIPELINE_PHASE"]
key = "opt_count" if phase == "optimization" else "gen_count"
data[key] = int(data.get(key) or 0) + 1
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(data.get("gen_count") or 0, data.get("opt_count") or 0)
' 2>/dev/null || echo "0 0")
  local gen_count opt_count
  read -r gen_count opt_count <<<"${counts}"
  PIPELINE_GEN_COUNT="${gen_count:-0}"
  PIPELINE_OPT_COUNT="${opt_count:-0}"
  if [[ "${PIPELINE_PHASE}" == "optimization" ]]; then
    PIPELINE_ATTEMPT="${opt_count:-1}"
  else
    PIPELINE_ATTEMPT="${gen_count:-1}"
  fi
  pipeline_status_write
  echo "[pipeline-budget] phase=${PIPELINE_PHASE} attempt=${PIPELINE_ATTEMPT}/${PIPELINE_LIMIT}"
}

pipeline_budget_exhausted() {
  [[ "${PIPELINE_ATTEMPT}" =~ ^[0-9]+$ ]] && [[ "${PIPELINE_LIMIT}" =~ ^[0-9]+$ ]] && [[ "${PIPELINE_ATTEMPT}" -ge "${PIPELINE_LIMIT}" ]]
}

pipeline_status_write() {
  [[ -n "${ARTIFACTS_DIR:-}" ]] || return 0
  mkdir -p "${ARTIFACTS_DIR}" 2>/dev/null || return 0
  PIPELINE_STATUS_FILE="${ARTIFACTS_DIR}/pipeline_budget_status.json"
  PIPELINE_STATUS_FILE="${PIPELINE_STATUS_FILE}" \
  SESSION_ID="${SESSION_ID:-}" TASK_ID="${TASK_ID:-}" OP_NAME="${OP_NAME:-}" \
  PIPELINE_PHASE="${PIPELINE_PHASE}" PIPELINE_ATTEMPT="${PIPELINE_ATTEMPT}" PIPELINE_LIMIT="${PIPELINE_LIMIT}" \
  PIPELINE_GEN_COUNT="${PIPELINE_GEN_COUNT}" PIPELINE_OPT_COUNT="${PIPELINE_OPT_COUNT}" \
  PIPELINE_FIRST_SUCCESS="${PIPELINE_FIRST_SUCCESS}" python3 - <<'PY' 2>/dev/null || true
import json
import os
import time
from pathlib import Path

def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

attempt = to_int(os.environ.get("PIPELINE_ATTEMPT"), 0)
limit = to_int(os.environ.get("PIPELINE_LIMIT"), 0)
payload = {
    "schema_version": 1,
    "session_id": os.environ.get("SESSION_ID") or None,
    "task_id": os.environ.get("TASK_ID") or None,
    "op_name": os.environ.get("OP_NAME") or None,
    "phase": os.environ.get("PIPELINE_PHASE") or None,
    "attempt": attempt,
    "limit": limit,
    "gen_count": to_int(os.environ.get("PIPELINE_GEN_COUNT"), 0),
    "opt_count": to_int(os.environ.get("PIPELINE_OPT_COUNT"), 0),
    "first_success": str(os.environ.get("PIPELINE_FIRST_SUCCESS") or "").lower() in {"1", "true", "yes"},
    "limit_exhausted": bool(limit and attempt > limit),
    "updated_at_unix": time.time(),
}
path = Path(os.environ["PIPELINE_STATUS_FILE"])
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

emit_pipeline_budget_exhausted() {
  local current_success="${1:-false}"
  pipeline_budget_exhausted || return 0
  if [[ "${PIPELINE_PHASE}" == "generation" && "${current_success}" == "true" ]]; then
    return 0
  fi
  echo "[pipeline-budget] LIMIT_EXHAUSTED phase=${PIPELINE_PHASE} attempt=${PIPELINE_ATTEMPT}/${PIPELINE_LIMIT}"
  echo "本阶段固定评测入口调用次数已经用完。必须立即停止当前任务。"
  echo "禁止继续分析错误、禁止总结修复方案、禁止恢复文件、禁止再次验证。"
  echo "best.py 已保存，当前工作文件状态不重要。"
  echo "禁止再调用 tools/triton_eval_pipeline.sh；禁止再调用 Bash/Skill/Read/Write/Edit/TaskUpdate 等任何工具。"
  echo "禁止读取 metrics_error.log 或 metrics.json；禁止继续修改实现；禁止继续验证或优化。"
  echo "现在只能输出最终简短结论并结束。"
}

# ----- 写 metrics.json（schema_version 2）-----
write_metrics() {
  local ast_ok="$1" corr_ok="$2" success="$3" fw_lat="$4" impl_lat="$5" speedup="$6" error="$7"
  if [[ -n "${error}" ]]; then printf "%s" "${error}" > "${OUT_DIR}/metrics_error.log"; else : > "${OUT_DIR}/metrics_error.log"; fi
  AST_OK="$ast_ok" CORR_OK="$corr_ok" SUCCESS="$success" FW_LAT="$fw_lat" IMPL_LAT="$impl_lat" \
  SPEEDUP="$speedup" ERROR_FILE="${OUT_DIR}/metrics_error.log" OUT_DIR="$OUT_DIR" OP_NAME="$OP_NAME" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
def b(x): return str(x).lower() in ("1","true","yes")
def n(x): return None if x == "" else float(x)
fw, impl, sp = os.environ.get("FW_LAT",""), os.environ.get("IMPL_LAT",""), os.environ.get("SPEEDUP","")
ef = Path(os.environ.get("ERROR_FILE",""))
full = ef.read_text(encoding="utf-8", errors="replace") if ef.exists() else ""
def classify(t):
    if not t: return None
    l = t.lower()
    if "implementation file missing" in l: return "implementation_missing"
    if "task file missing" in l: return "task_missing"
    if "ast退化" in l or "ast_check" in l or "ast check" in l: return "ast_check_failed"
    if "get_input" in l or ("filenotfounderror" in l and ".json" in l): return "input_load_failed"
    if "ub overflow" in l: return "triton_ub_overflow"
    compact = l.replace(" ", "")
    if (
        ("aclinit" in compact and (
            "invaliddeviceid" in compact or "deviceiderror" in compact
            or "getdevicecntfailed" in compact or "resource_busy" in compact
            or "rtgetdevmsgexecutionfailed" in compact
        ))
        or ("ptacallaclapifailed" in compact and "invaliddeviceid" in compact)
        or "inputerrordeviceid" in compact
        or ("invalid device id" in l and "torch_npu/csrc/core/npu/sys_ctrl" in l)
    ): return "npu_runtime_unavailable"
    if "bishenghir" in l or "bishengir" in l: return "triton_compile_failed"
    if "tritontostructured" in l or "pointer analysis" in l: return "triton_lowering_failed"
    if "correctness" in l or "mismatch" in l or "数值" in t: return "correctness_failed"
    if "benchmark" in l or "性能" in t: return "benchmark_failed"
    return "unknown"
perf = {"framework_latency_ms": float(fw), "impl_latency_ms": float(impl), "speedup_vs_torch": float(sp)} if (fw and impl and sp) else None
error_bytes = len(full.encode("utf-8")) if full else 0
error_sha256 = hashlib.sha256(full.encode("utf-8")).hexdigest() if full else None
metrics = {
    "schema_version": 2, "op_name": os.environ.get("OP_NAME",""),
    "success": b(os.environ["SUCCESS"]), "ast_check_ok": b(os.environ["AST_OK"]),
    "correctness_ok": b(os.environ["CORR_OK"]), "perf_data": perf,
    "error": None, "error_type": classify(full),
    "error_file": "metrics_error.log" if full else None,
    "error_bytes": error_bytes,
    "error_sha256": error_sha256,
    "error_truncated": False,
}
(out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  # ①短路用：记录本次评测对应的 impl+task 内容哈希（与刚写的 metrics.json 对应）。
  [[ -n "${CUR_HASH:-}" ]] && printf "%s" "${CUR_HASH}" > "${OUT_DIR}/.last_eval.hash" 2>/dev/null || true
}

# ----- 失败反馈：stdout 只做判定/路由；完整错误写入 metrics_error.log -----
# 目的：避免超长编译日志被 Claude Code 截断后吞掉停止指令。
emit_verdict_summary() {
  python3 - "${OUT_DIR}/metrics.json" <<'PY' || true
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    d = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    d = {}
perf = d.get("perf_data") if isinstance(d.get("perf_data"), dict) else {}
print(
    "[triton-eval] verdict — "
    f"success={d.get('success')} ast_check_ok={d.get('ast_check_ok')} "
    f"correctness_ok={d.get('correctness_ok')} error_type={d.get('error_type')} "
    f"speedup_vs_torch={perf.get('speedup_vs_torch')}"
)
PY
}

fail_hint() {               # 统一指引：未耗尽则读完整错误文件；耗尽则立即停止
  emit_verdict_summary
  if pipeline_budget_exhausted; then
    emit_pipeline_budget_exhausted false
    return 0
  fi
  echo "  ↳ 完整错误已写入 ${OUT_DIR}/metrics_error.log。下一步应以该文件为主要反馈，结合 src/、output/submission/、sketch.txt 和当前实现分析原因。"
  echo "  ↳ 可以进行必要的文案/说明阅读、代码阅读和定位；修复时只修改 output/submission/ 下的实现文件，然后重跑本固定入口。"
  echo "  ↳ metrics.json 仅作结构化摘要，通常无需读取；禁止修改 tools/、scripts/ 或 .agents/skills/ 下的评测/工具代码。"
}

# ----- 参数解析（只接受这些；没有 warmup/repeats/skip_framework 等危险开关）-----
OP_NAME="" IMPL_FILE="" TASK_FILE="" OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --op_name)   OP_NAME="$2";   shift 2 ;;
    --impl)      IMPL_FILE="$2"; shift 2 ;;
    --task)      TASK_FILE="$2"; shift 2 ;;
    --out_dir)   OUT_DIR="$2";   shift 2 ;;
    *) echo "[triton-eval] unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ----- 正常评测 -----
[[ -z "${OP_NAME}" ]] && { echo "[triton-eval] --op_name required" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-judge_out}"; mkdir -p "${OUT_DIR}"
pipeline_state_init

if [[ ! -f "${IMPL_FILE}" ]]; then
  write_metrics false false false "" "" "" "implementation file missing: ${IMPL_FILE}"
  echo "[triton-eval] implementation missing"
  fail_hint; exit 1
fi
if [[ ! -f "${TASK_FILE}" ]]; then
  write_metrics false false false "" "" "" "task file missing: ${TASK_FILE}"
  echo "[triton-eval] task(reference) missing"
  fail_hint; exit 1
fi

# ----- ①内容哈希短路：impl+task 与上次评测完全相同 → 复用上次 metrics.json，跳过 NPU verify+benchmark -----
# 避免对未改动代码重复运行 NPU verify+benchmark。哈希由 write_metrics 在每个终态写入，
# 与 metrics.json 对应。
CUR_HASH=$(python3 -c "import hashlib,sys;h=hashlib.sha256()
for p in sys.argv[1:]:
    h.update(open(p,'rb').read())
print(h.hexdigest())" "${IMPL_FILE}" "${TASK_FILE}" 2>/dev/null || echo "")
if [[ -n "${CUR_HASH}" && -f "${OUT_DIR}/.last_eval.hash" && -f "${OUT_DIR}/metrics.json" \
      && "${CUR_HASH}" == "$(cat "${OUT_DIR}/.last_eval.hash" 2>/dev/null)" ]]; then
  echo "[triton-eval] impl 未改动（与上次评测一致）→ 复用上次固定入口结果，跳过 NPU verify+benchmark"
  python3 -c "import json;d=json.load(open('${OUT_DIR}/metrics.json'));p=d.get('perf_data') or {};print('[triton-eval] cached verdict — success=%s ast_check_ok=%s correctness_ok=%s speedup_vs_torch=%s'%(d.get('success'),d.get('ast_check_ok'),d.get('correctness_ok'),p.get('speedup_vs_torch')))" 2>/dev/null || true
  CACHED_SUCCESS=$(python3 -c "import json;print('true' if json.load(open('${OUT_DIR}/metrics.json')).get('success') else 'false')" 2>/dev/null || echo "false")
  emit_pipeline_budget_exhausted "${CACHED_SUCCESS}"
  exit 0
fi

# verify_dir：verify.py/benchmark.py 按 import 名加载 {op}_torch / {op}_{IMPL_NAME}
VERIFY_DIR="${OUT_DIR}/verify_tmp"; rm -rf "${VERIFY_DIR}"; mkdir -p "${VERIFY_DIR}"
cp "${TASK_FILE}" "${VERIFY_DIR}/${OP_NAME}_torch.py"
cp "${IMPL_FILE}" "${VERIFY_DIR}/${OP_NAME}_${IMPL_NAME}.py"
# Step 1: AST 退化预检查（venv Python，免 NPU）
echo "[triton-eval] Step1 AST check"
if ! AST_OUT=$("${AST_CHECK_PYTHON}" "${VERIFIER_SCRIPTS}/validate_triton_impl.py" "${IMPL_FILE}" 2>&1); then
  rm -rf "${VERIFY_DIR}"; write_metrics false false false "" "" "" "AST退化检查失败: ${AST_OUT}"
  echo "[triton-eval] AST FAILED — 退化检测未通过（error_type=ast_check_failed）"
  fail_hint; exit 1
fi

# Step 2: 数值正确性（NPU，抢设备锁）
echo "[triton-eval] Step2 verify"
if ! VERIFY_ERR=$(run_npu_phase verify "${OPERATOR_PYTHON}" "${VERIFIER_SCRIPTS}/verify.py" \
      --op_name "${OP_NAME}" --verify_dir "${VERIFY_DIR}" --triton_impl_name "${IMPL_NAME}" \
      --timeout "${VERIFY_TIMEOUT}" 2>&1); then
  write_metrics true false false "" "" "" "数值验证失败: ${VERIFY_ERR}"
  echo "[triton-eval] verify FAILED"
  fail_hint; exit 1
fi

# Step 3: 性能（NPU，抢锁）—— 注意：不传 --skip_framework/--framework_latency_ms/--verify_not_required
echo "[triton-eval] Step3 benchmark"
PERF_JSON="${OUT_DIR}/perf_result.json"
if ! BENCH_ERR=$(run_npu_phase benchmark "${OPERATOR_PYTHON}" "${VERIFIER_SCRIPTS}/benchmark.py" \
      --op_name "${OP_NAME}" --verify_dir "${VERIFY_DIR}" --triton_impl_name "${IMPL_NAME}" \
      --warmup "${WARMUP}" --repeats "${REPEATS}" --output "${PERF_JSON}" 2>&1); then
  write_metrics true true false "" "" "" "性能测试失败: ${BENCH_ERR}"
  echo "[triton-eval] benchmark FAILED"
  fail_hint; exit 1
fi
rm -rf "${VERIFY_DIR}"

# Step 4: perf_result.json → metrics.json（speedup_vs_torch = 全 shape 几何平均，由 benchmark.py 算好）
FW=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['framework']['avg_latency_ms'])" 2>/dev/null || echo "")
IMPL_LAT=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['implementation']['avg_latency_ms'])" 2>/dev/null || echo "")
SP=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['speedup_vs_torch'])" 2>/dev/null || echo "")
write_metrics true true true "${FW}" "${IMPL_LAT}" "${SP}" ""
# ③成功也把结构化字段回显 stdout，agent 无需再单开一步读取评测状态文件
echo "[triton-eval] done — success=true ast_check_ok=true correctness_ok=true speedup_vs_torch=${SP}  (本次输出即最终反馈；不要另开一步读评测状态文件)"

# ----- 留存当前性能最好的正确实现 -----
# impl 落在 submission 同目录的 {op}_impl.best.py；metrics 落在 OUT_DIR/metrics.best.json。
# 仅在 speedup 比已存实现更高（或尚无已存实现）时更新；首个 success 必留存。
IMPL_BEST="${IMPL_FILE%.py}.best.py"
BEST_METRICS="${OUT_DIR}/metrics.best.json"
PREV_BEST_SP=$(python3 -c "import json;print(json.load(open('${BEST_METRICS}'))['perf_data']['speedup_vs_torch'])" 2>/dev/null || echo "")
UPDATE_BEST=$(PREV="${PREV_BEST_SP}" CUR="${SP}" python3 -c "
import os
prev, cur = os.environ.get('PREV',''), os.environ.get('CUR','')
try:
    print('1' if (prev == '' or float(cur) > float(prev)) else '0')
except Exception:
    print('1' if prev == '' else '0')
" 2>/dev/null || echo "1")
if [[ "${UPDATE_BEST}" == "1" ]]; then
  cp "${IMPL_FILE}" "${IMPL_BEST}" 2>/dev/null || true
  cp "${OUT_DIR}/metrics.json" "${BEST_METRICS}" 2>/dev/null || true
  echo "[triton-eval] verification passed; retained current implementation (speedup_vs_torch=${SP})"
else
  echo "[triton-eval] verification passed; current implementation did not improve speedup_vs_torch=${PREV_BEST_SP}"
fi
emit_pipeline_budget_exhausted true
