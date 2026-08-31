#!/usr/bin/env bash
set -uo pipefail

# 两个互斥模式:
#   默认:只把当前工程打成 candidate；绝不在评测前改写历史 best。
#   --promote:读取本次 metrics，将“实际被评测的 candidate”按 reward 顺序提升为 best。
# Agent 只调用固定评测入口；公开提交路径仍是 output/submission/{op}_impl.tar.gz。

OP_NAME="" CANDIDATE="" PUBLIC_TARBALL="" METRICS="" PROMOTE=0
LEGACY_VERIFIED="" LEGACY_SPEEDUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) CANDIDATE="$2"; shift 2;;
    --public) PUBLIC_TARBALL="$2"; shift 2;;
    --metrics) METRICS="$2"; shift 2;;
    --promote) PROMOTE=1; shift;;
    # 兼容旧调用的参数解析，但不再允许它们绕过 metrics 认证 best。
    --verified) LEGACY_VERIFIED=1; shift;;
    --speedup) LEGACY_SPEEDUP="$2"; shift 2;;
    -h|--help)
      echo "usage: bash tools/pack_submission.sh <op_name> [--candidate <tar>] [--public <tar>]"
      echo "       bash tools/pack_submission.sh <op_name> --promote --candidate <evaluated-tar> --metrics <metrics.json>"
      exit 0;;
    *) [[ -z "$OP_NAME" ]] && { OP_NAME="$1"; shift; } \
         || { echo "[pack] unknown arg: $1" >&2; exit 1; };;
  esac
done
[[ -z "$OP_NAME" ]] && { echo "[pack] usage: bash tools/pack_submission.sh <op_name>" >&2; exit 1; }

WORKDIR="${WORKDIR:-$PWD}"
WORKDIR="$(cd "$WORKDIR" 2>/dev/null && pwd)" \
  || { echo "[pack] FAILED: WORKDIR 不存在" >&2; exit 1; }
TASK_DIR="$WORKDIR/$OP_NAME"
SUB_DIR="$WORKDIR/output/submission"
DEFAULT_TARBALL="$SUB_DIR/${OP_NAME}_impl.tar.gz"
BEST_TARBALL="$SUB_DIR/${OP_NAME}_impl.best.tar.gz"
BEST_META="$SUB_DIR/.${OP_NAME}_impl.best.meta.json"
_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_TOOLS_DIR/env.sh" ]] && source "$_TOOLS_DIR/env.sh"
PY_BIN="${PY_BIN:-${AST_CHECK_PYTHON:-python3}}"

_abs_from_workdir() {
  case "$1" in
    /*) printf '%s\n' "$1";;
    *)  printf '%s/%s\n' "$WORKDIR" "$1";;
  esac
}

CANDIDATE="$(_abs_from_workdir "${CANDIDATE:-$DEFAULT_TARBALL}")"
PUBLIC_TARBALL="$(_abs_from_workdir "${PUBLIC_TARBALL:-$DEFAULT_TARBALL}")"
[[ -n "$METRICS" ]] && METRICS="$(_abs_from_workdir "$METRICS")"

_atomic_copy() {  # src dest
  local src="$1" dest="$2" tmp
  mkdir -p "$(dirname "$dest")" || return 1
  tmp=$(mktemp "${dest}.tmp.XXXXXX") || return 1
  if ! cp -f "$src" "$tmp"; then rm -f "$tmp"; return 1; fi
  chmod 0444 "$tmp" 2>/dev/null || true
  if ! mv -f "$tmp" "$dest"; then rm -f "$tmp"; return 1; fi
}

_mirror_best() {
  local sdir="${POLAR_RUNTIME_SESSION_DIR:-/polar/session}" dest
  [[ -d "$sdir" && -f "$BEST_TARBALL" ]] || return 0
  dest="$sdir/submission/${OP_NAME}_impl.best.tar.gz"
  _atomic_copy "$BEST_TARBALL" "$dest" \
    && echo "[pack] best 已镜像到 session 目录(宿主机可直读)" \
    || echo "[pack] WARN: best 镜像失败:$dest" >&2
}

if [[ "$PROMOTE" != "1" ]]; then
  [[ -z "$LEGACY_VERIFIED$LEGACY_SPEEDUP" ]] \
    || echo "[pack] WARN: --verified/--speedup 不再单独提升 best；best 只接受固定入口产出的 metrics" >&2
  if [[ ! -d "$TASK_DIR" ]]; then
    echo "[pack] FAILED: 工程目录不存在: $TASK_DIR(应为 {workdir}/{op_name}/)" >&2; exit 1
  fi
  MISSING=()
  [[ -f "$TASK_DIR/model_new_ascendc.py" ]] || MISSING+=("model_new_ascendc.py")
  [[ -d "$TASK_DIR/kernel" ]]              || MISSING+=("kernel/")
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[pack] FAILED: 缺少必需件: ${MISSING[*]}" >&2
    echo "[pack] 提交物必须含 model_new_ascendc.py + kernel/(op_host/ op_kernel/ ops.h register.cpp CMakeLists.txt setup.py)" >&2
    exit 1
  fi
  for want in kernel/CMakeLists.txt kernel/op_host kernel/op_kernel; do
    [[ -e "$TASK_DIR/$want" ]] || echo "[pack] WARN: 建议补齐 $want(judge 侧要从源码重编)"
  done

  # AST 只用于提示，不参与 best 排序。完整排序只能来自本 candidate 的 metrics。
  VALIDATOR=""
  for c in "$WORKDIR/.claude/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py" \
           "/opt/canonical/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py"; do
    [[ -f "$c" ]] && { VALIDATOR="$c"; break; }
  done
  if [[ -n "$VALIDATOR" ]]; then
    "$PY_BIN" "$VALIDATOR" "$TASK_DIR/model_new_ascendc.py" >/dev/null 2>&1 \
      || echo "[pack] WARN: 当前 candidate 未通过 AST 检查；仍打包供固定入口产生权威 metrics"
  elif ! grep -q "torch.ops.npu" "$TASK_DIR/model_new_ascendc.py" 2>/dev/null; then
    echo "[pack] WARN: 当前 candidate 未发现 torch.ops.npu 调用；仍打包供固定入口产生权威 metrics"
  fi

  mkdir -p "$(dirname "$CANDIDATE")" "$(dirname "$PUBLIC_TARBALL")"
  _TMP=$(mktemp "${CANDIDATE}.tmp.XXXXXX") \
    || { echo "[pack] FAILED: 无法创建 candidate 临时文件:$CANDIDATE" >&2; exit 1; }
  if ! (cd "$WORKDIR" && tar czf "$_TMP" \
          --exclude='build' --exclude='dist' --exclude='*.so' --exclude='*.a' \
          --exclude='*.o' --exclude='*.whl' --exclude='*.egg-info' \
          --exclude='__pycache__' --exclude='.eval_last.log' \
          "$OP_NAME" 2>/dev/null); then
    rm -f "$_TMP"
    echo "[pack] FAILED: tar 打包失败" >&2; exit 1
  fi
  chmod 0444 "$_TMP" 2>/dev/null || true
  mv -f "$_TMP" "$CANDIDATE" \
    || { rm -f "$_TMP"; echo "[pack] FAILED: candidate 原子落盘失败:$CANDIDATE" >&2; exit 1; }
  if [[ "$CANDIDATE" != "$PUBLIC_TARBALL" ]]; then
    _atomic_copy "$CANDIDATE" "$PUBLIC_TARBALL" \
      || { echo "[pack] FAILED: 公开提交物落盘失败:$PUBLIC_TARBALL" >&2; exit 1; }
  fi
  SIZE=$(du -h "$PUBLIC_TARBALL" 2>/dev/null | cut -f1)
  NFILES=$(tar tzf "$CANDIDATE" 2>/dev/null | grep -vc '/$')
  echo "[pack] 当前候选已打包:${PUBLIC_TARBALL#$WORKDIR/}；评测完成前不会更新 best | ${NFILES} 文件 ${SIZE}"
  echo "[pack] judge 取件顺序:${OP_NAME}_impl.best.tar.gz → ${OP_NAME}_impl.tar.gz"
  exit 0
fi

[[ -z "$LEGACY_VERIFIED$LEGACY_SPEEDUP" ]] \
  || { echo "[pack] FAILED: --promote 只接受 metrics，不能混用 --verified/--speedup" >&2; exit 1; }
[[ -n "$METRICS" && -f "$METRICS" ]] \
  || { echo "[pack] best 未更新:本次评测没有可读 metrics"; exit 0; }
[[ -f "$CANDIDATE" ]] \
  || { echo "[pack] best 未更新:实际评测 candidate 不存在:$CANDIDATE"; exit 0; }
mkdir -p "$SUB_DIR"

# 比较与替换在同一个文件锁内完成。reward_score 与 server 端 reward_from_metrics 顺序一致；
# 测试会逐档对账，避免以后只改 reward、忘记同步 best 选择器。
if ! PROMOTION=$(OP="$OP_NAME" METRICS="$METRICS" CANDIDATE="$CANDIDATE" \
  BEST="$BEST_TARBALL" META="$BEST_META" LOCK="$SUB_DIR/.${OP_NAME}_impl.best.lock" \
  "$PY_BIN" - <<'PY'
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

op = os.environ["OP"]
metrics_path = Path(os.environ["METRICS"])
candidate = Path(os.environ["CANDIDATE"])
best = Path(os.environ["BEST"])
meta_path = Path(os.environ["META"])
lock_path = Path(os.environ["LOCK"])

INFRA = {
    "npu_runtime_unavailable", "input_load_failed", "judge_container_failed",
    "judge_metrics_unreadable", "judge_no_metrics", "task_missing",
    "submission_fetch_failed", "profiler_unavailable", "judge_classification_failed",
}
RUN_FAILURES = {
    "op_not_registered", "ascendc_load_failed", "ascendc_run_crashed",
    "ascendc_run_timeout", "ascendc_launch_failed", "stateful_impl_detected",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def finite(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default

def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default

def score(metrics: dict):
    error_type = str(metrics.get("error_type") or "")
    if error_type in INFRA:
        return None, None, "infra"
    if bool(metrics.get("success")):
        speedup = finite((metrics.get("perf_data") or {}).get("speedup_vs_torch", 1.0))
        if speedup is None or speedup < 0:
            return None, None, "invalid success speedup"
        square = speedup * speedup
        reward = 0.75 + 0.25 * (square - 1.0) / (square + 1.0)
        if not math.isfinite(reward):
            return None, None, "invalid success reward"
        return reward, 3, f"success speedup={speedup}"
    if bool(metrics.get("correctness_ok")):
        return 0.4, 2, "correctness passed; benchmark failed"
    if not bool(metrics.get("ast_check_ok")):
        return 0.0, 0, error_type or "ast/submission failed"
    if error_type == "ascendc_compile_failed":
        return 0.1, 1, error_type
    if error_type in RUN_FAILURES:
        return 0.2, 1, error_type
    if error_type in {"correctness_failed", "output_precheck_failed"}:
        weight = finite(os.environ.get("POLAR_CASE_PASS_WEIGHT", "0.10") or "0.10", 0.10)
        if weight <= 0.0:
            return 0.35, 1, f"{error_type} cases=fixed"
        passed, total = metrics.get("cases_passed"), metrics.get("cases_total")
        if (isinstance(passed, int) and not isinstance(passed, bool)
                and isinstance(total, int) and not isinstance(total, bool)
                and 0 <= passed <= total and total > 0):
            reward = 0.3 + weight * min(passed / total, 0.999)
            return reward, 1, f"{error_type} cases={passed}/{total}"
        return 0.35, 1, f"{error_type} cases=unknown"
    return 0.25, 1, error_type or "unknown operator failure"

try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"SKIP\tmetrics unreadable:{type(exc).__name__}")
    raise SystemExit(0)
if not isinstance(metrics, dict):
    print("SKIP\tmetrics is not an object")
    raise SystemExit(0)
if str(metrics.get("op_name") or "") != op:
    print("SKIP\tmetrics 的 op_name 与 best 目标不一致")
    raise SystemExit(0)

current_score, tier, label = score(metrics)
if current_score is None:
    print(f"SKIP\t{label}")
    raise SystemExit(0)

candidate_hash = sha256(candidate)
metrics_hash = sha256(metrics_path)
evaluated_hash = str(metrics.get("evaluated_candidate_sha256") or "")
if not evaluated_hash or evaluated_hash != candidate_hash:
    print("SKIP\tmetrics 与实际 candidate 哈希不匹配；拒绝把旧结论贴到新源码")
    raise SystemExit(0)
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    previous = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except Exception:
            previous = {}

    if best.exists() and not previous:
        print("SKIP\t已有 best 但 meta 缺失/损坏；为避免覆盖未知历史版本而保留")
        raise SystemExit(0)
    if integer(previous.get("schema_version"), 0) >= 2:
        expected = str(previous.get("candidate_sha256") or "")
        actual = sha256(best) if best.exists() else ""
        if not expected or actual != expected:
            # 提升时先提交 meta、再替换 best：进程若在两步之间被 SIGKILL，judge
            # 看到的仍是旧 best。下次持锁进入时从不可变 candidate 完成事务。
            pending_raw = str(previous.get("candidate_path") or "")
            pending = Path(pending_raw) if pending_raw else None
            if pending is None or not pending.is_file() or sha256(pending) != expected:
                print("SKIP\tbest 提升事务不完整且 candidate 不可恢复；保留现状")
                raise SystemExit(0)
            fd, recover_name = tempfile.mkstemp(prefix=best.name + ".recover.", dir=str(best.parent))
            os.close(fd)
            recover = Path(recover_name)
            try:
                shutil.copyfile(pending, recover)
                os.chmod(recover, 0o444)
                os.replace(recover, best)
            finally:
                recover.unlink(missing_ok=True)

    previous_score = None
    if best.exists():
        previous_score = finite(previous.get("reward_score"))
        if previous_score is None:
            # 旧版 meta 没有精确失败分数。T1 用失败侧上界保守迁移，避免升级时
            # 用一个新编译失败覆盖未知的旧 correctness 候选。
            old_tier = integer(previous.get("tier"), 0)
            if old_tier >= 3:
                old_sp = finite(previous.get("speedup"))
                previous_score = (0.75 + 0.25 * (old_sp * old_sp - 1.0) / (old_sp * old_sp + 1.0)
                                  if old_sp is not None and old_sp >= 0 else 1.0)
            elif old_tier == 2:
                previous_score = 0.4
            elif old_tier == 1:
                previous_score = 0.399999
            else:
                previous_score = 0.0

    update = previous_score is None or current_score > previous_score
    if not update:
        print(f"KEEP\t{label};score={current_score:.9f};best={previous_score:.9f}")
        raise SystemExit(0)

    payload = {
        "schema_version": 2,
        "op_name": op,
        "tier": tier,
        "reward_score": current_score,
        "speedup": finite((metrics.get("perf_data") or {}).get("speedup_vs_torch")),
        "cases_passed": metrics.get("cases_passed"),
        "cases_total": metrics.get("cases_total"),
        "error_type": metrics.get("error_type"),
        "candidate_sha256": candidate_hash,
        "candidate_path": str(candidate.resolve()),
        "metrics_sha256": metrics_hash,
        "updated_at_unix": time.time(),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=meta_path.name + ".tmp.", dir=str(meta_path.parent))
    os.close(fd)
    tmp_meta = Path(tmp_name)
    try:
        tmp_meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    finally:
        tmp_meta.unlink(missing_ok=True)

    # meta 先落盘，best 后落盘。若在中间被杀，外部 judge 至多继续读旧 best；
    # 下次调用会依据 meta 中的不可变 candidate_path 自动完成这次提升。
    best.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=best.name + ".tmp.", dir=str(best.parent))
    os.close(fd)
    tmp_best = Path(tmp_name)
    try:
        shutil.copyfile(candidate, tmp_best)
        os.chmod(tmp_best, 0o444)
        os.replace(tmp_best, best)
    finally:
        tmp_best.unlink(missing_ok=True)
    print(f"UPDATE\t{label};score={current_score:.9f}")
PY
); then
  echo "[pack] FAILED: best 提升器执行失败" >&2
  exit 1
fi

ACTION="${PROMOTION%%$'\t'*}"
DETAIL="${PROMOTION#*$'\t'}"
case "$ACTION" in
  UPDATE)
    _mirror_best
    echo "[pack] best 已更新 — $DETAIL";;
  KEEP)
    echo "[pack] best 保持不变 — $DETAIL";;
  SKIP)
    echo "[pack] best 未更新 — $DETAIL";;
  *)
    echo "[pack] FAILED: best 提升器返回异常:$PROMOTION" >&2; exit 1;;
esac
echo "[pack] judge 取件顺序:${OP_NAME}_impl.best.tar.gz → ${OP_NAME}_impl.tar.gz"
exit 0
