#!/usr/bin/env bash
set -uo pipefail

OP_NAME="" VERIFIED="" SPEEDUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verified) VERIFIED=1; shift;;
    --speedup)  SPEEDUP="$2"; shift 2;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0;;
    *) [[ -z "$OP_NAME" ]] && { OP_NAME="$1"; shift; } || { echo "[pack] unknown arg: $1" >&2; exit 1; };;
  esac
done
[[ -z "$OP_NAME" ]] && { echo "[pack] usage: bash tools/pack_submission.sh <op_name>" >&2; exit 1; }

WORKDIR="${WORKDIR:-$PWD}"
TASK_DIR="$WORKDIR/$OP_NAME"
SUB_DIR="$WORKDIR/output/submission"
TARBALL="$SUB_DIR/${OP_NAME}_impl.tar.gz"
BEST_TARBALL="$SUB_DIR/${OP_NAME}_impl.best.tar.gz"
BEST_META="$SUB_DIR/.${OP_NAME}_impl.best.meta.json"
_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_TOOLS_DIR/env.sh" ]] && source "$_TOOLS_DIR/env.sh"
PY_BIN="${PY_BIN:-${AST_CHECK_PYTHON:-python3}}"

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

TIER=0
VALIDATOR=""
for c in "$WORKDIR/.claude/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py" \
         "/opt/canonical/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py"; do
  [[ -f "$c" ]] && { VALIDATOR="$c"; break; }
done
if [[ -n "$VALIDATOR" ]]; then
  if "$PY_BIN" "$VALIDATOR" "$TASK_DIR/model_new_ascendc.py" >/dev/null 2>&1; then TIER=1; fi
elif grep -q "torch.ops.npu" "$TASK_DIR/model_new_ascendc.py" 2>/dev/null; then
  TIER=1   # 找不到检测脚本时的保守回退(与 judge Step1 同判据)
fi
if [[ "$TIER" -ge 1 ]]; then
  if [[ -n "$VERIFIED" ]] || grep -qE "Result: *pass" "$TASK_DIR/.eval_last.log" 2>/dev/null; then TIER=2; fi
fi
if [[ "$TIER" -ge 2 ]]; then
  if [[ -z "$SPEEDUP" ]]; then
    for pj in "$TASK_DIR/performance.json" "$TASK_DIR/preformance.json"; do
      [[ -f "$pj" ]] && SPEEDUP=$("$PY_BIN" -c "import json;d=json.load(open('$pj'));v=d.get('geomean_speedup') or d.get('mean_speedup') or d.get('overall_speedup');print(float(v) if v else '')" 2>/dev/null) && [[ -n "$SPEEDUP" ]] && break
    done
  fi
  [[ -n "$SPEEDUP" ]] && TIER=3
fi

mkdir -p "$SUB_DIR"
if ! (cd "$WORKDIR" && tar czf "$TARBALL" \
        --exclude='build' --exclude='dist' --exclude='*.so' --exclude='*.a' \
        --exclude='*.o' --exclude='*.whl' --exclude='*.egg-info' \
        --exclude='__pycache__' --exclude='.eval_last.log' \
        "$OP_NAME" 2>/dev/null); then
  echo "[pack] FAILED: tar 打包失败" >&2; exit 1
fi
SIZE=$(du -h "$TARBALL" 2>/dev/null | cut -f1)
NFILES=$(tar tzf "$TARBALL" 2>/dev/null | grep -vc '/$')

UPDATED=$(TIER="$TIER" SP="${SPEEDUP:-}" META="$BEST_META" "$PY_BIN" - <<'PY'
import json, os, time
from pathlib import Path
tier = int(os.environ.get("TIER") or 0)
try: sp = float(os.environ.get("SP") or "nan")
except ValueError: sp = float("nan")
meta_path = Path(os.environ["META"])
prev = {}
if meta_path.exists():
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict): prev = loaded
    except Exception: prev = {}
ptier = int(prev.get("tier") or -1)
try: psp = float(prev.get("speedup"))
except (TypeError, ValueError): psp = float("nan")
if tier > ptier:                       update = True
elif tier < ptier:                     update = False
elif tier >= 3 and sp == sp:           update = (psp != psp) or (sp > psp)
else:                                  update = True
if update:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(
        {"tier": tier, "speedup": (sp if sp == sp else None), "updated_at_unix": time.time()},
        ensure_ascii=False, indent=2), encoding="utf-8")
print("1" if update else "0")
PY
)
TIER_DESC=("T0 残缺/退化(judge 0.2)" "T1 真调 torch.ops.npu(judge 0.3)" "T2 对拍通过(judge 0.4)" "T3 有 speedup(judge 0.75+)")
_mirror_best() {
  local sdir="${POLAR_RUNTIME_SESSION_DIR:-/polar/session}"
  [[ -d "$sdir" ]] || return 0
  mkdir -p "$sdir/submission" 2>/dev/null || return 0
  cp -f "$BEST_TARBALL" "$sdir/submission/${OP_NAME}_impl.best.tar.gz" 2>/dev/null \
    && echo "[pack] best 已镜像到 session 目录(宿主机可直读)"
}

if [[ "$UPDATED" == "1" ]]; then
  cp -f "$TARBALL" "$BEST_TARBALL"
  _mirror_best
  echo "[pack] ${TARBALL##*/} + best 已更新 — ${TIER_DESC[$TIER]}${SPEEDUP:+ speedup=$SPEEDUP} | ${NFILES} 文件 ${SIZE}"
else
  echo "[pack] ${TARBALL##*/} 已更新;best 保持不变(本次 ${TIER_DESC[$TIER]} 未超过历史最优)| ${NFILES} 文件 ${SIZE}"
fi
echo "[pack] judge 取件顺序:${OP_NAME}_impl.best.tar.gz → ${OP_NAME}_impl.tar.gz"
exit 0
