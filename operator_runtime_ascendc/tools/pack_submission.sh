#!/usr/bin/env bash
# =============================================================================
# pack_submission.sh — AscendC 提交物打包(agent 侧固定入口的一部分)
#
# 为什么需要它:AscendC 的提交物是一整个工程目录(15+ 文件),而 polar 跨 fresh-judge
# 边界只传**一个文件**(node.py 的 download_file 单文件语义)→ 必须打成 tarball。
# triton 的提交物天生是单个 .py,agent 直接编辑它 = 自始至终"已交卷";AscendC 这边
# 若只在收尾打包,session 被权重同步 abort(实测 66%)就全归零。
#
# 档位 = reward 阶梯的镜像(operator_reward.py:67-85):
#   T0 有目录但退化/残缺      → judge 0.2
#   T1 真调 torch.ops.npu.<op> → judge 0.3   判据: validate_ascendc_impl.py 退出 0
#   T2 数值对拍通过            → judge 0.4   判据: --verified 或 {op}/.eval_last.log 含 "Result: pass"
#   T3 测出 speedup            → judge 0.75+ 判据: --speedup 或 {op}/performance.json 的 overall_speedup
# 覆盖规则(只升不降):新档 > 旧档 → 更新 .best;同档且 T3 时 speedup 更高 → 更新;
#                     同档且 < T3 → 更新(视为修复进展);新档 < 旧档 → **不动 .best**。
# judge 的 submission_candidates 是 .best 优先,故交出去的必然是历史最好那一版。
#
# 用法: bash tools/pack_submission.sh <op_name> [--verified] [--speedup <float>]
# =============================================================================
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
# 打包全程不碰 NPU:AST 检查用免卡解释器(env.sh 里 AST_CHECK_PYTHON 可指向不装 torch_npu 的 python)
_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_TOOLS_DIR/env.sh" ]] && source "$_TOOLS_DIR/env.sh"
PY_BIN="${PY_BIN:-${AST_CHECK_PYTHON:-python3}}"

# ---- 必需件自检:缺了就明确说缺什么,不产出半残包 ----
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

# ---- 档位评级(客观证据优先,不采信自述)----
TIER=0
# T1: AST 退化检测 —— 纯 python、秒级、不占卡,pack 自己跑
VALIDATOR=""
for c in "$WORKDIR/.claude/skills/ascendc-translator/scripts/validate_ascendc_impl.py" \
         "/opt/canonical/skills/ascendc-translator/scripts/validate_ascendc_impl.py"; do
  [[ -f "$c" ]] && { VALIDATOR="$c"; break; }
done
if [[ -n "$VALIDATOR" ]]; then
  if "$PY_BIN" "$VALIDATOR" "$TASK_DIR/model_new_ascendc.py" >/dev/null 2>&1; then TIER=1; fi
elif grep -q "torch.ops.npu" "$TASK_DIR/model_new_ascendc.py" 2>/dev/null; then
  TIER=1   # 找不到检测脚本时的保守回退(与 judge Step1 同判据)
fi
# T2: 对拍通过
if [[ "$TIER" -ge 1 ]]; then
  if [[ -n "$VERIFIED" ]] || grep -qE "Result: *pass" "$TASK_DIR/.eval_last.log" 2>/dev/null; then TIER=2; fi
fi
# T3: 测出 speedup
if [[ "$TIER" -ge 2 ]]; then
  if [[ -z "$SPEEDUP" ]]; then
    for pj in "$TASK_DIR/performance.json" "$TASK_DIR/preformance.json"; do
      [[ -f "$pj" ]] && SPEEDUP=$("$PY_BIN" -c "import json;v=json.load(open('$pj')).get('overall_speedup');print(float(v) if v else '')" 2>/dev/null) && [[ -n "$SPEEDUP" ]] && break
    done
  fi
  [[ -n "$SPEEDUP" ]] && TIER=3
fi

# ---- 打当前版本(总是覆盖);排除二进制:judge 一律从源码重编,带上去毫无用处还撑大包 ----
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

# ---- 与历史最优比较,只升不降 ----
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
elif tier >= 3 and sp == sp:           update = (psp != psp) or (sp > psp)   # 同为 T3,比 speedup
else:                                  update = True                        # 同档 <T3,取最新(视为修复进展)
if update:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(
        {"tier": tier, "speedup": (sp if sp == sp else None), "updated_at_unix": time.time()},
        ensure_ascii=False, indent=2), encoding="utf-8")
print("1" if update else "0")
PY
)
TIER_DESC=("T0 残缺/退化(judge 0.2)" "T1 真调 torch.ops.npu(judge 0.3)" "T2 对拍通过(judge 0.4)" "T3 有 speedup(judge 0.75+)")
if [[ "$UPDATED" == "1" ]]; then
  cp -f "$TARBALL" "$BEST_TARBALL"
  echo "[pack] ${TARBALL##*/} + best 已更新 — ${TIER_DESC[$TIER]}${SPEEDUP:+ speedup=$SPEEDUP} | ${NFILES} 文件 ${SIZE}"
else
  echo "[pack] ${TARBALL##*/} 已更新;best 保持不变(本次 ${TIER_DESC[$TIER]} 未超过历史最优)| ${NFILES} 文件 ${SIZE}"
fi
echo "[pack] judge 取件顺序:${OP_NAME}_impl.best.tar.gz → ${OP_NAME}_impl.tar.gz"
exit 0
