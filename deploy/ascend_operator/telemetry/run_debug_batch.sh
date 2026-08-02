#!/usr/bin/env bash
# 批量【只编译】所有算子的最新提交,摸清 0% 通过的失败分布(编译败/编译过)。不需要 NPU,秒级。
# 宿主机跑:  bash run_debug_batch.sh
set -uo pipefail
REPO="/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server"
RT="$REPO/operator_runtime_t2a"
IMAGE="${IMAGE:-ascendc-tilelang:v1-aarch64}"
RUNS="$REPO/output/ascend_operator/runs"

# 每个算子取最新 .best.tar.gz,收集 op=>tarball
declare -A TARS
while IFS= read -r f; do
  op="$(basename "$f" | sed -E 's/_impl.*//')"
  [ -z "${TARS[$op]:-}" ] && TARS[$op]="$f"
done < <(find "$RUNS" -name "*_impl.best.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)

echo "[batch] 算子数: ${#TARS[@]}"
LIST=""
for op in "${!TARS[@]}"; do LIST+="$op|${TARS[$op]}"$'\n'; done

# 一个容器内循环编译,汇总
exec docker run --rm -i \
  -v "$RT":/opt/canonical:ro \
  -v "$RUNS":"$RUNS":ro \
  -e SOC_VERSION=ascend910b1 -e ASC_DEVKIT_DIR=/opt/asc-devkit \
  -e LIST="$LIST" \
  "$IMAGE" bash -c '
    set -uo pipefail
    SK=/opt/canonical/skills/tilelang2ascend-translator
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
    printf "%-26s %-10s %s\n" "OP" "BUILD" "首个编译错误"
    printf "%s\n" "$LIST" | while IFS="|" read -r op tar; do
      [ -z "$op" ] && continue
      W=/tmp/b_$op; rm -rf "$W"; mkdir -p "$W"; tar xzf "$tar" -C "$W" 2>/dev/null
      TASK="$(dirname "$(find "$W" -name model_new_ascendc.py | head -1)")"
      [ -z "$TASK" ] && { printf "%-26s %-10s %s\n" "$op" "NO_SRC" "-"; continue; }
      OUT=$(python3 "$SK/scripts/build_ascendc.py" "$TASK" -v ascend910b1 --build-type Release 2>&1)
      if find "$TASK/kernel" -name "*.so" | grep -q .; then
        printf "%-26s %-10s %s\n" "$op" "OK(.so)" "-"
      else
        ERR=$(printf "%s" "$OUT" | grep -iE "error:|Error [0-9]" | grep -ivE "gmake|make\[" | head -1 | cut -c1-90)
        printf "%-26s %-10s %s\n" "$op" "FAIL" "${ERR:-未捕获}"
      fi
    done
  '
