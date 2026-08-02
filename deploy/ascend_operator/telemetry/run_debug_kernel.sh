#!/usr/bin/env bash
# 一键:在判分沙箱容器里复现 kernel build→load→register,定位 op 未注册根因。
# 宿主机直接跑:  bash run_debug_kernel.sh            # 默认 3_Add
#               bash run_debug_kernel.sh 1_GELU     # 指定算子
#               OP=23_RepeatInterleave IMAGE=xxx bash run_debug_kernel.sh
# 自动找该算子最新的 submission tarball,拼好挂载与 env,不用手动复制路径。
set -uo pipefail

OP="${1:-${OP:-3_Add}}"
REPO="/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server"
RT="$REPO/operator_runtime_t2a"
IMAGE="${IMAGE:-ascendc-tilelang:v1-aarch64}"
DBG="$REPO/deploy/ascend_operator/telemetry/debug_kernel_load.sh"

# 找该算子最新的提交 tarball(优先 .best)
TAR="$(find "$REPO/output/ascend_operator/runs" -path "*${OP}*" -name "*impl*.best.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
[ -z "$TAR" ] && TAR="$(find "$REPO/output/ascend_operator/runs" -path "*${OP}*" -name "*impl*.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "$TAR" ]; then echo "找不到 $OP 的 submission tarball,换个算子或确认已有提交"; exit 1; fi
echo "[run_debug] OP=$OP"
echo "[run_debug] TAR=$TAR"
echo "[run_debug] IMAGE=$IMAGE"

# golden 数据集(STEP4 对拍要 golden model.py)
GOLDEN="${GOLDEN:-/home/docker/datasets/op_tasks/npukernelbench_level1_ascendc/op_tasks}"

# NPU 设备:STEP1 编译不必需,但 STEP4 对拍要卡。默认开(NPU_DEV=0 可关,只做 build/register 检查)。
NPU_ARGS=()
if [ "${NPU_DEV:-1}" = "1" ]; then
  for d in davinci0 davinci1 davinci2 davinci3 davinci_manager devmm_svm hisi_hdc upgrade; do
    [ -e "/dev/$d" ] && NPU_ARGS+=(--device "/dev/$d")
  done
  [ -d /usr/local/Ascend/driver ] && NPU_ARGS+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro)
  echo "[run_debug] NPU 设备: ${#NPU_ARGS[@]} 项(NPU_DEV=0 可关)"
fi
[ -d "$GOLDEN" ] && GOLDEN_ARG=(-v "$GOLDEN":/opt/golden:ro) || GOLDEN_ARG=()
echo "[run_debug] golden: ${GOLDEN:-无}"

exec docker run --rm -i \
  -v "$RT":/opt/canonical:ro \
  -v "$RT/tools":/opt/workspace/agent_workdir/tools:ro \
  -v "$TAR":/tmp/impl.tar.gz:ro \
  -v "$DBG":/tmp/dbg.sh:ro \
  "${GOLDEN_ARG[@]}" "${NPU_ARGS[@]}" \
  -e SOC_VERSION=ascend910b1 -e ASC_DEVKIT_DIR=/opt/asc-devkit -e GOLDEN_DIR=/opt/golden \
  "$IMAGE" bash /tmp/dbg.sh "$OP" /tmp/impl.tar.gz
