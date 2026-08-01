#!/usr/bin/env bash
# 一键拉起:采集 + polar。在宿主机(有 npu-smi + 卡)上跑。
#   1) 导出 POLAR_ENGINE_METRICS_DIR(与 vllm polar_telemetry 默认一致)
#   2) 后台起 npu-smi exporter(幂等,不重复起)
#   3) 调用现有 restart_polar_host.sh 起 polar
#
# engine(vllm)由 vime 侧另起——只需保证它加载的是含 polar_telemetry 的 vllm;engine_id 自动按 --port 区分,
# 落盘目录默认 = 下面的 POLAR_ENGINE_METRICS_DIR(共享 FS,免在 vime 端设)。改路径就 export 同名变量。
#
# 用法:  bash deploy/ascend_operator/telemetry/start_telemetry_run.sh
# 可调:  EXPORTER_PORT(默认9810) TELEMETRY_DISABLE=1(只起 polar 不起采集)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEPLOY_DIR="$(cd "${HERE}/.." && pwd -P)"

export POLAR_ENGINE_METRICS_DIR="${POLAR_ENGINE_METRICS_DIR:-/mnt/share/polar_engine_metrics}"
mkdir -p "${POLAR_ENGINE_METRICS_DIR}"
EXPORTER_PORT="${EXPORTER_PORT:-9810}"

if [[ "${TELEMETRY_DISABLE:-0}" != "1" ]]; then
  # --- 全套 T1–T8 常驻采集(幂等):T1 npu / T2 vllm / T3 host / T5–T8 aggregator ---
  # 复用 start_all_telemetry.sh;端口与落盘目录透传,POLAR_RUNS_ROOT 用其默认(算子 runs)或外部覆盖。
  EXPORTER_PORT="${EXPORTER_PORT}" POLAR_ENGINE_METRICS_DIR="${POLAR_ENGINE_METRICS_DIR}" \
    bash "${HERE}/start_all_telemetry.sh" || echo "[telemetry] ⚠️ start_all_telemetry 部分失败,polar 仍继续"
  echo "[telemetry] engine 落盘目录 POLAR_ENGINE_METRICS_DIR=${POLAR_ENGINE_METRICS_DIR}"
  echo "[telemetry] 记得:vime 侧 vllm 需重启以加载 polar_telemetry(T4,engine_id 自动按 --port)"
fi

# --- 起 polar(沿用现有入口;t2a run 默认 profile.t2a.yaml,可用 POLAR_PROFILE 覆盖)---
export POLAR_PROFILE="${POLAR_PROFILE:-${DEPLOY_DIR}/profile.t2a.yaml}"
echo "[telemetry] POLAR_PROFILE=${POLAR_PROFILE}"
echo "[telemetry] 拉起 polar ..."
exec bash "${DEPLOY_DIR}/restart_polar_host.sh" "$@"
