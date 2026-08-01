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
  # --- npu-smi exporter(幂等,只按本端口判;不因别端口的旧实例而跳过)---
  if pgrep -f "npu_smi_exporter.py .*--port ${EXPORTER_PORT}\b" >/dev/null 2>&1; then
    echo "[telemetry] npu-smi exporter 已在 :${EXPORTER_PORT} 运行,跳过"
  else
    PYBIN="${POLAR_PYTHON:-python3}"
    # T1:NPU 利用率/显存 → 落文件(--out;不依赖 Prometheus)+ 同时开 /metrics 端点
    nohup "${PYBIN}" "${HERE}/npu_smi_exporter.py" \
      --topology "${HERE}/card_topology.yaml" --port "${EXPORTER_PORT}" --interval 5 \
      --out "${POLAR_ENGINE_METRICS_DIR}/npu_state" \
      >"${HERE}/exporter.nohup.log" 2>&1 &
    echo "[telemetry] npu-smi exporter 起于 :${EXPORTER_PORT}(pid $!);T1 → ${POLAR_ENGINE_METRICS_DIR}/npu_state/npu_card.jsonl"
    echo "[telemetry] ⚠️ 确认 card_topology.yaml 的卡号/端口与本机一致(npu-smi info 核对)"
  fi
  # T2:vllm /metrics(TTFT/TPOT/KV/吞吐/抢占/prefix)→ 落文件。读 topology engine_endpoints。
  if pgrep -f 'vllm_metrics_poller\.py' >/dev/null 2>&1; then
    echo "[telemetry] vllm_metrics_poller 已在运行,跳过"
  else
    nohup "${POLAR_PYTHON:-python3}" "${HERE}/vllm_metrics_poller.py" \
      --topology "${HERE}/card_topology.yaml" \
      --out "${POLAR_ENGINE_METRICS_DIR}/vllm_state" --interval 5 \
      >"${HERE}/vllm_poller.nohup.log" 2>&1 &
    echo "[telemetry] vllm_metrics_poller 起(pid $!);T2 → ${POLAR_ENGINE_METRICS_DIR}/vllm_state/*.jsonl"
  fi
  echo "[telemetry] engine 落盘目录 POLAR_ENGINE_METRICS_DIR=${POLAR_ENGINE_METRICS_DIR}"
  echo "[telemetry] 记得:vime 侧 vllm 需重启以加载 polar_telemetry(engine_id 自动按 --port)"
fi

# --- 起 polar(沿用现有入口;t2a run 默认 profile.t2a.yaml,可用 POLAR_PROFILE 覆盖)---
export POLAR_PROFILE="${POLAR_PROFILE:-${DEPLOY_DIR}/profile.t2a.yaml}"
echo "[telemetry] POLAR_PROFILE=${POLAR_PROFILE}"
echo "[telemetry] 拉起 polar ..."
exec bash "${DEPLOY_DIR}/restart_polar_host.sh" "$@"
