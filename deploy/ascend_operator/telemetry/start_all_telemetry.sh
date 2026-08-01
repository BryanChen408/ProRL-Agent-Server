#!/usr/bin/env bash
# 一条命令起全套 T1–T8 常驻采集(独立进程,不重启 polar/vime)。幂等:已在跑的跳过。
# 覆盖变量:POLAR_ENGINE_METRICS_DIR / EXPORTER_PORT / POLAR_RUNS_ROOT / TELE_INTERVAL
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${POLAR_PYTHON:-python3}"
D="${POLAR_ENGINE_METRICS_DIR:-/mnt/share/polar_engine_metrics}"
PORT="${EXPORTER_PORT:-9811}"
INT="${TELE_INTERVAL:-5}"
# runs 父目录(算子场景默认);跑别的场景就 export POLAR_RUNS_ROOT 覆盖。
RUNS_ROOT="${POLAR_RUNS_ROOT:-/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs}"
mkdir -p "$D/npu_state" "$D/vllm_state" "$D/host_state"

start() { # name pgrep-pat cmd...
  local name="$1" pat="$2"; shift 2
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "[tele] $name 已在跑,跳过"
  else
    nohup "$@" >"$HERE/${name}.nohup.log" 2>&1 &
    echo "[tele] $name 起(pid $!,日志 $HERE/${name}.nohup.log)"
  fi
}

# T1 NPU 卡负载(利用率/显存/功耗/温度)
start npu_exporter "npu_smi_exporter\.py" \
  "$PY" "$HERE/npu_smi_exporter.py" --topology "$HERE/card_topology.yaml" \
  --port "$PORT" --interval "$INT" --out "$D/npu_state"

# T2 引擎态(TTFT/TPOT/吞吐/KV/prefix,含 histogram 桶→分位)
start vllm_poller "vllm_metrics_poller\.py" \
  "$PY" "$HERE/vllm_metrics_poller.py" --topology "$HERE/card_topology.yaml" \
  --out "$D/vllm_state" --interval "$INT"

# T3 主机 + 关键进程负载(rss/cpu/线程/fd + NFS/网卡 I/O)
start host_exporter "host_proc_exporter\.py" \
  "$PY" "$HERE/host_proc_exporter.py" --out "$D/host_state" --interval "$INT"

# T5–T8 派生表聚合(span/verify/rollout/step,自动扫最新 run_dir)
if [[ -n "$RUNS_ROOT" ]]; then
  start aggregator "telemetry_aggregator\.py" \
    "$PY" "$HERE/telemetry_aggregator.py" --runs-root "$RUNS_ROOT" \
    --engine-dir "$D" --interval "${AGG_INTERVAL:-30}"
else
  echo "[tele] ⚠️ 未设 POLAR_RUNS_ROOT → T5–T8 聚合未起。设好 runs 父目录后重跑本脚本即可补上。"
fi

echo
echo "[tele] 落盘根目录:$D"
echo "  T1 → $D/npu_state/npu_card.jsonl"
echo "  T2 → $D/vllm_state/infer-*.jsonl"
echo "  T3 → $D/host_state/host_proc.jsonl"
echo "  T4 → $D/engine-*.jsonl (引擎内 polar_telemetry,随 engine 起)"
echo "  T5–T8 → <run_dir>/telemetry_derived/*.jsonl"
