#!/usr/bin/env bash
# 拉起监控探针:检查两点 —— ①是否进入 rollout & 参数正确 ②性能采集是否使能。可反复跑。
#   bash monitor_launch.sh [<polar_run_dir>]   # 缺省取最新 run
set -uo pipefail
B=/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server
RUN_DIR="${1:-$(ls -1dt "$B/output/ascend_operator/runs/"*/ 2>/dev/null | head -1)}"
RUN_DIR="${RUN_DIR%/}"
echo "监控 run: $(basename "$RUN_DIR")  ($(stat -c '%y' "$RUN_DIR" 2>/dev/null | cut -c1-19))"
RR=$(find "$RUN_DIR/rollout_results" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
RUN_ID=$(basename "${RR:-}" | sed 's/^run_//')
TRAIN_LOG=$(ls -1t /mnt/share/c00937190/logs/train_${RUN_ID}.log 2>/dev/null | head -1)

echo
echo "========== ① 是否进入 rollout & 参数 =========="
if [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ]; then
  echo "vime 日志: $TRAIN_LOG"
  CMD=$(grep -m1 'train_async.py' "$TRAIN_LOG" 2>/dev/null)
  chk(){ echo "$CMD" | grep -q -- "$1" && echo "  ✅ $2" || echo "  ❌ 缺 $2 ($1)"; }
  chk 'debug-rollout-only' 'rollout-only(纯推理)'
  chk 'num-gpus-per-node 12' 'num-gpus-per-node 12'
  chk 'rollout-lb-proxy' 'LB proxy'
  chk 'npukernelbench_level1_ascendc' 'AscendC 数据集'
  chk 'rollout-max-context-len 262144' 'rollout ctx 262144'
  chk 'vllm-max-model-len 262144' 'vllm ctx 262144'
  echo "  -- 引擎注册(应 3 个,端口 15000/15001/15002 不撞)--"
  grep -oE 'Added worker[^"]*:[0-9]+|http://[0-9.]+:1500[0-9]' "$TRAIN_LOG" 2>/dev/null | sort -u | sed 's/^/     /' | head
  echo "  -- rollout 是否开跑 / 报错 --"
  grep -oE 'generate.*rollout|rollout_id|entering rollout|start_rollout' "$TRAIN_LOG" 2>/dev/null | tail -2 | sed 's/^/     /'
  grep -ciE 'http_400|docker cp.*exit|Traceback|CUDA|Error|重复注册|duplicate' "$TRAIN_LOG" 2>/dev/null | sed 's/^/     错误计数: /'
else
  echo "  ⏳ 尚无 vime train 日志(RUN_ID=${RUN_ID:-未知};vime 侧未起或用了别的 RUN_ID)"
fi
echo "  -- polar 侧 session 活动 --"
echo "     session 目录: $(find "$RUN_DIR/polar_sessions" -type d -name 'session-*' 2>/dev/null | wc -l) | ses_*.json: $(find "$RUN_DIR/rollout_results" -name 'ses_*.json' 2>/dev/null | wc -l)"
tail -2 "$RUN_DIR/logs/rollout.log" 2>/dev/null | sed 's/^/     rollout.log: /' | cut -c1-110

echo
echo "========== ② 性能采集是否使能 =========="
# a) npu-smi exporter
if ps aux 2>/dev/null | grep -q '[n]pu_smi_exporter'; then
  echo "  ✅ npu-smi exporter 在跑;/metrics 样本:"
  curl -s --max-time 3 http://127.0.0.1:9800/metrics 2>/dev/null | grep -E 'npu_aicore_util_pct|npu_hbm_used_mb' | head -3 | sed 's/^/     /' || echo "     ⚠️ /metrics 无响应"
else
  echo "  ❌ npu-smi exporter 未起(bash start_telemetry_run.sh 或单独起 npu_smi_exporter.py)"
fi
# b) gateway completion_metrics v2(trace_id / engine_url)
CM=$(find "$RUN_DIR" -name completion_metrics.jsonl 2>/dev/null | head -1)
if [ -n "$CM" ]; then
  row=$(tail -1 "$CM" 2>/dev/null)
  echo "  gateway completion_metrics: schema=$(echo "$row"|grep -oE '"schema_version":[0-9]+'|head -1) trace_id=$(echo "$row"|grep -oE '"trace_id":"[^"]*"'|head -1) engine_url=$(echo "$row"|grep -oE '"engine_url":"[^"]*"'|head -1)"
  echo "$row" | grep -q '"schema_version": 2\|"schema_version":2' && echo "  ✅ gateway v2 补丁已加载" || echo "  ❌ 还是 v1(gateway 未重启/未加载补丁)"
else
  echo "  ⏳ 尚无 completion_metrics.jsonl(还没推理请求)"
fi
# c) engine 侧 polar_telemetry
EM=$(find /mnt/share/polar_engine_metrics -name '*.jsonl' 2>/dev/null)
if [ -n "$EM" ]; then
  n=$(cat $EM 2>/dev/null | wc -l)
  echo "  ✅ engine_metrics 已写:$(echo "$EM"|wc -l) 文件 / $n 行;样本关键字段:"
  tail -1 $(echo "$EM"|head -1) 2>/dev/null | grep -oE '"(engine_id|trace_id|ttft_ms|prefill_ms|decode_ms|prefix_cache_hit_pct)":[^,]*' | sed 's/^/     /' | head
else
  echo "  ❌ engine_metrics 空(vllm 未加载 polar_telemetry,或引擎未重启,或还没请求)"
fi
# d) lease wait_seconds
LS=$(find "$RUN_DIR" -name 'npu_lease_status.*.json' 2>/dev/null | head -1)
if [ -n "$LS" ]; then
  grep -q wait_seconds "$LS" && echo "  ✅ npu_lease 记录 lease_wait(验证池饱和可采)" || echo "  ⚠️ npu_lease 无 wait_seconds(lease 补丁未加载)"
else
  echo "  ⏳ 尚无 npu_lease_status(还没跑到验证)"
fi
