#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

ROOT="${POLAR_OUTPUT_DIR}"
SLIME_ROOT=${SLIME_ROOT:-/workspace/slime-ascend}
TOPOLOGY=${POLAR_TOPOLOGY:-${POLAR_RUN_CONFIG_DIR}/topology.isolated.yaml}
ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://127.0.0.1:28080}
GATEWAY_URL=${POLAR_GATEWAY_URL:-http://127.0.0.1:28100}
ROUTER_PORT=${SGLANG_ROUTER_PORT:-24077}
RUN_ID=${RUN_ID:-smoke_isolated_$(date +%Y%m%d-%H%M%S)}

export POLAR_E2E_DIR="${ROOT}"
export POLAR_TOPOLOGY="${TOPOLOGY}"
export POLAR_ROLLOUT_URL="${ROLLOUT_URL}"
export POLAR_GATEWAY_URL="${GATEWAY_URL}"
export SGLANG_ROUTER_PORT="${ROUTER_PORT}"
export RUN_ID
export LOG_FILE="${LOG_FILE:-${POLAR_LOG_DIR}/train_polar_e2e_${RUN_ID}.log}"

echo "[stage] start isolated Polar services"
bash "${POLAR_DEPLOY_DIR}/start_polar_nohup.sh"

echo "[stage] probe Polar DockerRuntime"
python3 "${POLAR_DEPLOY_DIR}/tools/probe_gateway_runtime.py" \
  --gateway-url "${GATEWAY_URL}" \
  --image "${POLAR_RUNTIME_PROBE_IMAGE:-sandbox:v1}" \
  --pool "${POLAR_RUNTIME_PROBE_POOL:-8,9,10,11}" \
  --lock-dir "${POLAR_RUNTIME_PROBE_LOCK_DIR:-/dev/shm/polar-npu-locks}" \
  --skills-dir "${POLAR_RUNTIME_PROBE_SKILLS_DIR:-${POLAR_OPERATOR_RUNTIME_DIR}/tools}" \
  --timeout "${POLAR_RUNTIME_PROBE_TIMEOUT:-60}"

echo "[stage] launch smoke training RUN_ID=${RUN_ID} LOG_FILE=${LOG_FILE}"
exec bash "${SLIME_ROOT}/scripts/ascend_script/run-polar-e2e-smoke-qwen36-35b-a2.sh"
