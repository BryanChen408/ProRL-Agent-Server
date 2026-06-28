#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

ROOT="${POLAR_OUTPUT_DIR}"
POLAR_ROOT="${POLAR_REPO_ROOT}"
BASE_TOPOLOGY="${POLAR_TOPOLOGY_TEMPLATE:-${POLAR_DEPLOY_DIR}/topology.yaml}"
TOPOLOGY="${POLAR_TOPOLOGY:-${POLAR_RUN_CONFIG_DIR}/topology.rendered.yaml}"
LOG_DIR="${POLAR_LOG_DIR}"
GATEWAY_URL="${POLAR_GATEWAY_URL:-http://127.0.0.1:8100}"
ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://127.0.0.1:8080}"
SGLANG_ROUTER_URL="${SGLANG_ROUTER_URL:-}"
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS="${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-32768}"
export POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS="${POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS:-14400}"

mkdir -p "${LOG_DIR}" "${ROOT}/rollout_results"
mkdir -p "${POLAR_SESSION_BASE_DIR}" "${POLAR_OP_ASSETS_DIR}" "${POLAR_RUN_CONFIG_DIR}" "${POLAR_ROLLOUT_RESULTS_DIR}"
export POLAR_KEEP_SESSION_DIR="${POLAR_KEEP_SESSION_DIR:-1}"

cd "${POLAR_ROOT}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; BLUE=""
fi
start_log() { printf '%s[start]%s %-8s pid=%s log=%s\n' "${GREEN}" "${RESET}" "$1" "$2" "$3"; }
skip_log() { printf '%s[skip]%s %-9s %s\n' "${YELLOW}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }
info_log() { printf '[info] %s\n' "$*"; }

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[fatal] docker CLI is required where Polar gateway runs. Start this script from the host Docker environment, not from the slime training container." >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[fatal] docker daemon is not reachable from this environment; Polar DockerRuntime cannot start agent containers." >&2
    return 1
  fi
}

kill_residual() {
  local name="$1"
  local pattern="$2"
  local pids
  pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  cleanup_log "${name}" "residual pids ${pids//$'\n'/ }"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    cleanup_log "${name}" "force killing residual pids ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

find_python() {
  if [[ -n "${POLAR_PYTHON:-}" ]]; then
    printf '%s\n' "${POLAR_PYTHON}"
    return 0
  fi

  local candidates=(
    "${VIRTUAL_ENV:-}/bin/python"
    "/root/polar-venv/bin/python"
    "${POLAR_ROOT}/.venv/bin/python"
    python3.13
    python3.12
    python3.11
    python3
  )
  local candidate resolved
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      resolved="${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      resolved="$(command -v "${candidate}")"
    else
      continue
    fi
    if "${resolved}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done

  echo "No Python >= 3.11 found. Set POLAR_PYTHON=/path/to/python3.11." >&2
  return 1
}

PYTHON_BIN="$(find_python)"
info_log "python=${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
info_log "inference_request_timeout_seconds=${POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS}"
require_docker

if [[ -z "${SGLANG_ROUTER_URL}" ]]; then
  SGLANG_ROUTER_URL="http://${SGLANG_ROUTER_IP:-127.0.0.1}:${SGLANG_ROUTER_PORT:-4077}"
fi
render_host_args=()
if [[ -n "${POLAR_BIND_HOST:-}" ]]; then
  render_host_args=(--host "${POLAR_BIND_HOST}")
fi
"${PYTHON_BIN}" "${POLAR_DEPLOY_DIR}/tools/render_run_topology.py" \
  --topology "${BASE_TOPOLOGY}" \
  --output "${TOPOLOGY}" \
  --rollout-url "${ROLLOUT_URL}" \
  --gateway-url "${GATEWAY_URL}" \
  --router-url "${SGLANG_ROUTER_URL}" \
  "${render_host_args[@]}" \
  --operator-runtime-dir "${POLAR_OPERATOR_RUNTIME_DIR}" \
  --op-assets-dir "${POLAR_OP_ASSETS_DIR}" \
  --rollout-results-dir "${POLAR_ROLLOUT_RESULTS_DIR}" >/dev/null
info_log "topology=${TOPOLOGY}"

kill_residual "gateway" "from polar.cli import main.*serve_gateway.*${TOPOLOGY}"
kill_residual "rollout" "from polar.cli import main.*serve_rollout.*${TOPOLOGY}"
rm -f "${ROOT}/gateway.pid" "${ROOT}/rollout.pid"

start_one() {
  local name="$1"
  shift
  local pid_file="${ROOT}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    skip_log "${name}" "already running pid=$(cat "${pid_file}")"
    return 0
  fi

  PYTHONPATH="${POLAR_ROOT}/src:${PYTHONPATH:-}" \
    setsid nohup "${PYTHON_BIN}" -c 'from polar.cli import main; raise SystemExit(main())' "$@" \
    >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
  start_log "${name}" "$(cat "${pid_file}")" "${log_file}"
}

start_one rollout serve_rollout -c "${TOPOLOGY}"
start_one gateway serve_gateway -c "${TOPOLOGY}" --node-id ascend-node-01

POLAR_GATEWAY_URL="${GATEWAY_URL}" bash "${ROOT}/start_pipeline_budget_watcher.sh"
POLAR_GATEWAY_URL="${GATEWAY_URL}" bash "${ROOT}/start_observer.sh"

info_log "health: curl -s ${ROLLOUT_URL%/}/health"
info_log "health: curl -s ${GATEWAY_URL%/}/health"
