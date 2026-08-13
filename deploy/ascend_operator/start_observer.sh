#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

if [[ -z "${POLAR_TOPOLOGY:-}" && -f "${POLAR_DEPLOY_DIR}/profiles/profile.yaml" ]]; then
  if [[ -z "${POLAR_PYTHON:-}" && -x /root/polar-venv/bin/python ]]; then
    export POLAR_PYTHON=/root/polar-venv/bin/python
  fi
  source <("${POLAR_PYTHON:-python3}" "${POLAR_DEPLOY_DIR}/tools/load_polar_profile.py" \
    --profile "${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profiles/profile.yaml}" \
    --repo-root "${POLAR_REPO_ROOT}")
fi

ROOT="${POLAR_OUTPUT_DIR}"
POLAR_ROOT="${POLAR_REPO_ROOT}"
LOG_DIR="${POLAR_LOG_DIR}"
PID_FILE="${ROOT}/observer.pid"
NOHUP_LOG="${LOG_DIR}/observer.nohup.log"
OBSERVER_LOG="${LOG_DIR}/observer.log"
GATEWAY_URL="${POLAR_GATEWAY_URL:-http://127.0.0.1:8100}"
OBSERVER_HOST="${POLAR_OBSERVER_HOST:-0.0.0.0}"
# 不兜底 18088：真源是 profile 的 observer.port，兜底只会掩盖 loader 没跑。
OBSERVER_PORT="${POLAR_OBSERVER_PORT:?POLAR_OBSERVER_PORT 未设置：应由 load_polar_profile.py 从 profile 的 observer.port 导出}"

cd "${POLAR_DEPLOY_DIR}"
mkdir -p "${LOG_DIR}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; RED=""; BLUE=""
fi
ok_log() { printf '%s[ok]%s %-9s %s\n' "${GREEN}" "${RESET}" "$1" "$2"; }
fail_log() { printf '%s[fail]%s %-7s %s\n' "${RED}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }

find_python() {
  if [[ -n "${POLAR_PYTHON:-}" ]]; then
    printf '%s\n' "${POLAR_PYTHON}"
    return 0
  fi
  if [[ -x /root/polar-venv/bin/python ]]; then
    printf '%s\n' /root/polar-venv/bin/python
    return 0
  fi
  command -v python3
}

# 按 --port 精确匹配，别动同机其他 polar 的 observer（见 stop_observer.sh 同段注释）。
OBSERVER_PATTERN="polar_rollout_observer[.]py .*--port ${OBSERVER_PORT}([[:space:]]|\$)"

stop_existing() {
  local pids
  pids="$(pgrep -f "${OBSERVER_PATTERN}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    cleanup_log "observer" "residual pids ${pids//$'\n'/ } (port ${OBSERVER_PORT})"
    kill ${pids} 2>/dev/null || true
    sleep 1
    pids="$(pgrep -f "${OBSERVER_PATTERN}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      cleanup_log "observer" "force killing residual pids ${pids//$'\n'/ }"
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
}

stop_existing
rm -f "${PID_FILE}"

PYTHON_BIN="$(find_python)"
PYTHONPATH="${POLAR_ROOT}/src:${PYTHONPATH:-}" \
  setsid nohup "${PYTHON_BIN}" tools/polar_rollout_observer.py \
    --root "${ROOT}" \
    --results-dir "${POLAR_ROLLOUT_RESULTS_DIR}" \
    --gateway "${GATEWAY_URL}" \
    --host "${OBSERVER_HOST}" \
    --port "${OBSERVER_PORT}" \
    > "${NOHUP_LOG}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
sleep 1

if ! kill -0 "${pid}" 2>/dev/null; then
  fail_log "observer" "exited during startup"
  echo "--- ${NOHUP_LOG} ---"
  tail -n 80 "${NOHUP_LOG}" 2>/dev/null || true
  exit 1
fi

{
  printf 'Polar rollout observer: http://%s:%s\n' "${OBSERVER_HOST}" "${OBSERVER_PORT}"
  printf 'root=%s gateway=%s pid=%s\n' "${ROOT}" "${GATEWAY_URL}" "${pid}"
} > "${OBSERVER_LOG}"

ok_log "observer" "pid=${pid} url=http://${OBSERVER_HOST}:${OBSERVER_PORT} log=${OBSERVER_LOG}"
