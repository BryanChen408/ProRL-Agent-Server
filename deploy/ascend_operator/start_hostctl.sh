#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# hostctl 的端口表来自 profile（POLAR_*_PORT）。它常被单独手动拉起，不像
# start_observer.sh 那样必然有 restart_polar_host.sh 的环境，所以这里自己 source 一次。
if [[ -z "${POLAR_GATEWAY_PORT:-}" ]]; then
  POLAR_PROFILE="${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profile.yaml}"
  source <("${POLAR_PYTHON:-python3}" "${POLAR_DEPLOY_DIR}/tools/load_polar_profile.py" \
    --profile "${POLAR_PROFILE}" --repo-root "${POLAR_REPO_ROOT}")
fi

ROOT="${POLAR_OUTPUT_DIR}"
LOG_DIR="${ROOT}/hostctl/logs"
PID_FILE="${ROOT}/hostctl/hostctl.pid"
NOHUP_LOG="${LOG_DIR}/hostctl.nohup.log"

mkdir -p "${LOG_DIR}"
cd "${POLAR_DEPLOY_DIR}"

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

stop_residual() {
  local pids
  pids="$(pgrep -f "${POLAR_DEPLOY_DIR}/tools/hostctl_server.py" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${POLAR_DEPLOY_DIR}/tools/hostctl_server.py" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
}

stop_residual
rm -f "${PID_FILE}"

PYTHON_BIN="$(find_python)"
setsid nohup "${PYTHON_BIN}" "${POLAR_DEPLOY_DIR}/tools/hostctl_server.py" \
  --root "${ROOT}" \
  --repo-root "${POLAR_REPO_ROOT}" \
  --deploy-dir "${POLAR_DEPLOY_DIR}" \
  >"${NOHUP_LOG}" 2>&1 &
echo $! >"${PID_FILE}"
sleep 1

if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[fatal] hostctl exited during startup; see ${NOHUP_LOG}" >&2
  tail -n 80 "${NOHUP_LOG}" 2>/dev/null || true
  exit 1
fi

echo "[ok] hostctl pid=$(cat "${PID_FILE}") log=${NOHUP_LOG}"
echo "[ok] token=${ROOT}/hostctl/token"
