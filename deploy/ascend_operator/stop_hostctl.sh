#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

PID_FILE="${POLAR_OUTPUT_DIR}/hostctl/hostctl.pid"
HOSTCTL_SERVER="${POLAR_DEPLOY_DIR}/tools/hostctl_server.py"

stop_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  if ps -fp "${pid}" | grep -q "${HOSTCTL_SERVER}"; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
    echo "[stop] hostctl pid=${pid}"
  fi
}

if [[ -f "${PID_FILE}" ]]; then
  stop_pid "$(cat "${PID_FILE}" 2>/dev/null || true)"
fi

pids="$(pgrep -f "${HOSTCTL_SERVER}" 2>/dev/null || true)"
if [[ -n "${pids}" ]]; then
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${HOSTCTL_SERVER}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
fi

rm -f "${PID_FILE}"
