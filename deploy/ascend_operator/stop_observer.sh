#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

PID_FILE="${POLAR_OUTPUT_DIR}/observer.pid"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; BLUE=""
fi
stop_log() { printf '%s[stop]%s %-9s %s\n' "${GREEN}" "${RESET}" "$1" "$2"; }
skip_log() { printf '%s[skip]%s %-9s %s\n' "${YELLOW}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }

# 只杀监听本 profile 端口的 observer。原 pattern 是裸的 polar_rollout_observer.py，
# pgrep -f 全机匹配 —— 同机跑第二个 polar 时会连它的 observer 一起杀，即使端口不同。
# 与 stop_polar.sh 里 gateway/rollout 按 topology 路径区分的做法对齐。
OBSERVER_PORT="${POLAR_OBSERVER_PORT:?POLAR_OBSERVER_PORT 未设置：应由 load_polar_profile.py 从 profile 的 observer.port 导出}"
OBSERVER_PATTERN="polar_rollout_observer[.]py .*--port ${OBSERVER_PORT}([[:space:]]|\$)"

stop_pattern() {
  local pids
  pids="$(pgrep -f "${OBSERVER_PATTERN}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  cleanup_log "observer" "residual pids ${pids//$'\n'/ } (port ${OBSERVER_PORT})"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${OBSERVER_PATTERN}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    cleanup_log "observer" "force killing residual pids ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    if ps -fp "${pid}" | grep -q "polar_rollout_observer.py"; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
      stop_log "observer" "pid=${pid}"
    else
      skip_log "observer" "pid file does not point to observer: pid=${pid}"
    fi
  else
    skip_log "observer" "pid not running"
  fi
else
  skip_log "observer" "no pid file"
fi

stop_pattern
rm -f "${PID_FILE}"
