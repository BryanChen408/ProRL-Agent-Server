#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

PID_FILE="${POLAR_OUTPUT_DIR}/pipeline_budget_watcher.pid"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; BLUE=""
fi
stop_log() { printf '%s[stop]%s %-9s %s\n' "${GREEN}" "${RESET}" "$1" "$2"; }
skip_log() { printf '%s[skip]%s %-9s %s\n' "${YELLOW}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }

stop_pattern() {
  local pids
  pids="$(pgrep -f "polar_pipeline_budget_watcher.py" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  cleanup_log "watcher" "residual pids ${pids//$'\n'/ }"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "polar_pipeline_budget_watcher.py" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    cleanup_log "watcher" "force killing residual pids ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    if ps -fp "${pid}" | grep -q "polar_pipeline_budget_watcher.py"; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
      stop_log "watcher" "pid=${pid}"
    else
      skip_log "watcher" "pid file does not point to watcher: pid=${pid}"
    fi
  else
    skip_log "watcher" "pid not running"
  fi
else
  skip_log "watcher" "no pid file"
fi

stop_pattern
rm -f "${PID_FILE}"
