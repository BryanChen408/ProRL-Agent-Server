#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

ROOT="${POLAR_OUTPUT_DIR}"
TOPOLOGY="${POLAR_TOPOLOGY:-${POLAR_RUN_CONFIG_DIR}/topology.rendered.yaml}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; BLUE=""
fi
stop_log() { printf '%s[stop]%s %-9s %s\n' "${GREEN}" "${RESET}" "$1" "$2"; }
skip_log() { printf '%s[skip]%s %-9s %s\n' "${YELLOW}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }

bash "${POLAR_DEPLOY_DIR}/stop_pipeline_budget_watcher.sh" || true
bash "${POLAR_DEPLOY_DIR}/stop_observer.sh" || true

stop_by_pattern() {
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

for name in gateway rollout; do
  pid_file="${ROOT}/${name}.pid"
  if [[ ! -s "${pid_file}" ]]; then
    skip_log "${name}" "no pid file"
    continue
  fi

  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    stop_log "${name}" "pid=${pid}"
  else
    skip_log "${name}" "pid ${pid} not running"
  fi
  rm -f "${pid_file}"
done

stop_by_pattern "gateway" "from polar.cli import main.*serve_gateway.*${TOPOLOGY}"
stop_by_pattern "rollout" "from polar.cli import main.*serve_rollout.*${TOPOLOGY}"
