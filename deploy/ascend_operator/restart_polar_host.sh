#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

ROOT="${POLAR_OUTPUT_DIR}"
TOPOLOGY="${POLAR_TOPOLOGY:-${POLAR_RUN_CONFIG_DIR}/topology.rendered.yaml}"
ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://127.0.0.1:8080}"
GATEWAY_URL="${POLAR_GATEWAY_URL:-http://127.0.0.1:8100}"
EXTRA_STALE_PORTS="${POLAR_EXTRA_STALE_GATEWAY_PORTS:-8110}"
SKIP_INTERNAL_PORT_CLEANUP="${POLAR_SKIP_INTERNAL_PORT_CLEANUP:-0}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BLUE=$'\033[34m'
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; RED=""; BLUE=""
fi

section() { printf '\n%s== %s ==%s\n' "${BOLD}${BLUE}" "$1" "${RESET}"; }
ok() { printf '%s[ok]%s %s\n' "${GREEN}" "${RESET}" "$*"; }
info() { printf '%s[info]%s %s\n' "${DIM}" "${RESET}" "$*"; }
fail() { printf '%s[fail]%s %s\n' "${RED}" "${RESET}" "$*" >&2; }

mkdir -p "${ROOT}" "${POLAR_LOG_DIR}" "${POLAR_RUN_CONFIG_DIR}"
cd "${POLAR_DEPLOY_DIR}"

if [[ -x /root/polar-venv/bin/python ]]; then
  export POLAR_PYTHON=/root/polar-venv/bin/python
fi

section "Polar Host Restart"
info "root=${ROOT}"
info "topology=${TOPOLOGY}"
info "rollout=${ROLLOUT_URL}"
info "gateway=${GATEWAY_URL}"
info "stale_ports=${EXTRA_STALE_PORTS:-none}"
info "observer=http://0.0.0.0:${POLAR_OBSERVER_PORT:-18088}"
info "skip_internal_port_cleanup=${SKIP_INTERNAL_PORT_CLEANUP}"

url_port() {
  "${POLAR_PYTHON:-python3}" - "$1" <<'PY'
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
if url.port is not None:
    print(url.port)
elif url.scheme == "https":
    print(443)
else:
    print(80)
PY
}

port_pids() {
  local port="$1"
  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi
  fuser -n tcp "${port}" 2>/dev/null || true
}

kill_port_listeners() {
  local port="$1"
  local pids
  pids="$(port_pids "${port}")"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  info "killing listeners on tcp/${port}: ${pids//$'\n'/ }"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(port_pids "${port}")"
  if [[ -n "${pids}" ]]; then
    info "force killing listeners on tcp/${port}: ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

port_state() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp | grep -E ":${port}\\b" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
  elif command -v fuser >/dev/null 2>&1; then
    local pids
    pids="$(port_pids "${port}")"
    if [[ -n "${pids}" ]]; then
      printf 'tcp/%s listeners: %s\n' "${port}" "${pids//$'\n'/ }"
    fi
  fi
}

ROLLOUT_PORT="$(url_port "${ROLLOUT_URL}")"
GATEWAY_PORT="$(url_port "${GATEWAY_URL}")"

section "Stop Existing Services"
bash stop_polar.sh
if [[ "${SKIP_INTERNAL_PORT_CLEANUP}" == "1" ]]; then
  info "skipping internal fuser cleanup; caller is responsible for safe port cleanup"
else
  kill_port_listeners "${ROLLOUT_PORT}"
  kill_port_listeners "${GATEWAY_PORT}"
  for stale_port in ${EXTRA_STALE_PORTS//,/ }; do
    if [[ -n "${stale_port}" && "${stale_port}" =~ ^[0-9]+$ ]]; then
      kill_port_listeners "${stale_port}"
    fi
  done
fi

remaining_port_state="$(
  {
    port_state "${ROLLOUT_PORT}"
    port_state "${GATEWAY_PORT}"
    for stale_port in ${EXTRA_STALE_PORTS//,/ }; do
      if [[ -n "${stale_port}" && "${stale_port}" =~ ^[0-9]+$ ]]; then
        port_state "${stale_port}"
      fi
    done
  } | sed '/^[[:space:]]*$/d'
)"
if [[ -n "${remaining_port_state}" ]]; then
  fail "Polar ports are still occupied after stop_polar.sh"
  printf '%s\n' "${remaining_port_state}" >&2
  fail "Kill the listed old processes, then rerun this script."
  exit 1
fi
ok "ports ${ROLLOUT_PORT}/${GATEWAY_PORT}${EXTRA_STALE_PORTS:+ plus ${EXTRA_STALE_PORTS}} are free"

section "Start Services"
bash "${POLAR_DEPLOY_DIR}/start_polar_nohup.sh"

section "Health"
sleep 2
rollout_health="$(curl -fsS "${ROLLOUT_URL%/}/health")"
gateway_health="$(curl -fsS "${GATEWAY_URL%/}/health")"
nodes="$(curl -fsS "${ROLLOUT_URL%/}/nodes")"
ok "rollout  ${ROLLOUT_URL%/}/health  ${rollout_health}"
ok "gateway  ${GATEWAY_URL%/}/health  ${gateway_health}"
ok "nodes    ${ROLLOUT_URL%/}/nodes    ${nodes}"

section "Summary"
ok "Polar services restarted"
info "observer: http://<host-ip>:${POLAR_OBSERVER_PORT:-18088}"
info "logs: ${POLAR_LOG_DIR}/{rollout,gateway,pipeline_budget_watcher,observer}.log"
