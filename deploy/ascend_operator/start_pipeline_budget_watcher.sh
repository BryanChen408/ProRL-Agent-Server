#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

if [[ -z "${POLAR_TOPOLOGY:-}" && -f "${POLAR_DEPLOY_DIR}/profile.yaml" ]]; then
  if [[ -z "${POLAR_PYTHON:-}" && -x /root/polar-venv/bin/python ]]; then
    export POLAR_PYTHON=/root/polar-venv/bin/python
  fi
  source <("${POLAR_PYTHON:-python3}" "${POLAR_DEPLOY_DIR}/tools/load_polar_profile.py" \
    --profile "${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profile.yaml}" \
    --repo-root "${POLAR_REPO_ROOT}")
fi

ROOT="${POLAR_OUTPUT_DIR}"
LOG_DIR="${POLAR_LOG_DIR}"
PID_FILE="${ROOT}/pipeline_budget_watcher.pid"
NOHUP_LOG="${LOG_DIR}/pipeline_budget_watcher.nohup.log"
WATCH_LOG="${LOG_DIR}/pipeline_budget_watcher.log"
# 不兜底 127.0.0.1:8100：真源是 profile 的 service.gateway_url。兜底会让 watcher 去盯
# 一个错的 gateway，还会让下面按 --gateway 精确杀进程的 pattern 匹配错对象。
GATEWAY_URL="${POLAR_GATEWAY_URL:?POLAR_GATEWAY_URL 未设置：应由 load_polar_profile.py 从 profile 的 service.gateway_url 导出}"
GEN_MAX="${POLAR_GEN_PIPELINE_MAX:-6}"
OPT_MAX="${POLAR_OPT_PIPELINE_MAX:-3}"
INTERVAL="${POLAR_PIPELINE_WATCH_INTERVAL:-2}"

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

# 按 --gateway 精确匹配，别动同机其他 polar 的 watcher
# （见 stop_pipeline_budget_watcher.sh 同段注释）。
WATCHER_PATTERN="polar_pipeline_budget_watcher[.]py .*--gateway $(printf '%s' "${GATEWAY_URL}" | sed -e 's/[.]/[.]/g' -e 's|/|[/]|g')([[:space:]]|\$)"

stop_existing() {
  local pids
  pids="$(pgrep -f "${WATCHER_PATTERN}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    cleanup_log "watcher" "residual pids ${pids//$'\n'/ } (gateway ${GATEWAY_URL})"
    kill ${pids} 2>/dev/null || true
    sleep 1
    pids="$(pgrep -f "${WATCHER_PATTERN}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      cleanup_log "watcher" "force killing residual pids ${pids//$'\n'/ }"
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
}

stop_existing
rm -f "${PID_FILE}"

setsid nohup python3 tools/polar_pipeline_budget_watcher.py \
  --root "${ROOT}" \
  --gateway "${GATEWAY_URL}" \
  --session-base-dir "${POLAR_SESSION_BASE_DIR}" \
  --gen-max "${GEN_MAX}" \
  --opt-max "${OPT_MAX}" \
  --interval "${INTERVAL}" \
  --log-file "${WATCH_LOG}" \
  > "${NOHUP_LOG}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
sleep 1

if ! kill -0 "${pid}" 2>/dev/null; then
  fail_log "watcher" "exited during startup"
  echo "--- ${NOHUP_LOG} ---"
  tail -n 80 "${NOHUP_LOG}" 2>/dev/null || true
  exit 1
fi

ok_log "watcher" "pid=${pid} log=${WATCH_LOG}"
tail -n 20 "${WATCH_LOG}" 2>/dev/null || true
