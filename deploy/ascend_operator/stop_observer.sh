#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# pid 落在 <output_root>/runs/<run_id>/ 下，而 POLAR_OUTPUT_DIR 指向的是**本次调用**推导出的
# run 目录：run_id 只存在于启动那次的进程环境里，之后 load_polar_profile.py 会再生成一个新
# 时间戳。所以直接用它找 pid 必然落空（表现是 "[skip] observer no pid file"，而 observer
# 还在跑）。start_polar_nohup.sh 会把真正在跑的 run id 写进 <output_root>/current，这里读它。
# 与 stop_polar.sh 的做法一致；读不到就保持原行为。
_pid_root="${POLAR_OUTPUT_DIR}"
if [[ -n "${POLAR_OUTPUT_ROOT:-}" && -s "${POLAR_OUTPUT_ROOT}/current" ]]; then
  _cur_run="$(tr -d '[:space:]' < "${POLAR_OUTPUT_ROOT}/current")"
  [[ -n "${_cur_run}" && -d "${POLAR_OUTPUT_ROOT}/runs/${_cur_run}" ]] \
    && _pid_root="${POLAR_OUTPUT_ROOT}/runs/${_cur_run}"
fi
PID_FILE="${_pid_root}/observer.pid"

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
