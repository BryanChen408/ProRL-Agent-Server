#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

PID_FILE="${POLAR_OUTPUT_DIR}/hostctl/hostctl.pid"
HOSTCTL_SERVER="${POLAR_DEPLOY_DIR}/tools/hostctl_server.py"

# 只杀属于本实例的 hostctl。裸 pgrep -f "${HOSTCTL_SERVER}" 是全机匹配：POLAR_DEPLOY_DIR
# 由脚本位置推得，同机并存的两个 polar 共用同一个仓，所以那个 pattern 会同时命中对方的
# hostctl，停一个把另一个也杀掉。与 stop_observer.sh 按 --port、
# stop_pipeline_budget_watcher.sh 按 --gateway 区分的做法对齐。
#
# 区分用 output root 而不是完整的 --root 值：--root 是 <output_root>/runs/<run_id>，run_id
# 每次启动都变，用它做 pattern 会漏掉同实例上一轮残留的 hostctl。output root 按 profile
# 稳定，正是实例边界。
#
# 两种形态都要兼容：本脚本只 source _paths.sh、不跑 loader，此时 POLAR_OUTPUT_DIR 就是
# output root；而 start_hostctl.sh 跑过 loader，它启动时传的是 runs/<run_id>。所以 pattern
# 结尾允许紧跟空白（前者）或 /（后者）。
_out_root="${POLAR_OUTPUT_ROOT:-${POLAR_OUTPUT_DIR%%/runs/*}}"
_root_re="$(printf '%s' "${_out_root}" | sed -e 's/[.]/[.]/g' -e 's|/|[/]|g')"
HOSTCTL_PATTERN="hostctl_server[.]py .*--root ${_root_re}([[:space:]]|[/])"

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

pids="$(pgrep -f "${HOSTCTL_PATTERN}" 2>/dev/null || true)"
if [[ -n "${pids}" ]]; then
  echo "[cleanup] hostctl residual pids ${pids//$'\n'/ } (root ${_out_root})"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${HOSTCTL_PATTERN}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
fi

rm -f "${PID_FILE}"
