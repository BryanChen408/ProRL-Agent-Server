#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# hostctl 的端口表来自 profile（POLAR_*_PORT）。它常被单独手动拉起，不像
# start_observer.sh 那样必然有 restart_polar_host.sh 的环境，所以这里自己 source 一次。
if [[ -z "${POLAR_GATEWAY_PORT:-}" ]]; then
  # 优先用 launcher 渲染的运行时 profile。默认的 profile.yaml 是别的站点配置
  # （observer 18088、别的 host），手动拉 hostctl 时会拿到与在跑的服务不符的端口表。
  if [[ -z "${POLAR_PROFILE:-}" && -s /tmp/polar_profile_runtime.yaml ]]; then
    POLAR_PROFILE=/tmp/polar_profile_runtime.yaml
  fi
  POLAR_PROFILE="${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profiles/profile.yaml}"
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

# 只清本实例的 hostctl 残留。裸 pgrep -f "<deploy_dir>/tools/hostctl_server.py" 是全机匹配
# ——POLAR_DEPLOY_DIR 由脚本位置推得，同机并存的两个 polar 共用同一个仓，那个 pattern 会
# 连对方的 hostctl 一起杀。用 output root 区分（不是完整 --root 值：那里含 run_id，每次启动
# 都变，会漏掉同实例上一轮的残留）。与 stop_hostctl.sh 里同名 pattern 保持一致。
_out_root="${POLAR_OUTPUT_ROOT:-${POLAR_OUTPUT_DIR%%/runs/*}}"
_root_re="$(printf '%s' "${_out_root}" | sed -e 's/[.]/[.]/g' -e 's|/|[/]|g')"
HOSTCTL_PATTERN="hostctl_server[.]py .*--root ${_root_re}([[:space:]]|[/])"

stop_residual() {
  local pids
  pids="$(pgrep -f "${HOSTCTL_PATTERN}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[cleanup] hostctl residual pids ${pids//$'\n'/ } (root ${_out_root})"
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${HOSTCTL_PATTERN}" 2>/dev/null || true)"
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
