#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

if [[ -z "${POLAR_PYTHON:-}" && -x /root/polar-venv/bin/python ]]; then
  export POLAR_PYTHON=/root/polar-venv/bin/python
fi
POLAR_PROFILE="${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profiles/profile.yaml}"
PYTHON_FOR_PROFILE="${POLAR_PYTHON:-python3}"
source <("${PYTHON_FOR_PROFILE}" "${POLAR_DEPLOY_DIR}/tools/load_polar_profile.py" \
  --profile "${POLAR_PROFILE}" \
  --repo-root "${POLAR_REPO_ROOT}")

ROOT="${POLAR_OUTPUT_DIR}"
POLAR_ROOT="${POLAR_REPO_ROOT}"
TOPOLOGY="${POLAR_TOPOLOGY}"
LOG_DIR="${POLAR_LOG_DIR}"
GATEWAY_URL="${POLAR_GATEWAY_URL}"
ROLLOUT_URL="${POLAR_ROLLOUT_URL}"

mkdir -p "${LOG_DIR}" "${ROOT}/rollout_results"
mkdir -p "${POLAR_SESSION_BASE_DIR}" "${POLAR_OP_ASSETS_DIR}" "${POLAR_ROLLOUT_RESULTS_DIR}"
export POLAR_KEEP_SESSION_DIR="${POLAR_KEEP_SESSION_DIR:-1}"
export POLAR_MASK_REASONING="${POLAR_MASK_REASONING:-0}"

cd "${POLAR_ROOT}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
else
  RESET=""; GREEN=""; YELLOW=""; BLUE=""
fi
start_log() { printf '%s[start]%s %-8s pid=%s log=%s\n' "${GREEN}" "${RESET}" "$1" "$2" "$3"; }
skip_log() { printf '%s[skip]%s %-9s %s\n' "${YELLOW}" "${RESET}" "$1" "$2"; }
cleanup_log() { printf '%s[cleanup]%s %-6s %s\n' "${BLUE}" "${RESET}" "$1" "$2"; }
info_log() { printf '[info] %s\n' "$*"; }

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[fatal] docker CLI is required where Polar gateway runs. Start this script from the host Docker environment, not from the slime training container." >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[fatal] docker daemon is not reachable from this environment; Polar DockerRuntime cannot start agent containers." >&2
    return 1
  fi
}

kill_residual() {
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

find_python() {
  if [[ -n "${POLAR_PYTHON:-}" ]]; then
    printf '%s\n' "${POLAR_PYTHON}"
    return 0
  fi

  local candidates=(
    "${VIRTUAL_ENV:-}/bin/python"
    "/root/polar-venv/bin/python"
    "${POLAR_ROOT}/.venv/bin/python"
    python3.13
    python3.12
    python3.11
    python3
  )
  local candidate resolved
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      resolved="${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      resolved="$(command -v "${candidate}")"
    else
      continue
    fi
    if "${resolved}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done

  echo "No Python >= 3.11 found. Set POLAR_PYTHON=/path/to/python3.11." >&2
  return 1
}

PYTHON_BIN="$(find_python)"
info_log "python=${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
info_log "profile=${POLAR_PROFILE}"
info_log "topology=${TOPOLOGY}"
info_log "inference_request_timeout_seconds=${POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS}"
require_docker

kill_residual "gateway" "from polar.cli import main.*serve_gateway.*${TOPOLOGY}"
kill_residual "rollout" "from polar.cli import main.*serve_rollout.*${TOPOLOGY}"
rm -f "${ROOT}/gateway.pid" "${ROOT}/rollout.pid"

start_one() {
  local name="$1"
  shift
  local pid_file="${ROOT}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    skip_log "${name}" "already running pid=$(cat "${pid_file}")"
    return 0
  fi

  PYTHONPATH="${POLAR_ROOT}/src:${PYTHONPATH:-}" \
    setsid nohup "${PYTHON_BIN}" -c 'from polar.cli import main; raise SystemExit(main())' "$@" \
    >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
  start_log "${name}" "$(cat "${pid_file}")" "${log_file}"
}

# 当前实例指针。pid 落在 <output_root>/runs/<run_id>/ 下，而 run_id 只存在于本次启动的
# 进程环境里（launcher 是行内前缀赋值 POLAR_RUN_ID=...，load_polar_profile.py 只从环境变量
# 读，没有任何一处落盘）。于是之后在别的 shell 里跑 stop / hostctl 时，run_id 已经消失，
# load_polar_profile.py 会**再生成一个新的**，指向刚 mkdir 出来的空目录 —— 表现就是
# "no pid file" 而实际服务还在跑。
#
# 所以这里把 run id 记到 <output_root>/current。output_root 由 profile 的 paths.output_dir
# 推得、每份 profile 一个，所以这个指针天然是 per-instance 的，不需要额外的实例 id 概念。
#
# 写在启动之前：即使下面 start_one 中途失败，指针也指向本次 run，stop 仍能找到残留 pid。
# 原子写：stop 可能正在并发读。
if [[ -n "${POLAR_OUTPUT_ROOT:-}" && -n "${POLAR_RUN_ID:-}" ]]; then
  mkdir -p "${POLAR_OUTPUT_ROOT}"
  _cur_tmp="${POLAR_OUTPUT_ROOT}/.current.tmp.$$"
  if printf '%s\n' "${POLAR_RUN_ID}" > "${_cur_tmp}" && mv -f "${_cur_tmp}" "${POLAR_OUTPUT_ROOT}/current"; then
    info_log "current run pointer -> ${POLAR_OUTPUT_ROOT}/current (${POLAR_RUN_ID})"
  else
    rm -f "${_cur_tmp}"
    echo "[warn] 无法写当前实例指针 ${POLAR_OUTPUT_ROOT}/current；stop 将退回扫描历史 run" >&2
  fi
fi

start_one rollout serve_rollout -c "${TOPOLOGY}"
start_one gateway serve_gateway -c "${TOPOLOGY}" --node-id ascend-node-01

bash "${POLAR_DEPLOY_DIR}/start_pipeline_budget_watcher.sh"
bash "${POLAR_DEPLOY_DIR}/start_observer.sh"

info_log "health: curl -s ${ROLLOUT_URL%/}/health"
info_log "health: curl -s ${GATEWAY_URL%/}/health"
