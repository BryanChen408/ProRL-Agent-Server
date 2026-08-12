#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

if [[ -z "${POLAR_TOPOLOGY:-}" && -f "${POLAR_DEPLOY_DIR}/profile.yaml" ]]; then
  if [[ -z "${POLAR_PYTHON:-}" && -x /root/polar-venv/bin/python ]]; then
    export POLAR_PYTHON=/root/polar-venv/bin/python
  fi
  PYTHON_FOR_PROFILE="${POLAR_PYTHON:-python3}"
  source <("${PYTHON_FOR_PROFILE}" "${POLAR_DEPLOY_DIR}/tools/load_polar_profile.py" \
    --profile "${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profile.yaml}" \
    --repo-root "${POLAR_REPO_ROOT}")
fi

ROOT="${POLAR_OUTPUT_DIR}"
TOPOLOGY="${POLAR_TOPOLOGY:-${POLAR_OUTPUT_DIR}/run_artifacts/effective_topology.yaml}"

# POLAR_OUTPUT_DIR 指向的是**本次调用**推导出的 runs/<run_id>/，而不是在跑那个实例的目录：
# run_id 只存在于启动时的进程环境里，stop 时读不到，load_polar_profile.py:18 会再生成一个新的
# 时间戳。所以直接用它找 pid 必然落空 —— 表现是四个服务全打 "no pid file" 而服务还在跑，
# 且每次 stop 都在 runs/ 下留一个空目录。
#
# start_polar_nohup.sh 现在会把 run id 写进 <output_root>/current，这里读它来定位真正在跑的
# run。指针是 per-profile 的（output_root 由 profile 的 paths.output_dir 推得），所以同机并存
# 多个 polar 时各停自己的，不会互相误杀。
#
# 读不到指针就保持原行为：兼容用旧代码起、没写过 current 的实例。
if [[ -n "${POLAR_OUTPUT_ROOT:-}" && -s "${POLAR_OUTPUT_ROOT}/current" ]]; then
  _cur_run="$(tr -d '[:space:]' < "${POLAR_OUTPUT_ROOT}/current")"
  if [[ -n "${_cur_run}" && -d "${POLAR_OUTPUT_ROOT}/runs/${_cur_run}" ]]; then
    ROOT="${POLAR_OUTPUT_ROOT}/runs/${_cur_run}"
    TOPOLOGY="${ROOT}/run_artifacts/effective_topology.yaml"
  fi
fi

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

# 只杀监听本 profile 端口的实例。原来的 pattern 是
# "from polar.cli import main.*serve_gateway"，pgrep -f 会全机匹配 ——
# 同机跑第二个 polar（如 PD 分离那套）时会把别人的 gateway/rollout 一起杀掉。
# gateway/rollout 的命令行不含端口，所以按 topology 文件路径区分：
# 每个 run 的 topology 是 run 专属的绝对路径，天然唯一。                                                                                               
stop_by_pattern "gateway" "serve_gateway.*${TOPOLOGY}"
stop_by_pattern "rollout" "serve_rollout.*${TOPOLOGY}"
