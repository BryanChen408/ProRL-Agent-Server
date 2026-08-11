#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# 调用方显式给的 output root 优先于下面 loader 解析出来的。
# 原因：loader 读的是 POLAR_PROFILE（未给就是默认 profile.yaml），它会覆盖
# POLAR_OUTPUT_ROOT。而并存第二个 polar 时，launcher 只能把**本实例的** output root
# 传进来（source profile 还没渲染，端口是占位符，loader 会 ValueError —— 见 launcher 注释），
# 若被默认 profile 的值覆盖，就会去停别人的实例。
_caller_output_root="${POLAR_OUTPUT_ROOT:-}"

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

# 要停的 run 目录列表。不能只用 POLAR_OUTPUT_DIR：load_polar_profile.py:18 在
# POLAR_RUN_ID 未给时会生成一个**新的**时间戳 run id，于是 POLAR_OUTPUT_DIR 指向一个刚
# 造出来的空 runs/<新id>/，里面永远没有 pid —— 实测表现是四个服务全打 "no pid file"，
# 而端口上明明还有 polar 在跑（output/ascend_operator/runs/ 下那些自动 id 的空目录就是
# 这么来的）。
#
# POLAR_OUTPUT_ROOT 才是**按 profile 稳定**的那一层（由 profile 的 paths.output_dir 推得），
# 所以扫它下面所有 runs/*/ 找真正持有 pid 的那个 run。这也正是"只停自己"的隔离边界：
# 另一份 profile 有自己的 output_root，不会被扫到。
[[ -n "${_caller_output_root}" ]] && POLAR_OUTPUT_ROOT="${_caller_output_root}"

RUN_DIRS=()
if [[ -n "${POLAR_OUTPUT_ROOT:-}" && -d "${POLAR_OUTPUT_ROOT}/runs" ]]; then
  while IFS= read -r _d; do
    [[ -n "${_d}" ]] && RUN_DIRS+=("${_d%/}")
  done < <(find "${POLAR_OUTPUT_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
fi
# 兜底：POLAR_OUTPUT_ROOT 不可用时退回原行为，至少不比改之前差。
[[ ${#RUN_DIRS[@]} -eq 0 ]] && RUN_DIRS=("${ROOT}")

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

_stopped_any=0
for run_dir in "${RUN_DIRS[@]}"; do
  for name in gateway rollout; do
    pid_file="${run_dir}/${name}.pid"
    [[ -s "${pid_file}" ]] || continue

    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      stop_log "${name}" "pid=${pid} run=$(basename "${run_dir}")"
      _stopped_any=1
    else
      skip_log "${name}" "pid ${pid} not running (run=$(basename "${run_dir}"))"
    fi
    rm -f "${pid_file}"
  done

  # 只杀监听本 profile 端口的实例。原来的 pattern 是
  # "from polar.cli import main.*serve_gateway"，pgrep -f 会全机匹配 ——
  # 同机跑第二个 polar（如 PD 分离那套）时会把别人的 gateway/rollout 一起杀掉。
  # gateway/rollout 的命令行不含端口，所以按 topology 文件路径区分：
  # 每个 run 的 topology 是 run 专属的绝对路径，天然唯一。
  run_topology="${run_dir}/run_artifacts/effective_topology.yaml"
  stop_by_pattern "gateway" "serve_gateway.*${run_topology}"
  stop_by_pattern "rollout" "serve_rollout.*${run_topology}"
done

if [[ "${_stopped_any}" == "0" ]]; then
  skip_log "polar" "no running instance under ${POLAR_OUTPUT_ROOT:-${ROOT}}"
fi
