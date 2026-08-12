#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

# hostctl 的端口表来自 profile（POLAR_*_PORT）。它常被单独手动拉起，不像
# start_observer.sh 那样必然有 restart_polar_host.sh 的环境，所以这里自己 source 一次。
#
# 挑哪份 profile：默认的 profile.yaml 是别的站点配置（observer 18088、别的 host），
# 直接用会拿到与在跑的服务不符的端口表。原先的做法是硬编码读 /tmp/polar_profile_runtime.yaml
# —— 那是 launcher 的渲染产物，两个并存实例会抢同一个文件，且它跟 run 目录没有关联。
#
# 现在改为按实例发现：每个在跑的实例都在 <output_root>/current 记着自己的 run id，
# 而那个 run 目录里存着它实际使用的 profile（run_artifacts/effective_profile.yaml）。
# 扫 output/*/current 就能列出所有活实例。
#   恰好一个 → 直接用它
#   多于一个 → 明确报错要求显式给 POLAR_PROFILE，而不是静默挑一个挑错
# 显式给了 POLAR_PROFILE 就完全跳过发现逻辑。
if [[ -z "${POLAR_GATEWAY_PORT:-}" ]]; then
  if [[ -z "${POLAR_PROFILE:-}" ]]; then
    _found=()
    for _cur in "${POLAR_REPO_ROOT}"/output/*/current; do
      [[ -s "${_cur}" ]] || continue
      _root="$(dirname "${_cur}")"
      _rid="$(tr -d '[:space:]' < "${_cur}")"
      _prof="${_root}/runs/${_rid}/run_artifacts/effective_profile.yaml"
      [[ -s "${_prof}" ]] && _found+=("${_prof}")
    done

    if [[ ${#_found[@]} -eq 1 ]]; then
      POLAR_PROFILE="${_found[0]}"
      echo "[hostctl] 发现唯一在跑实例，使用其 profile：${POLAR_PROFILE}"
    elif [[ ${#_found[@]} -gt 1 ]]; then
      echo "ERROR: 发现 ${#_found[@]} 个在跑的 polar 实例，无法判断要给哪个拉 hostctl。" >&2
      printf '  %s\n' "${_found[@]}" >&2
      echo "  请显式指定：POLAR_PROFILE=<上面之一，或对应的源 profile> bash $0" >&2
      exit 1
    fi
  fi
  POLAR_PROFILE="${POLAR_PROFILE:-${POLAR_DEPLOY_DIR}/profile.yaml}"
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

stop_residual() {
  local pids
  pids="$(pgrep -f "${POLAR_DEPLOY_DIR}/tools/hostctl_server.py" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "${POLAR_DEPLOY_DIR}/tools/hostctl_server.py" 2>/dev/null || true)"
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
