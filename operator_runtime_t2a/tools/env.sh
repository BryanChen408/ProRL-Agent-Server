
# 解释器兜底:评测容器里 /usr/local/bin/python 和 PATH 上的 python3 都可能缺;
# 先探已知绝对安装位(含 /usr/local/python*/bin/python3),再退 PATH,最后才裸 python3(让下游报清晰错)。
_polar_py=""
for _c in /usr/local/bin/python /usr/local/python*/bin/python3 /usr/local/python*/bin/python /opt/*/bin/python3; do
  [ -x "${_c}" ] && { _polar_py="${_c}"; break; }
done
[ -z "${_polar_py}" ] && _polar_py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
: "${OPERATOR_PYTHON:=${_polar_py:-python3}}"   # verification_ascendc.py / performance.py(需 torch_npu、上 NPU)
: "${AST_CHECK_PYTHON:=${_polar_py:-python3}}"  # validate_ascendc_impl.py 纯 AST + 抢卡器本身,不碰 NPU
export OPERATOR_PYTHON AST_CHECK_PYTHON
unset _polar_py _c

: "${WORKSPACE_BASE:=/opt/workspace}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

: "${ASCEND_SETENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -n "${ASCEND_SETENV}" && -f "${ASCEND_SETENV}" ]]; then
  set +u; source "${ASCEND_SETENV}"; set -u
fi

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9382}"
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-${SOC_VERSION}}"
export BUILD_TYPE="${BUILD_TYPE:-Release}"

: "${BISHENGIR_BIN:=/usr/local/Ascend/cann-9.0.0/bin}"
if [ -n "${BISHENGIR_BIN}" ] && [ -d "${BISHENGIR_BIN}" ]; then
    export PATH="${BISHENGIR_BIN}:${PATH}"
fi
