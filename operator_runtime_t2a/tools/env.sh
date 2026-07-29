
: "${OPERATOR_PYTHON:=python3}"   # verification_ascendc.py / performance.py(需 torch_npu、上 NPU)
: "${AST_CHECK_PYTHON:=python3}"  # validate_ascendc_impl.py 纯 AST + 抢卡器本身,不碰 NPU
export OPERATOR_PYTHON AST_CHECK_PYTHON

: "${WORKSPACE_BASE:=/opt/workspace}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

: "${ASCEND_SETENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -n "${ASCEND_SETENV}" && -f "${ASCEND_SETENV}" ]]; then
  set +u; source "${ASCEND_SETENV}"; set -u
fi

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export SOC_VERSION="${SOC_VERSION:-ascend910b1}"
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-${SOC_VERSION}}"
export BUILD_TYPE="${BUILD_TYPE:-Release}"

: "${BISHENGIR_BIN:=/usr/local/Ascend/cann-9.0.0/bin}"
if [ -n "${BISHENGIR_BIN}" ] && [ -d "${BISHENGIR_BIN}" ]; then
    export PATH="${BISHENGIR_BIN}:${PATH}"
fi
