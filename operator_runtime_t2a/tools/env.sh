# shellcheck shell=bash
# AscendC 算子工具链环境配置。两个固定入口(agent 的 tools/ascendc_selfcheck.sh 与
# judge 的 tools/ascendc_eval_pipeline.sh)最先 source 本文件。
# Agent 禁止手动改/source;由固定入口自动加载。路径默认值按部署环境填,均可用容器 -e 覆盖。
# 逐字派生 operator_runtime/tools/env.sh(triton 那份),仅把 triton 专属项(TRITON_DEBUG /
# TRITON_ALLWAYS_COMPILE / .triton 缓存清理)换成 AscendC 专属项(SOC_VERSION / BUILD_TYPE)。

# --- Python(两个角色分开;部署时可把 AST 那个换成不装 torch_npu 的解释器,
#     让"免卡阶段"物理上碰不到 NPU。当前部署两者同指一个 python3)---
: "${OPERATOR_PYTHON:=python3}"   # verification_ascendc.py / performance.py(需 torch_npu、上 NPU)
: "${AST_CHECK_PYTHON:=python3}"  # validate_ascendc_impl.py 纯 AST + 抢卡器本身,不碰 NPU
export OPERATOR_PYTHON AST_CHECK_PYTHON

# --- Workspace ---
: "${WORKSPACE_BASE:=/opt/workspace}"

# --- Ascend / CANN 工具链(NPU 侧)---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

# toolkit set_env:默认指向标准路径,容器没全局配置时自动激活;-f guard,可用 -e ASCEND_SETENV 覆盖。
: "${ASCEND_SETENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -n "${ASCEND_SETENV}" && -f "${ASCEND_SETENV}" ]]; then
  set +u; source "${ASCEND_SETENV}"; set -u
fi

# --- AscendC 专属:SOC/arch 与编译类型(910B2C=A2=dav-2201,add_custom 已实测)---
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export SOC_VERSION="${SOC_VERSION:-ascend910b1}"
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-${SOC_VERSION}}"
export BUILD_TYPE="${BUILD_TYPE:-Release}"

# 用 if(条件假返回 0)而非 `[[…]] && export`(条件假返回 1)——后者作为 env.sh 末句会让
# `source env.sh` 返回 1,在固定入口的 set -e 下自爆。
: "${BISHENGIR_BIN:=/usr/local/Ascend/cann-9.0.0/bin}"
if [ -n "${BISHENGIR_BIN}" ] && [ -d "${BISHENGIR_BIN}" ]; then
    export PATH="${BISHENGIR_BIN}:${PATH}"
fi
