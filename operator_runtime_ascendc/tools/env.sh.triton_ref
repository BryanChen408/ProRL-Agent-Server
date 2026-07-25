# shellcheck shell=bash
# 算子工具链环境配置。固定评测入口 triton_eval_pipeline.sh 最先 source 本文件。
# Agent 禁止手动改/ source;由固定入口自动加载。路径默认值按部署环境填,均可用容器 -e 覆盖。

# --- Python(单一解释器)---
: "${OPERATOR_PYTHON:=/usr/local/bin/python}"   # 跑 verify.py / benchmark.py(需 torch_npu / 上 NPU)
: "${AST_CHECK_PYTHON:=/usr/local/bin/python}"  # validate_triton_impl.py 纯 AST,任意 python 即可
export OPERATOR_PYTHON AST_CHECK_PYTHON

# --- Workspace ---
: "${WORKSPACE_BASE:=/opt/workspace}"

rm -rf /root/.triton/cache/ 2>/dev/null || true

# --- Ascend / CANN 工具链(NPU 侧)---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

# toolkit set_env:默认指向标准路径,容器没全局配置时自动激活;-f guard,可用 -e ASCEND_SETENV 覆盖。
: "${ASCEND_SETENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -n "${ASCEND_SETENV}" && -f "${ASCEND_SETENV}" ]]; then
  set +u; source "${ASCEND_SETENV}"; set -u
fi

export TRITON_DEBUG="${TRITON_DEBUG:-1}"
export TRITON_ALLWAYS_COMPILE="${TRITON_ALLWAYS_COMPILE:-1}"

# 毕昇编译器 bin 目录;可用 -e BISHENGIR_BIN 覆盖。
: "${BISHENGIR_BIN:=/usr/local/Ascend/cann-9.0.0/bin}"

# 用 if(条件假返回 0)而非 `[[…]] && export`(条件假返回 1)——后者作为 env.sh 末句会让
# `source env.sh` 返回 1,在固定入口的 set -e 下自爆。
if [ -n "${BISHENGIR_BIN}" ] && [ -d "${BISHENGIR_BIN}" ]; then
    export PATH="${BISHENGIR_BIN}:${PATH}"
fi
