# syntax=docker/dockerfile:1
# Build context: this directory, a native Linux `claude` binary, and wheels/.
# BASE_IMAGE needs A5 CANN toolkit/ops; supply matching torch wheels if absent.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS common
USER root
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG CANN_HOME=/usr/local/Ascend/cann
ARG SOC_VERSION=Ascend950DT_9582
ENV ASCEND_HOME_PATH=${CANN_HOME} \
    ASCEND_SETENV=${CANN_HOME}/set_env.sh \
    SOC_VERSION=${SOC_VERSION} \
    ASCENDC_SOC_VERSION=${SOC_VERSION} \
    ASCEND_NPU_ARCH=dav-3510 \
    TILELANG_DEFAULT_TARGET=ascend \
    ACL_OP_INIT_MODE=1 \
    PIP_NO_CACHE_DIR=1

RUN [[ "${SOC_VERSION,,}" == ascend950* ]] || \
    { echo 'Pass --build-arg SOC_VERSION=<exact Ascend950 SKU supported by your toolkit>' >&2; exit 1; }
RUN test -f "${ASCEND_SETENV}" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git ca-certificates curl libnuma1 libgomp1 \
      libedit-dev zlib1g-dev util-linux libjemalloc2 && \
    rm -rf /var/lib/apt/lists/*

# Docker exec bypasses ENTRYPOINT; noninteractive Bash and login shells need CANN too.
COPY <<'ENV_SH' /etc/profile.d/polar-a5.sh
source "${ASCEND_SETENV}"
export PATH="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin:${PATH}"
ENV_SH
ENV BASH_ENV=/etc/profile.d/polar-a5.sh

# Keep large input wheels out of image layers. An empty directory preserves the base stack.
RUN --mount=type=bind,source=wheels,target=/tmp/torch-wheels <<'TORCH_SH'
shopt -s nullglob
wheels=(/tmp/torch-wheels/*.whl)
if (( ${#wheels[@]} )); then
    python3 -m pip install "${wheels[@]}"
fi
TORCH_SH

RUN mkdir -p /usr/local/share/polar-a5 && python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import sysconfig

assert sys.version_info >= (3, 10), "TileLang requires Python >= 3.10"
assert (Path(sysconfig.get_path("include")) / "Python.h").is_file(), "Missing matching Python development headers"
try:
    constraints = "".join(f"{name}=={version(name)}\n" for name in ("torch", "torch-npu"))
except PackageNotFoundError as error:
    raise SystemExit("Base lacks torch/torch_npu: put the matching A5/Python/CPU wheels in wheels/") from error
Path("/usr/local/share/polar-a5/torch-constraints.txt").write_text(constraints)
PY
RUN python3 -m pip install -c /usr/local/share/polar-a5/torch-constraints.txt \
      'cmake>=3.26.1,<4' ninja pybind11 packaging pytest && \
    command -v bisheng && command -v ld.lld && command -v msprof && \
    bisheng --version && cmake --version

FROM common AS wheel-builder
# Main TileLang's native 950 backend; the old tilelang-ascend wheel targets A2/A3.
ARG TILELANG_REF=562cbc7c8188d4521920e76b708d00168e778f3b
ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
WORKDIR /src/tilelang
RUN git init && \
    git remote add origin https://github.com/tile-ai/tilelang.git && \
    git fetch --depth 1 origin "${TILELANG_REF}" && git checkout --detach FETCH_HEAD && \
    git submodule update --init --recursive --depth 1 && \
    git rev-parse HEAD > /usr/local/share/polar-a5/tilelang-commit.txt
RUN USE_ASCEND=ON USE_CUDA=OFF \
    python3 -m pip wheel --no-deps --wheel-dir /wheels . && \
    cp examples/ascend/example_vecadd.py /usr/local/share/polar-a5/example_vecadd.py

FROM common AS runtime
COPY --from=wheel-builder /wheels/ /tmp/wheels/
COPY --from=wheel-builder /usr/local/share/polar-a5/ /usr/local/share/polar-a5/
# Remove the old distribution first if the supplied base includes TileLang.
RUN python3 -m pip uninstall -y tilelang && \
    python3 -m pip install -c /usr/local/share/polar-a5/torch-constraints.txt /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Same native-CLI installation as v2; the binary must match the target CPU architecture.
COPY --chmod=755 claude /usr/local/bin/claude
RUN claude --version && sha256sum /usr/local/bin/claude > /usr/local/share/polar-a5/claude-sha256.txt
RUN python3 - <<'PY'
from importlib.metadata import distribution, version
from pathlib import Path

dist = distribution("tilelang")
assert Path(dist.locate_file("tilelang/ascend/language/__init__.py")).is_file(), "Missing native Ascend dialect"
for line in Path("/usr/local/share/polar-a5/torch-constraints.txt").read_text().splitlines():
    name, expected = line.split("==")
    assert version(name) == expected, f"{name} changed while installing TileLang"
print("Package checks passed; run polar-a5-smoke on a mapped Ascend 950 device.")
PY

# Explicit on-card check, never executed during image build or container startup.
COPY --chmod=755 <<'SMOKE_SH' /usr/local/bin/polar-a5-smoke
#!/bin/bash
set -eo pipefail
source /etc/profile.d/polar-a5.sh
python3 - <<'PY'
import runpy
import sys
import torch
import torch_npu
import tilelang

assert torch.npu.is_available(), "NPU unavailable; check device/driver mounts"
name = torch.npu.get_device_name(0)
assert "950" in name, f"Expected Ascend 950, got {name}"
print("device:", name, "torch:", torch.__version__, "torch_npu:", torch_npu.__version__, "tilelang:", tilelang.__version__)
example = "/usr/local/share/polar-a5/example_vecadd.py"
sys.argv = [example, "--no-bench"]
runpy.run_path(example, run_name="__main__")
torch.npu.synchronize()
PY
SMOKE_SH

WORKDIR /opt/workspace/agent_workdir
# Polar launches `sleep infinity`; an inherited Bash entrypoint would break it.
ENTRYPOINT []
CMD ["bash"]
