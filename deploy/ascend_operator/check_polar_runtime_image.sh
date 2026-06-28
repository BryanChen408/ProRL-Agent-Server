#!/usr/bin/env bash
# Validate that a Polar DockerRuntime image/container has the dependencies needed
# for Claude Code + Triton Ascend operator generation and judging.
#
# Modes:
#   1. Inside a running runtime container:
#        bash check_polar_runtime_image.sh
#        bash check_polar_runtime_image.sh --with-npu
#
#   2. From a host/control container with Docker access:
#        bash check_polar_runtime_image.sh --image polar-op-image:v1
#        bash check_polar_runtime_image.sh --image polar-op-image:v1 --with-npu --device 11
#
# The image should contain CANN toolkit, Python, torch/torch_npu, triton, numpy,
# bash, and claude. Ascend driver/firmware/dcmi/npu-smi are expected to be
# mounted by DockerRuntime at runtime, not baked into the image.

set -u

IMAGE=""
EXPECT_NPU=0
ASCEND_DEVICE=""
SKIP_CLAUDE=0
INSIDE=0

usage() {
  cat <<'USAGE'
Validate that a Polar DockerRuntime image/container has the dependencies needed
for Claude Code + Triton Ascend operator generation and judging.

Modes:
  1. Inside a running runtime container:
       bash check_polar_runtime_image.sh
       bash check_polar_runtime_image.sh --with-npu

  2. From a host/control container with Docker access:
       bash check_polar_runtime_image.sh --image polar-op-image:v1
       bash check_polar_runtime_image.sh --image polar-op-image:v1 --with-npu --device 11

The image should contain CANN toolkit, Python, torch/torch_npu, triton, numpy,
bash, and claude. Ascend driver/firmware/dcmi/npu-smi are expected to be
mounted by DockerRuntime at runtime, not baked into the image.

Options:
  --image IMAGE       Run the check inside a temporary docker container.
  --with-npu          Require NPU visibility and run torch_npu + Triton kernel smoke.
  --device ID         Set ASCEND_RT_VISIBLE_DEVICES for --image or current check.
  --skip-claude       Do not require the claude executable.
  --inside            Internal flag used by --image mode.
  -h, --help          Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      [ -n "$IMAGE" ] || { echo "missing value for --image" >&2; exit 2; }
      shift 2
      ;;
    --with-npu)
      EXPECT_NPU=1
      shift
      ;;
    --device)
      ASCEND_DEVICE="${2:-}"
      [ -n "$ASCEND_DEVICE" ] || { echo "missing value for --device" >&2; exit 2; }
      shift 2
      ;;
    --skip-claude)
      SKIP_CLAUDE=1
      shift
      ;;
    --inside)
      INSIDE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$IMAGE" ] && [ "$INSIDE" -eq 0 ]; then
  command -v docker >/dev/null 2>&1 || { echo "docker is not available" >&2; exit 127; }

  docker_args=(run --rm -i --network host --ipc host --shm-size 16g)
  inside_args=(--inside)
  if [ "$EXPECT_NPU" -eq 1 ]; then
    docker_args+=(--privileged)
    docker_args+=(-v /dev:/dev)
    [ -e /usr/local/Ascend/driver ] && docker_args+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro)
    [ -e /usr/local/Ascend/firmware ] && docker_args+=(-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro)
    [ -e /usr/local/dcmi ] && docker_args+=(-v /usr/local/dcmi:/usr/local/dcmi:ro)
    [ -e /usr/local/bin/npu-smi ] && docker_args+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro)
    [ -e /etc/ascend_install.info ] && docker_args+=(-v /etc/ascend_install.info:/etc/ascend_install.info:ro)
    [ -e /usr/local/sbin ] && docker_args+=(-v /usr/local/sbin:/usr/local/sbin:ro)
    inside_args+=(--with-npu)
  fi
  if [ -n "$ASCEND_DEVICE" ]; then
    docker_args+=(-e "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICE")
    inside_args+=(--device "$ASCEND_DEVICE")
  fi
  if [ "$SKIP_CLAUDE" -eq 1 ]; then
    inside_args+=(--skip-claude)
  fi

  script_path="$(readlink -f "$0" 2>/dev/null || realpath "$0")"
  exec docker "${docker_args[@]}" "$IMAGE" bash -s -- "${inside_args[@]}" < "$script_path"
fi

if [ -n "$ASCEND_DEVICE" ]; then
  export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICE"
fi

PASS=0
FAIL=0
WARN=0
INFO=0

line() { printf '%s\n' "------------------------------------------------------------"; }
ok() { printf '[PASS] %s\n' "$*"; PASS=$((PASS + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL + 1)); }
warn() { printf '[WARN] %s\n' "$*"; WARN=$((WARN + 1)); }
info() { printf '[INFO] %s\n' "$*"; INFO=$((INFO + 1)); }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "############ Polar runtime image check ############"
echo "date: $(date 2>/dev/null || true)"
echo "host: $(hostname 2>/dev/null || true)"
echo "expect_npu=$EXPECT_NPU ascend_visible=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

TOOLKIT=/usr/local/Ascend/ascend-toolkit/set_env.sh
ATB=/usr/local/Ascend/nnal/atb/set_env.sh

# Source CANN before importing torch/torch_npu. Some torch_npu builds need the
# toolkit/driver library paths even for import-time backend registration.
set +u
[ -f "$TOOLKIT" ] && . "$TOOLKIT" >/dev/null 2>&1
[ -f "$ATB" ] && . "$ATB" >/dev/null 2>&1
set -u

line
echo "A. OS and base shell tools"
echo "arch: $(uname -m 2>/dev/null || true) kernel: $(uname -r 2>/dev/null || true)"
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "os: ${PRETTY_NAME:-unknown}"
else
  warn "missing /etc/os-release"
fi
glibc="$(ldd --version 2>/dev/null | head -1 || true)"
[ -n "$glibc" ] && info "glibc: $glibc" || warn "ldd/glibc not visible; native claude may not run on musl-like images"

for cmd in bash sh tar gzip grep sed awk find; do
  have_cmd "$cmd" && ok "command exists: $cmd" || fail "missing required command: $cmd"
done

line
echo "B. Python runtime"
PY=""
for cand in python3 python; do
  if have_cmd "$cand"; then
    PY="$(command -v "$cand")"
    break
  fi
done

if [ -z "$PY" ]; then
  fail "missing python3/python"
else
  ok "python: $PY ($("$PY" --version 2>&1))"
  "$PY" - <<'PYEOF'
import site
print("site-packages:", site.getsitepackages() if hasattr(site, "getsitepackages") else "n/a")
PYEOF
fi

if have_cmd pip3; then
  ok "pip: $(command -v pip3)"
elif have_cmd pip; then
  ok "pip: $(command -v pip)"
else
  warn "pip is not available; acceptable only if image is immutable and prebuilt"
fi

line
echo "C. Python packages"
if [ -n "$PY" ]; then
  if EXPECT_NPU="$EXPECT_NPU" "$PY" - <<'PYEOF'
import importlib.util as U
import os
import sys

expect_npu = os.environ.get("EXPECT_NPU") == "1"
required = ["torch", "torch_npu", "triton", "numpy"]
fatal = False

for name in required:
    spec = U.find_spec(name)
    if spec is None:
        print(f"[FAIL] missing package: {name}")
        fatal = True
        continue
    print(f"[PASS] package installed: {name} ({spec.origin})")
    try:
        mod = __import__(name)
        print(f"[PASS] import {name}: version={getattr(mod, '__version__', '?')}")
    except Exception as exc:
        # torch/triton/numpy must import even in image-only mode. torch_npu may
        # legitimately need runtime driver mounts, so keep it soft until
        # --with-npu is requested.
        hard_required = name in {"torch", "triton", "numpy"} or expect_npu
        tag = "[FAIL]" if hard_required else "[WARN]"
        print(f"{tag} import {name} raised {type(exc).__name__}: {str(exc)[:180]}")
        if hard_required:
            fatal = True

sys.exit(1 if fatal else 0)
PYEOF
  then
    ok "required Python packages are present"
  else
    fail "Python package check failed"
  fi
fi

line
echo "D. CANN toolkit and runtime mounts"

[ -f "$TOOLKIT" ] && ok "CANN toolkit exists: $TOOLKIT" || fail "missing CANN toolkit: $TOOLKIT"
[ -f "$ATB" ] && ok "optional ATB env exists: $ATB" || warn "optional ATB env missing: $ATB"

for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi /usr/local/bin/npu-smi /etc/ascend_install.info /usr/local/sbin; do
  if [ -e "$path" ]; then
    ok "runtime-mounted path visible: $path"
  elif [ "$EXPECT_NPU" -eq 1 ]; then
    fail "expected runtime-mounted path missing: $path"
  else
    info "runtime-mounted path not present in image-only mode: $path"
  fi
done

if have_cmd npu-smi; then
  ok "npu-smi on PATH: $(command -v npu-smi)"
  npu_smi_out="$(npu-smi info 2>&1)"
  npu_smi_rc=$?
  if [ "$npu_smi_rc" -eq 0 ]; then
    printf '%s\n' "$npu_smi_out" | sed -n '1,12p'
  elif [ "$EXPECT_NPU" -eq 1 ]; then
    fail "npu-smi info failed: $(printf '%s' "$npu_smi_out" | head -1)"
  else
    warn "npu-smi info failed in image-only mode: $(printf '%s' "$npu_smi_out" | head -1)"
  fi
elif [ "$EXPECT_NPU" -eq 1 ]; then
  fail "npu-smi not on PATH with --with-npu"
else
  info "npu-smi not on PATH in image-only mode"
fi

line
echo "E. Claude Code and helper tools"
if [ "$SKIP_CLAUDE" -eq 1 ]; then
  info "skipping claude requirement by request"
elif have_cmd claude; then
  ok "claude exists: $(command -v claude)"
  claude --version 2>&1 | head -1 | sed 's/^/[INFO] claude version: /' || warn "claude --version failed"
else
  fail "missing claude executable"
fi

have_cmd git && ok "git exists: $(git --version 2>&1)" || warn "git missing; usually useful for debug but not always hard-required"
have_cmd curl && ok "curl exists: $(command -v curl)" || warn "curl missing; useful for gateway/debug checks"

line
echo "F. NPU functional smoke"
if [ "$EXPECT_NPU" -ne 1 ]; then
  info "not requested; pass --with-npu to test torch_npu and Triton kernel execution"
elif [ -z "$PY" ]; then
  fail "cannot run NPU smoke without Python"
else
  # Third-party env scripts sometimes reference unset shell variables.
  set +u
  [ -f "$TOOLKIT" ] && . "$TOOLKIT" >/dev/null 2>&1
  [ -f "$ATB" ] && . "$ATB" >/dev/null 2>&1
  set -u
  smoke_py="$(mktemp /tmp/polar_runtime_npu_smoke.XXXXXX.py)"
  cat >"$smoke_py" <<'PYEOF'
import os
import sys

import torch
import torch_npu

print("ASCEND_RT_VISIBLE_DEVICES=", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))

if not hasattr(torch, "npu"):
    raise RuntimeError("torch.npu is not available after importing torch_npu")

available = torch.npu.is_available()
count = torch.npu.device_count() if available else 0
print("torch.npu.is_available=", available, "device_count=", count)
if not available or count < 1:
    raise RuntimeError("no visible NPU device")

torch.npu.set_device(0)
x = torch.randn(1024, device="npu")
y = torch.randn(1024, device="npu")
z = x + y
torch.npu.synchronize()
print("torch_npu add smoke max_abs=", float((z - (x + y)).abs().max().cpu()))

import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    xv = tl.load(x_ptr + offsets, mask=mask)
    yv = tl.load(y_ptr + offsets, mask=mask)
    tl.store(o_ptr + offsets, xv + yv, mask=mask)

n = 1024
out = torch.empty_like(x)
_add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.npu.synchronize()
err = (out - (x + y)).abs().max().cpu().item()
print("triton add smoke max_abs=", err)
if err > 1e-3:
    raise RuntimeError(f"Triton add result mismatch: {err}")
PYEOF
  if "$PY" "$smoke_py"; then
    rm -f "$smoke_py"
    ok "NPU torch_npu + Triton smoke passed"
  else
    rm -f "$smoke_py"
    fail "NPU torch_npu + Triton smoke failed"
  fi
fi

line
echo "############ Summary ############"
printf 'PASS=%d FAIL=%d WARN=%d INFO=%d\n' "$PASS" "$FAIL" "$WARN" "$INFO"

if [ "$FAIL" -eq 0 ]; then
  echo "RESULT=PASS"
  exit 0
fi

echo "RESULT=FAIL"
exit 1
