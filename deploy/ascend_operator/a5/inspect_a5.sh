#!/usr/bin/env bash
# Read-only host inventory. Optional argument: a local Docker image to inspect.
set -uo pipefail
export LC_ALL=C

if (( $# > 1 )); then
    printf 'Usage: bash %s [local-image:tag]\n' "$0" >&2
    exit 2
fi

section() { printf '\n===== %s =====\n' "$1"; }
run() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    if ! command -v "$1" >/dev/null 2>&1; then
        printf '[missing] %s\n' "$1"
        return 0
    fi
    local status=0
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 2s 20s "$@" 2>&1 || status=$?
    else
        "$@" 2>&1 || status=$?
    fi
    if (( status != 0 )); then
        printf '[unavailable/failed] exit=%s; continuing\n' "$status"
    fi
    return 0
}

section 'Host / CPU architecture'
run date -u '+%Y-%m-%dT%H:%M:%SZ'
run uname -m
run uname -r
run cat /etc/os-release
run lscpu
run free -h
run df -h / /tmp
if [[ -e /.dockerenv || -e /run/.containerenv ]]; then
    printf 'NOTE: This appears to be a container; host drivers/images may not be visible.\n'
fi

section 'NPU / driver / firmware'
if command -v npu-smi >/dev/null 2>&1; then
    run npu-smi info
else
    run /usr/local/Ascend/driver/tools/npu-smi info
fi
shopt -s nullglob
devices=()
for device in /dev/davinci* /dev/devmm_svm /dev/hisi_hdc; do
    [[ -e "$device" ]] && devices+=("$device")
done
if (( ${#devices[@]} )); then run ls -l "${devices[@]}"; fi
for info in /usr/local/Ascend/driver/version.info /usr/local/Ascend/firmware/version.info; do
    if [[ -f "$info" ]]; then run head -n 60 "$info"; fi
done

section 'Installed CANN / SDK locations (environment is not sourced)'
for root in /usr/local/Ascend/cann /usr/local/Ascend/cann-* \
            /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit/[0-9]*; do
    [[ -d "$root" ]] || continue
    run readlink -f "$root"
    for info in "$root"/version.info "$root"/*/version.info \
                "$root"/*/ascend_toolkit_install.info "$root"/ascend_toolkit_install.info; do
        [[ -f "$info" ]] && run head -n 40 "$info"
    done
    for setup in "$root"/set_env.sh "$root"/../set_env.sh; do
        [[ -f "$setup" ]] && printf 'CANN setup script: %s\n' "$setup"
    done
done

section 'Python package metadata (no torch/NPU import or kernel execution)'
run python3 - <<'PY'
import platform
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

print("python:", sys.executable)
print("version:", platform.python_version(), "architecture:", platform.machine())
header = Path(sysconfig.get_path("include")) / "Python.h"
print("Python.h:", header, "exists:", header.is_file())
for name in ("torch", "torch-npu", "tilelang", "triton-ascend", "pybind11", "cmake"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed in this Python")
PY

section 'Tools on PATH / Node executable format'
for tool in gcc cmake bisheng node npm claude; do
    run "$tool" --version
done
# This CANN msprof does not implement --version; locate it without invoking profiling.
printf 'msprof executable: %s\n' "$(command -v msprof 2>/dev/null || printf 'not on PATH')"
for tool in node claude; do
    path=$(command -v "$tool" 2>/dev/null || true)
    if [[ -f "$path" ]]; then run file -L "$path"; fi
done

section 'Docker (inspect only; no pull/build/run/exec)'
run docker version --format 'client={{.Client.Version}} server={{.Server.Version}}'
run docker image ls --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
if [[ -n "${1:-}" ]]; then
    run docker image inspect --format \
        'id={{.Id}} os={{.Os}} arch={{.Architecture}} workdir={{.Config.WorkingDir}} user={{.Config.User}}' -- "$1"
fi

section 'Done'
printf '%s\n' \
    'Package versions above describe the current host/container Python, not Docker image contents.' \
    'Missing tools or permission errors are retained in this report; they do not stop collection.' \
    'No packages installed, services restarted, shell profiles changed, or NPU kernels launched.'
