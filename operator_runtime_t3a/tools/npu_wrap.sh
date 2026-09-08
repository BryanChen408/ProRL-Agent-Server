#!/usr/bin/env bash
# [R2] 租约包装:设了 POLAR_NPU_LEASE_POOL 就抢锁执行,否则直通(与官方裸跑语义一致)
set -uo pipefail
if [[ -n "${POLAR_NPU_LEASE_POOL:-}" ]]; then
  exec python3 "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/npu_lease_exec.py" \
    --pool "$POLAR_NPU_LEASE_POOL" \
    --lock-dir "${POLAR_NPU_LOCK_DIR:-/dev/shm/npu-locks}" -- python "$@"
else
  exec python "$@"
fi
