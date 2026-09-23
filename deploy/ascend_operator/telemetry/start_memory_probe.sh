#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${HERE}/../../.." && pwd -P)"
OUT="${POLAR_MEMORY_PROBE_OUT:-${ROOT}/output/ascend_operator/memory_probe}"
SPY="${POLAR_PY_SPY:-${ROOT}/output/ascend_operator/memory_probe_tools/bin/py-spy}"
if [[ "$(cat /proc/1/comm)" != systemd || "$EUID" != 0 ]]; then
  echo 'Run this script as root on the host, outside the development container.' >&2
  exit 1
fi
if [[ ! -x "$SPY" ]]; then
  echo "Missing py-spy: ${SPY}; set POLAR_PY_SPY to an installed executable." >&2
  exit 1
fi
if systemctl is-active --quiet polar-memory-probe.service; then
  echo 'polar-memory-probe.service is already running'
  exit 0
fi
systemd-run --unit=polar-memory-probe --collect \
  --property=MemoryMax=512M --property=CPUQuota=20% --property=OOMScoreAdjust=-900 \
  /usr/bin/env python3 -u "${HERE}/memory_probe.py" --out "$OUT" --py-spy "$SPY" "$@"
systemctl --no-pager status polar-memory-probe.service
