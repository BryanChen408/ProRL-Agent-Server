#!/usr/bin/env python3
"""Read the host collector's snapshot; never query an NPU from rollout containers."""

import os
import sys
import time
from pathlib import Path


CACHE_DIR = Path(os.environ.get(
    "POLAR_NPU_SMI_CACHE_DIR",
    str(Path(os.environ.get("POLAR_NPU_LOCK_DIR", "/dev/shm/npu-locks")) / "npu-smi-snapshot"),
))


def main() -> int:
    if sys.argv[1:] != ["info"]:
        print("npu-smi: snapshot supports only 'info'; live queries are disabled, do not retry",
              file=sys.stderr)
        return 64
    try:
        # Open once so an atomic collector update cannot mix content and timestamp.
        with (CACHE_DIR / "npu-smi-info.out").open("rb") as snapshot:
            age = max(0, time.time() - os.fstat(snapshot.fileno()).st_mtime)
            data = snapshot.read()
        if not data:
            raise ValueError("empty snapshot")
    except (OSError, ValueError):
        print("npu-smi: snapshot unavailable; live queries are disabled, do not retry",
              file=sys.stderr)
        return 124
    print(f"npu-smi: host snapshot, age={age:.0f}s" + (" (stale)" if age > 60 else ""),
          file=sys.stderr)
    sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
