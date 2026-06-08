#!/usr/bin/env python3
"""On-node check: can a container get ONE Ascend card for COMPUTE (per-card, no privileged)?

Builds docker args with the REAL ``polar.runtime.ascend.ascend_create_args`` — the multi-container-safe
per-card recipe: physical card -> the container's ``davinci0``, ``ASCEND_RT_VISIBLE_DEVICES=0``,
**no --privileged, no -v /dev:/dev** (so concurrent containers don't fight the DCMI exclusive lock
-8005). Runs a throwaway container and does ``torch.ones(10, device='npu')`` inside.

Green ("NPU compute OK") = that card is usable for compute, which is exactly what operator
verification needs. Note: ``npu-smi`` / DCMI management is intentionally CLOSED inside this recipe —
that's expected; we only need compute.

    python examples/ascend/check_npu_passthrough.py --device-id 8 --dry-run        # just print the command
    python examples/ascend/check_npu_passthrough.py --device-id 8 --image <ascend-image>   # real check
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from polar.runtime.ascend import ascend_create_args  # noqa: E402

_CHECK = (
    "python -c \"import torch, torch_npu; "
    "x = torch.ones(10, device='npu'); "
    "print('NPU compute OK:', float(x.sum().item()))\""
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-id", required=True, help="ONE physical NPU card to test, e.g. 8")
    ap.add_argument("--image", default="polar-op-agent:latest", help="Ascend-ready image (torch_npu inside)")
    ap.add_argument("--entrypoint", default="bash",
                    help="override image ENTRYPOINT (default bash); some images set ENTRYPOINT=bash so "
                         "the CMD must start at -lc")
    ap.add_argument("--dry-run", action="store_true", help="print the docker command and exit")
    args = ap.parse_args()

    ascend = ascend_create_args({"device_id": args.device_id})
    cmd = ["docker", "run", "--rm", *ascend]
    if args.entrypoint:
        cmd += ["--entrypoint", args.entrypoint]
    cmd += [args.image, "-lc", _CHECK]
    print("[cmd] " + shlex.join(cmd) + "\n")
    if args.dry_run:
        print("[dry-run] not executed")
        return 0

    rc = subprocess.run(cmd).returncode
    ok = rc == 0
    print(f"\n[{'OK' if ok else 'FAIL'}] exit={rc} — 看到 'NPU compute OK' 即物理卡 {args.device_id} "
          f"可上算子(npu-smi 在本配方不可用是预期的,只验算力)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
