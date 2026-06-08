#!/usr/bin/env python3
"""On-node check: can a container get ONE Ascend card for COMPUTE?

Builds docker args with the REAL ``polar.runtime.ascend.ascend_create_args`` — the recipe validated on
Node-5-88 (+ the user's OpenHands worker): ``--privileged -v /dev:/dev`` (needed for device
enumeration on this host) + ``ASCEND_RT_VISIBLE_DEVICES=<device_id>`` to scope the process to one
physical card (concurrency-safe). Runs a throwaway container and does ``torch.ones(10, device='npu')``.

Green ("NPU compute OK") = that card is usable for compute, which is what operator verification needs.
(npu-smi works inside this recipe too, but we test compute since that's what matters.)

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
    print(f"\n[{'OK' if ok else 'FAIL'}] exit={rc} — 看到 'NPU compute OK' 即物理卡 {args.device_id} 可上算子")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
