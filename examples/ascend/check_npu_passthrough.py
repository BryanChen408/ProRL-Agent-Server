#!/usr/bin/env python3
"""On-node check: does the Ascend passthrough recipe actually expose NPU cards to a container?

Builds the docker args with the REAL ``polar.runtime.ascend.ascend_create_args`` (so this tests
that code, not a hand-copied command), starts a throwaway container, and runs ``npu-smi info`` +
lists ``/dev/davinci*`` inside. Green = the recipe + your host paths work, independent of Polar
rollout / reward. Zero third-party deps (stdlib only).

    # see the exact command first (no docker needed):
    python examples/ascend/check_npu_passthrough.py --device-ids 8,9,10,11 --dry-run
    # real check on the NPU host (use an Ascend-ready image — same one your operators run in):
    python examples/ascend/check_npu_passthrough.py --device-ids 8,9,10,11 --image <ascend-image>

Note: with ``-v /dev:/dev`` the container sees ALL cards in ``npu-smi`` — the per-operator
restriction to one card is done at run time by the in-container flock (``distributed_npu_lock``
sets ``ASCEND_RT_VISIBLE_DEVICES``), not at container level. So "npu-smi lists cards +
/dev/davinci* present" is the correct success signal for the *passthrough*.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

# Bootstrap src/ so we import the REAL ascend_create_args (no `pip install -e .` needed).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from polar.runtime.ascend import ascend_create_args  # noqa: E402

_CHECK = (
    'echo "== npu-smi info ==" ; npu-smi info 2>&1 | head -20 ; '
    'echo "== /dev/davinci* ==" ; ls -d /dev/davinci* 2>/dev/null || echo "(none)"'
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-ids", required=True, help="NPU pool, e.g. 8,9,10,11")
    ap.add_argument("--lock-dir", default="/tmp/npu-locks", help="host dir for flock locks (created if missing)")
    ap.add_argument("--image", default="polar-ascend-agent:latest",
                    help="Ascend-ready image (the one operators run in); npu-smi is bind-mounted from host")
    ap.add_argument("--entrypoint", default="bash",
                    help="override image ENTRYPOINT (default bash). Some Ascend/OpenHands images set "
                         "ENTRYPOINT=bash; then CMD must start at -lc, else you get "
                         "'bash: ...: cannot execute binary file' (exit 126).")
    ap.add_argument("--dry-run", action="store_true", help="print the docker command and exit")
    args = ap.parse_args()

    ascend = ascend_create_args({"device_ids": args.device_ids, "lock_dir": args.lock_dir})
    cmd = ["docker", "run", "--rm", *ascend]
    if args.entrypoint:
        cmd += ["--entrypoint", args.entrypoint]  # -> `<entrypoint> -lc '<CHECK>'`, avoids `bash bash -lc`
    cmd += [args.image, "-lc", _CHECK]
    print("[cmd] " + shlex.join(cmd) + "\n")  # copy-pasteable (CHECK stays one quoted arg)
    if args.dry_run:
        print("[dry-run] not executed")
        return 0

    os.makedirs(args.lock_dir, exist_ok=True)
    rc = subprocess.run(cmd).returncode
    ok = rc == 0
    print(f"\n[{'OK' if ok else 'FAIL'}] exit={rc} — 看到 npu-smi 列出卡 + /dev/davinci* 即透传成功 "
          f"(单卡限制由容器内 flock 运行时设,不在这一层)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
