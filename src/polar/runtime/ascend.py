"""Ascend NPU passthrough for the Docker rollout runtime (multi-container safe).

EMPIRICAL on Node-5-88 (and matching the user's proven OpenHands `remote_eval_worker`):

  * The recipe below — `--privileged -v /dev:/dev` (full device + management visibility) +
    `ASCEND_RT_VISIBLE_DEVICES=<one physical card>` — runs CONCURRENTLY with the training container
    with NO -8005. This is the working path.
  * The opposite (minimal per-card `--device=/dev/davinciN:/dev/davinci0`, no /dev, no privileged)
    FAILS here with aclInit 507899 `Resource_Busy` / `Get device cnt failed` / `chipType=0`: without
    the full /dev + privileged, the driver can't enumerate the device / read its chip type. So on
    this host, broad visibility is REQUIRED for init to work at all.

So concurrency-safety does NOT come from minimizing /dev exposure (that breaks enumeration); it comes
from two things:
  1. Each container is scoped to ONE card via `ASCEND_RT_VISIBLE_DEVICES=<N>` set BEFORE any NPU init
     (so torch_npu doesn't default to card 0 and collide). The host flock leases N.
  2. NOT running `npu-smi info` inside the hot path (it ignores RT and enumerates ALL cards globally,
     which is what actually fights the global DCMI lock -> -8005). The eval engine is torch-only, so
     this never happens.

Tradeoff: all cards are visible inside the container (soft, RT-based isolation rather than hard
device isolation). A generated Triton kernel honors ASCEND_RT_VISIBLE_DEVICES (it doesn't open device
files directly), and this is the user's proven OpenHands setup, so it's acceptable.

Card ALLOCATION is host-level: ``acquire_card`` flock's a free physical card from a pool (held by the
gateway process for the container lifetime, auto-released on crash) — the equivalent of OpenHands'
``npu_lease``. DockerRuntime acquires at start() and releases at stop(), then passes the leased card
as ``ASCEND_RT_VISIBLE_DEVICES``. Config: ``RuntimeSpec.kwargs['ascend'] = {pool, lock_dir}``.
"""

from __future__ import annotations

import fcntl
import os
from typing import Any

# Ascend driver/runtime mounts (host paths == the user's openhands worker). With -v /dev:/dev these
# give the runtime everything it needs to enumerate + run on the card.
_DRIVER_MOUNTS = (
    "/dev:/dev",
    "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
    "/usr/local/dcmi:/usr/local/dcmi:ro",
    "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
    "/etc/ascend_install.info:/etc/ascend_install.info:ro",
    "/usr/local/sbin:/usr/local/sbin:ro",
)


def parse_pool(spec: Any) -> list[str]:
    """'8,9,10,11' | [8,9,10,11] | '8-11' -> ['8','9','10','11']."""
    if isinstance(spec, (list, tuple)):
        return [str(x).strip() for x in spec if str(x).strip() != ""]
    s = str(spec or "").strip()
    if not s:
        return []
    if "-" in s and "," not in s:  # range form "8-11"
        lo, hi = s.split("-", 1)
        return [str(i) for i in range(int(lo), int(hi) + 1)]
    return [p.strip() for p in s.split(",") if p.strip() != ""]


def ascend_create_args(cfg: dict) -> list[str]:
    """``docker create`` args giving a container Ascend NPU access scoped to ONE physical card.

    cfg: ``{device_id: int|str (the assigned PHYSICAL card, e.g. 9), [shm_size], [mounts], [env]}``.
    `--privileged -v /dev:/dev` (required for device enumeration on this host) + full driver mounts +
    `ASCEND_RT_VISIBLE_DEVICES=<device_id>` (scopes the process to that one card -> concurrency-safe).
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"ascend cfg must be a dict, got {type(cfg).__name__}")
    device_id = str(cfg.get("device_id", "")).strip()
    if device_id == "":
        raise ValueError("ascend.device_id required (the physical NPU card to scope to, e.g. 9)")

    args: list[str] = ["--privileged"]
    if cfg.get("ipc", "host"):
        args += ["--ipc", str(cfg.get("ipc", "host"))]
    if cfg.get("shm_size", "500g"):
        args += ["--shm-size", str(cfg.get("shm_size", "500g"))]
    for mount in _DRIVER_MOUNTS:
        args += ["-v", mount]
    for mount in cfg.get("mounts", []) or []:
        args += ["-v", str(mount)]
    env = dict(cfg.get("env", {}) or {})
    # Always scope to the leased card, even if callers pass a generic env block.
    env["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    return args


class CardLock:
    """A held flock on one physical NPU card. ``release()`` (or process death) frees it."""

    def __init__(self, fd: int, device_id: str) -> None:
        self.fd = fd
        self.device_id = device_id

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass


def acquire_card(pool: list[str], lock_dir: str) -> CardLock:
    """flock the first FREE physical card in ``pool`` (held until release / process death).

    Raises RuntimeError if every card is locked. Crash-safe: flock auto-releases when the holding
    process dies, so a crashed rollout never leaks a card. This is OpenHands' ``npu_lease`` equivalent.
    """
    if not pool:
        raise ValueError("ascend.pool is empty — no NPU cards to allocate")
    os.makedirs(lock_dir, exist_ok=True)
    for dev in pool:
        path = os.path.join(lock_dir, f"npu{dev}.lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return CardLock(fd, dev)
        except (BlockingIOError, OSError):
            os.close(fd)
            continue
    raise RuntimeError(f"no free NPU card in pool {pool} (all {len(pool)} locked)")
