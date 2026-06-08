"""Ascend NPU per-card passthrough for the Docker rollout runtime (multi-container safe).

There are TWO ORTHOGONAL failures on share-disabled hosts (e.g. Node-5-88). They are independent;
the recipe below solves each with a different part (don't conflate them):

1. -8005 (DCMI_ERR_EXCLUSIVE) — a CONCURRENCY problem.
   `--privileged` / `-v /dev:/dev` exposes the GLOBAL management channel. A second container then
   fights the host's single DCMI management lock and its init is rejected (`dcmi module initialize
   failed ... -8005`). FIX = MINIMAL exposure: only the one card's davinci node + the shared
   davinci_manager/devmm_svm/hisi_hdc, NO /dev:/dev, NO --privileged. (Has nothing to do with #2.)

2. 107001 (aclInit "Invalid device ID ... deviceId:0") — a SINGLE-container init problem.
   torch_npu `_npu_init()` hard-builds its default context on LOGICAL device 0, which requires a
   `/dev/davinci0` NODE to exist. For HARD isolation we mount only ONE card; mounting it under its
   real name (e.g. davinci8) leaves the container with no davinci0 -> aclInit crashes. FIX = docker
   device-RENAME `--device=/dev/davinci8:/dev/davinci0`: the assigned physical card appears inside
   the container AS davinci0, so torch_npu's logical-0 init finds it; `ASCEND_RT_VISIBLE_DEVICES=0`
   then means "my one card is logical 0". (Like CUDA_VISIBLE_DEVICES, but Ascend additionally needs
   the davinci0 *file* present — so an env var alone is not enough when only one card is mounted;
   hence the rename. Mounting ALL cards would make davinci0 exist and avoid the rename, but only
   gives soft, env-based isolation — the rename is the price of one-card-per-container hard isolation,
   so a generated kernel cannot touch another rollout's card.)

Side effect of the minimal exposure: the DCMI/management channel is closed inside, so `npu-smi` does
NOT work there. That's fine — operator verification only needs compute (torch matmul, Triton
compile+run) and `torch.npu.max_memory_allocated` (a torch API, not DCMI), all of which work.

Card ALLOCATION is host-level: ``acquire_card`` flock's a free physical card from a pool (held by
the gateway process for the container lifetime, auto-released on crash). DockerRuntime acquires at
start() and releases at stop(). Config: ``RuntimeSpec.kwargs['ascend'] = {pool, lock_dir}``.
"""

from __future__ import annotations

import fcntl
import os
from typing import Any

# Per-card device nodes + driver mounts needed for compute (NOT the global /dev, NOT --privileged).
# Matches the validated Node-5-88 template.
_DRIVER_MOUNTS = (
    "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    "/usr/local/dcmi:/usr/local/dcmi:ro",
    "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
    "/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/",
    "/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info",
)
_SHARED_DEVICES = ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")


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
    """``docker create`` args giving a container ONE Ascend card, remapped to davinci0.

    cfg: ``{device_id: int|str (the assigned PHYSICAL card, e.g. 8), [mounts], [env]}``.
    No --privileged, no -v /dev:/dev. ``ASCEND_RT_VISIBLE_DEVICES=0`` aligns torch_npu's default
    device-0 with the single mounted card.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"ascend cfg must be a dict, got {type(cfg).__name__}")
    device_id = str(cfg.get("device_id", "")).strip()
    if device_id == "":
        raise ValueError("ascend.device_id required (ONE physical NPU card, e.g. 8)")

    args: list[str] = ["--device", f"/dev/davinci{device_id}:/dev/davinci0"]
    for dev in _SHARED_DEVICES:
        args += ["--device", dev]
    for mount in _DRIVER_MOUNTS:
        args += ["-v", mount]
    for mount in cfg.get("mounts", []) or []:
        args += ["-v", str(mount)]
    env = {"ASCEND_RT_VISIBLE_DEVICES": "0"}
    env.update(cfg.get("env", {}) or {})
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
    process dies, so a crashed rollout never leaks a card.
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
