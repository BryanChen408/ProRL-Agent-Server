"""Ascend NPU device passthrough for the Docker rollout runtime.

Centralizes the *one* proven device-mount recipe (matches the user's openhands_sdk
``remote_eval_worker._run_container``; host paths confirmed identical) so an agent **or**
judge container can compile + run Triton operators on Ascend cards.

Card *allocation* inside the container is done by the in-container flock pool
(``distributed_npu_lock``), which reads ``EVAL_DEVICE_IDS`` / ``EVAL_LOCK_DIR`` and sets
``ASCEND_RT_VISIBLE_DEVICES`` per operator run. Polar's job here is only to expose the
devices + the shared lock dir + that env — not to pick the card.

Enabled per session via ``RuntimeSpec.kwargs["ascend"]``::

    {"device_ids": "8,9,10,11", "lock_dir": "/shared/npu-locks",
     "shm_size": "500g", "ipc": "host", "mounts": [...], "env": {...}}

Pure arg-builder; unit-tested without Docker (see ``test_ascend_runtime.py``).
"""

from __future__ import annotations

# Fixed Ascend driver/runtime mounts (read-only). Host paths confirmed == openhands worker.
_ASCEND_RO_MOUNTS = (
    "/dev:/dev",
    "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
    "/usr/local/dcmi:/usr/local/dcmi:ro",
    "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
    "/etc/ascend_install.info:/etc/ascend_install.info:ro",
    "/usr/local/sbin:/usr/local/sbin:ro",
)
LOCK_MOUNT_TARGET = "/shared/device-locks"


def ascend_create_args(cfg: dict) -> list[str]:
    """``docker create`` args that give a container Ascend NPU access.

    cfg keys:
      * ``device_ids`` (required) — NPU pool the in-container flock picks from, e.g. ``"8,9,10,11"``
      * ``lock_dir``   (required) — host dir for cross-container flock locks (-> ``/shared/device-locks``)
      * ``shm_size``   (default ``"500g"``), ``ipc`` (default ``"host"``)
      * ``mounts``     — extra ``-v`` specs appended
      * ``env``        — extra ``-e K=V`` appended (overrides the EVAL_* defaults)

    Network/name/lifecycle are intentionally NOT set here — Polar's DockerRuntime owns those.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"ascend cfg must be a dict, got {type(cfg).__name__}")
    device_ids = str(cfg.get("device_ids") or "").strip()
    lock_dir = str(cfg.get("lock_dir") or "").strip()
    if not device_ids:
        raise ValueError("ascend.device_ids required (NPU pool, e.g. '8,9,10,11')")
    if not lock_dir:
        raise ValueError("ascend.lock_dir required (host dir for flock device locks)")

    args: list[str] = ["--privileged"]
    if cfg.get("ipc", "host"):
        args += ["--ipc", str(cfg.get("ipc", "host"))]
    if cfg.get("shm_size", "500g"):
        args += ["--shm-size", str(cfg.get("shm_size", "500g"))]
    for mount in _ASCEND_RO_MOUNTS:
        args += ["-v", mount]
    args += ["-v", f"{lock_dir}:{LOCK_MOUNT_TARGET}"]
    for mount in cfg.get("mounts", []) or []:
        args += ["-v", str(mount)]

    env = {
        "EVAL_LOCK_DIR": LOCK_MOUNT_TARGET,
        "EVAL_DEVICE_PREFIX": "npu",
        "EVAL_DEVICE_IDS": device_ids,
        "EVAL_ENV_NAME": "ASCEND_RT_VISIBLE_DEVICES",
    }
    env.update(cfg.get("env", {}) or {})
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    return args
