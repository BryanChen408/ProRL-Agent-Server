#!/usr/bin/env python3
"""Polar runtime hooks for CANNBot verifier scripts.

The module is inert unless Polar environment variables are present. It keeps
the native CANNBot verifier layout while adding two runtime controls:

- NPU lease via flock files named ``npu{device}.lock``.
- Per-session generation/optimization attempt budgets.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PIPELINE_STATUS_NAME = "pipeline_budget_status.json"
STATE_FILE_NAME = ".cannbot_pipeline_budget.json"
LOCK_FILE_NAME = ".cannbot_pipeline_budget.lock"


class BudgetExceeded(RuntimeError):
    """Raised when a verify attempt exceeds the configured Polar budget."""


def parse_pool(spec: str | None) -> list[str]:
    value = str(spec or "").strip()
    if not value:
        return []
    if "-" in value and "," not in value:
        lo, hi = value.split("-", 1)
        return [str(i) for i in range(int(lo), int(hi) + 1)]
    return [part.strip() for part in value.split(",") if part.strip()]


def phase_from_impl(triton_impl_name: str, explicit_phase: str | None = None) -> str:
    phase = str(explicit_phase or "").strip().lower()
    if phase in {"generation", "optimization"}:
        return phase
    return "generation" if triton_impl_name == "triton_ascend_impl" else "optimization"


def _int_env(name: str) -> int | None:
    if name not in os.environ:
        return None
    try:
        return int(os.environ[name])
    except Exception:
        return None


def _budget_limit(phase: str) -> int | None:
    if phase == "optimization":
        return _int_env("POLAR_OPT_PIPELINE_MAX")
    return _int_env("POLAR_GEN_PIPELINE_MAX")


def _budget_dir() -> Path | None:
    raw = os.environ.get("POLAR_BUDGET_DIR") or os.environ.get("ARTIFACTS_DIR") or os.environ.get("POLAR_ARTIFACTS_DIR")
    if not raw:
        return None
    return Path(raw)


def _status_path(budget_dir: Path) -> Path:
    explicit = os.environ.get("POLAR_PIPELINE_STATUS_FILE")
    if explicit:
        return Path(explicit)
    artifacts_dir = os.environ.get("ARTIFACTS_DIR") or os.environ.get("POLAR_ARTIFACTS_DIR")
    if artifacts_dir:
        return Path(artifacts_dir) / PIPELINE_STATUS_NAME
    return budget_dir / PIPELINE_STATUS_NAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _status_payload(
    *,
    phase: str,
    attempt: int,
    limit: int,
    gen_count: int,
    opt_count: int,
    op_name: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": os.environ.get("SESSION_ID") or None,
        "task_id": os.environ.get("TASK_ID") or None,
        "op_name": op_name or os.environ.get("OP_NAME") or None,
        "phase": phase,
        "attempt": attempt,
        "limit": limit,
        "gen_count": gen_count,
        "opt_count": opt_count,
        "first_success": phase == "optimization" or opt_count > 0,
        "limit_exhausted": bool(limit >= 0 and attempt > limit),
        "updated_at_unix": time.time(),
    }


@contextlib.contextmanager
def _locked_budget_state(budget_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    budget_dir.mkdir(parents=True, exist_ok=True)
    lock_path = budget_dir / LOCK_FILE_NAME
    state_path = budget_dir / STATE_FILE_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield state_path, _read_json(state_path)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def consume_verify_budget(phase: str, *, op_name: str | None = None) -> dict[str, Any] | None:
    """Increment and validate the current phase budget.

    Budgeting is disabled unless a budget directory and the matching max env are
    both present. ``max=0`` is valid and blocks the first attempt.
    """

    budget_dir = _budget_dir()
    limit = _budget_limit(phase)
    if budget_dir is None or limit is None or limit < 0:
        return None

    with _locked_budget_state(budget_dir) as (state_path, state):
        gen_count = int(state.get("gen_count") or 0)
        opt_count = int(state.get("opt_count") or 0)
        if phase == "optimization":
            opt_count += 1
            attempt = opt_count
        else:
            gen_count += 1
            attempt = gen_count
        state.update({"gen_count": gen_count, "opt_count": opt_count})
        _atomic_write_json(state_path, state)

        status = _status_payload(
            phase=phase,
            attempt=attempt,
            limit=limit,
            gen_count=gen_count,
            opt_count=opt_count,
            op_name=op_name,
        )
        _atomic_write_json(_status_path(budget_dir), status)

    if attempt > limit:
        raise BudgetExceeded(
            f"Polar pipeline budget exhausted: phase={phase} attempt={attempt}>{limit}"
        )
    return status


def attach_budget_status(phase: str, *, op_name: str | None = None) -> dict[str, Any] | None:
    """Rewrite current budget status without consuming another attempt."""

    budget_dir = _budget_dir()
    limit = _budget_limit(phase)
    if budget_dir is None or limit is None or limit < 0:
        return None

    with _locked_budget_state(budget_dir) as (_, state):
        gen_count = int(state.get("gen_count") or 0)
        opt_count = int(state.get("opt_count") or 0)
        attempt = opt_count if phase == "optimization" else gen_count
        status = _status_payload(
            phase=phase,
            attempt=attempt,
            limit=limit,
            gen_count=gen_count,
            opt_count=opt_count,
            op_name=op_name,
        )
        _atomic_write_json(_status_path(budget_dir), status)
        return status


@dataclass
class NpuLease:
    fd: int
    device_id: str
    lock_path: Path
    previous_visible_devices: str | None

    def release(self) -> None:
        if self.previous_visible_devices is None:
            os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        else:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = self.previous_visible_devices
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def _try_acquire(pool: list[str], lock_dir: Path) -> NpuLease | None:
    for device_id in pool:
        lock_path = lock_dir / f"npu{device_id}.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
            return NpuLease(fd=fd, device_id=device_id, lock_path=lock_path, previous_visible_devices=previous)
        except (BlockingIOError, OSError):
            os.close(fd)
    return None


def _lease_status_file(phase: str, work_dir: str | os.PathLike[str] | None) -> Path | None:
    explicit = os.environ.get("POLAR_NPU_LEASE_STATUS_FILE")
    if explicit:
        return Path(explicit)
    status_dir = os.environ.get("POLAR_NPU_LEASE_STATUS_DIR")
    if status_dir:
        return Path(status_dir) / f"npu_lease_status.{phase}.json"
    if work_dir:
        return Path(work_dir) / f"npu_lease_status.{phase}.json"
    return None


def _write_lease_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    payload = {**payload, "updated_at_unix": time.time()}
    _atomic_write_json(path, payload)


@contextlib.contextmanager
def npu_lease(
    phase: str,
    *,
    work_dir: str | os.PathLike[str] | None = None,
    poll_interval: float = 0.2,
) -> Iterator[NpuLease | None]:
    pool = parse_pool(os.environ.get("POLAR_NPU_LEASE_POOL"))
    if not pool:
        yield None
        return
    lock_dir = Path(os.environ.get("POLAR_NPU_LOCK_DIR") or "/tmp/npu-locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    status_file = _lease_status_file(phase, work_dir)
    base = {
        "schema_version": 1,
        "phase": phase,
        "pool": pool,
        "lock_dir": str(lock_dir),
        "pid": os.getpid(),
    }
    _write_lease_status(status_file, {**base, "state": "waiting"})

    lease = None
    while lease is None:
        lease = _try_acquire(pool, lock_dir)
        if lease is None:
            time.sleep(poll_interval)

    started_at = time.time()
    _write_lease_status(
        status_file,
        {
            **base,
            "state": "running",
            "device_id": lease.device_id,
            "lock_path": str(lease.lock_path),
            "started_at_unix": started_at,
        },
    )
    try:
        yield lease
        state = "completed"
        error = None
    except Exception as exc:
        state = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        lease.release()
        _write_lease_status(
            status_file,
            {
                **base,
                "state": state,
                "device_id": lease.device_id,
                "lock_path": str(lease.lock_path),
                "started_at_unix": started_at,
                "ended_at_unix": time.time(),
                "error": error,
            },
        )
