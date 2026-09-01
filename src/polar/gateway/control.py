"""Durable fail-closed control state for a Polar gateway node."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

from pydantic import BaseModel, ConfigDict, Field


class GatewayControlSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    paused: bool = False
    policy_namespace: str | None = Field(default=None, min_length=1, max_length=128)
    policy_epoch: int | None = Field(default=None, ge=0)
    epoch_enforced: bool = False
    transition_id: str | None = None
    updated_at: float = Field(default_factory=time.time)


class GatewayControlStore:
    """Persist pause/epoch state so a restart during training stays paused."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._snapshot = self._load()

    def snapshot(self) -> GatewayControlSnapshot:
        with self._lock:
            return self._snapshot.model_copy(deep=True)

    def update(
        self,
        *,
        paused: bool | None = None,
        policy_namespace: str | None = None,
        set_policy_namespace: bool = False,
        policy_epoch: int | None = None,
        set_policy_epoch: bool = False,
        epoch_enforced: bool | None = None,
        transition_id: str | None = None,
    ) -> GatewayControlSnapshot:
        with self._lock:
            updates: dict[str, object] = {"updated_at": time.time()}
            if paused is not None:
                updates["paused"] = bool(paused)
            if set_policy_namespace:
                updates["policy_namespace"] = policy_namespace
            if set_policy_epoch:
                updates["policy_epoch"] = policy_epoch
            if epoch_enforced is not None:
                updates["epoch_enforced"] = bool(epoch_enforced)
            if transition_id is not None:
                updates["transition_id"] = str(transition_id)
            self._snapshot = self._snapshot.model_copy(update=updates)
            self._persist_locked()
            return self._snapshot.model_copy(deep=True)

    def _load(self) -> GatewayControlSnapshot:
        if self.path is None or not self.path.exists():
            return GatewayControlSnapshot()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return GatewayControlSnapshot.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(f"invalid gateway control state at {self.path}: {exc}") from exc

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(self._snapshot.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
