"""Durable policy/weight transition state for Polar rollout coordination.

The record is intentionally small and synchronous.  Network reconciliation lives in
``polar.rollout.server``; this module only makes every local state transition durable
before the next distributed side effect is attempted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyTransitionError(RuntimeError):
    """Raised when a requested transition conflicts with durable state."""


class PolicyTransitionPhase(StrEnum):
    QUIESCING = "quiescing"
    ADMISSION_CLOSED = "admission_closed"
    READY_FOR_TRAINING = "ready_for_training"
    COMMITTING = "committing"
    ABORTING = "aborting"
    RECOVERY_REQUIRED = "recovery_required"
    SERVING = "serving"
    ABORTED = "aborted"
    QUIESCED = "quiesced"


class PolicyTransitionKind(StrEnum):
    UPDATE = "update"
    BOOTSTRAP = "bootstrap"
    INITIALIZE = "initialize"


_TERMINAL_PHASES = {
    PolicyTransitionPhase.SERVING,
    PolicyTransitionPhase.ABORTED,
    PolicyTransitionPhase.QUIESCED,
}


class PolicyTransitionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transition_id: str = Field(min_length=1, max_length=256)
    policy_namespace: str = Field(default="legacy", min_length=1, max_length=128)
    kind: PolicyTransitionKind = PolicyTransitionKind.UPDATE
    from_epoch: int = Field(ge=0)
    to_epoch: int = Field(ge=0)
    phase: PolicyTransitionPhase
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    verified_policy_epoch: int | None = Field(default=None, ge=0)
    from_engine_versions: dict[str, str] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    gateway_nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cancellation: dict[str, Any] = Field(default_factory=dict)
    engine_abort_confirmed: bool = False
    last_error: str | None = None


class PolicyTransitionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    active_namespace: str | None = Field(default=None, min_length=1, max_length=128)
    active_epoch: int | None = Field(default=None, ge=0)
    current: PolicyTransitionRecord | None = None


class PolicyTransitionStore:
    """Thread-safe state store with atomic replace persistence."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._snapshot = self._load()

    def snapshot(self) -> PolicyTransitionSnapshot:
        with self._lock:
            return self._snapshot.model_copy(deep=True)

    def start(
        self,
        *,
        transition_id: str,
        policy_namespace: str = "legacy",
        from_epoch: int,
        to_epoch: int,
        from_engine_versions: dict[str, str] | None = None,
        allow_epoch_reset: bool = False,
        kind: PolicyTransitionKind = PolicyTransitionKind.UPDATE,
    ) -> PolicyTransitionRecord:
        transition_id = str(transition_id).strip()
        if not transition_id:
            raise PolicyTransitionError("transition_id must be non-empty")
        policy_namespace = str(policy_namespace).strip()
        if not policy_namespace:
            raise PolicyTransitionError("policy_namespace must be non-empty")
        from_epoch = int(from_epoch)
        to_epoch = int(to_epoch)
        if from_epoch < 0 or to_epoch < 0:
            raise PolicyTransitionError("policy epochs must be non-negative")
        if to_epoch < from_epoch:
            raise PolicyTransitionError("to_epoch must not be less than from_epoch")

        with self._lock:
            current = self._snapshot.current
            if current is not None and current.transition_id == transition_id:
                if current.policy_namespace != policy_namespace or current.kind != kind:
                    raise PolicyTransitionError(
                        "transition_id was reused with a different namespace or kind"
                    )
                if (current.from_epoch, current.to_epoch) != (from_epoch, to_epoch):
                    raise PolicyTransitionError(
                        "transition_id was reused with different epochs: "
                        f"stored={current.from_epoch}->{current.to_epoch} "
                        f"requested={from_epoch}->{to_epoch}"
                    )
                if (
                    from_engine_versions is not None
                    and current.from_engine_versions != from_engine_versions
                ):
                    raise PolicyTransitionError(
                        "transition_id was reused with different engine evidence"
                    )
                return current.model_copy(deep=True)
            if current is not None and current.phase not in _TERMINAL_PHASES:
                raise PolicyTransitionError(
                    f"transition {current.transition_id} is still {current.phase}"
                )
            if (
                not allow_epoch_reset
                and self._snapshot.active_namespace is not None
                and self._snapshot.active_namespace != policy_namespace
            ):
                raise PolicyTransitionError(
                    "active policy namespace mismatch: "
                    f"stored={self._snapshot.active_namespace} requested={policy_namespace}"
                )
            if (
                not allow_epoch_reset
                and self._snapshot.active_epoch is not None
                and self._snapshot.active_epoch != from_epoch
            ):
                raise PolicyTransitionError(
                    "active policy epoch mismatch: "
                    f"stored={self._snapshot.active_epoch} requested={from_epoch}"
                )

            record = PolicyTransitionRecord(
                transition_id=transition_id,
                policy_namespace=policy_namespace,
                kind=kind,
                from_epoch=from_epoch,
                to_epoch=to_epoch,
                phase=PolicyTransitionPhase.QUIESCING,
                from_engine_versions=from_engine_versions or {},
            )
            self._snapshot.active_epoch = from_epoch
            self._snapshot.active_namespace = policy_namespace
            self._snapshot.current = record
            self._persist_locked()
            return record.model_copy(deep=True)

    def update(
        self,
        transition_id: str,
        *,
        phase: PolicyTransitionPhase | None = None,
        verified_policy_epoch: int | None = None,
        engine_versions: dict[str, str] | None = None,
        gateway_nodes: dict[str, dict[str, Any]] | None = None,
        cancellation: dict[str, Any] | None = None,
        engine_abort_confirmed: bool | None = None,
        last_error: str | None = None,
        clear_error: bool = False,
    ) -> PolicyTransitionRecord:
        with self._lock:
            current = self._require_current_locked(transition_id)
            updates: dict[str, Any] = {"updated_at": time.time()}
            if phase is not None:
                updates["phase"] = phase
            if verified_policy_epoch is not None:
                updates["verified_policy_epoch"] = int(verified_policy_epoch)
            if engine_versions is not None:
                updates["engine_versions"] = dict(engine_versions)
            if gateway_nodes is not None:
                updates["gateway_nodes"] = gateway_nodes
            if cancellation is not None:
                updates["cancellation"] = cancellation
            if engine_abort_confirmed is not None:
                updates["engine_abort_confirmed"] = bool(engine_abort_confirmed)
            if clear_error:
                updates["last_error"] = None
            elif last_error is not None:
                updates["last_error"] = last_error
            current = current.model_copy(update=updates, deep=True)
            self._snapshot.current = current
            if current.phase == PolicyTransitionPhase.SERVING:
                self._snapshot.active_epoch = current.to_epoch
                self._snapshot.active_namespace = current.policy_namespace
            elif current.phase == PolicyTransitionPhase.ABORTED:
                self._snapshot.active_epoch = current.from_epoch
                self._snapshot.active_namespace = current.policy_namespace
            self._persist_locked()
            return current.model_copy(deep=True)

    def _require_current_locked(self, transition_id: str) -> PolicyTransitionRecord:
        current = self._snapshot.current
        if current is None or current.transition_id != transition_id:
            raise PolicyTransitionError(f"unknown policy transition: {transition_id}")
        return current

    def _load(self) -> PolicyTransitionSnapshot:
        if self.path is None or not self.path.exists():
            return PolicyTransitionSnapshot()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            snapshot = PolicyTransitionSnapshot.model_validate(payload)
            # Backward-compatible migration for state written before namespaces were
            # introduced.  Legacy records remain isolated in the explicit namespace.
            if (
                snapshot.active_namespace is None
                and snapshot.active_epoch is not None
                and snapshot.current is not None
            ):
                snapshot = snapshot.model_copy(
                    update={"active_namespace": snapshot.current.policy_namespace}
                )
            return snapshot
        except Exception as exc:
            raise PolicyTransitionError(
                f"invalid durable policy transition state at {self.path}: {exc}"
            ) from exc

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        payload = self._snapshot.model_dump_json(indent=2)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
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
