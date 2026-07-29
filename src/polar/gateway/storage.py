"""In-memory session storage for gateway completion records."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from typing import Any

from polar.gateway.completion_writer import CompletionWriter
from polar.gateway.completion_metrics import (
    CompletionMetricsAggregate,
    build_completion_metric_event,
    combine_completion_metrics,
)
from polar.trajectory.models import CompletionRecord, CompletionSession


@dataclass(slots=True)
class _SessionState:
    session_id: str
    created_at: str | None = None
    completion_count: int = 0
    task_id: str | None = None
    model_requested: str | None = None
    model_used: str | None = None
    api_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    completions: list[CompletionRecord] = field(default_factory=list)
    # policy_version live when this session generated its FIRST turn.  The version-span
    # guard rejects the session's next request if current_version != gen_version.
    gen_version: int | None = None
    completion_metrics: CompletionMetricsAggregate = field(
        default_factory=CompletionMetricsAggregate
    )


@dataclass(slots=True)
class _ClosedSessionState:
    session_id: str
    closed_at: str
    reason: str | None = None


class SessionStore:
    """Thread-safe in-memory storage for active gateway sessions."""

    def __init__(self, *, completion_writer: CompletionWriter | None = None) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionState] = {}
        self._closed_sessions: dict[str, _ClosedSessionState] = {}
        self._drop_late_completions = self._env_flag(
            "POLAR_GATEWAY_DROP_LATE_COMPLETIONS",
            default=True,
        )
        self._closed_session_cache_size = self._env_int(
            "POLAR_GATEWAY_CLOSED_SESSION_CACHE_SIZE",
            default=10000,
        )
        self._late_message_drop_count = 0
        self._completion_writer = completion_writer
        # Live policy_version, bumped by the trainer at each weight sync (during the
        # engine pause, before resume) via POST /admin/policy_version.  Stamped onto
        # each completion and used for the version-span guard.  None = never set yet.
        self._current_policy_version: int | None = None

    def set_policy_version(self, version: int) -> None:
        """Set the live policy_version (called at each weight sync, during pause)."""
        with self._lock:
            self._current_policy_version = int(version)

    def get_policy_version(self) -> int | None:
        with self._lock:
            return self._current_policy_version

    def session_gen_version(self, session_id: str) -> int | None:
        """O(1) read of a session's first-generation version (version-span guard)."""
        with self._lock:
            state = self._sessions.get(session_id)
            return state.gen_version if state is not None else None

    def session_would_span(self, session_id: str) -> bool:
        """True iff this session already generated under a version != current, so its
        NEXT turn would span a weight update (mixed-weight).  O(1); reject before
        generating.  False for fresh sessions (no gen_version) and when version unset."""
        with self._lock:
            cur = self._current_policy_version
            state = self._sessions.get(session_id)
            gen = state.gen_version if state is not None else None
            return cur is not None and gen is not None and cur != gen

    def close(self) -> None:
        with self._lock:
            self._sessions.clear()

    def list_active_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._metadata_payload_locked(state)
                for state in self._sessions.values()
            ]

    def get_completions(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [c.model_dump(mode="json") for c in state.completions]

    def list_completion_metrics(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for state in self._sessions.values():
                if task_id and state.task_id != task_id:
                    continue
                if session_id and state.session_id != session_id:
                    continue
                rows.append(self._metrics_payload_locked(state))
            rows.sort(
                key=lambda row: (
                    str(row.get("task_id") or ""),
                    str(row.get("session_id") or ""),
                )
            )
            return rows

    def completion_metrics_summary(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        rows = self.list_completion_metrics(task_id=task_id, session_id=session_id)
        return {
            "summary": combine_completion_metrics(rows),
            "sessions": rows,
        }

    def ensure_session(
        self,
        session_id: str,
        model_requested: str | None,
        model_used: str | None,
        api_type: str | None,
        *,
        task_id: str | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or refresh session metadata."""
        with self._lock:
            state = self._get_or_create_session_locked(session_id, created_at=created_at)
            self._merge_metadata_locked(
                state,
                task_id=task_id,
                model_requested=model_requested,
                model_used=model_used,
                api_type=api_type,
                metadata=metadata,
            )
            return self._metadata_payload_locked(state)

    def save_message(
        self,
        session_id: str,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        original_request: dict[str, Any] | None = None,
        model_requested: str | None = None,
        model_used: str | None = None,
        api_type: str | None = None,
        task_id: str | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        streaming: bool = False,
    ) -> str | None:
        """Append one completion record to the in-memory session."""
        effective_model_used = model_used or request.get("model", "unknown")
        record = CompletionRecord.model_validate(
            {
                "completion_id": f"msg_{uuid.uuid4().hex[:12]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request": request,
                "original_request": original_request or {},
                "response": response,
                "metadata": dict(metadata or {}),
            }
        )

        with self._lock:
            if self._should_drop_late_message_locked(session_id):
                self._late_message_drop_count += 1
                return None
            state = self._get_or_create_session_locked(session_id, created_at=created_at)
            self._merge_metadata_locked(
                state,
                task_id=task_id,
                model_requested=model_requested,
                model_used=effective_model_used,
                api_type=api_type,
                metadata=metadata,
            )
            # version-span guard: stamp the live policy_version onto this completion and
            # record the session's first-generation version.  Used by the gateway entry
            # interception (reject next turn if current != gen_version) and the
            # prefix_merging cross-version fallback.
            if self._current_policy_version is not None:
                record.metadata["policy_version"] = self._current_policy_version
                if state.gen_version is None:
                    state.gen_version = self._current_policy_version
            state.completions.append(record)
            state.completion_count = len(state.completions)
            metric_event = build_completion_metric_event(
                session_id=session_id,
                task_id=state.task_id,
                completion_id=record.completion_id,
                sequence=state.completion_count,
                api_type=api_type,
                model_requested=model_requested,
                model_used=effective_model_used,
                response=response,
                latency_ms=latency_ms,
                streaming=streaming,
            )
            state.completion_metrics.update(metric_event)
            effective_task_id = state.task_id

        # Off the hot path: best-effort persist to disk.
        if self._completion_writer is not None:
            self._completion_writer.enqueue(
                task_id=effective_task_id,
                session_id=session_id,
                completion_id=record.completion_id,
                record={
                    "completion_id": record.completion_id,
                    "timestamp": record.timestamp,
                    "session_id": session_id,
                    "task_id": effective_task_id,
                    "api_type": api_type,
                    "model_requested": model_requested,
                    "model_used": effective_model_used,
                    "original_request": original_request or {},
                    "transformed_request": request,
                    "response": response,
                    # version-span: persist record.metadata (carries the live per-turn
                    # policy_version stamped in save_message), NOT the raw session-level
                    # scheduler metadata param -- else disk shows the constant submission
                    # version and cross-version can't be audited from the persisted files.
                    "metadata": dict(record.metadata),
                },
            )
            self._completion_writer.enqueue_metric(
                task_id=effective_task_id,
                session_id=session_id,
                completion_id=record.completion_id,
                metric=metric_event,
            )
        return record.completion_id

    def record_upstream_failure(self, session_id: str, kind: str) -> None:
        """Stamp an upstream (inference engine) failure onto the session.

        A 5xx from the engine never becomes a CompletionRecord — the request raised before
        ``should_save``, so the trajectory keeps no trace of it and the session looks like a
        normal, merely shorter one.  It is then scored as if the agent had produced that
        (truncated) work, which turns an engine outage into a real training signal.

        Observed: 56/64 sessions across three ascendc runs ended on
        ``API Error: 502 Upstream request failed`` yet were stored COMPLETED with reward 0.2.

        Counting here (not raising) keeps the request path unchanged; the trajectory builder
        decides what to do with it, mirroring the dev_09 abort discipline.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            failures = state.metadata.get("upstream_failures")
            if not isinstance(failures, dict):
                failures = {}
            failures[kind] = int(failures.get(kind) or 0) + 1
            state.metadata["upstream_failures"] = failures

    def mark_session_closed(self, session_id: str, *, reason: str | None = None) -> None:
        """Remember that a session is final so late upstream completions are ignored."""
        if not session_id:
            return
        with self._lock:
            self._mark_session_closed_locked(session_id, reason=reason)

    def is_session_closed(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._closed_sessions

    def late_completion_summary(self) -> dict[str, Any]:
        with self._lock:
            recent_closed = [
                {
                    "session_id": state.session_id,
                    "closed_at": state.closed_at,
                    "reason": state.reason,
                }
                for state in list(self._closed_sessions.values())[-20:]
            ]
            return {
                "drop_late_completions": self._drop_late_completions,
                "closed_session_count": len(self._closed_sessions),
                "closed_session_cache_size": self._closed_session_cache_size,
                "late_message_drop_count": self._late_message_drop_count,
                "recent_closed_sessions": recent_closed,
            }

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Return session metadata if present."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return self._metadata_payload_locked(state)

    def load_completion_session(self, session_id: str) -> CompletionSession:
        """Load the typed completion session from in-memory state."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return CompletionSession.model_validate(
                    {
                        "session_id": session_id,
                        "completion_count": 0,
                        "completions": [],
                    }
                )

            payload = self._metadata_payload_locked(state)
            payload["completions"] = [
                completion.model_dump(mode="python")
                for completion in state.completions
            ]
            return CompletionSession.model_validate(payload)

    def delete_session(self, session_id: str) -> int:
        """Drop a session and return how many messages were removed."""
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return 0
            return len(state.completions)

    def _mark_session_closed_locked(
        self,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> None:
        if not self._drop_late_completions:
            return
        self._closed_sessions[session_id] = _ClosedSessionState(
            session_id=session_id,
            closed_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        while len(self._closed_sessions) > self._closed_session_cache_size:
            self._closed_sessions.pop(next(iter(self._closed_sessions)))

    def _should_drop_late_message_locked(self, session_id: str) -> bool:
        return self._drop_late_completions and session_id in self._closed_sessions

    def _get_or_create_session_locked(
        self,
        session_id: str,
        *,
        created_at: str | None,
    ) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionState(
                session_id=session_id,
                created_at=created_at or datetime.now(timezone.utc).isoformat(),
            )
            self._sessions[session_id] = state
            return state

        if state.created_at is None:
            state.created_at = created_at or datetime.now(timezone.utc).isoformat()
        return state

    def _merge_metadata_locked(
        self,
        state: _SessionState,
        *,
        task_id: str | None,
        model_requested: str | None,
        model_used: str | None,
        api_type: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state.task_id = self._merge_field(state.task_id, task_id)
        state.model_requested = self._merge_field(state.model_requested, model_requested)
        state.model_used = self._merge_field(state.model_used, model_used)
        state.api_type = self._merge_field(state.api_type, api_type)
        if metadata:
            state.metadata.update(metadata)

    def _metadata_payload_locked(self, state: _SessionState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "created_at": state.created_at,
            "completion_count": len(state.completions),
            "task_id": state.task_id,
            "model_requested": state.model_requested,
            "model_used": state.model_used,
            "api_type": state.api_type,
            "metadata": dict(state.metadata),
            "completion_metrics": state.completion_metrics.as_dict(),
        }

    def _metrics_payload_locked(self, state: _SessionState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "task_id": state.task_id,
            "model_requested": state.model_requested,
            "model_used": state.model_used,
            "api_type": state.api_type,
            "completion_count": len(state.completions),
            "completion_metrics": state.completion_metrics.as_dict(),
        }

    @staticmethod
    def _merge_field(existing: Any, incoming: Any) -> Any:
        if incoming in (None, "", "unknown"):
            return existing
        if existing in (None, "", "unknown"):
            return incoming
        return existing

    @staticmethod
    def _env_flag(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_int(name: str, *, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(1, value)
