"""In-memory session storage for gateway completion records."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    completion_metrics: CompletionMetricsAggregate = field(
        default_factory=CompletionMetricsAggregate
    )


class SessionStore:
    """Thread-safe in-memory storage for active gateway sessions."""

    def __init__(self, *, completion_writer: CompletionWriter | None = None) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionState] = {}
        self._completion_writer = completion_writer

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
    ) -> str:
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
            state = self._get_or_create_session_locked(session_id, created_at=created_at)
            self._merge_metadata_locked(
                state,
                task_id=task_id,
                model_requested=model_requested,
                model_used=effective_model_used,
                api_type=api_type,
                metadata=metadata,
            )
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
                    "metadata": dict(metadata or {}),
                },
            )
            self._completion_writer.enqueue_metric(
                task_id=effective_task_id,
                session_id=session_id,
                completion_id=record.completion_id,
                metric=metric_event,
            )
        return record.completion_id

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
