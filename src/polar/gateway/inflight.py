"""In-flight generation request tracking for the gateway proxy."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from polar.gateway.proxy import UpstreamError


GenerationFactory = Callable[[], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class GenerationResult:
    response: dict[str, Any]
    latency_ms: float
    fingerprint: str
    coalesced: bool
    should_save: bool


@dataclass(slots=True)
class _GenerationEntry:
    session_id: str
    fingerprint: str
    task: asyncio.Task[dict[str, Any]]
    started_at: float
    latency_ms: float | None = None
    waiters: int = 0
    save_claimed: bool = False
    closed: bool = False
    close_reason: str | None = None


class InflightGenerationTracker:
    """Coalesce duplicate active upstream generation requests per session."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: dict[tuple[str, str], _GenerationEntry] = {}
        # Rollout session IDs are single-use.  Remember a closed ID even when there
        # was no tracked request at close time: otherwise DELETE can race between
        # the server's storage.closed check and ``run()`` registering its entry,
        # leaving an old-session request parked behind the generation pause and
        # able to execute after the next-policy resume.
        self._closed_sessions: dict[str, str | None] = {}
        self._coalesced_request_count = 0
        self._closed_generation_count = 0

    async def run(
        self,
        session_id: str,
        request: dict[str, Any],
        factory: GenerationFactory,
    ) -> GenerationResult:
        fingerprint = request_fingerprint(request)
        key = (session_id, fingerprint)
        async with self._lock:
            if session_id in self._closed_sessions:
                reason = self._closed_sessions[session_id]
                raise UpstreamError(
                    "Upstream generation rejected because session closed"
                    f" ({reason or 'closed'})"
                )
            entry = self._entries.get(key)
            if entry is None or (entry.task.done() and entry.waiters == 0):
                entry = self._create_entry(session_id, fingerprint, key, factory)
                self._entries[key] = entry
                coalesced = False
            else:
                self._coalesced_request_count += 1
                coalesced = True
            entry.waiters += 1

        try:
            try:
                response = await asyncio.shield(entry.task)
            except asyncio.CancelledError as exc:
                if entry.closed:
                    raise UpstreamError(
                        f"Upstream generation cancelled because session closed"
                        f" ({entry.close_reason or 'closed'})"
                    ) from exc
                raise
            async with self._lock:
                should_save = not entry.save_claimed and not entry.closed
                if should_save:
                    entry.save_claimed = True
                latency_ms = entry.latency_ms
            return GenerationResult(
                response=response,
                latency_ms=latency_ms or 0.0,
                fingerprint=fingerprint,
                coalesced=coalesced,
                should_save=should_save,
            )
        finally:
            # Keep local ownership bookkeeping non-suspending.  The request handler can
            # itself be cancelled again while session close is cancelling ``entry.task``;
            # an awaited lock here used to make that second cancellation leak a waiter
            # and retain an already-dead entry indefinitely.
            self._release_waiter(key, entry)

    async def close_session(self, session_id: str, *, reason: str | None = None) -> int:
        """Mark and cancel active upstream generations for a finalized session."""
        async with self._lock:
            self._closed_sessions[session_id] = reason
            entries = [
                (key, entry)
                for key, entry in self._entries.items()
                if entry.session_id == session_id
            ]
            for _key, entry in entries:
                if not entry.closed:
                    entry.closed = True
                    entry.close_reason = reason
                    self._closed_generation_count += 1
                if not entry.task.done():
                    entry.task.cancel()
            return len(entries)

    def status(self) -> dict[str, Any]:
        return {
            "active": len(self._entries),
            "closed_sessions": len(self._closed_sessions),
            "coalesced_request_count": self._coalesced_request_count,
            "closed_generation_count": self._closed_generation_count,
            "entries": [
                {
                    "session_id": entry.session_id,
                    "fingerprint": entry.fingerprint,
                    "waiters": entry.waiters,
                    "done": entry.task.done(),
                    "closed": entry.closed,
                    "close_reason": entry.close_reason,
                    "age_seconds": max(0.0, time.perf_counter() - entry.started_at),
                }
                for entry in self._entries.values()
            ],
        }

    def _create_entry(
        self,
        session_id: str,
        fingerprint: str,
        key: tuple[str, str],
        factory: GenerationFactory,
    ) -> _GenerationEntry:
        async def invoke() -> dict[str, Any]:
            started = time.perf_counter()
            try:
                return await factory()
            finally:
                entry.latency_ms = (time.perf_counter() - started) * 1000.0

        entry = _GenerationEntry(
            session_id=session_id,
            fingerprint=fingerprint,
            task=None,  # type: ignore[arg-type]
            started_at=time.perf_counter(),
        )
        entry.task = asyncio.create_task(invoke())
        entry.task.add_done_callback(
            lambda _task: asyncio.create_task(self._cleanup_done_entry(key, entry))
        )
        return entry

    def _release_waiter(
        self,
        key: tuple[str, str],
        entry: _GenerationEntry,
    ) -> None:
        # This tracker is event-loop-local, and every other mutation protected by
        # ``_lock`` has no suspension point inside its critical section.  Performing
        # this tiny mutation synchronously is therefore atomic with respect to them and,
        # importantly, cannot be interrupted by a repeated Task.cancel().
        entry.waiters = max(0, entry.waiters - 1)
        if entry.waiters == 0 and entry.task.done() and self._entries.get(key) is entry:
            self._entries.pop(key, None)

    async def _cleanup_done_entry(
        self,
        key: tuple[str, str],
        entry: _GenerationEntry,
    ) -> None:
        async with self._lock:
            if entry.waiters == 0 and self._entries.get(key) is entry:
                self._entries.pop(key, None)


def request_fingerprint(request: dict[str, Any]) -> str:
    """Stable semantic fingerprint for generation requests."""
    canonical = {
        key: value
        for key, value in request.items()
        if key not in {"stream", "stream_options"}
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
