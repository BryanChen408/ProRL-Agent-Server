"""Top-level task orchestration for rollout batches."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from polar.platform.events import EventBus
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import (
    SessionContext,
    SessionResult,
    SessionStatus,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from polar.rollout.pipeline import Pipeline

logger = logging.getLogger(__name__)

_CALLBACK_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int = 0
    errored_sessions: int = 0
    harness: str | None = None
    model: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    results: list[SessionResult] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)
    session_states: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, SessionContext] = field(default_factory=dict)
    background_task: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None


def _harness_from_request(request: TaskRequest) -> str | None:
    if request.agent and request.agent.harness:
        return request.agent.harness
    return None


def _model_from_request(request: TaskRequest) -> str | None:
    if request.agent and request.agent.model_name:
        return request.agent.model_name
    return None


def _mean_reward(results: list[SessionResult]) -> float | None:
    rewards: list[float] = []
    for r in results:
        traces = r.trajectory.traces
        if traces and traces[-1].reward is not None:
            try:
                rewards.append(float(traces[-1].reward))
            except (TypeError, ValueError):
                pass
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


def _mean_traces(results: list[SessionResult]) -> float | None:
    """Average number of traces per session for the task."""
    if not results:
        return None
    counts = [len(r.trajectory.traces) for r in results]
    return sum(counts) / len(counts)


def _mean_completions(results: list[SessionResult]) -> float | None:
    """Average number of raw completions (LLM requests) per session."""
    if not results:
        return None
    counts = [
        int(r.trajectory.metadata.get("record_count") or len(r.trajectory.traces))
        for r in results
    ]
    return sum(counts) / len(counts)


class RolloutManager:
    """Manage the lifecycle of rollout sessions for a single submitted task."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        scheduler: NodeScheduler,
        event_bus: EventBus | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler
        self.event_bus = event_bus or EventBus()
        self._tasks: dict[str, _TaskRecord] = {}
        self._cancelled_before_submit: dict[str, str] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
            except RuntimeError:
                return
        self.event_bus.publish_threadsafe(loop, event_type, payload)

    async def submit_task(self, request: TaskRequest) -> str:
        """Register a task and run it in the background. Returns task_id immediately."""
        self._loop = asyncio.get_running_loop()
        with self._lock:
            pre_cancel_reason = self._cancelled_before_submit.get(request.task_id)
            if pre_cancel_reason is not None:
                self._tasks[request.task_id] = _TaskRecord(
                    task_id=request.task_id,
                    status="cancelled",
                    total_sessions=request.num_samples,
                    harness=_harness_from_request(request),
                    model=_model_from_request(request),
                    cancel_requested=True,
                    cancel_reason=pre_cancel_reason,
                )
                logger.info(
                    "Task %s arrived after policy-cutoff cancellation; not starting",
                    request.task_id,
                )
                return request.task_id
            existing = self._tasks.get(request.task_id)
            if existing is not None and existing.status == "running":
                raise ValueError(f"task {request.task_id} is already running")
            self._tasks[request.task_id] = _TaskRecord(
                task_id=request.task_id,
                status="running",
                total_sessions=request.num_samples,
                harness=_harness_from_request(request),
                model=_model_from_request(request),
            )
        self._emit(
            "task.created",
            {
                "task_id": request.task_id,
                "status": "running",
                "harness": _harness_from_request(request),
                "model": _model_from_request(request),
                "num_samples": request.num_samples,
            },
        )
        background_task = asyncio.create_task(
            self._run_task_background(request),
            name=f"polar-rollout-{request.task_id}",
        )
        with self._lock:
            self._tasks[request.task_id].background_task = background_task
        return request.task_id

    async def _run_task_background(self, request: TaskRequest) -> None:
        """Execute a task in the background, updating the record on completion."""
        try:
            result = await self._execute_task(request)
            logger.info("Task %s completed with %d results", request.task_id, len(result.results))
        except asyncio.CancelledError:
            with self._lock:
                record = self._tasks.get(request.task_id)
                if record is not None:
                    record.status = "cancelled"
                    record.updated_at = time.time()
            self._emit(
                "task.completed",
                {"task_id": request.task_id, "status": "cancelled"},
            )
            return
        except Exception:
            logger.exception("Background task %s failed", request.task_id)
            self._emit("task.completed", {"task_id": request.task_id, "status": "failed"})
            return
        self._emit(
            "task.completed",
            {
                "task_id": request.task_id,
                "status": result.status,
                "completed_sessions": len(result.results),
            },
        )
        if request.callback_url:
            await self._post_callback(request.callback_url, result)

    async def _post_callback(self, callback_url: str, result: TaskResult) -> None:
        """Best-effort POST the terminal TaskResult to the trainer's callback URL."""
        try:
            async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    callback_url,
                    json=result.model_dump(mode="json"),
                )
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Callback POST to %s failed for task %s; trainer must fall back to polling",
                callback_url,
                result.task_id,
                exc_info=True,
            )

    def session_state_changed(self, task_id: str, session_id: str, status: str) -> None:
        """Update live task session-state cache from pipeline status polls."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                record.session_states[session_id] = status
                record.updated_at = time.time()

    async def cancel_tasks(
        self,
        task_ids: list[str],
        *,
        reason: str = "policy_cutoff",
    ) -> dict[str, Any]:
        """Logically cancel tasks and wait only for gateway cancellation acknowledgement.

        Cancelling the background task propagates into every ``Pipeline`` session.  Each
        session's cancellation handler closes its gateway session (which fences inference
        immediately); runtime/container teardown remains gateway-owned background cleanup.
        Late callbacks find no pending future and cannot resurrect a cancelled task.
        """
        requested = list(dict.fromkeys(str(task_id) for task_id in task_ids))
        tasks_to_cancel: list[asyncio.Task[None]] = []
        cancelled: list[str] = []
        already_terminal: list[str] = []
        cancelled_before_submit: list[str] = []
        missing: list[str] = []
        session_count = 0

        with self._lock:
            for task_id in requested:
                record = self._tasks.get(task_id)
                if record is None:
                    if reason == "policy_cutoff":
                        # The VIME asyncio worker can publish its local task ID just
                        # before the HTTP submit reaches us. Fence that single-use ID
                        # now so a late submit cannot create an orphan session after
                        # the boundary has already been acknowledged.
                        self._cancelled_before_submit[task_id] = reason
                        cancelled_before_submit.append(task_id)
                    else:
                        missing.append(task_id)
                    continue
                if record.status in {"completed", "failed", "cancelled"}:
                    already_terminal.append(task_id)
                    continue

                record.cancel_requested = True
                record.cancel_reason = reason
                record.status = "cancelled"
                record.updated_at = time.time()
                for session in record.sessions.values():
                    session.cancel_requested = True
                    session.cancel_reason = reason
                session_count += len(record.sessions)
                background_task = record.background_task
                if background_task is not None and not background_task.done():
                    tasks_to_cancel.append(background_task)
                cancelled.append(task_id)

        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        cancel_errors: dict[str, str] = {}
        with self._lock:
            for task_id in cancelled:
                record = self._tasks[task_id]
                for session in record.sessions.values():
                    if (
                        not session.cancel_acknowledged
                        and (
                            session.gateway_url is None
                            or session.rollout_result is not None
                        )
                    ):
                        # No gateway assignment means the session never existed remotely;
                        # a terminal rollout_result means normal completion won the cancel
                        # race and normal Pipeline cleanup already owns its teardown.
                        session.cancel_acknowledged = True
                    if session.cancel_error:
                        cancel_errors[session.session_id] = session.cancel_error
                    elif not session.cancel_acknowledged:
                        cancel_errors[session.session_id] = (
                            "gateway cancellation was not acknowledged"
                        )

        return {
            "requested": len(requested),
            "cancelled": len(cancelled),
            "already_terminal": len(already_terminal),
            "cancelled_before_submit": len(cancelled_before_submit),
            "missing": missing,
            "sessions_cancel_requested": session_count,
            "all_acknowledged": not missing and not cancel_errors,
            "errors": cancel_errors,
            "task_ids": {
                "cancelled": cancelled,
                "already_terminal": already_terminal,
                "cancelled_before_submit": cancelled_before_submit,
            },
        }

    async def cancel_task(
        self,
        task_id: str,
        *,
        reason: str = "sync_oversubscribe_abort",
    ) -> dict[str, object] | None:
        """Cancel one known task using the canonical batch cancellation path."""
        with self._lock:
            if task_id not in self._tasks:
                return None

        result = await self.cancel_tasks([task_id], reason=reason)
        with self._lock:
            status = self._tasks[task_id].status
        return {
            "task_id": task_id,
            "status": status,
            "all_cancelled": bool(result["all_acknowledged"]),
            "cancelled_sessions": int(result["sessions_cancel_requested"]),
            "failed_sessions": len(result["errors"]),
        }

    async def _execute_task(self, request: TaskRequest) -> TaskResult:
        sessions = [
            SessionContext(
                session_id=f"sk-polar-{uuid.uuid4()}",
                task_id=request.task_id,
                request=request,
                deadline_monotonic=time.monotonic() + request.timeout_seconds,
            )
            for _ in range(request.num_samples)
        ]
        with self._lock:
            record = self._tasks[request.task_id]
            record.sessions = {session.session_id: session for session in sessions}
            if record.cancel_requested:
                for session in sessions:
                    session.cancel_requested = True
                    session.cancel_reason = record.cancel_reason

        async def _on_result(result: SessionResult) -> None:
            result_path = self.pipeline.result_path_for(
                result.task_id,
                result.session_id,
                result.metadata,
            )
            with self._lock:
                record = self._tasks[request.task_id]
                if record.cancel_requested:
                    return
                record.completed_sessions += 1
                if result.status in {SessionStatus.ERROR, SessionStatus.TIMEOUT}:
                    record.errored_sessions += 1
                record.results.append(result)
                record.updated_at = time.time()
                record.session_states[result.session_id] = str(result.status)
                if result_path is not None:
                    record.result_paths.append(result_path)
            self._emit(
                "session.state_changed",
                {
                    "task_id": result.task_id,
                    "session_id": result.session_id,
                    "status": str(result.status),
                },
            )
            self._emit(
                "task.updated",
                {
                    "task_id": request.task_id,
                    "completed_sessions": record.completed_sessions,
                    "total_sessions": record.total_sessions,
                },
            )

        state_callback = (
            self.session_state_changed
            if bool(request.metadata.get("session_pool"))
            else None
        )
        try:
            results = await self.pipeline.run_batch(
                sessions,
                on_result=_on_result,
                on_state=state_callback,
            )
        except Exception:
            with self._lock:
                self._tasks[request.task_id].status = "failed"
                self._tasks[request.task_id].updated_at = time.time()
            raise

        ordered_results = list(results)
        with self._lock:
            record = self._tasks[request.task_id]
            if record.cancel_requested:
                return TaskResult(
                    task_id=request.task_id,
                    status="cancelled",
                    results=[],
                    result_paths=[],
                )
            record.status = "completed"
            record.completed_sessions = len(ordered_results)
            record.results = ordered_results
            record.updated_at = time.time()
            result_paths = list(record.result_paths)

        return TaskResult(
            task_id=request.task_id,
            status="completed",
            results=ordered_results,
            result_paths=result_paths,
        )

    def get_task(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            return TaskStatus(
                task_id=record.task_id,
                status=record.status,
                total_sessions=record.total_sessions,
                completed_sessions=record.completed_sessions,
                results=list(record.results),
                result_paths=list(record.result_paths),
            )

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for record in self._tasks.values():
                out.append({
                    "task_id": record.task_id,
                    "status": record.status,
                    "harness": record.harness,
                    "model": record.model,
                    "num_samples": record.total_sessions,
                    "completed_sessions": record.completed_sessions,
                    "errored_sessions": record.errored_sessions,
                    "mean_reward": _mean_reward(record.results),
                    "mean_traces": _mean_traces(record.results),
                    "mean_completions": _mean_completions(record.results),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "source": "live",
                })
            return out

    def list_sessions_for(self, task_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            existing = {r.session_id: r for r in record.results}
            out: list[dict[str, Any]] = []
            for session_id, status in record.session_states.items():
                result = existing.get(session_id)
                if result is not None:
                    traces = result.trajectory.traces
                    reward = traces[-1].reward if traces else None
                    out.append({
                        "session_id": result.session_id,
                        "task_id": result.task_id,
                        "status": str(result.status),
                        "node_id": result.node_id,
                        "reward": reward,
                        "timing": result.timing.model_dump(),
                        "error": result.error,
                    })
                else:
                    out.append({
                        "session_id": session_id,
                        "task_id": task_id,
                        "status": status,
                    })
            return out

    def status(self) -> dict[str, object]:
        with self._lock:
            task_statuses = {
                task_id: record.status
                for task_id, record in self._tasks.items()
            }
        return {
            "tasks": task_statuses,
            "pipeline": self.pipeline.status(),
            "nodes": self.scheduler.stats(),
        }
