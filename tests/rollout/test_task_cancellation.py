from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.agent.models import AgentSpec
from polar.gateway.dispatcher import ManagedSession, SessionDispatcher, SessionStage
from polar.gateway.node import GatewayNodeManager
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import SessionContext, SessionDispatchRequest, TaskRequest
from polar.rollout.pipeline import Pipeline
from polar.rollout.timer import StageTimer
from polar.runtime.models import RuntimeSpec


class _BlockingPipeline:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.sessions = []

    async def run_batch(self, sessions, *, on_result=None, on_state=None):
        self.sessions = list(sessions)
        self.started.set()
        await asyncio.Event().wait()

    def result_path_for(self, *args, **kwargs):
        return None


def _task_request() -> TaskRequest:
    return TaskRequest(
        task_id="task-old-policy",
        instruction="do work",
        num_samples=2,
        runtime=RuntimeSpec(image="sandbox:v1"),
        agent=AgentSpec(harness="codex"),
    )


def test_rollout_manager_policy_cutoff_cancels_all_owned_sessions() -> None:
    async def scenario() -> None:
        pipeline = _BlockingPipeline()
        manager = RolloutManager(
            pipeline=pipeline,
            scheduler=NodeScheduler(),
        )
        await manager.submit_task(_task_request())
        await pipeline.started.wait()

        result = await manager.cancel_tasks(
            ["task-old-policy"],
            reason="policy_cutoff",
        )

        assert result["all_acknowledged"] is True
        assert result["cancelled"] == 1
        assert result["sessions_cancel_requested"] == 2
        assert manager.get_task("task-old-policy").status == "cancelled"
        assert all(session.cancel_requested for session in pipeline.sessions)
        assert all(session.cancel_reason == "policy_cutoff" for session in pipeline.sessions)

    asyncio.run(scenario())


def test_rollout_manager_policy_cutoff_fences_submit_that_arrives_late() -> None:
    async def scenario() -> None:
        pipeline = _BlockingPipeline()
        manager = RolloutManager(
            pipeline=pipeline,
            scheduler=NodeScheduler(),
        )

        result = await manager.cancel_tasks(
            ["task-old-policy"],
            reason="policy_cutoff",
        )
        assert result["all_acknowledged"] is True
        assert result["cancelled_before_submit"] == 1

        task_id = await manager.submit_task(_task_request())

        assert task_id == "task-old-policy"
        assert manager.get_task(task_id).status == "cancelled"
        assert pipeline.sessions == []

    asyncio.run(scenario())


def test_rollout_manager_waits_for_real_pipeline_delete_ack() -> None:
    async def scenario() -> None:
        delete_calls = []
        dispatch_count = 0
        all_dispatched = asyncio.Event()

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

        class Client:
            async def delete(self, url, params=None):
                delete_calls.append((url, params))
                return Response()

        pipeline = Pipeline(
            callback_url="http://127.0.0.1:8080/callbacks/session_result",
            save_dir=None,
            scheduler=NodeScheduler(),
        )

        async def start() -> None:
            pipeline._client = Client()
            pipeline._started = True

        async def dispatch(session):
            nonlocal dispatch_count
            session.gateway_url = "http://127.0.0.1:8100"
            dispatch_count += 1
            if dispatch_count == 2:
                all_dispatched.set()
            await asyncio.Event().wait()

        pipeline.start = start
        pipeline._dispatch_session = dispatch
        manager = RolloutManager(
            pipeline=pipeline,
            scheduler=NodeScheduler(),
        )
        await manager.submit_task(_task_request())
        await all_dispatched.wait()

        result = await manager.cancel_tasks(
            ["task-old-policy"],
            reason="policy_cutoff",
        )

        assert result["all_acknowledged"] is True
        assert result["sessions_cancel_requested"] == 2
        assert len(delete_calls) == 2
        assert all(params == {"reason": "policy_cutoff"} for _, params in delete_calls)

    asyncio.run(scenario())


def test_rollout_manager_is_fail_closed_when_gateway_delete_fails() -> None:
    async def scenario() -> None:
        dispatched = asyncio.Event()

        class Response:
            status_code = 503

            def raise_for_status(self) -> None:
                raise RuntimeError("gateway unavailable")

        class Client:
            async def delete(self, url, params=None):
                return Response()

        request = _task_request().model_copy(update={"num_samples": 1})
        pipeline = Pipeline(
            callback_url="http://127.0.0.1:8080/callbacks/session_result",
            save_dir=None,
            scheduler=NodeScheduler(),
        )

        async def start() -> None:
            pipeline._client = Client()
            pipeline._started = True

        async def dispatch(session):
            session.gateway_url = "http://127.0.0.1:8100"
            dispatched.set()
            await asyncio.Event().wait()

        pipeline.start = start
        pipeline._dispatch_session = dispatch
        manager = RolloutManager(
            pipeline=pipeline,
            scheduler=NodeScheduler(),
        )
        await manager.submit_task(request)
        await dispatched.wait()

        result = await manager.cancel_tasks(
            ["task-old-policy"],
            reason="policy_cutoff",
        )

        assert result["all_acknowledged"] is False
        assert len(result["errors"]) == 1
        assert "gateway unavailable" in next(iter(result["errors"].values()))

    asyncio.run(scenario())


class _Runtime:
    def __init__(self) -> None:
        self.cancel_count = 0

    async def cancel(self) -> None:
        self.cancel_count += 1


class _Dispatcher:
    def __init__(self, managed: ManagedSession) -> None:
        self.managed = managed

    async def cancel(self, session_id: str, *, reason: str | None = None):
        assert session_id == self.managed.session_id
        self.managed.cancel_requested = True
        self.managed.cancel_reason = reason
        return self.managed


class _Inflight:
    def __init__(self) -> None:
        self.closed = []

    async def close_session(self, session_id: str, *, reason: str | None = None) -> int:
        self.closed.append((session_id, reason))
        return 1


def test_gateway_policy_cutoff_stops_agent_and_judge_runtimes(tmp_path: Path) -> None:
    async def scenario() -> None:
        agent_runtime = _Runtime()
        judge_runtime = _Runtime()
        request = SessionDispatchRequest(
            session_id="session-old-policy",
            task_id="task-old-policy",
            instruction="do work",
            remaining_timeout_seconds=60.0,
            runtime=RuntimeSpec(image="sandbox:v1"),
            agent=AgentSpec(harness="codex"),
        )
        managed = ManagedSession(
            request=request,
            timer=StageTimer(),
            session_dir=tmp_path,
            artifacts_dir=tmp_path,
            runtime=agent_runtime,
            eval_runtime=judge_runtime,
        )

        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        manager._dispatcher = _Dispatcher(managed)
        manager.inflight = _Inflight()
        manager._policy_cleanup_tasks = set()

        assert await manager.cancel(
            managed.session_id,
            reason="policy_cutoff",
        )
        await asyncio.sleep(0)
        cleanup_tasks = list(manager._policy_cleanup_tasks)
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)

        assert manager.inflight.closed == [
            ("session-old-policy", "policy_cutoff")
        ]
        assert agent_runtime.cancel_count == 1
        assert judge_runtime.cancel_count == 1

    asyncio.run(scenario())


def test_pipeline_policy_cutoff_waits_for_gateway_delete_ack() -> None:
    async def scenario() -> None:
        calls = []

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

        class Client:
            async def delete(self, url, params=None):
                calls.append((url, params))
                return Response()

        pipeline = Pipeline(
            callback_url="http://127.0.0.1:8080/callbacks/session_result",
            save_dir=None,
            scheduler=NodeScheduler(),
        )
        pipeline._client = Client()
        session = SessionContext(
            session_id="session-old-policy",
            task_id="task-old-policy",
            request=_task_request(),
            gateway_url="http://127.0.0.1:8100",
            cancel_requested=True,
            cancel_reason="policy_cutoff",
        )

        await pipeline._cleanup_session(
            session,
            reason="policy_cutoff",
            strict=True,
        )

        assert session.cancel_acknowledged is True
        assert session.cancel_error is None
        assert calls == [
            (
                "http://127.0.0.1:8100/sessions/session-old-policy",
                {"reason": "policy_cutoff"},
            )
        ]

    asyncio.run(scenario())


def test_policy_cutoff_tombstone_prevents_late_dispatch_enqueue(tmp_path: Path) -> None:
    async def scenario() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
        )
        await dispatcher.start()
        try:
            assert await dispatcher.cancel(
                "session-race",
                reason="policy_cutoff",
            ) is None
            request = SessionDispatchRequest(
                session_id="session-race",
                task_id="task-old-policy",
                instruction="do work",
                remaining_timeout_seconds=60.0,
                runtime=RuntimeSpec(image="sandbox:v1"),
                agent=AgentSpec(harness="codex"),
            )
            managed = ManagedSession(
                request=request,
                timer=StageTimer(),
                session_dir=tmp_path,
                artifacts_dir=tmp_path,
            )

            with pytest.raises(ValueError, match="cancelled before enqueue"):
                await dispatcher.enqueue(managed)
        finally:
            await dispatcher.stop()

    asyncio.run(scenario())


def test_policy_cutoff_does_not_release_unowned_ready_slot(tmp_path: Path) -> None:
    async def scenario() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
        )
        await dispatcher._ready_slots.acquire()
        request = SessionDispatchRequest(
            session_id="session-waiting-ready-slot",
            task_id="task-old-policy",
            instruction="do work",
            remaining_timeout_seconds=60.0,
            runtime=RuntimeSpec(image="sandbox:v1"),
            agent=AgentSpec(harness="codex"),
        )
        managed = ManagedSession(
            request=request,
            timer=StageTimer(),
            session_dir=tmp_path,
            artifacts_dir=tmp_path,
            stage=SessionStage.READY,
        )
        dispatcher._sessions[managed.session_id] = managed
        waiter = asyncio.create_task(dispatcher._acquire_ready_slot(managed))
        await asyncio.sleep(0)

        await dispatcher.cancel(managed.session_id, reason="policy_cutoff")

        assert await waiter is False
        assert dispatcher._ready_slots._value == 0
        dispatcher._ready_slots.release()
        assert dispatcher._ready_slots._value == 1

    asyncio.run(scenario())
