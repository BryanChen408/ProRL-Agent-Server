from __future__ import annotations

import asyncio

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import TaskRequest


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


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="task-1",
        instruction="do work",
        num_samples=1,
        agent=AgentSpec(harness="codex"),
    )


def test_manager_single_task_cancel_adapts_batch_result() -> None:
    async def _run() -> None:
        pipeline = _BlockingPipeline()
        manager = RolloutManager(pipeline=pipeline, scheduler=NodeScheduler())
        await manager.submit_task(_request())
        await pipeline.started.wait()

        result = await manager.cancel_task("task-1")

        assert result == {
            "task_id": "task-1",
            "status": "cancelled",
            "all_cancelled": True,
            "cancelled_sessions": 1,
            "failed_sessions": 0,
        }
        assert pipeline.sessions[0].cancel_requested is True
        assert pipeline.sessions[0].cancel_reason == "sync_oversubscribe_abort"

        repeated = await manager.cancel_task("task-1")
        assert repeated is not None
        assert repeated["status"] == "cancelled"
        assert repeated["all_cancelled"] is True
        assert repeated["cancelled_sessions"] == 0

    asyncio.run(_run())


def test_manager_single_task_cancel_rejects_unknown_task() -> None:
    manager = RolloutManager(
        pipeline=_BlockingPipeline(),
        scheduler=NodeScheduler(),
    )
    assert asyncio.run(manager.cancel_task("missing")) is None
