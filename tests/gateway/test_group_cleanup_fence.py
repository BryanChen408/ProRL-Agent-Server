"""Exercise cancellation acknowledgements across the real HTTP conversion layers."""

import asyncio

import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_discard_judge import setup, fake_runtime
from polar.gateway import server
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager, _TaskRecord
from polar.rollout.models import SessionContext, TaskRequest
from polar.rollout.pipeline import Pipeline


@pytest.mark.parametrize("first_reason", ["policy_cutoff", "group_unacceptable"])
@pytest.mark.parametrize("retire_dispatcher", [False, True])
def test_policy_fence_cannot_confirm_group_cleanup(
    tmp_path, monkeypatch, first_reason, retire_dispatcher
):
    async def run():
        node, managed = setup(tmp_path)
        entered, allow_stop = asyncio.Event(), asyncio.Event()

        async def stop():
            entered.set()
            await allow_stop.wait()

        managed.runtime = fake_runtime()
        managed.runtime.cancel = AsyncMock(side_effect=stop)
        registry = SessionRegistry()
        registry.register(managed.session_id, task_id="task")
        storage = SessionStore()
        storage.ensure_session(managed.session_id, None, None, None, task_id="task")
        node.release_session_affinity_best_effort = AsyncMock(return_value=True)
        state = SimpleNamespace(
            node_manager=node, session_registry=registry, storage=storage,
            discarded_sessions=set(),
            inflight=SimpleNamespace(status=lambda: {"entries": []}, close_session=AsyncMock()),
        )
        monkeypatch.setattr(server, "_state", state)
        pipeline = Pipeline.__new__(Pipeline)
        pipeline._new_client = lambda: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app)
        )
        session = SessionContext(
            session_id=managed.session_id, task_id="task",
            request=TaskRequest(task_id="task", instruction="test", agent=managed.request.agent),
            gateway_url="http://isolated", cancel_requested=True,
            cancel_reason=first_reason,
        )
        manager = RolloutManager(pipeline=pipeline, scheduler=NodeScheduler())
        manager._tasks["task"] = _TaskRecord(
            task_id="task", status="cancelled", total_sessions=1,
            cancel_requested=True, cancel_reason=first_reason,
            sessions={session.session_id: session},
        )
        try:
            if first_reason == "group_unacceptable":
                result = await manager.cancel_tasks(["task"], reason=first_reason)
                assert not result["cleanup_confirmed"]
            # This acknowledgement must return while Docker teardown is blocked.
            await asyncio.wait_for(
                pipeline._cleanup_session(session, reason="policy_cutoff", strict=True), 1
            )
            await entered.wait()
            assert session.cancel_acknowledged and not allow_stop.is_set()
            result = await manager.cancel_tasks(["task"], reason="group_unacceptable")
            assert not result["cleanup_confirmed"]
            assert not session.group_cleanup_acknowledged
            pending_error = session.group_cleanup_error
            assert pending_error
            # A delayed policy retry cannot erase the independent cleanup error.
            await pipeline._cleanup_session(session, reason="policy_cutoff", strict=True)
            assert session.group_cleanup_error == pending_error
            assert not session.group_cleanup_acknowledged

            allow_stop.set()
            await asyncio.gather(*node._policy_cleanup_tasks)
            if retire_dispatcher:
                node._dispatcher._sessions.pop(managed.session_id)
            # Runtime completion alone does not prove router capacity was released.
            node.release_session_affinity_best_effort.return_value = False
            result = await manager.cancel_tasks(["task"], reason="group_unacceptable")
            assert not result["cleanup_confirmed"]
            node.release_session_affinity_best_effort.return_value = True
            result = await manager.cancel_tasks(["task"], reason="group_unacceptable")
            assert result["cleanup_confirmed"]
            assert session.group_cleanup_acknowledged
            assert session.group_cleanup_error is None
            assert managed.session_id in state.discarded_sessions
            again = await manager.cancel_tasks(["task"], reason="group_unacceptable")
            assert again["cleanup_confirmed"]
            managed.runtime.cancel.assert_awaited_once()
        finally:
            allow_stop.set()
            await asyncio.gather(*node._policy_cleanup_tasks)

    asyncio.run(run())


def test_failed_policy_teardown_does_not_create_completion_receipt(tmp_path):
    async def run():
        node, managed = setup(tmp_path)
        managed.runtime = fake_runtime()
        managed.runtime.cancel = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
        await node.cancel(managed.session_id, reason="policy_cutoff")
        await asyncio.gather(*node._policy_cleanup_tasks)
        assert managed.session_id not in getattr(node, "_completed_runtime_cleanups", set())

    asyncio.run(run())


@pytest.mark.parametrize("failure", [202, 404, 503, "lost_ack"])
@pytest.mark.parametrize("policy_order", ["before", "after", "concurrent"])
def test_cleanup_failure_never_borrows_policy_ack(failure, policy_order):
    async def run():
        calls = []
        repaired = False

        async def handler(request):
            reason = request.url.params["reason"]
            calls.append(reason)
            if reason == "policy_cutoff" or repaired:
                return httpx.Response(200, json={})
            if failure == "lost_ack":
                raise httpx.ReadError("response lost after server cleanup")
            return httpx.Response(failure, json={})

        pipeline = Pipeline.__new__(Pipeline)
        pipeline._new_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        session = SessionContext(
            session_id="s", task_id="t", gateway_url="http://isolated",
            request=TaskRequest(task_id="t", instruction="test", agent={"harness": "codex"}),
        )
        policy = lambda: pipeline._cleanup_session(session, reason="policy_cutoff", strict=True)
        group = lambda: pipeline._cleanup_session(session, reason="group_unacceptable", strict=True)
        if policy_order == "concurrent":
            results = await asyncio.gather(policy(), group(), return_exceptions=True)
            assert isinstance(results[1], Exception)
        else:
            if policy_order == "before":
                await policy()
            with pytest.raises(Exception):
                await group()
            if policy_order == "after":
                await policy()
        assert session.cancel_acknowledged
        assert not session.group_cleanup_acknowledged
        assert session.group_cleanup_error
        repaired = True
        await group()
        assert session.group_cleanup_acknowledged and session.group_cleanup_error is None

    asyncio.run(run())


def test_late_pending_response_cannot_permanently_erase_successful_cleanup():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                entered.set()
                await release.wait()
                return httpx.Response(202, json={})
            return httpx.Response(200, json={})

        pipeline = Pipeline.__new__(Pipeline)
        pipeline._new_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        session = SessionContext(
            session_id="s", task_id="t", gateway_url="http://isolated",
            request=TaskRequest(task_id="t", instruction="test", agent={"harness": "codex"}),
        )
        old = asyncio.create_task(pipeline._cleanup_session(session, reason="group_unacceptable", strict=True))
        await entered.wait()
        await pipeline._cleanup_session(session, reason="group_unacceptable", strict=True)
        release.set()
        with pytest.raises(RuntimeError):
            await old
        # A conservative stale pending error may cause a retry, but cannot stick forever.
        await pipeline._cleanup_session(session, reason="group_unacceptable", strict=True)
        assert session.group_cleanup_acknowledged and session.group_cleanup_error is None

    asyncio.run(run())


@pytest.mark.parametrize("forget_receipt", [False, True])
def test_group_arrives_after_policy_cleanup_and_session_retirement(tmp_path, monkeypatch, forget_receipt):
    async def run():
        node, managed = setup(tmp_path)
        managed.runtime = fake_runtime()
        registry, storage = SessionRegistry(), SessionStore()
        registry.register(managed.session_id, task_id="task")
        storage.ensure_session(managed.session_id, None, None, None, task_id="task")
        node.release_session_affinity_best_effort = AsyncMock(return_value=True)
        state = SimpleNamespace(
            node_manager=node, session_registry=registry, storage=storage,
            discarded_sessions=set(),
            inflight=SimpleNamespace(status=lambda: {"entries": []}, close_session=AsyncMock()),
        )
        monkeypatch.setattr(server, "_state", state)
        await server.delete_session(managed.session_id, reason="policy_cutoff")
        await asyncio.gather(*node._policy_cleanup_tasks)
        assert registry.get(managed.session_id) is None
        node._dispatcher._sessions.pop(managed.session_id)
        if forget_receipt:
            # Simulate lost in-memory proof after Gateway restart; never invent success.
            node._completed_runtime_cleanups.clear()
            with pytest.raises(server.HTTPException) as error:
                await server.delete_session(managed.session_id, reason="group_unacceptable")
            assert error.value.status_code == 404
        else:
            result = await server.delete_session(managed.session_id, reason="group_unacceptable")
            assert result.deleted
            managed.runtime.cancel.assert_awaited_once()

    asyncio.run(run())
