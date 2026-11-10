"""Deterministic lifecycle orderings; Docker commands and engine work are isolated."""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from test_discard_judge import setup
from polar.gateway import server
from polar.gateway.control import GatewayControlStore
from polar.gateway.engine import VLLMEngine
from polar.gateway.inflight import InflightGenerationTracker
from polar.gateway.proxy import InferenceClient, UpstreamError
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager, _TaskRecord
from polar.rollout.models import SessionContext, TaskRequest
from polar.rollout.pipeline import Pipeline
from polar.runtime.docker import DockerRuntime


ORDERS = list(itertools.permutations(("policy", "group", "postrun"))) + [("simultaneous",)]
RELEASE_ORDERS = list(itertools.permutations(("inference", "agent", "judge")))


async def settle():
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.mark.parametrize("order", ORDERS, ids=lambda x: "-".join(x))
@pytest.mark.parametrize("release_order", RELEASE_ORDERS, ids=lambda x: "-".join(x))
@pytest.mark.parametrize("retire_dispatcher", [False, True])
def test_release_order_matrix(tmp_path, monkeypatch, order, release_order, retire_dispatcher):
    async def run():
        node, managed = setup(tmp_path)
        gates = {name: asyncio.Event() for name in ("inference", "agent", "judge")}
        commands, leases, runtimes = {}, {}, {}
        for name in ("agent", "judge"):
            runtime = DockerRuntime.__new__(DockerRuntime)
            runtime._destroyed = False
            runtime._chmod_needed = False
            runtime._active_process = None
            runtime._container_name = "isolated-" + name
            runtime._npu_lock = Mock()
            leases[name] = runtime._npu_lock
            commands[name] = []

            async def command(*args, _name=name, **kwargs):
                commands[_name].append(args[1])
                if args[1] == "rm":
                    await gates[_name].wait()
                return 0, "", ""

            runtime._run_local_command = command
            runtimes[name] = runtime
        managed.runtime, managed.eval_runtime = runtimes["agent"], runtimes["judge"]
        tracker = InflightGenerationTracker()
        node.inflight = tracker
        registry, storage = SessionRegistry(), SessionStore()
        registry.register(managed.session_id, task_id="task")
        storage.ensure_session(managed.session_id, None, None, None, task_id="task")
        node.release_session_affinity_best_effort = AsyncMock(return_value=False)
        proxy = InferenceClient("http://127.0.0.1:1", VLLMEngine())
        state = SimpleNamespace(
            node_manager=node, session_registry=registry, storage=storage,
            discarded_sessions=set(), inflight=tracker,
            control=GatewayControlStore(None), inference=proxy,
        )
        monkeypatch.setattr(server, "_state", state)
        pipeline = Pipeline.__new__(Pipeline)
        pipeline._new_client = lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app))
        session = SessionContext(
            session_id=managed.session_id, task_id="task",
            request=TaskRequest(task_id="task", instruction="test", agent=managed.request.agent),
            gateway_url="http://isolated",
        )
        manager = RolloutManager(pipeline=pipeline, scheduler=NodeScheduler())
        manager._tasks["task"] = _TaskRecord(
            task_id="task", status="running", total_sessions=1, sessions={session.session_id: session},
        )
        entered = asyncio.Event()
        survivor_gate = asyncio.Event()

        async def generating():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Model the cancellation-safe close of an upstream HTTP response.
                while not gates["inference"].is_set():
                    try:
                        await gates["inference"].wait()
                    except asyncio.CancelledError:
                        pass

        async def survivor():
            await survivor_gate.wait()
            return {"ok": True}

        old = asyncio.create_task(tracker.run(managed.session_id, {}, generating))
        other = asyncio.create_task(tracker.run("unrelated-session", {}, survivor))
        await entered.wait()
        postrun = []

        async def action(name):
            if name == "postrun":
                postrun.extend(asyncio.create_task(rt.stop()) for rt in runtimes.values())
            elif name == "policy":
                paused = await server.pause_inference_generation(
                    timeout_seconds=0, wait_for_drain=False, transition_id="matrix"
                )
                assert paused["paused"]
                await pipeline._cleanup_session(session, reason="policy_cutoff", strict=True)
                assert not all(gate.is_set() for gate in gates.values())
            else:
                ack = await manager.cancel_tasks(["task"], reason="group_unacceptable")
                assert not ack["cleanup_confirmed"]

        try:
            if order == ("simultaneous",):
                await asyncio.wait_for(asyncio.gather(*(action(x) for x in ("policy", "group", "postrun"))), 2)
            else:
                for name in order:
                    await asyncio.wait_for(action(name), 2)
            assert session.cancel_acknowledged
            assert not session.group_cleanup_acknowledged
            assert managed.cancel_reason == "group_unacceptable"
            assert not other.done()
            # Resume admission is not blocked on slow old Docker teardown.
            await proxy.resume_generation()
            forbidden = AsyncMock(return_value={})
            with pytest.raises(UpstreamError, match="closed"):
                await tracker.run(managed.session_id, {"late": True}, forbidden)
            forbidden.assert_not_awaited()
            fresh = await tracker.run("new-policy-session", {}, AsyncMock(return_value={"ok": True}))
            assert fresh.response["ok"]
            node.release_session_affinity_best_effort.reset_mock()
            for name in release_order:
                gates[name].set()
                await settle()
                if retire_dispatcher and all(gates[n].is_set() for n in ("agent", "judge")):
                    await asyncio.gather(*node._policy_cleanup_tasks)
                    node._dispatcher._sessions.pop(managed.session_id, None)
                ack = await manager.cancel_tasks(["task"], reason="group_unacceptable")
                # Even after every runtime exits, router ownership is still pending.
                assert not ack["cleanup_confirmed"]
                if not all(gate.is_set() for gate in gates.values()):
                    node.release_session_affinity_best_effort.assert_not_awaited()
                for runtime_name in ("agent", "judge"):
                    if not gates[runtime_name].is_set():
                        leases[runtime_name].release.assert_not_called()
            await asyncio.gather(*node._policy_cleanup_tasks, *postrun)
            await asyncio.gather(old, return_exceptions=True)
            node.release_session_affinity_best_effort.return_value = True
            acks = await asyncio.gather(*(
                manager.cancel_tasks(["task"], reason="group_unacceptable") for _ in range(8)
            ))
            assert all(ack["cleanup_confirmed"] for ack in acks)
            for name in ("agent", "judge"):
                assert commands[name] == ["kill", "rm"]
                leases[name].release.assert_called_once()
            assert not other.done()
            survivor_gate.set()
            assert (await other).response["ok"]
            assert tracker.status()["active"] == 0
        finally:
            for gate in gates.values():
                gate.set()
            survivor_gate.set()
            await asyncio.gather(old, other, *postrun, *node._policy_cleanup_tasks, return_exceptions=True)
            await proxy.close()

    asyncio.run(run())
