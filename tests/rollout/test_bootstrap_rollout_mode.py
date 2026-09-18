from __future__ import annotations

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from polar.gateway import server as gateway
from polar.gateway.control import GatewayControlStore
from polar.gateway.engine import get_engine
from polar.gateway.inflight import InflightGenerationTracker
from polar.gateway.proxy import InferenceClient
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout import server
from polar.rollout.models import (
    PolicyBootstrapBeginRequest, PolicyQuiesceRequest, PolicyTransitionBeginRequest,
    PolicyTransitionCommitRequest, PolicyTransitionDrainRequest,
)
from polar.rollout.policy_transition import PolicyTransitionError, PolicyTransitionStore


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    for key in ("POLAR_PARTIAL_ROLLOUT", "POLAR_PARTIAL_CHECKPOINT_DIR", "POLAR_CONTROL_DIR"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "topology.yaml"
    path.write_text(f"""
rollout:
  save_dir: {tmp_path / 'results'}
gateway:
  nodes:
    - id: n1
      public_url: http://n1:8100
      inference:
        engine: vllm
    - id: n2
      public_url: http://n2:8100
      inference:
        engine: vllm
""")
    server.configure_server(str(path))
    state = server.get_state()
    nodes = {}
    for node in state.topology.gateway.nodes:
        nodes[node.id] = SimpleNamespace(
            topology=state.topology, node=node,
            inference=InferenceClient("http://engine", get_engine(node.engine)),
            control=GatewayControlStore(tmp_path / f"{node.id}.json"),
            inflight=InflightGenerationTracker(), storage=SessionStore(),
            node_manager=SimpleNamespace(partial_rollout=None),
            session_registry=SessionRegistry(),
        )
    selected = ContextVar("gateway")
    monkeypatch.setattr(gateway, "get_state", selected.get)

    class FleetTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            token = selected.set(nodes[request.url.host])
            try:
                response = await httpx.ASGITransport(app=gateway.app).handle_async_request(request)
                node = nodes[request.url.host]
                if request.url.path.endswith("/rollout-mode") and getattr(node, "lose_ack", False):
                    node.lose_ack = False
                    raise httpx.ReadError("acknowledgement lost", request=request)
                return response
            finally:
                selected.reset(token)

    client = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: client(transport=FleetTransport(), **kw))
    return state, nodes


def bootstrap(namespace="run-a", partial=True):
    return PolicyBootstrapBeginRequest(
        transition_id=f"boot-{namespace}", policy_namespace=namespace, epoch=0,
        partial_rollout=partial, partial_rollout_protocol=2 if partial else 0,
    )


async def commit_bootstrap():
    ready = await server.confirm_policy_transition_drained(
        "boot-run-a", PolicyTransitionDrainRequest(wait_timeout_seconds=0, engine_abort_confirmed=True),
    )
    assert ready["phase"] == "ready_for_training"
    committed = await server.commit_policy_transition(
        "boot-run-a", PolicyTransitionCommitRequest(policy_namespace="run-a", verified_policy_epoch=0, engine_versions={"engine": "1"}),
    )
    assert committed["phase"] == "serving"


def test_env_free_bootstrap_idempotency_updates_and_next_normal_run(fleet):
    state, nodes = fleet

    async def run():
        nodes["n2"].lose_ack = True
        result = await server.begin_policy_bootstrap(bootstrap())
        assert result["phase"] == "admission_closed"
        assert result["gateway_nodes"]["n2"]["source"] == "reconciled_status"
        assert result["rollout_mode_negotiated"] is True
        managers = {}
        for name, node in nodes.items():
            partial = node.inference.partial
            assert partial is node.node_manager.partial_rollout
            managers[name] = partial
            assert partial.checkpoint_dir.is_dir()
            assert partial.checkpoint_dir.is_relative_to(state.topology.rollout.save_dir)
            assert name in partial.checkpoint_dir.name
            assert node.inflight._retain_completed
            saved = GatewayControlStore(node.control.path).snapshot()
            assert saved.partial_rollout is True
            assert saved.rollout_namespace == "run-a"
            restored = InferenceClient("http://engine", get_engine("vllm"),
                                       initially_paused=saved.paused, partial_rollout=saved.partial_rollout,
                                       partial_checkpoint_dir=saved.partial_checkpoint_dir)
            assert restored.partial.checkpoint_dir == partial.checkpoint_dir
            assert restored.generation_status()["paused"]
        assert nodes["n1"].inference.partial.checkpoint_dir != nodes["n2"].inference.partial.checkpoint_dir
        await server.begin_policy_bootstrap(bootstrap())
        assert all(node.inference.partial is managers[name] for name, node in nodes.items())
        await commit_bootstrap()
        assert all(node.inference.partial.policy_version() == 0 for node in nodes.values())
        await server._validate_node_rollout_mode("n1", "http://n1:8100")

        # A trainer cannot silently replace a live run or change its mode.
        for request in (bootstrap("run-b", False), bootstrap("run-a", False)):
            with pytest.raises(HTTPException) as error:
                await server.begin_policy_bootstrap(request)
            assert error.value.status_code == 409
        restored_store = PolicyTransitionStore(state.policy_transitions.path)
        with pytest.raises(PolicyTransitionError, match="immutable"):
            restored_store.start(transition_id="bad-update", policy_namespace="run-a", from_epoch=0, to_epoch=1)

        await server.quiesce_policy(PolicyQuiesceRequest(policy_namespace="run-a", epoch=0))
        normal = await server.begin_policy_bootstrap(bootstrap("run-b", False))
        assert normal["phase"] == "admission_closed"
        for node in nodes.values():
            assert node.inference.partial is None
            assert node.node_manager.partial_rollout is None
            assert not node.inflight._retain_completed
            assert GatewayControlStore(node.control.path).snapshot().partial_rollout is False

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["unsupported", "active", "unwritable"])
def test_one_failed_gateway_keeps_admission_closed(fleet, monkeypatch, tmp_path, failure):
    state, nodes = fleet
    node = nodes["n2"]
    if failure == "unsupported":
        node.inference.engine = get_engine("sglang")
    elif failure == "active":
        node.session_registry = SimpleNamespace(active_sessions=lambda: ["active-session"])
    else:
        obstacle = tmp_path / "file"
        obstacle.write_text("not a directory")
        original = gateway._partial_checkpoint_path
        monkeypatch.setattr(gateway, "_partial_checkpoint_path", lambda s, ns: obstacle / "child" if s is node else original(s, ns))
    result = asyncio.run(server.begin_policy_bootstrap(bootstrap()))
    assert result["phase"] == "quiescing"
    assert result["gateway_nodes"]["n2"]["status"] == "error"
    assert state.manager.status()["policy_admission"]["closed"] is True
    assert all(n.inference.generation_status()["paused"] for n in nodes.values())
    assert node.control.snapshot().partial_rollout is None


def test_update_preserves_manager_and_unconfigured_restart_cannot_rejoin(fleet):
    _, nodes = fleet

    async def run():
        await server.begin_policy_bootstrap(bootstrap())
        await commit_bootstrap()
        original = nodes["n1"].inference.partial
        update = await server.begin_policy_transition(PolicyTransitionBeginRequest(
            transition_id="update", policy_namespace="run-a", from_epoch=0, to_epoch=1,
            engine_versions={"engine": "1"}, partial_rollout=True,
        ))
        assert update["phase"] == "admission_closed"
        assert update["rollout_mode_negotiated"] is True
        assert nodes["n1"].inference.partial is original
        nodes["n2"].inference = InferenceClient("http://engine", get_engine("vllm"))
        with pytest.raises(HTTPException, match="acknowledged"):
            await server._validate_node_rollout_mode("n2")
        with pytest.raises(HTTPException, match="topology"):
            await server._validate_node_rollout_mode("n3")

    asyncio.run(run())


def test_rejects_wrong_protocol_before_closing_admission(fleet):
    state, _ = fleet
    request = bootstrap().model_copy(update={"partial_rollout_protocol": 1})
    with pytest.raises(HTTPException, match="protocol"):
        asyncio.run(server.begin_policy_bootstrap(request))
    assert state.policy_transitions.snapshot().current is None


@pytest.mark.parametrize("partial", [False, True])
def test_gateway_build_restores_negotiated_mode_over_environment(fleet, monkeypatch, tmp_path, partial):
    state, _ = fleet
    monkeypatch.setenv("POLAR_CONTROL_DIR", str(tmp_path / "control"))
    # Stale manual environment must not override an explicitly negotiated mode.
    monkeypatch.setenv("POLAR_PARTIAL_ROLLOUT", "0" if partial else "1")
    store = GatewayControlStore(tmp_path / "control" / "gateway-n1.json")
    store.update(
        paused=True, transition_id="owner", rollout_namespace="run-a",
        partial_rollout=partial, partial_checkpoint_dir=str(tmp_path / "checkpoints") if partial else None,
        policy_namespace="run-a", set_policy_namespace=True,
        policy_epoch=3, set_policy_epoch=True, epoch_enforced=True,
    )

    async def run():
        restored = gateway._build_state(state.topology, "n1")
        try:
            assert restored.inference.generation_status()["paused"] is True
            assert restored.inference.generation_status()["rollout_namespace"] == "run-a"
            assert (restored.inference.partial is not None) is partial
            assert restored.node_manager.partial_rollout is restored.inference.partial
            assert restored.inflight._retain_completed is partial
            if partial:
                assert restored.inference.partial.policy_version() == 3
        finally:
            await restored.node_manager.close()

    asyncio.run(run())
