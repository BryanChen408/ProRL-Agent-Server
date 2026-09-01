from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from polar.rollout import server
from polar.rollout.models import (
    PolicyEpochInitializeRequest,
    PolicyTransitionBeginRequest,
    PolicyTransitionCommitRequest,
    PolicyTransitionDrainRequest,
    PolicyTransitionFailRequest,
)
from polar.rollout.policy_transition import (
    PolicyTransitionError,
    PolicyTransitionPhase,
    PolicyTransitionStore,
)


def _topology(tmp_path: Path) -> Path:
    path = tmp_path / "topology.yaml"
    path.write_text(
        f"""
rollout:
  public_url: http://127.0.0.1:8080
  save_dir: {tmp_path / 'results'}
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip()
    )
    return path


def test_policy_transition_store_is_idempotent_and_durable(tmp_path: Path) -> None:
    path = tmp_path / "control" / "transition.json"
    store = PolicyTransitionStore(path)
    first = store.start(transition_id="t-0-1", from_epoch=0, to_epoch=1)
    duplicate = store.start(transition_id="t-0-1", from_epoch=0, to_epoch=1)
    assert duplicate == first

    ready = store.update(
        "t-0-1",
        phase=PolicyTransitionPhase.READY_FOR_TRAINING,
    )
    restored = PolicyTransitionStore(path).snapshot()
    assert restored.active_epoch == 0
    assert restored.current == ready

    try:
        store.start(transition_id="other", from_epoch=0, to_epoch=1)
    except PolicyTransitionError as exc:
        assert "still ready_for_training" in str(exc)
    else:
        raise AssertionError("a second concurrent transition must be rejected")


def test_new_trainer_can_reinitialize_after_terminal_previous_run(tmp_path: Path) -> None:
    store = PolicyTransitionStore(tmp_path / "transition.json")
    store.start(
        transition_id="old-update",
        from_epoch=4,
        to_epoch=5,
        from_engine_versions={"engine-000": "12"},
    )
    store.update(
        "old-update",
        phase=PolicyTransitionPhase.SERVING,
        verified_policy_epoch=5,
        engine_versions={"engine-000": "13"},
    )

    initialized = store.start(
        transition_id="new-init",
        from_epoch=0,
        to_epoch=0,
        from_engine_versions={"engine-000": "1"},
        allow_epoch_reset=True,
    )
    assert initialized.phase == PolicyTransitionPhase.QUIESCING
    assert store.snapshot().active_epoch == 0


def test_epoch_reuse_requires_the_same_run_namespace(tmp_path: Path) -> None:
    store = PolicyTransitionStore(tmp_path / "transition.json")
    store.start(
        transition_id="run-a-update",
        policy_namespace="run-a",
        from_epoch=0,
        to_epoch=1,
    )
    store.update("run-a-update", phase=PolicyTransitionPhase.SERVING)

    try:
        store.start(
            transition_id="run-b-update",
            policy_namespace="run-b",
            from_epoch=1,
            to_epoch=2,
        )
    except PolicyTransitionError as exc:
        assert "namespace mismatch" in str(exc)
    else:
        raise AssertionError("a normal update cannot silently cross trainer namespaces")

    reset = store.start(
        transition_id="run-b-bootstrap",
        policy_namespace="run-b",
        from_epoch=0,
        to_epoch=0,
        allow_epoch_reset=True,
    )
    assert reset.policy_namespace == "run-b"


def test_engine_version_evidence_rejects_partial_or_mixed_sync() -> None:
    try:
        server._validate_engine_versions(
            {"engine-000": "8", "engine-001": "9"}
        )
    except PolicyTransitionError as exc:
        assert "mixed weight versions" in str(exc)
    else:
        raise AssertionError("mixed engine versions must fail closed")

    try:
        server._validate_engine_versions(
            {"engine-000": "8"},
            previous={"engine-000": "7", "engine-001": "7"},
            require_advanced=True,
        )
    except PolicyTransitionError as exc:
        assert "engine set changed" in str(exc)
    else:
        raise AssertionError("missing engine evidence must fail closed")

    for after in ("7", "6"):
        try:
            server._validate_engine_versions(
                {"engine-000": after},
                previous={"engine-000": "7"},
                require_advanced=True,
            )
        except PolicyTransitionError as exc:
            assert "did not advance monotonically" in str(exc)
        else:
            raise AssertionError("equal/decreasing engine versions must fail closed")


def test_gateway_control_reconciles_lost_post_ack(monkeypatch) -> None:
    node = SimpleNamespace(id="n1", public_url="http://gateway")
    state = SimpleNamespace(
        topology=SimpleNamespace(gateway=SimpleNamespace(nodes=(node,)))
    )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):  # noqa: ANN201
            return {
                "paused": True,
                "drained": False,
                "inflight": 2,
                "transition_id": "t-0-1",
            }

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs):  # noqa: ANN201
            raise httpx.ReadError(
                "response connection disappeared after apply",
                request=httpx.Request("POST", "http://gateway/admin/inference/pause"),
            )

        async def get(self, *args, **kwargs):  # noqa: ANN201
            return Response()

    monkeypatch.setattr(server, "get_state", lambda: state)
    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    result = asyncio.run(
        server._apply_gateway_control(action="pause", transition_id="t-0-1")
    )
    assert result[0]["status"] == "ok"
    assert result[0]["source"] == "reconciled_status"
    assert "response connection disappeared" in result[0]["post_error"]


def test_gateway_control_does_not_mask_explicit_4xx_with_get(monkeypatch) -> None:
    node = SimpleNamespace(id="n1", public_url="http://gateway")
    state = SimpleNamespace(
        topology=SimpleNamespace(gateway=SimpleNamespace(nodes=(node,)))
    )

    class Response:
        status_code = 409
        text = "transition owner mismatch"

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs):  # noqa: ANN201
            return Response()

        async def get(self, *args, **kwargs):  # noqa: ANN201
            raise AssertionError("authoritative 4xx must not be reconciled with GET")

    monkeypatch.setattr(server, "get_state", lambda: state)
    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    result = asyncio.run(
        server._apply_gateway_control(action="pause", transition_id="new-owner")
    )
    assert result[0]["status"] == "error"
    assert result[0]["source"] == "post"
    assert "HTTP 409" in result[0]["post_error"]


def test_policy_transition_begin_commit_and_duplicate_commit(monkeypatch, tmp_path: Path) -> None:
    server.configure_server(str(_topology(tmp_path)))
    state = server.get_state()
    actions: list[tuple[str, str, int | None]] = []
    gateway_inflight = {"count": 0}

    async def apply(
        *,
        action: str,
        transition_id: str,
        policy_namespace: str | None = None,
        epoch: int | None = None,
        allow_paused_transition_takeover: bool = False,
    ):
        del allow_paused_transition_takeover
        actions.append((action, transition_id, epoch))
        payload = {
            "transition_id": transition_id,
            "policy_namespace": policy_namespace,
            "policy_version": epoch if epoch is not None else state.policy_transitions.snapshot().active_epoch,
            "epoch_enforced": True,
            "paused": action != "resume",
            "drained": True,
            "inflight": 0,
        }
        return [{"node_id": "n1", "status": "ok", "response": payload}]

    async def observe():
        current = state.policy_transitions.snapshot().current
        assert current is not None
        return [{
            "node_id": "n1",
            "status": "ok",
            "response": {
                    "transition_id": current.transition_id,
                    "paused": True,
                    "drained": gateway_inflight["count"] == 0,
                    "inflight": gateway_inflight["count"],
            },
        }]

    monkeypatch.setattr(server, "_apply_gateway_control", apply)
    monkeypatch.setattr(server, "_observe_gateway_control", observe)

    initialized = asyncio.run(
        server.initialize_policy_epoch(
            PolicyEpochInitializeRequest(
                transition_id="init-0",
                epoch=0,
                engine_versions={"engine-000": "1"},
            )
        )
    )
    assert initialized["phase"] == "serving"

    ready = asyncio.run(
        server.begin_policy_transition(
            PolicyTransitionBeginRequest(
                transition_id="t-0-1",
                from_epoch=0,
                to_epoch=1,
                engine_versions={"engine-000": "1"},
            )
        )
    )
    assert ready["phase"] == "admission_closed"
    assert state.manager.status()["policy_admission"]["closed"] is True

    # The vLLM abort RPC has completed, but its old gateway HTTP coroutine may
    # still be unwinding. That cleanup is not an NPU-safety prerequisite.
    gateway_inflight["count"] = 2
    ready = asyncio.run(
        server.confirm_policy_transition_drained(
            "t-0-1",
            PolicyTransitionDrainRequest(
                wait_timeout_seconds=0,
                engine_abort_confirmed=True,
            ),
        )
    )
    assert ready["phase"] == "ready_for_training"
    assert ready["engine_abort_confirmed"] is True

    request = PolicyTransitionCommitRequest(
        verified_policy_epoch=1,
        engine_versions={"engine-000": "2"},
    )
    committed = asyncio.run(server.commit_policy_transition("t-0-1", request))
    duplicate = asyncio.run(server.commit_policy_transition("t-0-1", request))
    assert committed["phase"] == "serving"
    assert duplicate["phase"] == "serving"
    assert committed["active_epoch"] == 1
    admission = state.manager.status()["policy_admission"]
    assert admission["closed"] is False
    assert admission["active_epoch"] == 1

    restored = PolicyTransitionStore(
        tmp_path / "results" / "_control" / "policy-transition.json"
    ).snapshot()
    assert restored.active_epoch == 1
    assert actions.count(("resume", "t-0-1", 1)) == 1


def test_ambiguous_engine_failure_stays_fail_closed(monkeypatch, tmp_path: Path) -> None:
    server.configure_server(str(_topology(tmp_path)))
    state = server.get_state()

    async def apply(
        *,
        action: str,
        transition_id: str,
        policy_namespace: str | None = None,
        epoch: int | None = None,
        allow_paused_transition_takeover: bool = False,
    ):
        del allow_paused_transition_takeover
        return [{
            "node_id": "n1",
            "status": "ok",
            "response": {
                "transition_id": transition_id,
                "policy_namespace": policy_namespace,
                "policy_version": epoch,
                "epoch_enforced": True,
                "paused": action != "resume",
                "drained": True,
                "inflight": 0,
            },
        }]

    async def observe():
        current = state.policy_transitions.snapshot().current
        assert current is not None
        return [{
            "node_id": "n1",
            "status": "ok",
            "response": {
                "transition_id": current.transition_id,
                "paused": True,
                "drained": True,
                "inflight": 0,
            },
        }]

    monkeypatch.setattr(server, "_apply_gateway_control", apply)
    monkeypatch.setattr(server, "_observe_gateway_control", observe)
    asyncio.run(
        server.initialize_policy_epoch(
            PolicyEpochInitializeRequest(
                transition_id="init-0",
                epoch=0,
                engine_versions={"engine-000": "1"},
            )
        )
    )
    asyncio.run(
        server.begin_policy_transition(
            PolicyTransitionBeginRequest(
                transition_id="t-0-1",
                from_epoch=0,
                to_epoch=1,
                engine_versions={"engine-000": "1"},
            )
        )
    )
    failed = asyncio.run(
        server.fail_policy_transition(
            "t-0-1",
            PolicyTransitionFailRequest(reason="weight update outcome unknown"),
        )
    )
    assert failed["phase"] == "recovery_required"
    assert failed["active_epoch"] == 0
    assert state.manager.status()["policy_admission"]["closed"] is True
