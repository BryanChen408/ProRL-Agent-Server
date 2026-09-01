from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from polar.gateway import server
from polar.gateway.control import GatewayControlStore
from polar.gateway.proxy import InferenceClient, UpstreamError
from polar.gateway.storage import SessionStore


def test_gateway_control_state_survives_restart(tmp_path) -> None:
    path = tmp_path / "control" / "gateway.json"
    store = GatewayControlStore(path)
    store.update(
        paused=True,
        policy_namespace="run-a",
        set_policy_namespace=True,
        policy_epoch=4,
        set_policy_epoch=True,
        epoch_enforced=True,
        transition_id="t-3-4",
    )

    restored = GatewayControlStore(path).snapshot()
    assert restored.paused is True
    assert restored.policy_namespace == "run-a"
    assert restored.policy_epoch == 4
    assert restored.epoch_enforced is True
    assert restored.transition_id == "t-3-4"


def test_request_waiting_behind_pause_rechecks_epoch_before_engine_admission() -> None:
    async def run() -> None:
        client = InferenceClient(
            "http://unused",
            SimpleNamespace(name="fake"),
            initially_paused=True,
        )
        completion = asyncio.create_task(
            client.completion({}, generation_guard=lambda: False)
        )
        await asyncio.sleep(0)
        assert completion.done() is False

        await client.resume_generation()
        with pytest.raises(UpstreamError, match="policy epoch fence"):
            await completion
        assert client.generation_status()["inflight"] == 0
        assert client.generation_status()["drained"] is True

    asyncio.run(run())


def test_epoch_update_requires_current_transition_owner(monkeypatch, tmp_path) -> None:
    control = GatewayControlStore(tmp_path / "gateway.json")
    control.update(paused=True, transition_id="owner")
    state = SimpleNamespace(storage=SessionStore(), control=control)
    monkeypatch.setattr(server, "get_state", lambda: state)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.set_policy_version(
                1,
                policy_namespace="run-a",
                transition_id="stale-writer",
                enforce_epoch=True,
            )
        )
    assert exc_info.value.status_code == 409
    assert state.storage.get_policy_version() is None

    result = asyncio.run(
        server.set_policy_version(
            1,
            policy_namespace="run-a",
            transition_id="owner",
            enforce_epoch=True,
        )
    )
    assert result["policy_namespace"] == "run-a"
    assert result["policy_version"] == 1
    assert result["epoch_enforced"] is True


def test_paused_gateway_rejects_competing_transition(monkeypatch, tmp_path) -> None:
    control = GatewayControlStore(tmp_path / "gateway.json")
    control.update(paused=True, transition_id="owner")

    class Inference:
        async def pause_generation(self, **kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("competing transition must be rejected before side effect")

    state = SimpleNamespace(
        storage=SessionStore(),
        control=control,
        inference=Inference(),
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.pause_inference_generation(
                wait_for_drain=False,
                transition_id="stale-writer",
            )
        )
    assert exc_info.value.status_code == 409


def test_bootstrap_can_take_over_an_already_paused_previous_run(monkeypatch, tmp_path) -> None:
    control = GatewayControlStore(tmp_path / "gateway.json")
    control.update(paused=True, transition_id="old-run")

    class Inference:
        async def pause_generation(self, **kwargs):  # noqa: ANN003, ANN201
            return {"paused": True, "drained": True, "inflight": 0}

    state = SimpleNamespace(
        storage=SessionStore(),
        control=control,
        inference=Inference(),
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    result = asyncio.run(
        server.pause_inference_generation(
            wait_for_drain=False,
            transition_id="new-run",
            allow_paused_transition_takeover=True,
        )
    )
    assert result["paused"] is True
    assert result["transition_id"] == "new-run"


def test_namespace_is_part_of_the_epoch_fence() -> None:
    storage = SessionStore()
    storage.set_policy_version(
        7,
        policy_namespace="run-new",
        enforce_epoch=True,
    )
    state = SimpleNamespace(storage=storage)

    assert server._policy_epoch_rejection(
        state,
        {"policy_namespace": "run-old", "policy_version": 7},
    ) == ("run-old", 7, "run-new", 7)
    assert server._policy_epoch_rejection(
        state,
        {"policy_namespace": "run-new", "policy_version": 7},
    ) is None
