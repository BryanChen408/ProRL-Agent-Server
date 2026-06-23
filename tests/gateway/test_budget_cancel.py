from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from polar.agent.models import AgentSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionResult, SessionStatus
from polar.rollout.timer import StageTimer
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import Trace, Trajectory


def _request() -> SessionDispatchRequest:
    return SessionDispatchRequest(
        session_id="s",
        task_id="t",
        instruction="do work",
        remaining_timeout_seconds=60.0,
        runtime=RuntimeSpec(image="sandbox:v1"),
        agent=AgentSpec(harness="codex"),
    )


def _managed(tmp_path: Path, *, reason: str | None = "pipeline_budget_exceeded") -> ManagedSession:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return ManagedSession(
        request=_request(),
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=artifacts_dir,
        cancel_requested=True,
        cancel_reason=reason,
    )


def test_budget_cancel_builds_trainable_partial_result(tmp_path: Path) -> None:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node-test"
    trace = Trace(response_ids=[1], loss_mask=[1], reward=0.3)
    built = SessionResult(
        session_id="s",
        task_id="t",
        status=SessionStatus.ERROR,
        trajectory=Trajectory(status="ERROR", traces=[trace], error="pipeline budget exceeded"),
        metadata={"base": True},
        error="pipeline budget exceeded",
    )

    async def build_session_result(managed):
        return built

    manager._build_session_result = build_session_result

    result = asyncio.run(manager._build_cancelled_session_result(_managed(tmp_path)))

    assert result.status == SessionStatus.COMPLETED
    assert result.error is None
    assert result.metadata["cancelled_partial"] is True
    assert result.metadata["termination_reason"] == "pipeline_budget_exceeded"
    assert result.trajectory.status == "COMPLETED"
    assert result.trajectory.error is None
    assert result.trajectory.metadata["cancelled_partial"] is True
    assert result.trajectory.traces[0].reward == 0.3


def test_manual_cancel_stays_error(tmp_path: Path) -> None:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node-test"

    result = asyncio.run(manager._build_cancelled_session_result(_managed(tmp_path, reason=None)))

    assert result.status == SessionStatus.ERROR
    assert result.error == "session cancelled"
    assert result.trajectory.traces == []


def test_budget_cancel_without_traces_stays_error(tmp_path: Path) -> None:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node-test"
    built = SessionResult(
        session_id="s",
        task_id="t",
        status=SessionStatus.ERROR,
        trajectory=Trajectory(status="ERROR", traces=[]),
        error="pipeline budget exceeded",
    )

    async def build_session_result(managed):
        return built

    manager._build_session_result = build_session_result

    result = asyncio.run(manager._build_cancelled_session_result(_managed(tmp_path)))

    assert result.status == SessionStatus.ERROR
    assert "before any trainable traces" in result.error


def test_budget_delete_preserves_active_storage(monkeypatch) -> None:
    from polar.gateway import server

    class FakeNodeManager:
        def __init__(self):
            self.calls = []

        async def cancel(self, session_id: str, *, reason: str | None = None) -> bool:
            self.calls.append((session_id, reason))
            return True

    storage = SessionStore()
    storage.ensure_session("s", None, None, None, task_id="t")
    storage.save_message("s", {"model": "m"}, {"choices": []}, task_id="t")
    registry = SessionRegistry()
    registry.register("s", task_id="t", status=SessionStatus.RUNNING)
    node_manager = FakeNodeManager()
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=node_manager,
            session_registry=registry,
            storage=storage,
        ),
    )

    response = asyncio.run(server.delete_session("s", reason="pipeline_budget_exceeded"))

    assert response.deleted is True
    assert response.messages_deleted == 0
    assert node_manager.calls == [("s", "pipeline_budget_exceeded")]
    assert registry.get("s") is not None
    assert len(storage.load_completion_session("s").completions) == 1


def test_manual_delete_removes_storage(monkeypatch) -> None:
    from polar.gateway import server

    class FakeNodeManager:
        async def cancel(self, session_id: str, *, reason: str | None = None) -> bool:
            return True

    storage = SessionStore()
    storage.ensure_session("s", None, None, None, task_id="t")
    storage.save_message("s", {"model": "m"}, {"choices": []}, task_id="t")
    registry = SessionRegistry()
    registry.register("s", task_id="t", status=SessionStatus.RUNNING)
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=FakeNodeManager(),
            session_registry=registry,
            storage=storage,
        ),
    )

    response = asyncio.run(server.delete_session("s"))

    assert response.deleted is True
    assert response.messages_deleted == 1
    assert registry.get("s") is None
    assert storage.load_completion_session("s").completions == []
