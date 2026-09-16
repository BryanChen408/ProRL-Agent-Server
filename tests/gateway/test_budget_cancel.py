from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from polar.agent.models import AgentRunResult, AgentSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionResult, SessionStatus
from polar.rollout.timer import StageTimer
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import (
    CompletionRecord, CompletionSession, EvaluatorSpec, StrategySpec, Trace, Trajectory,
)
from polar.trajectory.registry import default_builder_registry


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


@pytest.mark.parametrize("ending", ["context", "output", "output_not_error", "timeout", "other_400", "502", "policy_cutoff", "judge_failure", "output_judge_failure"])
def test_budget_prefix_is_judged_and_accepted_without_hiding_other_failures(tmp_path, ending):
    command = "bash tools/ascendc_eval_pipeline.sh --op_name OP --out_dir judge_out"
    call = {"id": "p1", "type": "function", "function": {
        "name": "Bash", "arguments": json.dumps({"command": command}),
    }}
    records = []
    for i, prompt in enumerate([[1, 2], [1, 2, 10, 99, 50]]):
        records.append(CompletionRecord(
            completion_id=f"c{i}", request={"system": "harness", "messages": []},
            response={"choices": [{
                "input_token_ids": prompt, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "work", "tool_calls": [call] if i == 0 else []},
                "logprobs": {"content": [
                    {"token_id": 10 + i, "logprob": -0.1}, {"token_id": 99, "logprob": -0.2},
                ]},
            }]}, metadata={"policy_version": 0},
        ))
    session = CompletionSession(session_id="s", completions=records)
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node-test"
    manager.storage = SimpleNamespace(load_completion_session=lambda sid: session.model_copy(deep=True))
    manager.builders = default_builder_registry()
    manager.session_registry = SessionRegistry()
    manager.session_registry.register("s", task_id="t")
    managed = _managed(tmp_path, reason=None)
    managed.cancel_requested = ending == "policy_cutoff"
    managed.cancel_reason = "policy_cutoff" if managed.cancel_requested else None
    managed.request.agent = AgentSpec(harness="claude_code")
    managed.request.builder = StrategySpec(strategy="prefix_merging", config={"end_of_turn_token_id": 99})
    managed.request.evaluator = EvaluatorSpec(strategy="operator_judge", postrun_timeout_seconds=30)
    managed.agent_result = AgentRunResult(
        status="timeout" if ending == "timeout" else "failed", return_code=1, error="agent stopped",
    )
    log = tmp_path / "logs/agent/claude-code.txt"
    log.parent.mkdir(parents=True)
    events = [{"type": "user", "message": {"content": [{
        "type": "tool_result", "tool_use_id": "p1",
        "content": "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=2.0",
    }]}}]
    if ending != "timeout":
        events.append({"type": "result", "is_error": True, "api_error_status": 502 if ending == "502" else 400,
                       "result": "maximum context length is 262144" if ending in {"context", "policy_cutoff", "judge_failure"} else "invalid request"})
        if ending.startswith("output"):
            events[-1].update(
                is_error=ending != "output_not_error", api_error_status=None,
                result="API Error: Claude's response exceeded the 32768 output token maximum. "
                       "To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.",
            )
    log.write_text("\n".join(json.dumps(e) for e in events) + '\n{"partial":')
    judged = []

    async def judge(request, trajectory, **kwargs):
        judged.append(trajectory)
        if ending in {"judge_failure", "output_judge_failure"}:
            raise RuntimeError("judge container unavailable")
        for trace in trajectory.traces:
            trace.reward = 0.9
        return trajectory

    manager._run_eval = judge

    async def run():
        managed.execution_deadline = asyncio.get_running_loop().time() + 30
        manager._start_postrun_deadline(managed)
        return await manager._build_session_result(managed)

    result = asyncio.run(run())
    assert len(judged) == 1
    if ending in {"judge_failure", "output_judge_failure"}:
        assert result.status == "ERROR"
        assert "judge container unavailable" in result.error
        assert result.trajectory.metadata["completed_pipeline_prefix"]["tool_call_id"] == "p1"
        return
    assert result.trajectory.traces[0].reward == 0.9
    if ending in {"context", "output", "timeout"}:
        assert result.status == "COMPLETED"
        assert result.error is None
        assert result.trajectory.metadata["completed_pipeline_prefix"]["tool_call_id"] == "p1"
        assert result.trajectory.traces[0].loss_mask == [1, 1, 0, 0, 0]
        if ending == "output":
            assert result.trajectory.metadata["termination_reason"] == "agent_output_limit_exceeded"
            assert result.trajectory.metadata["agent_error"] == "agent stopped"
    else:
        assert result.status == "ERROR"
        assert "completed_pipeline_prefix" not in result.trajectory.metadata


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
            self.affinity_releases = []

        async def cancel(self, session_id: str, *, reason: str | None = None) -> bool:
            self.calls.append((session_id, reason))
            return True

        async def release_session_affinity_best_effort(self, session_id: str) -> None:
            self.affinity_releases.append(session_id)

    class FakeInflight:
        def __init__(self):
            self.calls = []

        async def close_session(self, session_id: str, *, reason: str | None = None) -> int:
            self.calls.append((session_id, reason))
            return 0

    storage = SessionStore()
    storage.ensure_session("s", None, None, None, task_id="t")
    storage.save_message("s", {"model": "m"}, {"choices": []}, task_id="t")
    registry = SessionRegistry()
    registry.register("s", task_id="t", status=SessionStatus.RUNNING)
    node_manager = FakeNodeManager()
    inflight = FakeInflight()
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=node_manager,
            session_registry=registry,
            storage=storage,
            inflight=inflight,
        ),
    )

    response = asyncio.run(server.delete_session("s", reason="pipeline_budget_exceeded"))

    assert response.deleted is True
    assert response.messages_deleted == 0
    assert node_manager.calls == [("s", "pipeline_budget_exceeded")]
    assert node_manager.affinity_releases == []
    assert inflight.calls == []
    assert registry.get("s") is not None
    assert len(storage.load_completion_session("s").completions) == 1
    assert storage.is_session_closed("s") is False


def test_manual_delete_removes_storage(monkeypatch) -> None:
    from polar.gateway import server

    class FakeNodeManager:
        def __init__(self):
            self.calls = []
            self.affinity_releases = []

        async def cancel(self, session_id: str, *, reason: str | None = None) -> bool:
            self.calls.append((session_id, reason))
            return True

        async def release_session_affinity_best_effort(self, session_id: str) -> None:
            self.affinity_releases.append(session_id)

    class FakeInflight:
        def __init__(self):
            self.calls = []

        async def close_session(self, session_id: str, *, reason: str | None = None) -> int:
            self.calls.append((session_id, reason))
            return 1

    storage = SessionStore()
    storage.ensure_session("s", None, None, None, task_id="t")
    storage.save_message("s", {"model": "m"}, {"choices": []}, task_id="t")
    registry = SessionRegistry()
    registry.register("s", task_id="t", status=SessionStatus.RUNNING)
    node_manager = FakeNodeManager()
    inflight = FakeInflight()
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=node_manager,
            session_registry=registry,
            storage=storage,
            inflight=inflight,
        ),
    )

    response = asyncio.run(server.delete_session("s"))

    assert response.deleted is True
    assert response.messages_deleted == 1
    assert node_manager.calls == [("s", None)]
    assert node_manager.affinity_releases == ["s"]
    assert inflight.calls == [("s", "delete_session")]
    assert registry.get("s") is None
    assert storage.load_completion_session("s").completions == []
    assert storage.is_session_closed("s") is True
    assert storage.save_message("s", {"model": "m"}, {"choices": []}, task_id="t") is None
    assert storage.late_completion_summary()["late_message_drop_count"] == 1
