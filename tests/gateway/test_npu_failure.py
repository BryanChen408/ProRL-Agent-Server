import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from polar.agent.models import AgentRunResult, AgentSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest
from polar.rollout.timer import StageTimer
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.models import EvaluatorSpec
from polar.trajectory.models import Trace, Trajectory


def managed_session(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return ManagedSession(
        request=SessionDispatchRequest(
            session_id="s", task_id="t", instruction="work", remaining_timeout_seconds=60,
            runtime=RuntimeSpec(image="unused"), agent=AgentSpec(harness="claude_code"),
            evaluator=EvaluatorSpec(strategy="operator_judge", config={
                "op_name": "op", "judge_command": "bash tools/ascendc_eval_pipeline.sh --op_name op",
                "reward_scheme": "correctness_banded",
            }),
        ), timer=StageTimer(), session_dir=tmp_path, artifacts_dir=artifacts,
    )


def test_no_best_discards_without_judge_or_trajectory_build(tmp_path):
    managed = managed_session(tmp_path)
    managed.agent_result = AgentRunResult(status="failed", return_code=125,
        metadata={"termination_reason": "npu_execution_failed"})
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node"
    # No storage/build/evaluator configured: none may be accessed on this path.
    result = asyncio.run(manager._build_session_result(managed))
    assert result.status == "ERROR"
    assert result.metadata["discarded"] is True
    assert result.trajectory.traces == []
    assert "evaluation" not in result.trajectory.metadata


def test_gateway_stops_agent_while_its_npu_tool_is_still_running(tmp_path):
    managed = managed_session(tmp_path)
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    finished = []

    async def execute(*args, **kwargs):
        (managed.artifacts_dir / "npu_failure.json").write_text(json.dumps({
            "error_type": "npu_execution_failed", "error_code": "507035", "device_id": "1",
        }))
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    runtime = SimpleNamespace(exec=execute, cancel=AsyncMock())

    async def run():
        managed.execution_deadline = asyncio.get_running_loop().time() + 30
        return await asyncio.wait_for(manager._run_exec_inputs(
            runtime, [ExecInput(command="agent")], {}, managed), timeout=3)

    result = asyncio.run(run())
    assert result.status == "failed" and result.return_code == 125
    assert result.metadata["termination_reason"] == "npu_execution_failed"
    runtime.cancel.assert_awaited_once()
    assert finished == [True]


@pytest.mark.parametrize("judge_fails", [False, True])
def test_best_still_goes_through_original_judge(tmp_path, monkeypatch, judge_fails):
    managed = managed_session(tmp_path)
    managed.agent_result = AgentRunResult(status="failed", return_code=125,
        metadata={"termination_reason": "npu_execution_failed"})
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.node_id = "node"
    manager.session_registry = SimpleNamespace(set_status=lambda *args: None)
    manager._build_trajectory = lambda *args: Trajectory(status="COMPLETED", traces=[Trace()],
        metadata={"completed_pipeline_prefix": {"tool_call_id": "p1"}})
    monkeypatch.setattr("polar.gateway.node.selected_candidate", lambda *args: tmp_path / "best.tar.gz")

    async def judge(request, trajectory, **kwargs):
        if judge_fails:
            raise RuntimeError("judge failed")
        trajectory.traces[0].reward = .37
        return trajectory

    manager._run_eval = AsyncMock(side_effect=judge)

    async def run():
        managed.execution_deadline = asyncio.get_running_loop().time() + 30
        return await manager._build_session_result(managed)

    result = asyncio.run(run())
    manager._run_eval.assert_awaited_once()
    if judge_fails:
        assert result.status == "ERROR"
    else:
        assert result.status == "COMPLETED"
        assert result.trajectory.traces[0].reward == .37
