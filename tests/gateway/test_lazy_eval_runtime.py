from __future__ import annotations

import asyncio
import json
from pathlib import Path

from polar.agent.models import AgentRunResult, AgentSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecInput, ExecResult, RuntimeSpec
from polar.trajectory.models import EvaluatorSpec, Trajectory
from polar.trajectory.registry import default_evaluator_registry


OP = "add"
SUB = f"output/submission/{OP}_impl.py"
WORKDIR = "/work"
METRICS = "judge_out/metrics.json"


class FakeRuntime(BaseRuntime):
    def __init__(
        self,
        name: str,
        events: list[str],
        session_dir: Path,
        *,
        files: dict[str, str] | None = None,
    ) -> None:
        super().__init__(RuntimeSpec(image="sandbox:v1"), name, session_dir)
        self.name = name
        self.events = events
        self.files = files or {}
        self.uploads: list[tuple[str, str]] = []
        self.stop_count = 0

    @property
    def runtime_id(self) -> str:
        return self.name

    async def start(self) -> None:
        self.events.append(f"{self.name}.start")

    async def stop(self) -> None:
        self.stop_count += 1
        self.events.append(f"{self.name}.stop")

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        self.events.append(f"{self.name}.exec:{command}")
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        self.events.append(f"{self.name}.upload:{remote_path}")
        self.uploads.append((local_path, remote_path))
        self.files[remote_path] = Path(local_path).read_text()

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        self.events.append(f"{self.name}.upload_dir:{remote_path}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        self.events.append(f"{self.name}.download:{remote_path}")
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_text(self.files[remote_path])

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        self.events.append(f"{self.name}.download_dir:{remote_path}")


class FakeHarness:
    async def setup(self, runtime: BaseRuntime) -> None:
        return None

    def run_steps(self, instruction: str) -> list[ExecInput]:
        return [ExecInput(command="run-agent")]

    async def postprocess(
        self, runtime: BaseRuntime, result: AgentRunResult
    ) -> None:
        return None

    def postrun_steps(self) -> list[ExecInput]:
        return [ExecInput(command="postrun-cleanup")]


def _request(*, lazy: bool) -> SessionDispatchRequest:
    config = {
        "op_name": OP,
        "judge_command": "bash pipeline.sh",
        "metrics_path": METRICS,
        "workdir": WORKDIR,
    }
    if lazy:
        config["lazy_refresh_runtime"] = True
    return SessionDispatchRequest(
        session_id="s",
        task_id="t",
        instruction="do work",
        remaining_timeout_seconds=60.0,
        runtime=RuntimeSpec(image="sandbox:v1"),
        agent=AgentSpec(harness="codex"),
        evaluator=EvaluatorSpec(
            strategy="operator_judge",
            refresh_runtime=True,
            config=config,
            runtime=RuntimeSpec(image="sandbox:v1"),
        ),
    )


def _managed(
    request: SessionDispatchRequest,
    runtime: BaseRuntime,
    tmp_path: Path,
) -> ManagedSession:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=artifacts_dir,
        runtime=runtime,
        execution_deadline=9999999999.0,
    )


def _run_manager() -> GatewayNodeManager:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager._await_with_budget = _await_direct
    manager._runtime_env = lambda *args, **kwargs: {}
    manager._remaining_budget = lambda managed: 60.0
    manager._resolve_agent_harness = lambda request: FakeHarness()
    manager._run_exec_inputs = _run_agent_success
    return manager


async def _await_direct(awaitable, managed):
    return await awaitable


async def _run_agent_success(runtime, steps, env, managed):
    return AgentRunResult(status="completed", return_code=0)


def test_lazy_refresh_runtime_skips_run_stage_eval_prewarm(tmp_path: Path) -> None:
    events: list[str] = []
    manager = _run_manager()
    prewarm_calls: list[str] = []
    manager._start_eval_prewarm = lambda managed: prewarm_calls.append("prewarm")
    managed = _managed(_request(lazy=True), FakeRuntime("agent", events, tmp_path), tmp_path)

    asyncio.run(manager._handle_run(managed))

    assert prewarm_calls == []
    assert managed.agent_result is not None
    assert managed.postrun_steps == [ExecInput(command="postrun-cleanup")]


def test_non_lazy_refresh_runtime_still_prewarms_during_run(tmp_path: Path) -> None:
    events: list[str] = []
    manager = _run_manager()
    prewarm_calls: list[str] = []
    manager._start_eval_prewarm = lambda managed: prewarm_calls.append("prewarm")
    managed = _managed(_request(lazy=False), FakeRuntime("agent", events, tmp_path), tmp_path)

    asyncio.run(manager._handle_run(managed))

    assert prewarm_calls == ["prewarm"]
    assert managed.agent_result is not None


def test_lazy_eval_stops_agent_before_starting_judge_and_uploads_submission(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    request = _request(lazy=True)
    agent = FakeRuntime(
        "agent",
        events,
        tmp_path / "agent",
        files={f"{WORKDIR}/{SUB}": "# final kernel"},
    )
    judge = FakeRuntime(
        "judge",
        events,
        tmp_path / "judge",
        files={
            f"{WORKDIR}/{METRICS}": json.dumps(
                {"success": True, "perf_data": {"speedup_vs_torch": 2.0}}
            )
        },
    )
    manager = _run_manager()
    manager.evaluators = default_evaluator_registry()
    monkeypatch.setattr("polar.gateway.node.create_runtime", lambda *args: judge)
    managed = _managed(request, agent, tmp_path)
    managed.postrun_steps = [ExecInput(command="postrun-cleanup")]
    trajectory = Trajectory(status="COMPLETED", traces=[])
    agent_result = AgentRunResult(status="completed", return_code=0)

    updated = asyncio.run(
        manager._run_lazy_eval(
            request,
            trajectory,
            agent_result=agent_result,
            managed=managed,
            eval_runtime_spec=request.evaluator.runtime,
        )
    )

    assert updated.metadata["evaluation"]["outcome_reward"] == 1.0
    assert events.index("agent.stop") < events.index("judge.start")
    assert f"judge.upload:{WORKDIR}/{SUB}" in events
    assert f"judge.exec:{request.evaluator.config['judge_command']}" in events
    assert "judge.stop" in events
    assert agent.stop_count == 1
    assert judge.stop_count == 1
    assert managed.runtime is None
    assert managed.postrun_steps == []
    local_upload, remote_upload = judge.uploads[0]
    assert remote_upload == f"{WORKDIR}/{SUB}"
    assert Path(local_upload).read_text() == "# final kernel"
