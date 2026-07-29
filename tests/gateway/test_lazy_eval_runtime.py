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
from polar.runtime.models import ExecInput, ExecResult, PrepareAction, RuntimeSpec
from polar.trajectory.models import EvaluatorSpec, Trajectory
from polar.trajectory.registry import default_evaluator_registry


OP = "add"
SUB = f"output/submission/{OP}_impl.py"
WORKDIR = "/work"
METRICS = "judge_out/metrics.json"
POOL = "8,9"
LOCK_DIR = "/dev/shm/polar-npu-locks"
PIPELINE_ENV = {
    "POLAR_NPU_LEASE_POOL": POOL,
    "POLAR_NPU_LOCK_DIR": LOCK_DIR,
}


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
        self.exec_calls: list[dict[str, object]] = []
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
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env or {}),
                "timeout_sec": timeout_sec,
            }
        )
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


def _request(
    *,
    lazy: bool,
    eval_runtime: RuntimeSpec | None = None,
    evaluator_env: dict[str, str] | None = None,
) -> SessionDispatchRequest:
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
            env=evaluator_env or {},
            runtime=eval_runtime or RuntimeSpec(image="sandbox:v1"),
        ),
    )


def _pipeline_lease_eval_runtime() -> RuntimeSpec:
    return RuntimeSpec(
        image="sandbox:v1",
        workdir=WORKDIR,
        env=dict(PIPELINE_ENV),
        eval_prepare=[
            PrepareAction(
                type="exec",
                command="prepare-judge",
                cwd=WORKDIR,
                env={"PREPARE_ENV": "1"},
            )
        ],
        kwargs={
            "ascend": {
                "pool": POOL,
                "lock_dir": LOCK_DIR,
                "lease_at_start": False,
            },
            "volumes": [
                "/opt/canonical:/opt/canonical:ro",
                "/readonly_tools:/work/tools:ro",
            ],
        },
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


async def _ready_runtime(runtime: BaseRuntime) -> BaseRuntime:
    return runtime


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


def test_eval_prewarm_uses_pipeline_lease_eval_runtime_spec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    eval_runtime = _pipeline_lease_eval_runtime()
    request = _request(lazy=False, eval_runtime=eval_runtime)
    judge = FakeRuntime("judge", events, tmp_path / "judge")
    captured: list[tuple[RuntimeSpec, str, Path]] = []

    def fake_create_runtime(
        runtime_spec: RuntimeSpec,
        session_id: str,
        session_dir: Path,
    ) -> BaseRuntime:
        captured.append((runtime_spec, session_id, session_dir))
        return judge

    monkeypatch.setattr("polar.gateway.node.create_runtime", fake_create_runtime)
    manager = _run_manager()
    managed = _managed(request, FakeRuntime("agent", events, tmp_path / "agent"), tmp_path)

    prepared = asyncio.run(manager._prepare_eval_runtime(managed))

    assert prepared is judge
    assert len(captured) == 1
    runtime_spec, session_id, session_dir = captured[0]
    assert session_id == "s-eval"
    assert session_dir == tmp_path / "eval_runtime"
    assert runtime_spec.kwargs["ascend"] == {
        "pool": POOL,
        "lock_dir": LOCK_DIR,
        "lease_at_start": False,
    }
    assert runtime_spec.env == PIPELINE_ENV
    assert events == ["judge.start", "judge.exec:prepare-judge"]
    assert judge.exec_calls == [
        {
            "command": "prepare-judge",
            "cwd": WORKDIR,
            "env": {"PREPARE_ENV": "1"},
            "timeout_sec": 60.0,
        }
    ]


def test_non_lazy_fresh_judge_receives_pipeline_lease_env(tmp_path: Path) -> None:
    events: list[str] = []
    request = _request(
        lazy=False,
        eval_runtime=_pipeline_lease_eval_runtime(),
    )
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
    managed = _managed(request, agent, tmp_path)
    trajectory = Trajectory(status="COMPLETED", traces=[])
    agent_result = AgentRunResult(status="completed", return_code=0)

    async def run_eval() -> Trajectory:
        managed.eval_prewarm_task = asyncio.create_task(_ready_runtime(judge))
        return await manager._run_eval(
            request,
            trajectory,
            agent_result=agent_result,
            managed=managed,
        )

    updated = asyncio.run(run_eval())

    assert updated.metadata["evaluation"]["outcome_reward"] == 0.9
    judge_calls = [
        call
        for call in judge.exec_calls
        if call["command"] == request.evaluator.config["judge_command"]
    ]
    assert len(judge_calls) == 1
    assert judge_calls[0]["cwd"] == WORKDIR
    assert judge_calls[0]["env"] == PIPELINE_ENV


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

    assert updated.metadata["evaluation"]["outcome_reward"] == 0.9
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


def test_cannbot_lazy_submission_candidates_prefer_phase5_final() -> None:
    manager = _run_manager()
    evaluator = EvaluatorSpec(
        strategy="operator_judge",
        refresh_runtime=True,
        config={
            "op_name": OP,
            "judge_mode": "cannbot",
            "workdir": WORKDIR,
        },
    )

    candidates = manager._operator_judge_submission_candidates(evaluator)

    assert candidates == [
        (f"{OP}_generated.py", f"{WORKDIR}/{OP}_generated.py"),
        ("output/optimized_code.py", f"{WORKDIR}/output/optimized_code.py"),
        ("output/generated_code.py", f"{WORKDIR}/output/generated_code.py"),
    ]


def test_lazy_fresh_judge_uses_pipeline_lease_spec_after_agent_stop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    request = _request(
        lazy=True,
        eval_runtime=_pipeline_lease_eval_runtime(),
    )
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
    captured: list[tuple[RuntimeSpec, list[str]]] = []

    def fake_create_runtime(
        runtime_spec: RuntimeSpec,
        _session_id: str,
        _session_dir: Path,
    ) -> BaseRuntime:
        captured.append((runtime_spec, list(events)))
        return judge

    manager = _run_manager()
    manager.evaluators = default_evaluator_registry()
    monkeypatch.setattr("polar.gateway.node.create_runtime", fake_create_runtime)
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

    assert updated.metadata["evaluation"]["outcome_reward"] == 0.9
    assert len(captured) == 1
    runtime_spec, events_at_create = captured[0]
    assert runtime_spec.kwargs["ascend"]["lease_at_start"] is False
    assert "agent.stop" in events_at_create
    assert "judge.start" not in events_at_create
    assert events.index("agent.stop") < events.index("judge.start")
    judge_calls = [
        call
        for call in judge.exec_calls
        if call["command"] == request.evaluator.config["judge_command"]
    ]
    assert len(judge_calls) == 1
    assert judge_calls[0]["env"] == PIPELINE_ENV


def test_evaluator_env_overrides_runtime_env_for_judge_command(tmp_path: Path) -> None:
    events: list[str] = []
    request = _request(
        lazy=False,
        eval_runtime=_pipeline_lease_eval_runtime(),
        evaluator_env={"POLAR_NPU_LEASE_POOL": "10", "EXTRA": "1"},
    )
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
    managed = _managed(request, agent, tmp_path)
    trajectory = Trajectory(status="COMPLETED", traces=[])
    agent_result = AgentRunResult(status="completed", return_code=0)

    async def run_eval() -> Trajectory:
        managed.eval_prewarm_task = asyncio.create_task(_ready_runtime(judge))
        return await manager._run_eval(
            request,
            trajectory,
            agent_result=agent_result,
            managed=managed,
        )

    updated = asyncio.run(run_eval())

    assert updated.metadata["evaluation"]["outcome_reward"] == 0.9
    judge_calls = [
        call
        for call in judge.exec_calls
        if call["command"] == request.evaluator.config["judge_command"]
    ]
    assert len(judge_calls) == 1
    assert judge_calls[0]["env"] == {
        **PIPELINE_ENV,
        "POLAR_NPU_LEASE_POOL": "10",
        "EXTRA": "1",
    }
