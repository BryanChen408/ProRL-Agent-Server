from __future__ import annotations

from polar.agent.models import AgentSpec
from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import EvaluatorSpec


def _runtime(pool: str) -> RuntimeSpec:
    return RuntimeSpec(
        backend="docker",
        image="sandbox:v1",
        kwargs={"ascend": {"pool": pool, "lock_dir": "/locks"}},
    )


def _manager(default_runtime: RuntimeSpec | None = None) -> GatewayNodeManager:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.default_runtime = default_runtime
    return manager


def _request(
    *,
    runtime: RuntimeSpec | None,
    evaluator_runtime: RuntimeSpec | None,
) -> SessionDispatchRequest:
    return SessionDispatchRequest(
        session_id="s",
        task_id="t",
        instruction="do work",
        remaining_timeout_seconds=60.0,
        runtime=runtime,
        agent=AgentSpec(harness="codex"),
        evaluator=EvaluatorSpec(
            strategy="operator_judge",
            refresh_runtime=True,
            runtime=evaluator_runtime,
        ),
    )


def test_eval_runtime_override_uses_evaluator_runtime_pool() -> None:
    agent_runtime = _runtime("8,9")
    judge_runtime = _runtime("10,11")
    request = _request(runtime=agent_runtime, evaluator_runtime=judge_runtime)
    manager = _manager()

    assert manager._resolve_runtime_spec(request) is agent_runtime
    assert manager._resolve_eval_runtime_spec(request) is judge_runtime
    assert manager._resolve_eval_runtime_spec(request).kwargs["ascend"]["pool"] == "10,11"


def test_eval_runtime_override_falls_back_to_agent_runtime() -> None:
    agent_runtime = _runtime("8,9")
    request = _request(runtime=agent_runtime, evaluator_runtime=None)
    manager = _manager()

    assert manager._resolve_eval_runtime_spec(request) is agent_runtime


def test_eval_runtime_override_falls_back_to_default_runtime() -> None:
    default_runtime = _runtime("10,11")
    request = _request(runtime=None, evaluator_runtime=None)
    manager = _manager(default_runtime)

    assert manager._resolve_eval_runtime_spec(request) is default_runtime
