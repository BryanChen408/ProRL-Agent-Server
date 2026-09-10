from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import SessionContext, SessionDispatchRequest, SessionResult, TaskRequest
from polar.rollout.pipeline import Pipeline
from polar.trajectory.models import EvaluatorSpec, Trajectory


def test_rollout_dispatch_and_wait_include_independent_postrun_budget(tmp_path):
    task = TaskRequest(
        task_id="t", instruction="solve", timeout_seconds=0.02,
        agent=AgentSpec(harness="codex"),
        evaluator=EvaluatorSpec(strategy="operator_judge", postrun_timeout_seconds=1),
    )
    session = SessionContext(session_id="s", task_id="t", request=task, deadline_monotonic=time.monotonic() + 1)
    pipeline = Pipeline(
        callback_url="http://rollout/callback", save_dir=str(tmp_path),
        scheduler=SimpleNamespace(acquire_node=lambda: SimpleNamespace(node_id="n", gateway_url="http://gateway")),
        callback_grace_seconds=0, dispatch_poll_interval_seconds=0.01,
    )
    result = SessionResult(session_id="s", task_id="t", status="COMPLETED", trajectory=Trajectory(status="COMPLETED"))

    def dispatch(request):
        parsed = SessionDispatchRequest.model_validate_json(request.content)
        assert parsed.remaining_timeout_seconds == 0.02
        assert parsed.evaluator.postrun_timeout_seconds == 1
        return httpx.Response(200, json={"session_id": "s"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as client:
            pipeline._client = client
            dispatched = await pipeline._dispatch_session(session)
            started = time.monotonic()

            async def poll(*args, **kwargs):
                if time.monotonic() - started > 0.08:
                    return "COMPLETED", result
                return "EVALUATING", None

            pipeline._poll_session_state = poll
            return await pipeline._wait_for_result(
                session, dispatched, asyncio.get_running_loop().create_future(), lambda *args: None,
            )

    assert asyncio.run(run()) == result


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_postrun_budget_must_be_finite_and_positive(budget):
    with pytest.raises(ValueError):
        EvaluatorSpec(strategy="operator_judge", postrun_timeout_seconds=budget)


def test_pipeline_result_paths_are_scoped_by_training_run(tmp_path) -> None:
    pipeline = Pipeline(
        callback_url="http://127.0.0.1:8080/callbacks/session_result",
        save_dir=str(tmp_path),
        scheduler=NodeScheduler(),
    )

    path = pipeline.result_path_for(
        "train-a-polar-op-0-0",
        "sk-session",
        {"run_id": "train-a"},
    )

    assert path == str(
        tmp_path
        / "run_train-a"
        / "task_train-a-polar-op-0-0"
        / "ses_sk-session.json"
    )


def test_pipeline_legacy_result_paths_stay_flat(tmp_path) -> None:
    pipeline = Pipeline(
        callback_url="http://127.0.0.1:8080/callbacks/session_result",
        save_dir=str(tmp_path),
        scheduler=NodeScheduler(),
    )

    path = pipeline.result_path_for("polar-op-0-0", "sk-session")

    assert path == str(tmp_path / "task_polar-op-0-0" / "ses_sk-session.json")
