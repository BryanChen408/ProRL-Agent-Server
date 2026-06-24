from __future__ import annotations

import asyncio
from enum import Enum
from types import SimpleNamespace

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming, TaskResult
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge.config import PolarSlimeConfig
from slime_bridge.rollout import (
    _chunk_task_payloads,
    _convert_task_result_to_samples,
    _merge_task_results,
    _submit_payload_in_chunks,
)


class FakeSample:
    class Status(str, Enum):
        COMPLETED = "completed"
        ABORTED = "aborted"
        FAILED = "failed"
        TRUNCATED = "truncated"

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _config(max_sessions_per_task: int | None = 2) -> PolarSlimeConfig:
    return PolarSlimeConfig(
        rollout_server_url="http://rollout:8080",
        task_template={"agent": {"harness": "claude_code", "model_name": "qwen"}},
        task_id_template="task-{rollout_id}",
        instruction_template=None,
        reward_key="score",
        max_concurrency=1,
        max_session_concurrency=4,
        max_async_level=1,
        max_sessions_per_task=max_sessions_per_task,
        max_off_policy_steps=2,
        request_timeout=None,
        callback_host="127.0.0.1",
        scoring_mode="group",
        min_complete_accept_fraction=0.0,
        tokenizer_name_or_path=None,
        add_generation_prompt=True,
        eval_dataset_name="eval",
        scheduler_mode="group",
        max_active_sessions=4,
        session_pool_pause_policy="drain_open_groups",
    )


def _session_result(task_id: str, index: int) -> SessionResult:
    trace = Trace(
        prompt_ids=[10 + index],
        response_ids=[20 + index],
        loss_mask=[1],
        prompt_messages=[{"role": "user", "content": f"prompt {index}"}],
        response_messages=[{"role": "assistant", "content": f"answer {index}"}],
        response_logprobs=[-0.1 - index],
        reward=float(index),
    )
    return SessionResult(
        session_id=f"session-{index}",
        task_id=task_id,
        status=SessionStatus.COMPLETED,
        node_id="node-a",
        timing=SessionTiming(init_ms=1.0, run_ms=2.0, postrun_ms=3.0),
        trajectory=Trajectory(status="COMPLETED", traces=[trace]),
    )


def test_chunk_task_payloads_splits_session_fanout_and_preserves_parent_metadata() -> None:
    payload = {
        "task_id": "task-0",
        "instruction": "generate",
        "num_samples": 4,
        "metadata": {"group_id": 7},
    }

    chunks = _chunk_task_payloads(payload, max_sessions_per_task=2)

    assert [chunk["task_id"] for chunk in chunks] == ["task-0--part000", "task-0--part001"]
    assert [chunk["num_samples"] for chunk in chunks] == [2, 2]
    assert [chunk["metadata"]["chunk_start"] for chunk in chunks] == [0, 2]
    assert all(chunk["metadata"]["parent_task_id"] == "task-0" for chunk in chunks)
    assert all(chunk["metadata"]["group_id"] == 7 for chunk in chunks)


def test_submit_payload_in_chunks_runs_child_tasks_sequentially_and_merges_results() -> None:
    async def run() -> None:
        payload = {"task_id": "task-0", "num_samples": 4, "metadata": {}}
        submitted: list[tuple[str, int]] = []

        async def submit_one(chunk: dict) -> TaskResult:
            task_id = str(chunk["task_id"])
            start = int(chunk["metadata"]["chunk_start"])
            size = int(chunk["num_samples"])
            submitted.append((task_id, size))
            return TaskResult(
                task_id=task_id,
                status="completed",
                results=[_session_result(task_id, start + offset) for offset in range(size)],
                result_paths=[f"/tmp/{task_id}.json"],
            )

        merged = await _submit_payload_in_chunks(
            payload,
            max_sessions_per_task=2,
            submit_one=submit_one,
        )

        assert submitted == [("task-0--part000", 2), ("task-0--part001", 2)]
        assert merged.task_id == "task-0"
        assert merged.status == "completed"
        assert [result.session_id for result in merged.results] == [
            "session-0",
            "session-1",
            "session-2",
            "session-3",
        ]

    asyncio.run(run())


def test_merged_chunk_results_convert_to_one_ordered_slime_group(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    child_results = [
        TaskResult(
            task_id="task-0--part000",
            status="completed",
            results=[_session_result("task-0--part000", 0), _session_result("task-0--part000", 1)],
        ),
        TaskResult(
            task_id="task-0--part001",
            status="completed",
            results=[_session_result("task-0--part001", 2), _session_result("task-0--part001", 3)],
        ),
    ]
    merged = _merge_task_results("task-0", child_results)
    group = [
        SimpleNamespace(index=100 + idx, group_index=5)
        for idx in range(4)
    ]

    samples = _convert_task_result_to_samples(_config(), merged, group)

    assert [sample.index for sample in samples] == [100, 101, 102, 103]
    assert [sample.group_index for sample in samples] == [5, 5, 5, 5]
    assert [sample.metadata["polar"]["session_id"] for sample in samples] == [
        "session-0",
        "session-1",
        "session-2",
        "session-3",
    ]
    assert [sample.reward["score"] for sample in samples] == [0.0, 1.0, 2.0, 3.0]
