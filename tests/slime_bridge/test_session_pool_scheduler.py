from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming, TaskResult
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge.config import PolarSlimeConfig
from slime_bridge.rollout import (
    PolarLowCompleteAcceptFractionError,
    PolarRolloutSchedulerError,
    _build_session_unit_payload,
    _completed_group_from_session_accumulator,
    _flatten_session_pool_units,
    _new_session_group_accumulator,
    _next_session_pool_unit,
    _record_session_unit_result,
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


def _args(**overrides) -> SimpleNamespace:
    base = {
        "polar_task_id_template": "task-{rollout_id}-{sample.group_index}",
        "polar_task_template": {
            "agent": {"harness": "codex", "model_name": "qwen"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(*, threshold: float = 0.0) -> PolarSlimeConfig:
    return PolarSlimeConfig(
        rollout_server_url="http://rollout:8080",
        task_template={
            "agent": {"harness": "codex", "model_name": "qwen"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
        task_id_template="task-{rollout_id}-{sample.group_index}",
        instruction_template=None,
        reward_key="score",
        max_concurrency=4,
        max_session_concurrency=32,
        max_async_level=1,
        max_sessions_per_task=None,
        max_off_policy_steps=2,
        request_timeout=None,
        callback_host="127.0.0.1",
        scoring_mode="group",
        min_complete_accept_fraction=threshold,
        tokenizer_name_or_path=None,
        add_generation_prompt=True,
        eval_dataset_name="eval",
        scheduler_mode="session_pool",
        max_active_sessions=16,
        session_pool_pause_policy="drain_open_groups",
    )


def _groups(group_count: int = 4, group_size: int = 8) -> list[list[SimpleNamespace]]:
    return [
        [
            SimpleNamespace(
                prompt=f"prompt {group_pos}-{sample_pos}",
                group_index=100 + group_pos,
                index=1000 + group_pos * group_size + sample_pos,
                metadata={"instance_id": f"inst-{group_pos}-{sample_pos}"},
            )
            for sample_pos in range(group_size)
        ]
        for group_pos in range(group_count)
    ]


def _session_result(
    task_id: str,
    sample_pos: int,
    *,
    status: SessionStatus = SessionStatus.COMPLETED,
    trainable: bool = True,
) -> SessionResult:
    if trainable:
        trace = Trace(
            prompt_ids=[10 + sample_pos],
            response_ids=[20 + sample_pos],
            loss_mask=[1],
            prompt_messages=[{"role": "user", "content": f"prompt {sample_pos}"}],
            response_messages=[{"role": "assistant", "content": f"answer {sample_pos}"}],
            response_logprobs=[-0.1],
            reward=float(sample_pos),
        )
    else:
        trace = Trace(prompt_ids=[], response_ids=[], loss_mask=[])
    return SessionResult(
        session_id=f"session-{sample_pos}",
        task_id=task_id,
        status=status,
        node_id="node-a",
        timing=SessionTiming(init_ms=1.0, run_ms=2.0, postrun_ms=3.0),
        metadata={
            "session_pool": True,
            "parent_task_id": "task-10-100",
            "sample_pos": sample_pos,
            "group_size": 8,
            "policy_version": 3,
            "rollout_step": 7,
        },
        trajectory=Trajectory(
            status="COMPLETED" if status == SessionStatus.COMPLETED else status.value,
            traces=[trace],
        ),
    )


def _task_result(unit, *, status: str = "completed", session_status: SessionStatus = SessionStatus.COMPLETED) -> TaskResult:
    return TaskResult(
        task_id=unit.task_id,
        status=status,
        results=[_session_result(unit.task_id, unit.sample_pos, status=session_status)],
        result_paths=[f"/tmp/{unit.task_id}.json"],
    )


def test_session_pool_flattens_4_by_8_into_32_units_with_unique_task_ids() -> None:
    units = _flatten_session_pool_units(
        args=_args(),
        config=_config(),
        groups=_groups(4, 8),
        first_group_id=10,
        submitted_rollout_id=7,
        policy_version=3,
    )

    assert len(units) == 32
    assert [(unit.group_pos, unit.sample_pos) for unit in units[:10]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 0),
        (1, 1),
    ]
    assert len({unit.task_id for unit in units}) == 32
    assert units[0].task_id == "task-10-100--g000010-sp000"
    assert units[8].task_id == "task-11-101--g000011-sp000"


def test_session_pool_unit_payload_is_one_polar_session_with_scheduler_metadata() -> None:
    unit = _flatten_session_pool_units(
        args=_args(),
        config=_config(),
        groups=_groups(1, 8),
        first_group_id=10,
        submitted_rollout_id=7,
        policy_version=3,
    )[5]

    payload = _build_session_unit_payload(args=_args(), config=_config(), unit=unit)

    assert payload["task_id"] == "task-10-100--g000010-sp005"
    assert payload["num_samples"] == 1
    assert payload["metadata"]["group_id"] == 10
    assert payload["metadata"]["policy_version"] == 3
    assert payload["metadata"]["rollout_step"] == 7
    assert payload["metadata"]["session_pool"] is True
    assert payload["metadata"]["parent_task_id"] == "task-10-100"
    assert payload["metadata"]["sample_pos"] == 5
    assert payload["metadata"]["group_size"] == 8


def test_group_contiguous_helper_submits_one_group_before_opening_next() -> None:
    group = _groups(1, 8)[0]
    accumulator = _new_session_group_accumulator(
        args=_args(),
        config=_config(),
        group_id=10,
        group_pos=0,
        group=group,
        submitted_rollout_id=7,
        policy_version=3,
    )

    units = [_next_session_pool_unit(accumulator) for _ in range(8)]

    assert [unit.sample_pos for unit in units if unit is not None] == list(range(8))
    assert accumulator.fully_submitted is True
    assert _next_session_pool_unit(accumulator) is None


def test_out_of_order_completion_reconstructs_original_group_order(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    group = _groups(1, 8)[0]
    accumulator = _new_session_group_accumulator(
        args=_args(),
        config=_config(),
        group_id=10,
        group_pos=0,
        group=group,
        submitted_rollout_id=7,
        policy_version=3,
    )
    units = [_next_session_pool_unit(accumulator) for _ in range(8)]
    for unit in reversed(units):
        _record_session_unit_result(
            config=_config(),
            accumulator=accumulator,
            unit=unit,
            task_result=_task_result(unit),
        )

    completed = _completed_group_from_session_accumulator(_config(), accumulator)

    assert completed.task_id == "task-10-100"
    assert completed.session_count == 8
    assert [sample.index for sample in completed.samples] == [sample.index for sample in group]
    assert [sample.group_index for sample in completed.samples] == [100] * 8
    assert [sample.metadata["polar"]["sample_pos"] for sample in completed.samples] == list(range(8))
    assert all(sample.metadata["polar"]["session_pool"] is True for sample in completed.samples)


def test_timeout_session_stays_in_original_slot_without_replacement(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    group = _groups(1, 4)[0]
    accumulator = _new_session_group_accumulator(
        args=_args(),
        config=_config(),
        group_id=10,
        group_pos=0,
        group=group,
        submitted_rollout_id=7,
        policy_version=3,
    )
    units = [_next_session_pool_unit(accumulator) for _ in range(4)]

    for unit in units:
        _record_session_unit_result(
            config=_config(),
            accumulator=accumulator,
            unit=unit,
            task_result=_task_result(
                unit,
                session_status=SessionStatus.TIMEOUT if unit.sample_pos == 2 else SessionStatus.COMPLETED,
            ),
        )

    completed = _completed_group_from_session_accumulator(_config(), accumulator)

    assert completed.session_count == 4
    assert [sample.index for sample in completed.samples] == [sample.index for sample in group]
    assert completed.samples[2].status == FakeSample.Status.ABORTED
    assert completed.samples[2].metadata["polar"]["session_status"] == SessionStatus.TIMEOUT


def test_hard_unit_failure_raises_instead_of_leaving_accumulator_hanging(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    group = _groups(1, 4)[0]
    accumulator = _new_session_group_accumulator(
        args=_args(),
        config=_config(),
        group_id=10,
        group_pos=0,
        group=group,
        submitted_rollout_id=7,
        policy_version=3,
    )
    unit = _next_session_pool_unit(accumulator)

    with pytest.raises(PolarRolloutSchedulerError, match="task status=failed"):
        _record_session_unit_result(
            config=_config(),
            accumulator=accumulator,
            unit=unit,
            task_result=_task_result(unit, status="failed"),
        )

    assert accumulator.completed_count == 0


def test_low_complete_accept_fraction_uses_original_session_count(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    group = _groups(1, 4)[0]
    accumulator = _new_session_group_accumulator(
        args=_args(),
        config=_config(threshold=0.75),
        group_id=10,
        group_pos=0,
        group=group,
        submitted_rollout_id=7,
        policy_version=3,
    )
    units = [_next_session_pool_unit(accumulator) for _ in range(4)]
    for unit in units:
        _record_session_unit_result(
            config=_config(threshold=0.75),
            accumulator=accumulator,
            unit=unit,
            task_result=TaskResult(
                task_id=unit.task_id,
                status="completed",
                results=[
                    _session_result(
                        unit.task_id,
                        unit.sample_pos,
                        status=SessionStatus.COMPLETED if unit.sample_pos < 2 else SessionStatus.TIMEOUT,
                        trainable=unit.sample_pos < 2,
                    )
                ],
            ),
        )

    with pytest.raises(PolarLowCompleteAcceptFractionError, match="2/4"):
        _completed_group_from_session_accumulator(_config(threshold=0.75), accumulator)
