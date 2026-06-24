from __future__ import annotations

import asyncio
from enum import Enum
from types import SimpleNamespace

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming, TaskResult
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge import rollout as rollout_module
from slime_bridge.config import PolarSlimeConfig
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    _CompletedGroup,
    PolarLowCompleteAcceptFractionError,
    PolarRolloutSchedulerError,
    _build_session_unit_payload,
    _completed_group_from_session_accumulator,
    _flatten_session_pool_units,
    finish_policy_update,
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
        "polar_rollout_url": "http://rollout:8080",
        "polar_task_id_template": "task-{rollout_id}-{sample.group_index}",
        "polar_task_template": {
            "agent": {"harness": "codex", "model_name": "qwen"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
        "polar_scheduler_mode": "session_pool",
        "polar_max_active_sessions": 16,
        "polar_session_pool_pause_policy": "drain_open_groups",
        "polar_max_async_level": 1,
        "rollout_batch_size": 4,
        "n_samples_per_prompt": 8,
        "update_weights_interval": 1,
        "polar_callback_host": "127.0.0.1",
        "polar_min_complete_accept_fraction": 0.0,
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


class FakeDataSource:
    def __init__(self, groups: list[list[SimpleNamespace]]) -> None:
        self.groups = list(groups)

    def get_samples(self, num_samples: int) -> list[list[SimpleNamespace]]:
        del num_samples
        if not self.groups:
            return []
        return [self.groups.pop(0)]


class _NoopCallbackServer:
    def __init__(self) -> None:
        self.should_exit = False


class ControlledSessionPoolWorker(AsyncPolarRolloutWorker):
    def __init__(self, args: SimpleNamespace, data_source: FakeDataSource) -> None:
        super().__init__(args, data_source)
        self.submitted_units = []
        self.inflight: dict[str, asyncio.Future[TaskResult]] = {}
        self.max_seen_active = 0
        self.stop_after_groups = 4

    async def _start_callback_listener(self):  # noqa: ANN202
        server = _NoopCallbackServer()

        async def idle() -> None:
            while not server.should_exit:
                await asyncio.sleep(0.01)

        return server, asyncio.create_task(idle())

    async def _submit_session_unit(self, client, unit):  # noqa: ANN001, ANN202
        del client
        self.submitted_units.append(unit)
        self.max_seen_active = max(self.max_seen_active, len(self.inflight) + 1)
        future: asyncio.Future[TaskResult] = asyncio.get_running_loop().create_future()
        self.inflight[unit.task_id] = future
        return await future

    async def _emit_completed(self, completed):  # noqa: ANN001, ANN202
        await super()._emit_completed(completed)
        if self.output_queue.qsize() >= self.stop_after_groups:
            self._running = False

    def complete_oldest(self, count: int = 1) -> None:
        for _ in range(count):
            if not self.inflight:
                return
            task_id, future = next(iter(self.inflight.items()))
            unit = next(unit for unit in self.submitted_units if unit.task_id == task_id)
            self.inflight.pop(task_id)
            future.set_result(_task_result(unit))

    def complete_task(self, task_id: str, *, status: str = "completed") -> None:
        future = self.inflight.pop(task_id)
        unit = next(unit for unit in self.submitted_units if unit.task_id == task_id)
        future.set_result(_task_result(unit, status=status))

    def complete_all(self) -> None:
        while self.inflight:
            self.complete_oldest()


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:  # noqa: ANN001
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


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


def test_session_pool_worker_caps_active_sessions_and_admits_group_contiguous(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)

    async def run() -> None:
        worker = ControlledSessionPoolWorker(
            _args(polar_max_active_sessions=16, rollout_batch_size=4, n_samples_per_prompt=8),
            FakeDataSource(_groups(4, 8)),
        )
        loop_task = asyncio.create_task(worker._async_session_pool_loop())
        await _wait_until(lambda: len(worker.submitted_units) == 16)

        assert worker.max_seen_active == 16
        assert [(unit.group_pos, unit.sample_pos) for unit in worker.submitted_units] == [
            *((0, idx) for idx in range(8)),
            *((1, idx) for idx in range(8)),
        ]
        assert worker.snapshot_metrics()["polar/session_pool/partial_open_groups"] == 0.0

        worker.complete_task(worker.submitted_units[0].task_id)
        await _wait_until(lambda: len(worker.submitted_units) == 17)

        assert (worker.submitted_units[-1].group_pos, worker.submitted_units[-1].sample_pos) == (2, 0)
        assert worker.snapshot_metrics()["polar/session_pool/partial_open_groups"] == 1.0
        assert worker.max_seen_active == 16

        worker.complete_all()
        worker.stop_after_groups = 0
        worker._running = False
        await asyncio.wait_for(loop_task, timeout=2.0)

    asyncio.run(run())


def test_session_pool_worker_emits_completed_groups_in_original_group_order(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)

    async def run() -> None:
        worker = ControlledSessionPoolWorker(
            _args(polar_max_active_sessions=8, rollout_batch_size=2, n_samples_per_prompt=4),
            FakeDataSource(_groups(2, 4)),
        )
        loop_task = asyncio.create_task(worker._async_session_pool_loop())
        await _wait_until(lambda: len(worker.submitted_units) == 8)

        for unit in [unit for unit in worker.submitted_units if unit.group_pos == 1]:
            worker.complete_task(unit.task_id)
        await asyncio.sleep(0.05)
        assert worker.output_queue.qsize() == 0

        for unit in [unit for unit in worker.submitted_units if unit.group_pos == 0]:
            worker.complete_task(unit.task_id)
        await _wait_until(lambda: worker.output_queue.qsize() == 2)

        first = worker.output_queue.get_nowait()
        second = worker.output_queue.get_nowait()
        assert [first.group_id, second.group_id] == [0, 1]

        worker._running = False
        await asyncio.wait_for(loop_task, timeout=2.0)

    asyncio.run(run())


def test_session_pool_worker_hard_unit_failure_drops_group_without_hanging(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)

    async def run() -> None:
        worker = ControlledSessionPoolWorker(
            _args(polar_max_active_sessions=4, rollout_batch_size=1, n_samples_per_prompt=4),
            FakeDataSource(_groups(1, 4)),
        )
        loop_task = asyncio.create_task(worker._async_session_pool_loop())
        await _wait_until(lambda: len(worker.submitted_units) == 4)

        failed_task_id = worker.submitted_units[0].task_id
        worker.complete_task(failed_task_id, status="failed")
        for unit in worker.submitted_units[1:]:
            worker.complete_task(unit.task_id)
        await _wait_until(
            lambda: worker.snapshot_metrics().get("polar/session_pool/dropped_groups", 0.0) == 1.0
        )

        assert worker.output_queue.qsize() == 0
        assert worker.snapshot_metrics()["polar/dropped_groups"] == 1.0
        assert worker.snapshot_metrics()["polar/dropped_sessions"] == 4.0

        worker._running = False
        await asyncio.wait_for(loop_task, timeout=2.0)

    asyncio.run(run())


def test_session_pool_prepare_policy_update_drains_open_groups_without_opening_new(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)

    async def run() -> None:
        worker = ControlledSessionPoolWorker(
            _args(polar_max_active_sessions=2, rollout_batch_size=2, n_samples_per_prompt=4),
            FakeDataSource(_groups(2, 4)),
        )
        loop_task = asyncio.create_task(worker._async_session_pool_loop())
        await _wait_until(lambda: len(worker.submitted_units) == 2)

        worker.begin_policy_update_drain(policy_version=9)
        worker.complete_oldest(2)
        await _wait_until(lambda: len(worker.submitted_units) == 4)

        assert [(unit.group_pos, unit.sample_pos) for unit in worker.submitted_units] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
        ]

        worker.complete_all()
        await asyncio.to_thread(worker.wait_for_policy_update_drain, timeout=2.0)
        assert worker.output_queue.qsize() == 1
        assert not worker.submitted_units[0].policy_version == 9

        worker.update_policy_version(9)
        worker.finish_policy_update_drain()
        await _wait_until(lambda: len(worker.submitted_units) == 6)

        assert [(unit.group_pos, unit.sample_pos) for unit in worker.submitted_units[-2:]] == [(1, 0), (1, 1)]
        assert {unit.policy_version for unit in worker.submitted_units[-2:]} == {9}

        worker.complete_all()
        worker._running = False
        await asyncio.wait_for(loop_task, timeout=2.0)

    asyncio.run(run())


def test_finish_policy_update_updates_local_worker_version_before_resume(monkeypatch) -> None:
    worker = ControlledSessionPoolWorker(
        _args(polar_max_active_sessions=2, rollout_batch_size=1, n_samples_per_prompt=2),
        FakeDataSource([]),
    )
    worker.begin_policy_update_drain(policy_version=9)
    monkeypatch.setattr(rollout_module, "_global_async_worker", worker)

    def fake_resume(args) -> None:  # noqa: ANN001
        del args
        assert worker.snapshot_metrics()["polar/scheduler/policy_version"] == 9.0

    monkeypatch.setattr(rollout_module, "_resume_gateway_generation", fake_resume)

    finish_policy_update(_args(), 9)

    assert worker.snapshot_metrics()["polar/scheduler/policy_version"] == 9.0
    assert worker._session_pool_draining() is False


def test_session_pool_staleness_filtering_happens_in_drain_completed() -> None:
    worker = ControlledSessionPoolWorker(
        _args(rollout_batch_size=1, n_samples_per_prompt=1, update_weights_interval=1),
        FakeDataSource([]),
    )
    stale = _CompletedGroup(
        group_id=1,
        group=[],
        samples=[],
        task_id="task-1",
        submitted_rollout_id=1,
        policy_version=0,
        session_count=1,
    )
    fresh = _CompletedGroup(
        group_id=2,
        group=[],
        samples=[SimpleNamespace(metadata={"polar": {}}, train_metadata={})],
        task_id="task-2",
        submitted_rollout_id=3,
        policy_version=3,
        session_count=1,
    )
    worker.output_queue.put(stale)
    worker.output_queue.put(fresh)

    accepted = worker.drain_completed(max_groups=2, rollout_id=3)

    assert [group.group_id for group in accepted] == [2]
    assert worker.snapshot_metrics()["polar/stale_groups"] == 1.0
    assert worker.snapshot_metrics()["polar/dropped_stale_groups"] == 1.0
