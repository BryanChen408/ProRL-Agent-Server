import asyncio
import threading
from types import SimpleNamespace

import pytest

from polar.rollout import server
from polar.rollout.manager import RolloutManager, _TaskRecord
from polar.rollout.policy_transition import PolicyTransitionPhase, PolicyTransitionStore


@pytest.mark.parametrize(
    "engine_ack,drained", [(False, False), (False, True), (True, False), (True, True)]
)
def test_partial_training_requires_both_engine_abort_and_local_drain(
    monkeypatch, engine_ack, drained
):
    async def run():
        store = PolicyTransitionStore(None)
        record = store.start(transition_id="t", from_epoch=0, to_epoch=1, partial_rollout=True)
        record = store.update(
            "t", phase=PolicyTransitionPhase.ADMISSION_CLOSED, engine_abort_confirmed=engine_ack
        )

        async def observe():
            return [
                {
                    "node_id": "n",
                    "status": "ok",
                    "response": {
                        "paused": True,
                        "transition_id": "t",
                        "drained": drained,
                        "inflight": 0 if drained else 1,
                    },
                }
            ]

        monkeypatch.setattr(server, "_observe_gateway_control", observe)
        state = SimpleNamespace(policy_transitions=store)
        record = await server._drive_confirm_drained(state, record, wait_timeout_seconds=0)
        expected = (
            PolicyTransitionPhase.READY_FOR_TRAINING
            if engine_ack and drained
            else PolicyTransitionPhase.ADMISSION_CLOSED
        )
        assert record.phase == expected

    asyncio.run(run())


def test_expiry_uses_group_oldest_epoch_even_for_new_sibling():
    manager = object.__new__(RolloutManager)
    manager._lock = threading.RLock()
    manager._active_policy_namespace = "run"
    manager._tasks = {}
    for name, epoch, oldest, namespace in (
        ("old", 0, 0, "run"),
        ("new-sibling-of-old", 1, 0, "run"),
        ("retain", 1, 1, "run"),
        ("foreign", 1, 1, "other"),
    ):
        manager._tasks[name] = _TaskRecord(
            name,
            "running",
            1,
            policy_version=epoch,
            group_policy_version=oldest,
            policy_namespace=namespace,
            partial_rollout=True,
        )
    assert manager.expired_partial_tasks(list(manager._tasks), 2) == [
        "old",
        "new-sibling-of-old",
        "foreign",
    ]
