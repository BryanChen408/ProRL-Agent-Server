from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import NodeRegistrationRequest, NodeStageMetrics


def _register(
    scheduler: NodeScheduler,
    node_id: str,
    *,
    max_init_workers: int = 4,
    max_run_workers: int = 2,
    max_postrun_workers: int = 2,
) -> None:
    scheduler.register_node(
        NodeRegistrationRequest(
            node_id=node_id,
            gateway_url=f"http://127.0.0.1/{node_id}",
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
            heartbeat_interval_seconds=30,
        )
    )


def test_acquire_prefers_lower_run_pressure_before_init_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "busy-run", max_run_workers=2)
    _register(scheduler, "free-run", max_run_workers=2)
    scheduler.heartbeat(
        "busy-run",
        metrics=NodeStageMetrics(run_inflight=1, init_queue_depth=0),
    )
    scheduler.heartbeat(
        "free-run",
        metrics=NodeStageMetrics(run_inflight=0, init_queue_depth=3),
    )

    selected = scheduler.acquire_node()

    assert selected is not None
    assert selected.node_id == "free-run"
    assert selected.dispatch_reservations == 1


def test_release_reservation_decrements_dispatch_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")

    assert scheduler.acquire_node().dispatch_reservations == 1
    assert scheduler.release_reservation("node-a").dispatch_reservations == 0


def test_draining_node_does_not_receive_new_sessions() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")
    _register(scheduler, "node-b")

    scheduler.acquire_node()
    drained = scheduler.drain_node("node-a")
    selected = scheduler.acquire_node()

    assert drained.draining is True
    assert selected is not None
    assert selected.node_id == "node-b"


def test_stale_nodes_become_ineligible() -> None:
    scheduler = NodeScheduler(stale_factor=2.5)
    _register(scheduler, "node-a")
    scheduler._nodes["node-a"].last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=10)

    assert scheduler.acquire_node() is None
    assert scheduler.get_node("node-a").healthy is False


def test_postrun_backlog_blocks_admission() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a", max_postrun_workers=2)
    scheduler.heartbeat(
        "node-a",
        metrics=NodeStageMetrics(postrun_queue_depth=4),
    )

    assert scheduler.acquire_node() is None


def test_strict_scheduler_rejects_unknown_dynamic_node() -> None:
    scheduler = NodeScheduler(
        bootstrap_nodes=[
            {
                "node_id": "ascend-node-01",
                "gateway_url": "http://127.0.0.1:8100",
                "max_init_workers": 8,
                "max_run_workers": 32,
                "max_postrun_workers": 32,
                "heartbeat_interval_seconds": 30,
            }
        ],
        allow_dynamic_nodes=False,
    )

    _register(scheduler, "ascend-node-01", max_run_workers=32)

    try:
        _register(scheduler, "profile-node-01")
    except KeyError as exc:
        assert "profile-node-01" in str(exc)
    else:
        raise AssertionError("strict scheduler accepted an unknown dynamic node")

    nodes = scheduler.list_nodes()
    assert [node.node_id for node in nodes] == ["ascend-node-01"]
    assert nodes[0].healthy is True
