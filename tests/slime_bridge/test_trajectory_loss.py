from __future__ import annotations

import torch

from slime_bridge.trajectory_loss import get_trajectory_pg_loss_reducer


def test_trajectory_pg_loss_reducer_weights_traces_by_trajectory(monkeypatch) -> None:
    monkeypatch.setattr(
        "slime_bridge.trajectory_loss.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    masks = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])]
    reducer = get_trajectory_pg_loss_reducer(
        total_lengths=[4, 4, 4],
        response_lengths=[2, 2, 2],
        loss_masks=masks,
        trajectory_keys=[[0, 0], [0, 0], [0, 1]],
        trajectory_trace_counts=[2, 2, 1],
        trajectory_loss_scale=1.5,
    )

    # Trace means: 1, 3, 10. Trajectory means: (1 + 3) / 2 = 2, and 10.
    # Slime divides the reducer output by the flat dynamic global batch size (3),
    # so the final PG loss is (2 + 10) / 2 = 6.
    loss = torch.tensor([1.0, 1.0, 3.0, 3.0, 10.0, 10.0])

    assert torch.isclose(reducer(loss), torch.tensor(18.0))
    assert torch.isclose(reducer(loss) / 3, torch.tensor(6.0))


def test_trajectory_pg_loss_reducer_falls_back_to_sample_mean_without_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "slime_bridge.trajectory_loss.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    masks = [torch.tensor([1, 1]), torch.tensor([1, 1])]
    reducer = get_trajectory_pg_loss_reducer(
        total_lengths=[4, 4],
        response_lengths=[2, 2],
        loss_masks=masks,
    )

    loss = torch.tensor([1.0, 1.0, 3.0, 3.0])

    assert torch.isclose(reducer(loss), torch.tensor(4.0))
