from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime_bridge.reward_post_process import post_process_rewards


class FakeSample:
    def __init__(
        self,
        *,
        group_index: int,
        index: int,
        reward: float,
        status: str = "COMPLETED",
    ) -> None:
        self.group_index = group_index
        self.index = index
        self._reward = reward
        self.status = status

    def get_reward_value(self, args) -> float:
        return self._reward


def _args(**overrides):
    base = {
        "rewards_normalization": True,
        "advantage_estimator": "grpo",
        "grpo_std_normalization": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_trajectory_reward_normalization_uses_sessions_not_trace_count() -> None:
    samples = [
        *(FakeSample(group_index=0, index=0, reward=1.0) for _ in range(3)),
        *(FakeSample(group_index=0, index=1, reward=0.6) for _ in range(2)),
        *(FakeSample(group_index=0, index=2, reward=0.2) for _ in range(3)),
        FakeSample(group_index=0, index=3, reward=0.8),
    ]

    raw_rewards, rewards = post_process_rewards(_args(), samples)

    assert raw_rewards == [1.0, 1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.2, 0.8]
    assert rewards == pytest.approx(
        [0.35, 0.35, 0.35, -0.05, -0.05, -0.45, -0.45, -0.45, 0.15]
    )


def test_failed_trajectories_do_not_pollute_group_baseline() -> None:
    samples = [
        FakeSample(group_index=0, index=0, reward=1.0),
        FakeSample(group_index=0, index=0, reward=1.0),
        FakeSample(group_index=0, index=1, reward=-100.0, status="ABORTED"),
        FakeSample(group_index=0, index=2, reward=0.5),
        FakeSample(group_index=0, index=2, reward=0.5),
    ]

    raw_rewards, rewards = post_process_rewards(_args(), samples)

    assert raw_rewards == [1.0, 1.0, -100.0, 0.5, 0.5]
    assert rewards == pytest.approx([0.25, 0.25, 0.0, -0.25, -0.25])
