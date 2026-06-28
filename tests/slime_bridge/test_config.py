from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime_bridge.config import (
    render_instruction,
    render_task_payload,
    render_topology_template,
    resolve_polar_slime_config,
    resolve_sglang_router_base_url,
)


def _args(**overrides):
    base = {
        "polar_rollout_url": "http://rollout:8080/",
        "polar_task_template": {
            "agent": {"harness": "codex", "model_name": "{args.model_name}"},
            "runtime": {"image": "{sample.metadata.image}"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
        "polar_task_id_template": "task-{rollout_id}-{sample.group_index}",
        "polar_instruction_template": "Instruction: {instruction}",
        "polar_reward_key": "score",
        "polar_max_async_level": 2,
        "rollout_batch_size": 3,
        "n_samples_per_prompt": 4,
        "update_weights_interval": 5,
        "polar_request_timeout": 60,
        "polar_callback_host": "127.0.0.1",
        "polar_scoring_mode": "group",
        "polar_min_complete_accept_fraction": 0.0,
        "hf_checkpoint": "tokenizer-name",
        "polar_add_generation_prompt": True,
        "polar_eval_dataset_name": "eval",
        "model_name": "openai/gpt-test",
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_polar_slime_config_computes_concurrency_and_normalizes_url() -> None:
    config = resolve_polar_slime_config(_args())

    assert config.rollout_server_url == "http://rollout:8080"
    assert config.max_concurrency == 6
    assert config.max_session_concurrency == 24
    assert config.max_sessions_per_task is None
    assert config.scheduler_mode == "group"
    assert config.max_active_sessions == 24
    assert config.session_pool_pause_policy == "drain_open_groups"
    assert config.max_off_policy_steps == 7
    assert config.request_timeout == 60.0
    assert config.min_complete_accept_fraction == 0.0


def test_resolve_polar_slime_config_accepts_url_only_operator_samples_mode() -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_url="http://rollout:8081/",
            polar_rollout_url=None,
            polar_task_template={},
            polar_submit_mode="operator_samples",
            polar_profile="operator_npu",
            rollout_max_async_level=3,
            rollout_scheduler_mode="session_pool",
            rollout_max_active_sessions=32,
            rollout_release_on_postrun=True,
            rollout_request_timeout=120,
            rollout_min_complete_accept_fraction=0.7,
        )
    )

    assert config.rollout_server_url == "http://rollout:8081"
    assert config.submit_mode == "operator_samples"
    assert config.operator_profile == "operator_npu"
    assert config.max_concurrency == 9
    assert config.scheduler_mode == "session_pool"
    assert config.max_active_sessions == 32
    assert config.session_pool_release_on_postrun is True
    assert config.request_timeout == 120.0
    assert config.min_complete_accept_fraction == 0.7


def test_resolve_polar_slime_config_does_not_limit_sessions_by_device_pool() -> None:
    assert resolve_polar_slime_config(_args(polar_device_pool="8,9")).max_sessions_per_task is None
    assert resolve_polar_slime_config(_args(polar_device_pool="8-11")).max_sessions_per_task is None
    assert resolve_polar_slime_config(_args(polar_device_pool=[8, 9, 10])).max_sessions_per_task is None


def test_resolve_polar_slime_config_allows_explicit_max_sessions_per_task() -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_device_pool="8,9",
            polar_max_sessions_per_task=1,
        )
    )

    assert config.max_sessions_per_task == 1


def test_resolve_polar_slime_config_rejects_invalid_max_sessions_per_task() -> None:
    with pytest.raises(ValueError, match="polar_max_sessions_per_task"):
        resolve_polar_slime_config(_args(polar_max_sessions_per_task=0))


def test_resolve_polar_slime_config_accepts_session_pool_scheduler() -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_scheduler_mode="session_pool",
            polar_max_active_sessions=16,
            polar_session_pool_pause_policy="drain_open_groups",
        )
    )

    assert config.scheduler_mode == "session_pool"
    assert config.max_active_sessions == 16
    assert config.max_session_concurrency == 24
    assert config.max_sessions_per_task is None
    assert config.session_pool_pause_policy == "drain_open_groups"
    assert config.session_pool_release_on_postrun is False


@pytest.mark.parametrize("value", [True, "true", "1", "yes", "on"])
def test_resolve_polar_slime_config_accepts_session_pool_release_flag(value) -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_scheduler_mode="session_pool",
            polar_session_pool_release_on_postrun=value,
        )
    )

    assert config.session_pool_release_on_postrun is True


def test_resolve_polar_slime_config_rejects_invalid_session_pool_release_flag() -> None:
    with pytest.raises(ValueError, match="rollout_release_on_postrun"):
        resolve_polar_slime_config(
            _args(
                polar_scheduler_mode="session_pool",
                polar_session_pool_release_on_postrun="maybe",
            )
        )


def test_resolve_polar_slime_config_rejects_invalid_scheduler_mode() -> None:
    with pytest.raises(ValueError, match="scheduler_mode"):
        resolve_polar_slime_config(_args(polar_scheduler_mode="round_robin"))


@pytest.mark.parametrize("value", [0, -1])
def test_resolve_polar_slime_config_rejects_invalid_max_active_sessions(value) -> None:
    with pytest.raises(ValueError, match="max_active_sessions"):
        resolve_polar_slime_config(_args(polar_max_active_sessions=value))


def test_resolve_polar_slime_config_rejects_unimplemented_session_pool_pause_policy() -> None:
    with pytest.raises(ValueError, match="session_pool_pause_policy"):
        resolve_polar_slime_config(_args(polar_session_pool_pause_policy="drop_partial_group"))


def test_resolve_polar_slime_config_requires_agent_template() -> None:
    with pytest.raises(ValueError, match="agent spec"):
        resolve_polar_slime_config(
            _args(polar_task_template={}, polar_submit_mode="task_request")
        )


def test_resolve_polar_slime_config_accepts_complete_fraction_threshold() -> None:
    config = resolve_polar_slime_config(
        _args(polar_min_complete_accept_fraction=0.8)
    )

    assert config.min_complete_accept_fraction == 0.8


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_resolve_polar_slime_config_rejects_invalid_complete_fraction(value) -> None:
    with pytest.raises(ValueError, match="polar_min_complete_accept_fraction"):
        resolve_polar_slime_config(_args(polar_min_complete_accept_fraction=value))


def test_render_task_payload_resolves_args_and_sample_placeholders() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "abc123"},
        group_index=9,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=2,
        task_position=0,
        num_rollouts=4,
    )

    assert payload["task_id"] == "task-2-9"
    assert payload["instruction"] == "Fix the bug"
    assert payload["num_samples"] == 4
    assert payload["agent"]["model_name"] == "openai/gpt-test"
    assert payload["runtime"]["image"] == "runtime:latest"
    assert payload["metadata"]["instance"] == "abc123"


def test_render_instruction_uses_optional_template() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)

    rendered = render_instruction(
        args=args,
        config=config,
        sample=SimpleNamespace(metadata={}),
        prompt_text="Fix the bug",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )

    assert rendered == "Instruction: Fix the bug"


def test_resolve_sglang_router_base_url_requires_both_ip_and_port() -> None:
    assert resolve_sglang_router_base_url(_args()) == "http://127.0.0.1:30000"
    assert resolve_sglang_router_base_url(_args(sglang_router_port=None)) is None


def test_render_topology_template_emits_inference_block(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout: {host: 127.0.0.1, port: 8080, public_url: http://127.0.0.1:8080}
gateway:
  nodes:
    - id: n1
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: Qwen/Qwen3.5-4B
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
""".strip()
    )
    rendered = render_topology_template(str(topology_path), _args())
    node = rendered["gateway"]["nodes"][0]
    assert node["inference"] == {"engine": "sglang", "base_url": "http://127.0.0.1:30000"}
    assert "sglang" not in node


def test_render_topology_template_preserves_operator_profiles(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
  default_operator_profile: operator_npu
  operator_profiles:
    operator_npu:
      agent: {harness: claude_code}
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
""".strip()
    )

    rendered = render_topology_template(str(topology_path), _args())

    assert rendered["rollout"]["default_operator_profile"] == "operator_npu"
    assert rendered["rollout"]["operator_profiles"]["operator_npu"]["agent"]["harness"] == "claude_code"


def test_render_topology_template_can_override_model_served(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout: {host: 127.0.0.1, port: 8080, public_url: http://127.0.0.1:8080}
gateway:
  nodes:
    - id: n1
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: placeholder
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
""".strip()
    )

    rendered = render_topology_template(
        str(topology_path),
        _args(polar_model_served_name="/models/Qwen3"),
    )

    assert rendered["gateway"]["nodes"][0]["model_served"] == "/models/Qwen3"
