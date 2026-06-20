from __future__ import annotations

import json
from types import SimpleNamespace

from polar.config import TopologyConfig
from polar.rollout.models import TaskRequest
from slime_bridge.config import (
    render_task_payload,
    render_topology_template,
    resolve_polar_slime_config,
)


def _mainline_args(**overrides):
    base = {
        "polar_rollout_url": "http://127.0.0.1:8080",
        "polar_reward_key": "score",
        "polar_max_async_level": 1,
        "polar_request_timeout": 4000,
        "polar_callback_host": "127.0.0.1",
        "polar_scoring_mode": "group",
        "polar_min_complete_accept_fraction": 0.6,
        "polar_add_generation_prompt": True,
        "polar_task_id_template": "polar-op-{rollout_id}-{sample.group_index}",
        "polar_op_image": "sandbox:v1",
        "polar_op_backend": "docker",
        "polar_op_model_name": "claude-opus-4-5",
        "polar_builder_strategy": "prefix_merging",
        "polar_builder_config": {},
        "polar_disallowed_tools": "AskUserQuestion CronCreate CronDelete CronList EnterPlanMode EnterWorktree ExitPlanMode ExitWorktree NotebookEdit ScheduleWakeup WebFetch WebSearch",
        "polar_device_pool": "0",
        "polar_eval_device_pool": "1",
        "polar_lock_dir": "/dev/shm/npu-locks",
        "polar_skills_dir": "/opt/polar-skills",
        "polar_tasks_dir": "/opt/polar-tasks",
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 2,
        "update_weights_interval": 1,
        "hf_checkpoint": "qwen-slime",
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 4077,
        "polar_model_served_name": "qwen-slime",
        "polar_task_template": {
            "timeout_seconds": 3600.0,
            "runtime": {
                "backend": "{args.polar_op_backend}",
                "image": "{args.polar_op_image}",
                "network": "host",
                "workdir": "/opt/workspace/agent_workdir",
                "env": {
                    "DISABLE_AUTOUPDATER": "1",
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768",
                    "POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS": "32768",
                },
                "kwargs": {
                    "ascend": {
                        "pool": "{args.polar_device_pool}",
                        "lock_dir": "{args.polar_lock_dir}",
                    },
                    "volumes": ["{args.polar_skills_dir}:/opt/canonical:ro"],
                },
                "prepare": [
                    {
                        "type": "upload_file",
                        "source": "{args.polar_tasks_dir}/{sample.metadata.op_name}.py",
                        "target": "/opt/workspace/agent_workdir/src/{sample.metadata.op_name}.py",
                    },
                    {
                        "type": "exec",
                        "command": "python3 /opt/canonical/tools/prepare_operator_workdir.py --op-name {sample.metadata.op_name} --workdir /opt/workspace/agent_workdir --require-claude",
                    },
                ],
                "eval_prepare": [
                    {
                        "type": "upload_file",
                        "source": "{args.polar_tasks_dir}/{sample.metadata.op_name}.py",
                        "target": "/opt/workspace/agent_workdir/src/{sample.metadata.op_name}.py",
                    },
                    {
                        "type": "exec",
                        "command": "python3 /opt/canonical/tools/prepare_operator_workdir.py --op-name {sample.metadata.op_name} --workdir /opt/workspace/agent_workdir --no-stub",
                    },
                ],
            },
            "agent": {
                "harness": "claude_code",
                "model_name": "{args.polar_op_model_name}",
                "skills_path": "/opt/canonical/skills",
                "settings": {
                    "disallowed_tools": "{args.polar_disallowed_tools}",
                },
            },
            "evaluator": {
                "strategy": "operator_judge",
                "refresh_runtime": True,
                "runtime": {
                    "backend": "{args.polar_op_backend}",
                    "image": "{args.polar_op_image}",
                    "network": "host",
                    "workdir": "/opt/workspace/agent_workdir",
                    "env": {
                        "DISABLE_AUTOUPDATER": "1",
                    },
                    "kwargs": {
                        "ascend": {
                            "pool": "{args.polar_eval_device_pool}",
                            "lock_dir": "{args.polar_lock_dir}",
                        },
                        "volumes": ["{args.polar_skills_dir}:/opt/canonical:ro"],
                    },
                    "eval_prepare": [
                        {
                            "type": "upload_file",
                            "source": "{args.polar_tasks_dir}/{sample.metadata.op_name}.py",
                            "target": "/opt/workspace/agent_workdir/src/{sample.metadata.op_name}.py",
                        },
                        {
                            "type": "exec",
                            "command": "python3 /opt/canonical/tools/prepare_operator_workdir.py --op-name {sample.metadata.op_name} --workdir /opt/workspace/agent_workdir --no-stub",
                        },
                    ],
                },
                "config": {
                    "op_name": "{sample.metadata.op_name}",
                    "judge_command": "bash tools/triton_eval_pipeline.sh --op_name {sample.metadata.op_name} --impl output/submission/{sample.metadata.op_name}_impl.py --task src/{sample.metadata.op_name}.py --out_dir judge_out",
                    "submission_path": "output/submission/{sample.metadata.op_name}_impl.py",
                    "metrics_path": "judge_out/metrics.json",
                    "workdir": "/opt/workspace/agent_workdir",
                },
            },
            "builder": {
                "strategy": "{args.polar_builder_strategy}",
                "config": "{args.polar_builder_config}",
            },
        },
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_mainline_operator_payload_is_polar_docker_runtime_contract() -> None:
    args = _mainline_args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt=[{"role": "user", "content": "implement operator"}],
        response="",
        label="kernelbench_l1_19_19_ReLU",
        metadata={"op_name": "kernelbench_l1_19_19_ReLU"},
        index=0,
        group_index=3,
        status=None,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="implement operator",
        rollout_id=5,
        task_position=0,
        num_rollouts=2,
    )
    request = TaskRequest(**payload)

    assert request.task_id == "polar-op-5-3"
    assert request.num_samples == 2
    assert request.runtime is not None
    assert request.runtime.backend == "docker"
    assert request.runtime.image == "sandbox:v1"
    assert request.runtime.network == "host"
    assert request.runtime.kwargs["ascend"] == {
        "pool": "0",
        "lock_dir": "/dev/shm/npu-locks",
    }
    assert request.runtime.kwargs["volumes"] == ["/opt/polar-skills:/opt/canonical:ro"]
    assert request.runtime.prepare[1].command is not None
    assert "prepare_operator_workdir.py" in request.runtime.prepare[1].command
    assert "--require-claude" in request.runtime.prepare[1].command
    assert request.runtime.eval_prepare is not None
    assert request.runtime.eval_prepare[1].command is not None
    assert "--no-stub" in request.runtime.eval_prepare[1].command
    assert request.agent.harness == "claude_code"
    assert request.agent.skills_path == "/opt/canonical/skills"
    assert request.evaluator is not None
    assert request.evaluator.strategy == "operator_judge"
    assert request.evaluator.refresh_runtime is True
    assert request.evaluator.runtime is not None
    assert request.evaluator.runtime.kwargs["ascend"] == {
        "pool": "1",
        "lock_dir": "/dev/shm/npu-locks",
    }
    assert request.evaluator.runtime.prepare == []
    assert request.evaluator.runtime.eval_prepare is not None
    assert "--no-stub" in request.evaluator.runtime.eval_prepare[1].command
    assert request.builder.strategy == "prefix_merging"
    assert request.builder.config == {}
    banned = set(request.agent.settings["disallowed_tools"].split())
    assert "Agent" not in banned
    assert not any(tool.startswith("Task") for tool in banned)

    blob = json.dumps(payload, sort_keys=True)
    for excluded in (
        "polar_faith",
        "faithfulness",
        "rllm",
        "LocalRuntime",
        "deploy_dood_B",
        "polar-op-image",
    ):
        assert excluded not in blob


def test_mainline_topology_renders_slime_router_and_model_served(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout: {host: 127.0.0.1, port: 8080, public_url: http://127.0.0.1:8080}
gateway:
  nodes:
    - id: polar-local
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: placeholder
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
""".strip()
    )
    TopologyConfig.load(topology_path)

    rendered = render_topology_template(topology_path, _mainline_args())
    node = rendered["gateway"]["nodes"][0]

    assert node["model_served"] == "qwen-slime"
    assert node["inference"] == {
        "engine": "sglang",
        "base_url": "http://127.0.0.1:4077",
    }
    assert not node["inference"]["base_url"].endswith("/v1")
