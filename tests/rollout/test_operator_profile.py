from __future__ import annotations

from polar.config import TopologyConfig
from polar.rollout.models import OperatorSample, OperatorSampleRequest
from polar.rollout.operator_profile import expand_operator_sample_request


def test_operator_profile_expands_thin_request_to_task_request(tmp_path) -> None:
    task = tmp_path / "op.py"
    task.write_text("task")
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        f"""
rollout:
  public_url: http://127.0.0.1:8080
  default_operator_profile: operator_npu
  operator_profiles:
    operator_npu:
      timeout_seconds: 123
      runtime:
        backend: docker
        image: sandbox:v1
        prepare:
          - type: upload_file
            source: "{tmp_path}/{{op_name}}.py"
            target: "/work/src/{{op_name}}.py"
      agent:
        harness: claude_code
        model_name: claude-opus-4-5
        skills_path: /opt/canonical/skills
      evaluator:
        strategy: operator_judge
        refresh_runtime: true
        config:
          op_name: "{{op_name}}"
          judge_command: "bash tools/triton_eval_pipeline.sh --op_name {{op_name}}"
      builder:
        strategy: prefix_merging
      metadata:
        profile_meta: "{{sample.metadata.case_id}}"
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip()
    )
    rollout = TopologyConfig.load(topology_path).rollout

    task_request = expand_operator_sample_request(
        OperatorSampleRequest(
            task_id="task-1",
            instruction="do it",
            num_samples=2,
            sample=OperatorSample(
                op_name="op",
                group_index=3,
                index=4,
                metadata={"case_id": "abc"},
            ),
            metadata={"policy_version": 7},
        ),
        rollout,
    )

    assert task_request.task_id == "task-1"
    assert task_request.instruction == "do it"
    assert task_request.num_samples == 2
    assert task_request.timeout_seconds == 123
    assert task_request.runtime is not None
    assert task_request.runtime.prepare[0].source == f"{tmp_path}/op.py"
    assert task_request.agent.harness == "claude_code"
    assert task_request.evaluator is not None
    assert task_request.evaluator.config["op_name"] == "op"
    assert task_request.metadata["operator_profile"] == "operator_npu"
    assert task_request.metadata["op_name"] == "op"
    assert task_request.metadata["group_index"] == 3
    assert task_request.metadata["sample_index"] == 4
    assert task_request.metadata["policy_version"] == 7
    assert task_request.metadata["profile_meta"] == "abc"


def test_operator_profile_request_profile_overrides_default(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
  default_operator_profile: a
  operator_profiles:
    a:
      agent: {harness: codex}
    b:
      agent: {harness: claude_code}
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip()
    )
    rollout = TopologyConfig.load(topology_path).rollout

    task_request = expand_operator_sample_request(
        OperatorSampleRequest(
            task_id="task-1",
            instruction="do it",
            profile="b",
            sample=OperatorSample(op_name="op"),
        ),
        rollout,
    )

    assert task_request.agent.harness == "claude_code"
    assert task_request.metadata["operator_profile"] == "b"
