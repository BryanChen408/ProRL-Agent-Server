from __future__ import annotations

import hashlib

import pytest

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


def test_operator_profile_rewrites_task_uploads_to_request_source_cache(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
  save_dir: output/ascend_operator/rollout_results
  default_operator_profile: operator_npu
  operator_profiles:
    operator_npu:
      operator_task_cache_dir: "{cache_dir}"
      runtime:
        backend: docker
        image: sandbox:v1
        prepare:
          - type: upload_file
            source: "output/ascend_operator/op_assets/op_tasks/{op_name}.py"
            target: "/work/src/{op_name}.py"
        eval_prepare:
          - type: upload_file
            source: "output/ascend_operator/op_assets/op_tasks/{op_name}.py"
            target: "/work/src/{op_name}.py"
      agent:
        harness: claude_code
      evaluator:
        strategy: operator_judge
        runtime:
          backend: docker
          image: sandbox:v1
          eval_prepare:
            - type: upload_file
              source: "output/ascend_operator/op_assets/op_tasks/{op_name}.py"
              target: "/work/src/{op_name}.py"
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip().format(cache_dir=tmp_path / "task_cache", op_name="{op_name}")
    )
    rollout = TopologyConfig.load(topology_path).rollout
    source = "class Model: pass\n"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    task_request = expand_operator_sample_request(
        OperatorSampleRequest(
            task_id="task-1",
            instruction="do it",
            sample=OperatorSample(
                op_name="op",
                task_source=source,
                task_source_sha256=digest,
            ),
        ),
        rollout,
    )

    cache_path = tmp_path / "task_cache" / f"{digest}.py"
    assert cache_path.read_text(encoding="utf-8") == source
    assert task_request.runtime is not None
    assert task_request.runtime.prepare[0].source == str(cache_path)
    assert task_request.runtime.eval_prepare is not None
    assert task_request.runtime.eval_prepare[0].source == str(cache_path)
    assert task_request.evaluator is not None
    assert task_request.evaluator.runtime is not None
    assert task_request.evaluator.runtime.eval_prepare is not None
    assert task_request.evaluator.runtime.eval_prepare[0].source == str(cache_path)


def test_operator_profile_rejects_task_source_hash_mismatch(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
  default_operator_profile: operator_npu
  operator_profiles:
    operator_npu:
      runtime:
        backend: docker
        image: sandbox:v1
        prepare:
          - type: upload_file
            source: "output/ascend_operator/op_assets/op_tasks/{op_name}.py"
            target: "/work/src/{op_name}.py"
      agent:
        harness: claude_code
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip()
    )
    rollout = TopologyConfig.load(topology_path).rollout

    with pytest.raises(ValueError, match="task_source_sha256"):
        expand_operator_sample_request(
            OperatorSampleRequest(
                task_id="task-1",
                instruction="do it",
                sample=OperatorSample(
                    op_name="op",
                    task_source="class Model: pass\n",
                    task_source_sha256="0" * 64,
                ),
            ),
            rollout,
        )
