from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "ascend" / "polar_dockerruntime_e2e"
SCRIPT = EXAMPLE / "check_render_contract.py"
POLAR_ROOT = ROOT
PLAN_TASK_DISALLOWED_TOOLS = (
    "AskUserQuestion CronCreate CronDelete CronList EnterPlanMode EnterWorktree "
    "ExitPlanMode ExitWorktree NotebookEdit ScheduleWakeup TaskCreate TaskGet "
    "TaskList TaskOutput TaskStop TaskUpdate TodoWrite WebFetch WebSearch"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_render_contract", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(**overrides):
    payload = {
        "runtime": {
            "backend": "docker",
            "image": "sandbox:v1",
            "env": {
                "POLAR_NPU_LEASE_POOL": "0",
                "POLAR_NPU_LOCK_DIR": "/dev/shm/npu-locks",
            },
            "kwargs": {
                "ascend": {
                    "pool": "0",
                    "lock_dir": "/dev/shm/npu-locks",
                    "lease_at_start": False,
                },
                "volumes": [
                    "/skills:/opt/canonical:ro",
                    "/readonly_tools:/opt/workspace/agent_workdir/tools:ro",
                ],
            },
            "prepare": [
                {
                    "type": "exec",
                    "command": (
                        "python3 /opt/canonical/runtime/prepare_operator_workdir.py "
                        "--require-claude --no-stub --readonly-tools"
                    ),
                }
            ],
            "eval_prepare": [
                {
                    "type": "exec",
                    "command": (
                        "python3 /opt/canonical/runtime/prepare_operator_workdir.py "
                        "--no-stub --readonly-tools"
                    ),
                }
            ],
        },
        "agent": {
            "harness": "claude_code",
            "skills_path": "/opt/canonical/skills",
            "settings": {
                "max_turns": 45,
                "append_system_prompt": (
                    "Follow ./CLAUDE.md. The user instruction provides the concrete operator name "
                    "and src/output paths. Do not ask user questions in this non-interactive run."
                ),
                "disallowed_tools": PLAN_TASK_DISALLOWED_TOOLS,
            },
        },
        "evaluator": {
            "strategy": "operator_judge",
            "refresh_runtime": True,
            "config": {"lazy_refresh_runtime": True},
            "runtime": {
                "backend": "docker",
                "image": "sandbox:v1",
                "env": {
                    "POLAR_NPU_LEASE_POOL": "1",
                    "POLAR_NPU_LOCK_DIR": "/dev/shm/npu-locks",
                },
                "kwargs": {
                    "ascend": {
                        "pool": "1",
                        "lock_dir": "/dev/shm/npu-locks",
                        "lease_at_start": False,
                    },
                    "volumes": [
                        "/skills:/opt/canonical:ro",
                        "/readonly_tools:/opt/workspace/agent_workdir/tools:ro",
                    ],
                },
                "eval_prepare": [
                    {
                        "type": "exec",
                        "command": (
                            "python3 /opt/canonical/runtime/prepare_operator_workdir.py "
                            "--no-stub --readonly-tools"
                        ),
                    }
                ],
            },
        },
        "builder": {"strategy": "prefix_merging", "config": {}},
    }
    payload.update(overrides)
    return payload


def test_payload_contract_accepts_mainline_payload() -> None:
    module = _load_module()

    module._assert_payload_contract(
        _payload(),
        image="sandbox:v1",
        skills_dir=Path("/skills"),
        readonly_tools_dir=Path("/readonly_tools"),
        pool="0",
        eval_pool="1",
        lock_dir="/dev/shm/npu-locks",
    )


def test_payload_contract_rejects_non_mainline_image() -> None:
    module = _load_module()

    with pytest.raises(SystemExit, match="sandbox:v1"):
        module._assert_payload_contract(
            _payload(runtime={**_payload()["runtime"], "image": "polar-op-image"}),
            image="polar-op-image",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_excluded_terms() -> None:
    module = _load_module()
    payload = _payload(metadata={"backend": "rllm"})

    with pytest.raises(SystemExit, match="excluded mainline term"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_subagent_or_workflow_tools_banned() -> None:
    module = _load_module()
    payload = _payload(
        agent={
            "harness": "claude_code",
            "skills_path": "/opt/canonical/skills",
            "settings": {
                "max_turns": 45,
                "append_system_prompt": (
                    "Follow ./CLAUDE.md. The user instruction provides the concrete operator name "
                    "and src/output paths. Do not ask user questions in this non-interactive run."
                ),
                "disallowed_tools": f"{PLAN_TASK_DISALLOWED_TOOLS} Agent",
            },
        }
    )

    with pytest.raises(SystemExit, match="sub-agent"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_plan_task_tools_not_banned() -> None:
    module = _load_module()
    payload = _payload(
        agent={
            "harness": "claude_code",
            "skills_path": "/opt/canonical/skills",
            "settings": {
                "max_turns": 45,
                "append_system_prompt": (
                    "Follow ./CLAUDE.md. The user instruction provides the concrete operator name "
                    "and src/output paths. Do not ask user questions in this non-interactive run."
                ),
                "disallowed_tools": "AskUserQuestion WebSearch",
            },
        }
    )

    with pytest.raises(SystemExit, match="plan/task tools"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_non_prefix_merging_default() -> None:
    module = _load_module()
    payload = _payload(builder={"strategy": "per_request", "config": {}})

    with pytest.raises(SystemExit, match="prefix_merging"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_missing_lazy_judge_flag() -> None:
    module = _load_module()
    payload = _payload()
    payload["evaluator"] = {
        key: value
        for key, value in payload["evaluator"].items()
        if key != "config"
    }

    with pytest.raises(SystemExit, match="lazy_refresh_runtime"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_payload_contract_rejects_non_dict_builder_config() -> None:
    module = _load_module()
    payload = _payload(builder={"strategy": "prefix_merging", "config": "not-a-dict"})

    with pytest.raises(SystemExit, match="builder.config"):
        module._assert_payload_contract(
            payload,
            image="sandbox:v1",
            skills_dir=Path("/skills"),
            readonly_tools_dir=Path("/readonly_tools"),
            pool="0",
            eval_pool="1",
            lock_dir="/dev/shm/npu-locks",
        )


def test_safe_op_check_rejects_unsafe_operator_name(tmp_path: Path) -> None:
    module = _load_module()
    tasks_dir = tmp_path / "op_tasks"
    tasks_dir.mkdir()
    (tasks_dir / "safe.py").write_text("class Model: pass\n")

    assert module._assert_safe_op({"metadata": {"op_name": "safe"}}, tasks_dir) == "safe"
    with pytest.raises(SystemExit, match="unsafe/missing op_name"):
        module._assert_safe_op({"metadata": {"op_name": "../bad"}}, tasks_dir)


def test_load_first_task_rejects_empty_jsonl(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "operator_tasks.jsonl"
    path.write_text("\n")

    with pytest.raises(SystemExit, match="empty task jsonl"):
        module._load_first_task(path)


def test_render_contract_rejects_runtime_workers_tied_to_eval_card_count(tmp_path: Path) -> None:
    if not (POLAR_ROOT / "src" / "polar").is_dir():
        pytest.skip(f"Polar repo not found: {POLAR_ROOT}")

    module = _load_module()
    fixture = EXAMPLE / "fixtures" / "operator_assets"
    topology = yaml.safe_load((EXAMPLE / "topology.yaml").read_text(encoding="utf-8"))
    topology["gateway"]["nodes"][0]["max_run_workers"] = 1
    topology["gateway"]["nodes"][0]["max_postrun_workers"] = 1
    bad_topology = tmp_path / "topology.yaml"
    bad_topology.write_text(yaml.safe_dump(topology), encoding="utf-8")

    with pytest.raises(SystemExit, match="max_run_workers"):
        module.main(
            [
                "--polar-root",
                str(POLAR_ROOT),
                "--config",
                str(EXAMPLE / "polar_config.yaml"),
                "--topology",
                str(bad_topology),
                "--skills-dir",
                "/home/docker/polar_e2e/operator_runtime",
                "--tasks-dir",
                str(fixture / "op_tasks"),
                "--task-jsonl",
                str(fixture / "operator_tasks.jsonl"),
                "--image",
                "sandbox:v1",
                "--device-pool",
                "0",
                "--lock-dir",
                "/dev/shm/npu-locks",
                "--model-served",
                "model-served-placeholder",
                "--router-ip",
                "127.0.0.1",
                "--router-port",
                "4077",
            ]
        )


def test_render_contract_main_with_bundled_fixture() -> None:
    if not (POLAR_ROOT / "src" / "polar").is_dir():
        pytest.skip(f"Polar repo not found: {POLAR_ROOT}")

    module = _load_module()
    fixture = EXAMPLE / "fixtures" / "operator_assets"

    rc = module.main(
        [
            "--polar-root",
            str(POLAR_ROOT),
            "--config",
            str(EXAMPLE / "polar_config.yaml"),
            "--topology",
            str(EXAMPLE / "topology.yaml"),
            "--skills-dir",
            "/home/docker/polar_e2e/operator_runtime",
            "--tasks-dir",
            str(fixture / "op_tasks"),
            "--task-jsonl",
            str(fixture / "operator_tasks.jsonl"),
            "--image",
            "sandbox:v1",
            "--device-pool",
            "0",
            "--lock-dir",
            "/dev/shm/npu-locks",
            "--model-served",
            "model-served-placeholder",
            "--router-ip",
            "127.0.0.1",
            "--router-port",
            "4077",
        ]
    )

    assert rc == 0
