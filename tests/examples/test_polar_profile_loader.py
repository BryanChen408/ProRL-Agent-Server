from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from polar.agent.models import AgentSpec
from polar.agent.presets.claude_code import ClaudeCodeHarness


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "ascend_operator" / "tools" / "load_polar_profile.py"


def _load_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("export "):
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        env[key] = value.strip("'")
    return env


@pytest.mark.parametrize("name", ["t2a", "t3a"])
def test_real_profiles_separate_attempt_credit_from_call_limits(tmp_path, name):
    profile = yaml.safe_load((ROOT / f"deploy/ascend_operator/profile.{name}.yaml").read_text())
    profile["paths"]["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    proc = subprocess.run(
        ["python3", str(SCRIPT), "--profile", str(path), "--repo-root", str(ROOT)],
        env={**os.environ, "POLAR_RUN_ID": "test", "POLAR_T3A_ATTEMPT_SPANS": "0"},
        capture_output=True, text=True, check=True,
    )
    env = _load_env(proc.stdout)
    topology = yaml.safe_load(Path(env["POLAR_TOPOLOGY"]).read_text())
    op = topology["rollout"]["operator_profiles"]["operator_npu"]
    command = ClaudeCodeHarness(AgentSpec(**op["agent"])).run_steps("test operator")[0].command
    assert op["timeout_seconds"] == profile["operator"]["timeout_seconds"]
    assert topology["gateway"]["nodes"][0]["inference"]["base_url"] == profile["service"]["sglang_router_url"]
    if name == "t3a":
        assert op["evaluator"]["postrun_timeout_seconds"] == 5400
        assert op["evaluator"]["config"]["judge_timeout"] == 5400
        assert env["POLAR_T3A_ATTEMPT_SPANS"] == "1"
        assert env["POLAR_PIPELINE_BUDGET_ENABLED"] == "0"
        assert env["POLAR_GEN_PIPELINE_MAX"] == env["POLAR_OPT_PIPELINE_MAX"] == ""
        assert "POLAR_GEN_PIPELINE_MAX" not in op["runtime"]["env"]
        assert "POLAR_OPT_PIPELINE_MAX" not in op["runtime"]["env"]
        assert "max_turns" not in op["agent"]["settings"]
        assert "--max-turns" not in command
    else:
        assert "postrun_timeout_seconds" not in op["evaluator"]
        assert env["POLAR_T3A_ATTEMPT_SPANS"] == "0"
        assert env["POLAR_PIPELINE_BUDGET_ENABLED"] == "1"
        assert env["POLAR_GEN_PIPELINE_MAX"] == str(profile["operator_runtime"]["budget"]["generation_max"])
        assert env["POLAR_OPT_PIPELINE_MAX"] == str(profile["operator_runtime"]["budget"]["optimization_max"])
        assert op["runtime"]["env"]["POLAR_GEN_PIPELINE_MAX"] == env["POLAR_GEN_PIPELINE_MAX"]
        assert op["agent"]["settings"]["max_turns"] == 150
        assert "--max-turns 150" in command


def test_time_only_start_stops_old_watcher_without_starting_another(tmp_path):
    # Run the actual launcher with isolated paths and fake process controls.
    script = tmp_path / "start_pipeline_budget_watcher.sh"
    script.write_text((ROOT / "deploy/ascend_operator/start_pipeline_budget_watcher.sh").read_text())
    (tmp_path / "_paths.sh").write_text(
        'export POLAR_DEPLOY_DIR="$PWD" POLAR_OUTPUT_DIR="$PWD/out" POLAR_LOG_DIR="$PWD/out/logs"\n'
        'pgrep() { printf "123456\\n"; }\n'
        'kill() { printf "%s\\n" "$*" >> "$PWD/stopped"; }\n'
        'sleep() { :; }\n'
        'setsid() { touch "$PWD/unexpected_start"; return 1; }\n'
    )
    (tmp_path / "out").mkdir()
    pid = tmp_path / "out/pipeline_budget_watcher.pid"
    pid.write_text("123456\n")
    proc = subprocess.run(
        ["bash", str(script)], cwd=tmp_path,
        env={**os.environ, "POLAR_TOPOLOGY": "already-rendered", "POLAR_PIPELINE_BUDGET_ENABLED": "0"},
        capture_output=True, text=True, check=True,
    )
    assert "call limits disabled" in proc.stdout
    assert (tmp_path / "stopped").read_text().splitlines() == ["123456", "-9 123456"]
    assert not pid.exists()
    assert not (tmp_path / "unexpected_start").exists()


def test_profile_loader_derives_topology_and_sidecar_env(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = repo / "operator_runtime"
    runtime.mkdir(parents=True)
    profile = repo / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "bind_host": "0.0.0.0",
                    "rollout_url": "http://10.0.0.1:8080",
                    "gateway_url": "http://10.0.0.1:8100",
                    "sglang_router_url": "http://10.0.0.2:4077",
                },
                "paths": {
                    "output_dir": "out",
                },
                "operator_runtime": {
                    "workflow": "legacy",
                    "budget": {
                        "generation_max": 5,
                        "optimization_max": 3,
                        "interval_seconds": 7,
                    },
                    "npu_lease": {
                        "enabled": True,
                        "pool": [4, 5],
                        "lock_dir": "/tmp/npu-locks",
                    },
                },
                "observer": {"host": "0.0.0.0", "port": 18088},
                "gateway": {
                    "node_id": "node-a",
                    "release_session_affinity": True,
                    "max_init_workers": 2,
                    "max_run_workers": 4,
                    "max_postrun_workers": 6,
                    "completion_persistence": {
                        "enabled": True,
                        "max_field_bytes": 67108864,
                        "queue_size": 4096,
                    },
                },
                "operator": {
                    "profile": "operator_npu",
                    "runtime": {},
                    "agent": {
                        "model_name": "claude-test",
                        "allowed_tools": "Bash Read Edit Write Skill Grep Glob",
                        "disallowed_tools": "Agent Workflow WebSearch",
                    },
                    "evaluator": {
                        "judge_command": "bash tools/triton_eval_pipeline.sh --op_name {op_name}",
                        "submission_path": "output/submission/{op_name}_impl.py",
                        "metrics_path": "judge_out/metrics.json",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env_vars = os.environ.copy()
    env_vars["POLAR_RUN_ID"] = "unit-run"
    proc = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(SCRIPT), "--profile", str(profile), "--repo-root", str(repo)],
        check=True,
        text=True,
        capture_output=True,
        env=env_vars,
    )
    env = _load_env(proc.stdout)
    topology = yaml.safe_load(Path(env["POLAR_TOPOLOGY"]).read_text(encoding="utf-8"))
    op_profile = topology["rollout"]["operator_profiles"]["operator_npu"]

    assert env["POLAR_GATEWAY_URL"] == "http://10.0.0.1:8100"
    assert env["POLAR_RUN_ID"] == "unit-run"
    assert env["POLAR_OUTPUT_ROOT"] == str((repo / "out").resolve())
    assert env["POLAR_OUTPUT_DIR"] == str((repo / "out" / "runs" / "unit-run").resolve())
    assert env["POLAR_GEN_PIPELINE_MAX"] == "5"
    assert env["POLAR_OPT_PIPELINE_MAX"] == "3"
    assert env["POLAR_PIPELINE_WATCH_INTERVAL"] == "7"
    assert topology["rollout"]["save_dir"] == str((repo / "out" / "runs" / "unit-run" / "rollout_results").resolve())
    assert topology["gateway"]["nodes"][0]["inference"]["base_url"] == "http://10.0.0.2:4077"
    assert topology["gateway"]["nodes"][0]["session_affinity_release_url"] == (
        "http://10.0.0.2:4077/vime/release_sticky_session"
    )
    assert topology["gateway"]["nodes"][0]["max_run_workers"] == 4
    assert topology["gateway"]["completion_persistence"] == {
        "enabled": True,
        "max_field_bytes": 67108864,
        "queue_size": 4096,
    }
    assert op_profile["runtime"]["env"]["POLAR_GEN_PIPELINE_MAX"] == "5"
    assert "POLAR_OPT_PIPELINE_MAX" not in op_profile["evaluator"]["runtime"]["env"]
    assert op_profile["runtime"]["kwargs"]["ascend"]["pool"] == "4,5"
    assert op_profile["runtime"]["kwargs"]["volumes"][0] == f"{runtime.resolve()}:/opt/canonical:ro"
    assert op_profile["agent"]["settings"]["allowed_tools"] == (
        "Bash Read Edit Write Skill Grep Glob"
    )
    assert op_profile["agent"]["settings"]["disallowed_tools"] == "Agent Workflow WebSearch"


def test_profile_loader_derives_cannbot_runtime_from_workflow(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = repo / "operator_runtime" / "cannbot"
    (runtime / "skills").mkdir(parents=True)
    (runtime / "AGENTS.md").write_text("workflow", encoding="utf-8")
    profile = repo / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "rollout_url": "http://10.0.0.1:8080",
                    "gateway_url": "http://10.0.0.1:8100",
                    "sglang_router_url": "http://10.0.0.2:4077",
                },
                "paths": {"output_dir": "out"},
                "operator_runtime": {
                    "workflow": "cannbot",
                    "budget": {"generation_max": 3, "optimization_max": 1, "interval_seconds": 2},
                    "npu_lease": {"enabled": True, "pool": "8-9", "lock_dir": "/dev/shm/npu-locks"},
                },
                "operator": {
                    "profile": "operator_npu",
                    "runtime": {"image": "sandbox:v1"},
                    "agent": {"model_name": "claude-test"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env_vars = os.environ.copy()
    env_vars["POLAR_RUN_ID"] = "cannbot-run"
    proc = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(SCRIPT), "--profile", str(profile), "--repo-root", str(repo)],
        check=True,
        text=True,
        capture_output=True,
        env=env_vars,
    )
    env = _load_env(proc.stdout)
    topology = yaml.safe_load(Path(env["POLAR_TOPOLOGY"]).read_text(encoding="utf-8"))
    op_profile = topology["rollout"]["operator_profiles"]["operator_npu"]

    assert "session_affinity_release_url" not in topology["gateway"]["nodes"][0]
    assert op_profile["operator_runtime_dir"] == str(runtime.resolve())
    assert op_profile["runtime"]["kwargs"]["volumes"] == [f"{runtime.resolve()}:/opt/canonical:ro"]
    assert op_profile["runtime"]["prepare"] == [
        {
            "type": "upload_file",
            "source": str((repo / "out" / "op_assets" / "op_tasks" / "{op_name}.py").resolve()),
            "target": "/opt/workspace/agent_workdir/input/{op_name}.py",
        },
        {
            "type": "exec",
            "cwd": "/opt/workspace/agent_workdir",
            "command": (
                "mkdir -p input output && cp /opt/canonical/AGENTS.md CLAUDE.md "
                "&& ln -sfn /polar/session/.claude .claude"
            ),
        },
    ]
    assert op_profile["runtime"]["env"]["POLAR_GEN_PIPELINE_MAX"] == "3"
    assert op_profile["runtime"]["env"]["POLAR_OPT_PIPELINE_MAX"] == "1"
    assert op_profile["runtime"]["env"]["POLAR_NPU_LEASE_POOL"] == "8-9"
    assert "POLAR_GEN_PIPELINE_MAX" not in op_profile["evaluator"]["runtime"]["env"]
    assert "allowed_tools" not in op_profile["agent"]["settings"]
    assert op_profile["evaluator"]["config"] == {
        "lazy_refresh_runtime": True,
        "op_name": "{op_name}",
        "judge_mode": "cannbot",
        "task_path": "input/{op_name}.py",
        "cannbot_runtime_root": "/opt/canonical",
        "workdir": "/opt/workspace/agent_workdir",
    }
