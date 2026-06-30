from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


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
                    "operator_runtime_dir": "operator_runtime",
                },
                "pipeline_budget": {
                    "generation_max": 5,
                    "optimization_max": 3,
                    "interval_seconds": 7,
                },
                "observer": {"host": "0.0.0.0", "port": 18088},
                "gateway": {"node_id": "node-a", "max_init_workers": 2, "max_run_workers": 4, "max_postrun_workers": 6},
                "operator": {
                    "profile": "operator_npu",
                    "runtime": {"npu_pool": "4,5", "npu_lock_dir": "/tmp/npu-locks"},
                    "agent": {"model_name": "claude-test"},
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

    proc = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(SCRIPT), "--profile", str(profile), "--repo-root", str(repo)],
        check=True,
        text=True,
        capture_output=True,
    )
    env = _load_env(proc.stdout)
    topology = yaml.safe_load(Path(env["POLAR_TOPOLOGY"]).read_text(encoding="utf-8"))
    op_profile = topology["rollout"]["operator_profiles"]["operator_npu"]

    assert env["POLAR_GATEWAY_URL"] == "http://10.0.0.1:8100"
    assert env["POLAR_GEN_PIPELINE_MAX"] == "5"
    assert env["POLAR_OPT_PIPELINE_MAX"] == "3"
    assert env["POLAR_PIPELINE_WATCH_INTERVAL"] == "7"
    assert topology["gateway"]["nodes"][0]["inference"]["base_url"] == "http://10.0.0.2:4077"
    assert topology["gateway"]["nodes"][0]["max_run_workers"] == 4
    assert op_profile["runtime"]["env"]["POLAR_GEN_PIPELINE_MAX"] == "5"
    assert op_profile["evaluator"]["runtime"]["env"]["POLAR_OPT_PIPELINE_MAX"] == "3"
    assert op_profile["runtime"]["kwargs"]["ascend"]["pool"] == "4,5"
    assert op_profile["runtime"]["kwargs"]["volumes"][0] == f"{runtime.resolve()}:/opt/canonical:ro"
