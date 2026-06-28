from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "ascend" / "polar_dockerruntime_e2e" / "tools" / "render_run_topology.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_render_run_topology", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_run_topology_overrides_service_ports_and_preserves_workers(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "topology.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "rollout": {
                    "host": "127.0.0.1",
                    "port": 8080,
                    "public_url": "http://127.0.0.1:8080",
                    "save_dir": "/home/docker/polar_e2e/rollout_results",
                },
                "gateway": {
                    "heartbeat_interval_seconds": 30,
                    "nodes": [
                        {
                            "id": "ascend-node-01",
                            "host": "127.0.0.1",
                            "port": 8100,
                            "public_url": "http://127.0.0.1:8100",
                            "max_init_workers": 8,
                            "max_run_workers": 32,
                            "max_postrun_workers": 32,
                            "inference": {"engine": "sglang", "base_url": "http://127.0.0.1:4077"},
                        }
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = module.render_topology_file(
        base,
        tmp_path / "topology.isolated.yaml",
        rollout_port=28080,
        gateway_port=28100,
        router_port=24077,
    )
    rendered = yaml.safe_load(out.read_text(encoding="utf-8"))
    node = rendered["gateway"]["nodes"][0]

    assert rendered["rollout"]["port"] == 28080
    assert rendered["rollout"]["public_url"] == "http://127.0.0.1:28080"
    assert rendered["rollout"]["save_dir"] == "/home/docker/polar_e2e/rollout_results"
    assert node["port"] == 28100
    assert node["public_url"] == "http://127.0.0.1:28100"
    assert node["inference"]["base_url"] == "http://127.0.0.1:24077"
    assert node["max_init_workers"] == 8
    assert node["max_run_workers"] == 32
    assert node["max_postrun_workers"] == 32
