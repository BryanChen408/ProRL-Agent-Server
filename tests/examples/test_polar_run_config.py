from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "ascend" / "polar_dockerruntime_e2e" / "tools" / "render_run_config.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_render_run_config", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_run_config_adds_run_id_prefix_without_mutating_base(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "polar_config.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "polar_task_id_template": "polar-op-{rollout_id}-{sample.group_index}",
                "polar_device_pool": "8,9,10,11",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = module.render_config_file(base, "regular 2026/06/23", tmp_path / "run_configs")
    rendered = yaml.safe_load(out.read_text(encoding="utf-8"))
    original = yaml.safe_load(base.read_text(encoding="utf-8"))

    assert out.name == "polar_config.regular_2026_06_23.yaml"
    assert rendered["polar_run_id"] == "regular_2026_06_23"
    assert rendered["polar_task_id_template"] == (
        "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
    )
    assert original["polar_task_id_template"] == "polar-op-{rollout_id}-{sample.group_index}"


def test_render_run_config_does_not_duplicate_existing_run_prefix(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "polar_config.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "polar_task_id_template": "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = module.render_config_file(base, "smoke_1", tmp_path / "run_configs")
    rendered = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert rendered["polar_task_id_template"] == (
        "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
    )


def test_render_run_config_can_override_polar_service_urls(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "polar_config.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "polar_rollout_url": "http://127.0.0.1:8080",
                "polar_gateway_url": "http://127.0.0.1:8100",
                "polar_task_id_template": "polar-op-{rollout_id}-{sample.group_index}",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = module.render_config_file(
        base,
        "isolated",
        tmp_path / "run_configs",
        rollout_url="http://127.0.0.1:28080/",
        gateway_url="http://127.0.0.1:28100/",
    )
    rendered = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert rendered["polar_rollout_url"] == "http://127.0.0.1:28080"
    assert rendered["polar_gateway_url"] == "http://127.0.0.1:28100"
