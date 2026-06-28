#!/usr/bin/env python3
"""Render a run-scoped Polar config without mutating the base config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


RUN_ID_VAR = "{args.polar_run_id}"


def sanitize_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = text.strip("._-")
    return text or "run"


def render_config(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    out = dict(config)
    safe_run_id = sanitize_run_id(run_id)
    out["polar_run_id"] = safe_run_id

    template = str(out.get("polar_task_id_template") or "polar-op-{rollout_id}-{sample.group_index}")
    if RUN_ID_VAR not in template:
        template = f"{RUN_ID_VAR}-{template}"
    out["polar_task_id_template"] = template
    return out


def apply_url_overrides(
    config: dict[str, Any],
    rollout_url: str | None = None,
    gateway_url: str | None = None,
    callback_host: str | None = None,
) -> dict[str, Any]:
    out = dict(config)
    if rollout_url:
        out["polar_url"] = rollout_url.rstrip("/")
        out["polar_rollout_url"] = rollout_url.rstrip("/")
    if gateway_url:
        out["polar_gateway_url"] = gateway_url.rstrip("/")
    if callback_host:
        out["polar_callback_host"] = callback_host.strip()
    return out


def apply_pool_overrides(
    config: dict[str, Any],
    device_pool: str | None = None,
    eval_device_pool: str | None = None,
) -> dict[str, Any]:
    out = dict(config)
    if device_pool:
        out["polar_device_pool"] = device_pool
    if eval_device_pool:
        out["polar_eval_device_pool"] = eval_device_pool
    elif device_pool:
        out["polar_eval_device_pool"] = device_pool
    return out


def render_config_file(
    config_path: Path,
    run_id: str,
    output_dir: Path,
    rollout_url: str | None = None,
    gateway_url: str | None = None,
    callback_host: str | None = None,
    device_pool: str | None = None,
    eval_device_pool: str | None = None,
) -> Path:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    safe_run_id = sanitize_run_id(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"polar_config.{safe_run_id}.yaml"
    rendered = render_config(data, safe_run_id)
    rendered = apply_url_overrides(
        rendered,
        rollout_url=rollout_url,
        gateway_url=gateway_url,
        callback_host=callback_host,
    )
    rendered = apply_pool_overrides(
        rendered,
        device_pool=device_pool,
        eval_device_pool=eval_device_pool,
    )
    out_path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-url")
    parser.add_argument("--gateway-url")
    parser.add_argument("--callback-host")
    parser.add_argument("--device-pool")
    parser.add_argument("--eval-device-pool")
    args = parser.parse_args(argv)

    print(
        render_config_file(
            args.config,
            args.run_id,
            args.output_dir,
            rollout_url=args.rollout_url,
            gateway_url=args.gateway_url,
            callback_host=args.callback_host,
            device_pool=args.device_pool,
            eval_device_pool=args.eval_device_pool,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
