#!/usr/bin/env python3
"""Render a Polar topology for a run.

Committed topology files stay repo-relative so the Polar repository can be
moved. This renderer writes the run-scoped topology consumed by DockerRuntime,
where host volume sources must be absolute paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


def _url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return _repo_root() / "output" / "ascend_operator"


def _parse_url(value: str) -> tuple[str, int]:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"invalid URL: {value!r}")
    if parsed.port is None:
        return parsed.hostname, 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port


def _replace_path_prefix(text: str, replacements: dict[str, Path]) -> str:
    normalized = text[2:] if text.startswith("./") else text
    for prefix, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        prefix_text = prefix[2:] if prefix.startswith("./") else prefix
        if normalized == prefix_text:
            return str(replacement)
        for sep in ("/", ":"):
            marker = prefix_text + sep
            if normalized.startswith(marker):
                return str(replacement) + normalized[len(prefix_text) :]
    return text


def _render_paths(value: Any, replacements: dict[str, Path]) -> Any:
    if isinstance(value, str):
        return _replace_path_prefix(value, replacements)
    if isinstance(value, list):
        return [_render_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _render_paths(item, replacements) for key, item in value.items()}
    return value


def render_topology(
    topology: dict[str, Any],
    *,
    rollout_port: int | None = None,
    gateway_port: int | None = None,
    router_port: int | None = None,
    host: str | None = None,
    rollout_url: str | None = None,
    gateway_url: str | None = None,
    router_url: str | None = None,
    operator_runtime_dir: Path | None = None,
    op_assets_dir: Path | None = None,
    rollout_results_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = _default_output_root()
    operator_runtime_dir = (operator_runtime_dir or (_repo_root() / "operator_runtime")).resolve()
    op_assets_dir = (op_assets_dir or (output_root / "op_assets")).resolve()
    rollout_results_dir = (rollout_results_dir or (output_root / "rollout_results")).resolve()

    replacements = {
        "operator_runtime/tools": operator_runtime_dir / "tools",
        "operator_runtime": operator_runtime_dir,
        "output/ascend_operator/op_assets": op_assets_dir,
        "output/ascend_operator/rollout_results": rollout_results_dir,
        "op_assets": op_assets_dir,
        "rollout_results": rollout_results_dir,
    }

    out = _render_paths(dict(topology), replacements)
    rollout = dict(out.get("rollout") or {})
    if rollout_url:
        rollout_host, rollout_port = _parse_url(rollout_url)
        rollout["public_url"] = _url(rollout_host, rollout_port)
        rollout["port"] = rollout_port
    if host:
        rollout["host"] = host
        if rollout_port is not None:
            rollout["public_url"] = _url(host, rollout_port)
    if rollout_port is not None:
        rollout["port"] = rollout_port
        if not rollout_url and host:
            rollout["public_url"] = _url(host, rollout_port)
    out["rollout"] = rollout

    gateway = dict(out.get("gateway") or {})
    gateway_host = host
    if gateway_url:
        gateway_host, gateway_port = _parse_url(gateway_url)
    if router_url:
        router_base_url = router_url.rstrip("/")
    elif router_port is not None and host:
        router_base_url = _url(host, router_port)
    else:
        router_base_url = None
    nodes = []
    for node in gateway.get("nodes") or []:
        item = dict(node)
        if host:
            item["host"] = host
        if gateway_port is not None:
            item["port"] = gateway_port
        if gateway_url:
            item["public_url"] = gateway_url.rstrip("/")
        elif gateway_host and gateway_port is not None:
            item["public_url"] = _url(gateway_host, gateway_port)
        inference = dict(item.get("inference") or {})
        if router_base_url:
            inference["base_url"] = router_base_url
        item["inference"] = inference
        nodes.append(item)
    gateway["nodes"] = nodes
    out["gateway"] = gateway
    return out


def render_topology_file(
    topology_path: Path,
    output_path: Path,
    *,
    rollout_port: int | None = None,
    gateway_port: int | None = None,
    router_port: int | None = None,
    host: str | None = None,
    rollout_url: str | None = None,
    gateway_url: str | None = None,
    router_url: str | None = None,
    operator_runtime_dir: Path | None = None,
    op_assets_dir: Path | None = None,
    rollout_results_dir: Path | None = None,
) -> Path:
    data = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{topology_path} must contain a YAML mapping")
    rendered = render_topology(
        data,
        rollout_port=rollout_port,
        gateway_port=gateway_port,
        router_port=router_port,
        host=host,
        rollout_url=rollout_url,
        gateway_url=gateway_url,
        router_url=router_url,
        operator_runtime_dir=operator_runtime_dir,
        op_assets_dir=op_assets_dir,
        rollout_results_dir=rollout_results_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-port", type=int)
    parser.add_argument("--gateway-port", type=int)
    parser.add_argument("--router-port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--rollout-url")
    parser.add_argument("--gateway-url")
    parser.add_argument("--router-url")
    parser.add_argument("--operator-runtime-dir", type=Path)
    parser.add_argument("--op-assets-dir", type=Path)
    parser.add_argument("--rollout-results-dir", type=Path)
    args = parser.parse_args(argv)

    print(
        render_topology_file(
            args.topology,
            args.output,
            rollout_port=args.rollout_port,
            gateway_port=args.gateway_port,
            router_port=args.router_port,
            host=args.host,
            rollout_url=args.rollout_url,
            gateway_url=args.gateway_url,
            router_url=args.router_url,
            operator_runtime_dir=args.operator_runtime_dir,
            op_assets_dir=args.op_assets_dir,
            rollout_results_dir=args.rollout_results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
