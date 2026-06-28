#!/usr/bin/env python3
"""Render a Polar topology for an isolated E2E run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def render_topology(
    topology: dict[str, Any],
    *,
    rollout_port: int,
    gateway_port: int,
    router_port: int,
    host: str = "127.0.0.1",
) -> dict[str, Any]:
    out = dict(topology)
    rollout = dict(out.get("rollout") or {})
    rollout["host"] = host
    rollout["port"] = rollout_port
    rollout["public_url"] = _url(host, rollout_port)
    out["rollout"] = rollout

    gateway = dict(out.get("gateway") or {})
    nodes = []
    for node in gateway.get("nodes") or []:
        item = dict(node)
        item["host"] = host
        item["port"] = gateway_port
        item["public_url"] = _url(host, gateway_port)
        inference = dict(item.get("inference") or {})
        inference["base_url"] = _url(host, router_port)
        item["inference"] = inference
        nodes.append(item)
    gateway["nodes"] = nodes
    out["gateway"] = gateway
    return out


def render_topology_file(
    topology_path: Path,
    output_path: Path,
    *,
    rollout_port: int,
    gateway_port: int,
    router_port: int,
    host: str = "127.0.0.1",
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
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-port", type=int, required=True)
    parser.add_argument("--gateway-port", type=int, required=True)
    parser.add_argument("--router-port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    print(
        render_topology_file(
            args.topology,
            args.output,
            rollout_port=args.rollout_port,
            gateway_port=args.gateway_port,
            router_port=args.router_port,
            host=args.host,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
