#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import secrets
import shlex
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _safe_run_id(value: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _run_id() -> str:
    return _safe_run_id(os.environ.get("POLAR_RUN_ID") or os.environ.get("RUN_ID") or "")


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _pool_spec(value: object, default: object) -> str:
    raw = default if value is None else value
    if isinstance(raw, (list, tuple)):
        return ",".join(str(item).strip() for item in raw if str(item).strip())
    return str(raw)


def _operator_runtime_dir(*, repo: Path, workflow: str, paths: dict) -> Path:
    if workflow == "cannbot":
        return (repo / "operator_runtime" / "cannbot").resolve()
    configured = paths.get("operator_runtime_dir", "operator_runtime")
    value = Path(configured)
    return value if value.is_absolute() else (repo / value).resolve()


def _repo_path(repo: Path, value: object) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (repo / path).resolve())


def _runtime_volumes(operator_runtime_dir: Path, workflow: str) -> list[str]:
    volumes = [f"{operator_runtime_dir}:/opt/canonical:ro"]
    tools_dir = operator_runtime_dir / "tools"
    if workflow == "legacy" or tools_dir.is_dir():
        volumes.append(f"{tools_dir}:/opt/workspace/agent_workdir/tools:ro")
    return volumes


def _operator_prepare(
    *,
    workflow: str,
    upload_source: str,
    workdir: str,
    require_claude: bool,
) -> list[dict]:
    if workflow == "cannbot":
        return [
            {
                "type": "upload_file",
                "source": upload_source,
                "target": f"{workdir}/input/{{op_name}}.py",
            },
            {
                "type": "exec",
                "cwd": workdir,
                "command": "mkdir -p input output && cp /opt/canonical/AGENTS.md CLAUDE.md && ln -sfn /polar/session/.claude .claude",
            },
        ]

    command = (
        "python3 /opt/canonical/runtime/prepare_operator_workdir.py "
        "--op-name {op_name} --workdir /opt/workspace/agent_workdir "
        "--no-stub --readonly-tools"
    )
    if require_claude:
        command += " --require-claude"
    return [
        {
            "type": "upload_file",
            "source": upload_source,
            "target": f"{workdir}/src/{{op_name}}.py",
        },
        {"type": "exec", "command": command},
    ]


def _evaluator_config(*, workflow: str, evaluator: dict, workdir: str) -> dict:
    if workflow == "cannbot":
        return {
            "lazy_refresh_runtime": True,
            "op_name": "{op_name}",
            "judge_mode": "cannbot",
            "task_path": "input/{op_name}.py",
            "cannbot_runtime_root": "/opt/canonical",
            "workdir": workdir,
        }
    return {
        "lazy_refresh_runtime": True,
        "op_name": "{op_name}",
        "judge_command": str(evaluator.get("judge_command")),
        "submission_path": str(evaluator.get("submission_path")),
        "metrics_path": str(evaluator.get("metrics_path")),
        "workdir": workdir,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = (repo / profile_path).resolve()
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}

    def path(value: str | Path) -> Path:
        value = Path(value)
        return value if value.is_absolute() else (repo / value).resolve()

    service = _mapping(profile.get("service"))
    paths = _mapping(profile.get("paths"))
    operator_runtime = _mapping(profile.get("operator_runtime"))
    legacy_budget = _mapping(profile.get("pipeline_budget"))
    budget = _mapping(operator_runtime.get("budget"))
    npu_lease = _mapping(operator_runtime.get("npu_lease"))
    observer = _mapping(profile.get("observer"))
    gateway = _mapping(profile.get("gateway"))
    completion_persistence = gateway.get("completion_persistence") or {}
    operator = _mapping(profile.get("operator"))
    runtime = _mapping(operator.get("runtime"))
    agent = _mapping(operator.get("agent"))
    evaluator = _mapping(operator.get("evaluator"))
    workflow = str(operator_runtime.get("workflow") or "legacy").strip().lower()
    if workflow not in {"legacy", "cannbot"}:
        raise SystemExit(f"unsupported operator_runtime.workflow: {workflow!r}")

    output_root = path(paths.get("output_dir", "output/ascend_operator"))
    run_id = _run_id()
    run_root_dir = path(paths.get("run_root_dir", output_root / "runs"))
    output_dir = path(paths.get("run_dir", run_root_dir / run_id))
    operator_runtime_dir = _operator_runtime_dir(repo=repo, workflow=workflow, paths=paths)
    log_dir = path(paths.get("log_dir", output_dir / "logs"))
    op_assets_dir = path(paths.get("op_assets_dir", output_root / "op_assets"))
    rollout_results_dir = path(paths.get("rollout_results_dir", output_dir / "rollout_results"))
    session_base_dir = path(paths.get("session_base_dir", output_dir / "polar_sessions"))
    run_artifact_dir = path(paths.get("run_artifact_dir", output_dir / "run_artifacts"))
    topology_path = path(paths.get("effective_topology", run_artifact_dir / "effective_topology.yaml"))
    for directory in (output_root, run_root_dir, output_dir, log_dir, op_assets_dir, rollout_results_dir, session_base_dir, run_artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rollout_url = str(service.get("rollout_url", "http://127.0.0.1:8080")).rstrip("/")
    gateway_url = str(service.get("gateway_url", "http://127.0.0.1:8100")).rstrip("/")
    router_url = str(service.get("sglang_router_url", "http://127.0.0.1:4077")).rstrip("/")
    bind_host = str(service.get("bind_host", "0.0.0.0"))
    rollout_port = urlparse(rollout_url).port or 8080
    gateway_port = urlparse(gateway_url).port or 8100
    profile_name = str(operator.get("profile", "operator_npu"))
    lease_enabled = bool(npu_lease.get("enabled", True))
    npu_pool = _pool_spec(npu_lease.get("pool"), runtime.get("npu_pool", "0"))
    npu_lock_dir = _repo_path(repo, npu_lease.get("lock_dir", runtime.get("npu_lock_dir", "/dev/shm/npu-locks")))
    gen_max = str(budget.get("generation_max", legacy_budget.get("generation_max", 6)))
    opt_max = str(budget.get("optimization_max", legacy_budget.get("optimization_max", 3)))
    watch_interval = str(budget.get("interval_seconds", legacy_budget.get("interval_seconds", 2)))
    max_tokens = str(agent.get("max_output_tokens", 32768))
    timeout_ms = str(agent.get("inference_timeout_ms", 14400000))

    runtime_env = {
        "DISABLE_AUTOUPDATER": "1",
        "API_TIMEOUT_MS": timeout_ms,
        "CLAUDE_CODE_MAX_RETRIES": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": max_tokens,
        "POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS": max_tokens,
        "POLAR_GEN_PIPELINE_MAX": gen_max,
        "POLAR_OPT_PIPELINE_MAX": opt_max,
    }
    if lease_enabled:
        runtime_env.update(
            {
                "POLAR_NPU_LEASE_POOL": npu_pool,
                "POLAR_NPU_LOCK_DIR": npu_lock_dir,
            }
        )
    eval_env = {
        "DISABLE_AUTOUPDATER": "1",
    }
    if lease_enabled:
        eval_env.update(
            {
                "POLAR_NPU_LEASE_POOL": npu_pool,
                "POLAR_NPU_LOCK_DIR": npu_lock_dir,
            }
        )
    volumes = _runtime_volumes(operator_runtime_dir, workflow)
    upload_source = str(op_assets_dir / "op_tasks" / "{op_name}.py")
    workdir = str(runtime.get("workdir", "/opt/workspace/agent_workdir"))
    prepare = _operator_prepare(
        workflow=workflow,
        upload_source=upload_source,
        workdir=workdir,
        require_claude=True,
    )
    eval_prepare = _operator_prepare(
        workflow=workflow,
        upload_source=upload_source,
        workdir=workdir,
        require_claude=False,
    )
    runtime_spec = {
        "backend": "docker",
        "image": str(runtime.get("image", "sandbox:v1")),
        "network": str(runtime.get("network", "host")),
        "workdir": workdir,
        "env": runtime_env,
        "kwargs": {"ascend": {"pool": npu_pool, "lock_dir": npu_lock_dir, "lease_at_start": False}, "volumes": volumes},
        "prepare": prepare,
        "eval_prepare": eval_prepare,
    }
    evaluator_runtime = {
        "backend": runtime_spec["backend"],
        "image": runtime_spec["image"],
        "network": runtime_spec["network"],
        "workdir": runtime_spec["workdir"],
        "env": eval_env,
        "kwargs": runtime_spec["kwargs"],
        "eval_prepare": eval_prepare,
    }
    topology = {
        "rollout": {
            "host": bind_host,
            "port": rollout_port,
            "public_url": rollout_url,
            "save_dir": str(rollout_results_dir),
            "default_operator_profile": profile_name,
            "operator_profiles": {
                profile_name: {
                    "timeout_seconds": float(operator.get("timeout_seconds", 3600.0)),
                    "operator_runtime_dir": str(operator_runtime_dir),
                    "runtime": runtime_spec,
                    "agent": {
                        "harness": "claude_code",
                        "model_name": str(agent.get("model_name", "claude-opus-4-5")),
                        "skills_path": "/opt/canonical/skills",
                        "settings": {
                            "max_turns": int(agent.get("max_turns", 45)),
                            "disallowed_tools": str(agent.get("disallowed_tools", "")),
                            "append_system_prompt": str(agent.get("append_system_prompt", "")),
                        },
                    },
                    "evaluator": {
                        "strategy": "operator_judge",
                        "refresh_runtime": True,
                        "runtime": evaluator_runtime,
                        "config": _evaluator_config(
                            workflow=workflow,
                            evaluator=evaluator,
                            workdir=str(runtime_spec["workdir"]),
                        ),
                    },
                    "builder": {"strategy": "prefix_merging", "config": {}},
                }
            },
        },
        "gateway": {
            "heartbeat_interval_seconds": 30,
            "rollout_server_url": rollout_url,
            "completion_persistence": {
                "enabled": bool(completion_persistence.get("enabled", True)),
                "max_field_bytes": int(completion_persistence.get("max_field_bytes", 64 * 1024 * 1024)),
                "queue_size": int(completion_persistence.get("queue_size", 4096)),
            },
            "nodes": [
                {
                    "id": str(gateway.get("node_id", "ascend-node-01")),
                    "host": bind_host,
                    "port": gateway_port,
                    "public_url": gateway_url,
                    "max_init_workers": int(gateway.get("max_init_workers", 8)),
                    "max_run_workers": int(gateway.get("max_run_workers", 32)),
                    "max_postrun_workers": int(gateway.get("max_postrun_workers", 32)),
                    "model_served": "",
                    "inference": {"engine": "sglang", "base_url": router_url},
                }
            ],
        },
    }
    topology_path.write_text(yaml.safe_dump(topology, sort_keys=False), encoding="utf-8")

    env = {
        "POLAR_PROFILE": profile_path,
        "POLAR_RUN_ID": run_id,
        "POLAR_OUTPUT_ROOT": output_root,
        "POLAR_OUTPUT_DIR": output_dir,
        "POLAR_LOG_DIR": log_dir,
        "POLAR_OP_ASSETS_DIR": op_assets_dir,
        "POLAR_ROLLOUT_RESULTS_DIR": rollout_results_dir,
        "POLAR_SESSION_BASE_DIR": session_base_dir,
        "POLAR_TOPOLOGY": topology_path,
        "POLAR_ROLLOUT_URL": rollout_url,
        "POLAR_GATEWAY_URL": gateway_url,
        "SGLANG_ROUTER_URL": router_url,
        "POLAR_OBSERVER_HOST": str(observer.get("host", "0.0.0.0")),
        "POLAR_OBSERVER_PORT": str(observer.get("port", 18088)),
        "POLAR_GEN_PIPELINE_MAX": gen_max,
        "POLAR_OPT_PIPELINE_MAX": opt_max,
        "POLAR_PIPELINE_WATCH_INTERVAL": watch_interval,
        "POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS": max_tokens,
        "POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS": str(int(timeout_ms) // 1000),
    }
    for key, value in env.items():
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
