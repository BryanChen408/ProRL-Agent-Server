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


def _runtime_volumes(operator_runtime_dir: Path, workflow: str,
                     asc_devkit_dir: str | None = None) -> list[str]:
    volumes = [f"{operator_runtime_dir}:/opt/canonical:ro"]
    tools_dir = operator_runtime_dir / "tools"
    if workflow == "legacy" or tools_dir.is_dir():
        volumes.append(f"{tools_dir}:/opt/workspace/agent_workdir/tools:ro")
    # asc-devkit:上游 init.sh Step 4 会 clone 的算子开发资料仓($ASC_DEVKIT_DIR)。
    # 只读挂载而非拷进 workdir —— 97M,每 session 拷一份不划算,上游那边也是一份 clone 挂着。
    if asc_devkit_dir:
        volumes.append(f"{asc_devkit_dir}:/opt/asc-devkit:ro")
    return volumes


def _operator_prepare(
    *,
    workflow: str,
    upload_source: str,
    workdir: str,
    require_claude: bool,
    backend: str = "triton",
    task_assets_dir: str | None = None,
    only_project_skills: bool = False,
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
    if backend == "ascendc":
        command += " --backend ascendc"
        if only_project_skills:
            # 规则:非本项目(canonical/skills)提供的 CLI 自带 skill 一律关掉 —— prepare 据此
            # 写 .claude/settings.json 的 skillOverrides={name: off}。名单在 prepare 里(CLI 无通配符),
            # 不配 agent.only_project_skills = 不加这个参数 = prepare 不写文件,行为一字不变。
            command += " --only-project-skills"
    if require_claude:
        command += " --require-claude"
    target = (
        f"{workdir}/input/{{op_name}}.py" if backend == "ascendc"
        else f"{workdir}/src/{{op_name}}.py"
    )
    actions: list[dict] = [
        {"type": "upload_file", "source": upload_source, "target": target},
    ]
    if backend == "ascendc" and task_assets_dir:
        # NPUKernelBench 的 model.py 用 get_input_groups() 读**同名 .json**(用例规格),
        # 必须与 {op}.py 并排落在 input/。{op}.py 走 vime 的 sample.task_source(内容寻址
        # 缓存,polar 会把这条 upload 的 source 改写成 cache 路径);.json 没有这条通道,
        # 直接从数据集目录上传 —— polar 的 _is_operator_task_upload_action 只匹配 .py,
        # 不会改写这条。triton 侧不配 task_assets_dir,动作列表一字不变。
        actions.append(
            {
                "type": "upload_file",
                "source": f"{task_assets_dir.rstrip('/')}/{{op_name}}.json",
                "target": f"{workdir}/input/{{op_name}}.json",
            }
        )
    actions.append({"type": "exec", "command": command})
    return actions


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
    config = {
        "lazy_refresh_runtime": True,
        "op_name": "{op_name}",
        "judge_command": str(evaluator.get("judge_command")),
        "submission_path": str(evaluator.get("submission_path")),
        "metrics_path": str(evaluator.get("metrics_path")),
        "workdir": workdir,
    }
    # submission_candidates(operator_judge.py:76 的 EvaluatorSpec.config 字段):按顺序取第一个
    # 存在的文件跨 fresh-judge 边界。ascendc 用它实现"优先取历史最优包(.best.tar.gz)",
    # 不透传的话 profile 里配了也等于没配(judge 永远只评最后一次打的包)。
    # triton 的 profile 不配这个键 → 不产生该字段,行为一字不变。
    candidates = [str(c).strip() for c in (evaluator.get("submission_candidates") or []) if str(c).strip()]
    if candidates:
        config["submission_candidates"] = candidates
    return config


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
    if workflow not in {"legacy", "cannbot", "task_request"}:
        raise SystemExit(f"unsupported operator_runtime.workflow: {workflow!r}")
    # backend: triton(默认,原样)| ascendc(prepare 上传 input/{op}.py + --backend ascendc)
    backend = str(operator_runtime.get("backend") or "triton").strip().lower()
    # task_request: the trainer submits self-contained tasks (runtime + agent + evaluator) via
    # --polar-task-template, so the server runs a BARE gateway+inference topology with no baked
    # operator. Used by the SWE-Gym coding-agent (codex) pipeline; see profile.swe-8b.yaml.
    task_request = workflow == "task_request"

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
    # profile 显式追加/覆盖 agent 容器 env(operator.runtime.env);profile 不配=一字不变。
    runtime_env.update(
        {str(k): str(v) for k, v in _mapping(runtime.get("env")).items()}
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
    # asc-devkit 宿主机路径(profile 不配=不挂载,行为一字不变)
    asc_devkit_dir = str(operator_runtime.get("asc_devkit_dir") or "").strip() or None
    if asc_devkit_dir:
        asc_devkit_dir = _repo_path(repo, asc_devkit_dir)
    volumes = _runtime_volumes(operator_runtime_dir, workflow, asc_devkit_dir)
    upload_source = str(op_assets_dir / "op_tasks" / "{op_name}.py")
    workdir = str(runtime.get("workdir", "/opt/workspace/agent_workdir"))
    # ascendc 专用:算子同名 .json(用例规格)所在的数据集目录;triton 侧不配=不产生该动作
    task_assets_dir = str(operator_runtime.get("task_assets_dir") or "").strip() or None
    # ascendc 专用:只保留 canonical/skills 的 skill(CLI 自带的关掉);triton 侧不配=不产生该参数
    only_project_skills = bool(agent.get("only_project_skills"))
    prepare = _operator_prepare(
        workflow=workflow,
        upload_source=upload_source,
        workdir=workdir,
        require_claude=True,
        backend=backend,
        task_assets_dir=task_assets_dir,
        only_project_skills=only_project_skills,
    )
    eval_prepare = _operator_prepare(
        workflow=workflow,
        upload_source=upload_source,
        workdir=workdir,
        require_claude=False,
        backend=backend,
        task_assets_dir=task_assets_dir,
        only_project_skills=only_project_skills,
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
    # rollout block: bare (host/port/url/save_dir) for task_request; operator_samples adds the
    # baked default_operator_profile + operator_profiles (runtime/agent/evaluator).
    rollout_cfg: dict = {
        "host": bind_host,
        "port": rollout_port,
        "public_url": rollout_url,
        "save_dir": str(rollout_results_dir),
    }
    # skills_path:polar 的 claude_code harness 把它拷进 agent HOME($CLAUDE_CONFIG_DIR/skills)
    # = **用户级**安装点。ascendc 的 prepare 已把同一份拷进 workdir/.claude/skills(**项目级**,
    # CLAUDE.md 与任务 prompt 里的命令走相对路径 `.claude/skills/...`,必须有这份)。两处都装
    # → 同一个 skill 在 prompt 里列两遍(会话无 MCP 时 CLI 不按 name 去重)。profile 显式写
    # `skills_path: ""` 即只留项目级那份;不写该字段=保持原值(triton 的 prepare 不拷 skills,靠它)。
    skills_path = (
        str(agent.get("skills_path") or "").strip()
        if "skills_path" in agent
        else "/opt/canonical/skills"
    )
    agent_block: dict = {
        "harness": "claude_code",
        "model_name": str(agent.get("model_name", "claude-opus-4-5")),
    }
    if skills_path:  # 键序保持与原来一致(harness/model_name/skills_path/settings)
        agent_block["skills_path"] = skills_path
    agent_block["settings"] = {
        "max_turns": int(agent.get("max_turns", 45)),
        "disallowed_tools": str(agent.get("disallowed_tools", "")),
        "append_system_prompt": str(agent.get("append_system_prompt", "")),
    }
    if not task_request:
        rollout_cfg["default_operator_profile"] = profile_name
        rollout_cfg["operator_profiles"] = {
            profile_name: {
                "timeout_seconds": float(operator.get("timeout_seconds", 3600.0)),
                "operator_runtime_dir": str(operator_runtime_dir),
                "runtime": runtime_spec,
                "agent": agent_block,
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
        }
    topology = {
        "rollout": rollout_cfg,
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
                    "model_served": str(service.get("model_served", "")),
                    "inference": {
                        "engine": str(service.get("inference_engine", "sglang")),
                        "base_url": router_url,
                    },
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
