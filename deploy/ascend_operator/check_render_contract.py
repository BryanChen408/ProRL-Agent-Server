#!/usr/bin/env python3
"""Validate the Polar DockerRuntime render contract without starting services."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

SAFE_OP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXCLUDED_MAINLINE_TERMS = (
    "polar_faith",
    "faithfulness",
    "rllm",
    "LocalRuntime",
    "deploy_dood_B",
    "polar-op-image",
)


def _load_render_topology_file():
    path = Path(__file__).resolve().parent / "tools" / "render_run_topology.py"
    spec = importlib.util.spec_from_file_location("polar_render_run_topology", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load render_run_topology.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_topology_file


def _parse_pool_count(spec: str) -> int:
    text = str(spec or "").strip()
    if not text:
        return 0
    if "-" in text and "," not in text:
        lo, hi = text.split("-", 1)
        return int(hi) - int(lo) + 1
    return len([part for part in text.split(",") if part.strip()])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser.add_argument("--polar-root", type=Path, default=repo_root)
    parser.add_argument("--config", type=Path, default=here / "polar_config.yaml")
    parser.add_argument("--topology", type=Path, default=here / "topology.yaml")
    parser.add_argument("--skills-dir", type=Path, default=repo_root / "operator_runtime")
    parser.add_argument("--readonly-tools-dir", type=Path, default=repo_root / "operator_runtime" / "tools")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task-jsonl", type=Path, required=True)
    parser.add_argument("--image", default="sandbox:v1")
    parser.add_argument("--device-pool", default="0")
    parser.add_argument("--eval-device-pool", default=None)
    parser.add_argument("--lock-dir", default="/dev/shm/npu-locks")
    parser.add_argument("--model-served", default="model-served-placeholder")
    parser.add_argument("--router-ip", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=4077)
    return parser.parse_args(argv)


def _load_first_task(task_jsonl: Path) -> dict[str, Any]:
    if not task_jsonl.is_file():
        raise SystemExit(f"missing task jsonl: {task_jsonl}")
    for line in task_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    raise SystemExit(f"empty task jsonl: {task_jsonl}")


def _assert_safe_op(row: dict[str, Any], tasks_dir: Path) -> str:
    metadata = row.get("metadata") or {}
    op = metadata.get("op_name")
    if not isinstance(op, str) or not SAFE_OP_NAME_RE.fullmatch(op):
        raise SystemExit(f"unsafe/missing op_name in first row: {op!r}")
    task = tasks_dir / f"{op}.py"
    if not task.is_file():
        raise SystemExit(f"missing task file for first row: {task}")
    return op


def _assert_payload_contract(
    payload: dict[str, Any],
    *,
    image: str,
    skills_dir: Path,
    readonly_tools_dir: Path,
    pool: str,
    eval_pool: str,
    lock_dir: str,
) -> None:
    if image != "sandbox:v1":
        raise SystemExit(f"mainline runtime image must be sandbox:v1, got {image!r}")

    runtime = payload.get("runtime") or {}
    if runtime.get("backend") != "docker":
        raise SystemExit(f"runtime.backend must be docker, got {runtime.get('backend')!r}")
    if runtime.get("image") != "sandbox:v1":
        raise SystemExit(f"runtime.image must be sandbox:v1, got {runtime.get('image')!r}")

    kwargs = runtime.get("kwargs") or {}
    expected_ascend = {"pool": pool, "lock_dir": lock_dir, "lease_at_start": False}
    if kwargs.get("ascend") != expected_ascend:
        raise SystemExit(f"unexpected runtime.kwargs.ascend: {kwargs.get('ascend')!r}")
    runtime_env = runtime.get("env") or {}
    if runtime_env.get("POLAR_NPU_LEASE_POOL") != pool:
        raise SystemExit(
            f"runtime.env.POLAR_NPU_LEASE_POOL must be {pool!r}, got {runtime_env.get('POLAR_NPU_LEASE_POOL')!r}"
        )
    if runtime_env.get("POLAR_NPU_LOCK_DIR") != lock_dir:
        raise SystemExit(
            f"runtime.env.POLAR_NPU_LOCK_DIR must be {lock_dir!r}, got {runtime_env.get('POLAR_NPU_LOCK_DIR')!r}"
        )
    expected_volumes = [
        f"{skills_dir}:/opt/canonical:ro",
        f"{readonly_tools_dir}:/opt/workspace/agent_workdir/tools:ro",
    ]
    if kwargs.get("volumes") != expected_volumes:
        raise SystemExit(f"unexpected runtime.kwargs.volumes: {kwargs.get('volumes')!r}")
    prepare = runtime.get("prepare") or []
    eval_prepare = runtime.get("eval_prepare") or []
    prepare_command = " ".join(str(item.get("command", "")) for item in prepare)
    eval_prepare_command = " ".join(str(item.get("command", "")) for item in eval_prepare)
    if (
        "prepare_operator_workdir.py" not in prepare_command
        or "--require-claude" not in prepare_command
        or "--no-stub" not in prepare_command
    ):
        raise SystemExit(
            "agent prepare must use prepare_operator_workdir.py --require-claude --no-stub: "
            f"{prepare_command!r}"
        )
    if "prepare_operator_workdir.py" not in eval_prepare_command or "--no-stub" not in eval_prepare_command:
        raise SystemExit(f"eval_prepare must use prepare_operator_workdir.py --no-stub: {eval_prepare_command!r}")
    if "--readonly-tools" not in prepare_command or "--readonly-tools" not in eval_prepare_command:
        raise SystemExit("agent runtime prepare/eval_prepare must use --readonly-tools")

    agent = payload.get("agent") or {}
    if agent.get("harness") != "claude_code":
        raise SystemExit(f"agent.harness must be claude_code, got {agent.get('harness')!r}")
    if agent.get("skills_path") != "/opt/canonical/skills":
        raise SystemExit(f"agent.skills_path must be /opt/canonical/skills, got {agent.get('skills_path')!r}")
    settings = agent.get("settings") or {}
    try:
        max_turns = int(settings.get("max_turns"))
    except (TypeError, ValueError):
        raise SystemExit(f"agent.settings.max_turns must be a positive integer, got {settings.get('max_turns')!r}")
    if max_turns <= 0:
        raise SystemExit(f"agent.settings.max_turns must be a positive integer, got {settings.get('max_turns')!r}")
    append_prompt = str(settings.get("append_system_prompt", ""))
    if "Follow ./CLAUDE.md" not in append_prompt or "operator name and src/output paths" not in append_prompt:
        raise SystemExit(f"agent append_system_prompt must stay minimal and delegate workflow to CLAUDE.md, got {append_prompt!r}")
    blocked_prompt_terms = ("Phase 2", "Phase 3", "canonical pipeline", "Do NOT edit")
    leaked_prompt_terms = [term for term in blocked_prompt_terms if term in append_prompt]
    if leaked_prompt_terms:
        raise SystemExit(f"agent append_system_prompt leaked control/hardening terms: {leaked_prompt_terms}")
    banned_tools = set(str(settings.get("disallowed_tools", "")).split())
    if "Agent" in banned_tools or "Workflow" in banned_tools:
        raise SystemExit("sub-agent smoke config must not ban Agent or Workflow")
    required_plan_bans = {
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TaskUpdate",
        "TodoWrite",
    }
    missing_plan_bans = sorted(required_plan_bans - banned_tools)
    if missing_plan_bans:
        raise SystemExit(f"plan/task tools must be banned in smoke config, missing: {missing_plan_bans}")

    evaluator = payload.get("evaluator") or {}
    if evaluator.get("strategy") != "operator_judge":
        raise SystemExit(f"evaluator.strategy must be operator_judge, got {evaluator.get('strategy')!r}")
    if evaluator.get("refresh_runtime") is not True:
        raise SystemExit("operator_judge must set refresh_runtime=true")
    evaluator_config = evaluator.get("config") or {}
    if evaluator_config.get("lazy_refresh_runtime") is not True:
        raise SystemExit(
            "operator_judge must set config.lazy_refresh_runtime=true to start fresh judge at postrun"
        )
    eval_runtime = evaluator.get("runtime") or {}
    if eval_runtime.get("backend") != "docker":
        raise SystemExit(f"evaluator.runtime.backend must be docker, got {eval_runtime.get('backend')!r}")
    if eval_runtime.get("image") != "sandbox:v1":
        raise SystemExit(f"evaluator.runtime.image must be sandbox:v1, got {eval_runtime.get('image')!r}")
    eval_kwargs = eval_runtime.get("kwargs") or {}
    expected_eval_ascend = {"pool": eval_pool, "lock_dir": lock_dir, "lease_at_start": False}
    if eval_kwargs.get("ascend") != expected_eval_ascend:
        raise SystemExit(f"unexpected evaluator.runtime.kwargs.ascend: {eval_kwargs.get('ascend')!r}")
    eval_env = eval_runtime.get("env") or {}
    if eval_env.get("POLAR_NPU_LEASE_POOL") != eval_pool:
        raise SystemExit(
            "evaluator.runtime.env.POLAR_NPU_LEASE_POOL must be "
            f"{eval_pool!r}, got {eval_env.get('POLAR_NPU_LEASE_POOL')!r}"
        )
    if eval_env.get("POLAR_NPU_LOCK_DIR") != lock_dir:
        raise SystemExit(
            "evaluator.runtime.env.POLAR_NPU_LOCK_DIR must be "
            f"{lock_dir!r}, got {eval_env.get('POLAR_NPU_LOCK_DIR')!r}"
        )
    if eval_kwargs.get("volumes") != expected_volumes:
        raise SystemExit(f"unexpected evaluator.runtime.kwargs.volumes: {eval_kwargs.get('volumes')!r}")
    eval_prepare = eval_runtime.get("eval_prepare") or []
    eval_runtime_prepare_command = " ".join(str(item.get("command", "")) for item in eval_prepare)
    if "prepare_operator_workdir.py" not in eval_runtime_prepare_command or "--no-stub" not in eval_runtime_prepare_command:
        raise SystemExit(
            "evaluator runtime eval_prepare must use prepare_operator_workdir.py --no-stub: "
            f"{eval_runtime_prepare_command!r}"
        )
    if "--readonly-tools" not in eval_runtime_prepare_command:
        raise SystemExit("evaluator runtime eval_prepare must use --readonly-tools")
    builder = payload.get("builder") or {}
    if builder.get("strategy") != "prefix_merging":
        raise SystemExit(f"builder.strategy must default to prefix_merging, got {builder.get('strategy')!r}")
    if not isinstance(builder.get("config"), dict):
        raise SystemExit(f"builder.config must render as a dict, got {type(builder.get('config')).__name__}")

    blob = json.dumps(payload, sort_keys=True)
    leaked = [term for term in EXCLUDED_MAINLINE_TERMS if term in blob]
    if leaked:
        raise SystemExit(f"excluded mainline term leaked into payload: {leaked}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, str(args.polar_root / "src"))

    from polar.config import TopologyConfig
    from polar.rollout.models import OperatorSample, OperatorSampleRequest, TaskRequest
    from polar.rollout.operator_profile import expand_operator_sample_request
    from slime_bridge._messages import prompt_to_instruction_text
    from slime_bridge.config import (
        render_instruction,
        render_task_payload,
        render_topology_template,
        resolve_polar_slime_config,
    )

    render_topology_file = _load_render_topology_file()
    with tempfile.TemporaryDirectory(prefix="polar-render-contract.") as tmp:
        rendered_topology_path = render_topology_file(
            args.topology,
            Path(tmp) / "topology.rendered.yaml",
            router_url=f"http://{args.router_ip}:{args.router_port}",
            operator_runtime_dir=args.skills_dir,
            op_assets_dir=args.tasks_dir.parent,
            rollout_results_dir=Path(tmp) / "rollout_results",
        )
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cfg["polar_topology_path"] = str(rendered_topology_path)
        cfg.update(
            polar_skills_dir=str(args.skills_dir),
            polar_readonly_tools_dir=str(args.readonly_tools_dir),
            operator_tasks_dir=str(args.tasks_dir),
            polar_op_image=args.image,
            polar_device_pool=args.device_pool,
            polar_eval_device_pool=args.eval_device_pool or args.device_pool,
            polar_lock_dir=args.lock_dir,
            polar_model_served_name=args.model_served,
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            update_weights_interval=1,
            hf_checkpoint=args.model_served,
            sglang_router_ip=args.router_ip,
            sglang_router_port=args.router_port,
        )
        runtime_args = SimpleNamespace(**cfg)
        config = resolve_polar_slime_config(runtime_args)

        row = _load_first_task(args.task_jsonl)
        op = _assert_safe_op(row, args.tasks_dir)
        sample = SimpleNamespace(
            prompt=row["prompt"],
            response="",
            label=row.get("label"),
            metadata=row.get("metadata") or {},
            index=0,
            group_index=0,
            status=None,
        )
        instruction = render_instruction(
            args=runtime_args,
            config=config,
            sample=sample,
            prompt_text=prompt_to_instruction_text(sample.prompt),
            rollout_id=7,
            task_position=0,
            num_rollouts=2,
        )
        payload = render_task_payload(
            args=runtime_args,
            config=config,
            sample=sample,
            instruction=instruction,
            rollout_id=7,
            task_position=0,
            num_rollouts=2,
        )
        if config.submit_mode == "operator_samples":
            topology = TopologyConfig.load(rendered_topology_path)
            profile_request = OperatorSampleRequest(
                task_id=str(payload["task_id"]),
                instruction=str(payload["instruction"]),
                num_samples=int(payload.get("num_samples") or 1),
                profile=config.operator_profile,
                timeout_seconds=payload.get("timeout_seconds"),
                sample=OperatorSample(
                    op_name=op,
                    group_index=sample.group_index,
                    index=sample.index,
                    metadata=sample.metadata,
                ),
                metadata=payload.get("metadata") or {},
            )
            payload = expand_operator_sample_request(profile_request, topology.rollout).model_dump(mode="python")
        else:
            TaskRequest(**payload)

        expected_pool = args.device_pool
        expected_eval_pool = args.eval_device_pool or args.device_pool
        if config.submit_mode == "operator_samples":
            runtime_ascend = ((payload.get("runtime") or {}).get("kwargs") or {}).get("ascend") or {}
            eval_ascend = (
                (((payload.get("evaluator") or {}).get("runtime") or {}).get("kwargs") or {})
                .get("ascend")
                or {}
            )
            expected_pool = str(runtime_ascend.get("pool") or expected_pool)
            expected_eval_pool = str(eval_ascend.get("pool") or expected_eval_pool)
        _assert_payload_contract(
            payload,
            image=args.image,
            skills_dir=args.skills_dir,
            readonly_tools_dir=args.readonly_tools_dir,
            pool=expected_pool,
            eval_pool=expected_eval_pool,
            lock_dir=args.lock_dir,
        )

        TopologyConfig.load(rendered_topology_path)
        rendered = render_topology_template(rendered_topology_path, runtime_args)
        expected_base_url = f"http://{args.router_ip}:{args.router_port}"
        eval_pool_count = _parse_pool_count(args.eval_device_pool or args.device_pool)
        for node in rendered["gateway"]["nodes"]:
            if node["model_served"] != args.model_served:
                raise SystemExit(f"topology model_served mismatch: {node['model_served']!r}")
            max_run_workers = int(node.get("max_run_workers") or 0)
            max_postrun_workers = int(node.get("max_postrun_workers") or 0)
            if max_run_workers < 2:
                raise SystemExit(
                    f"pipeline-lease topology must allow logical session concurrency; "
                    f"max_run_workers must be >= 2 for this fixture, got {max_run_workers!r}"
                )
            if eval_pool_count and max_run_workers <= eval_pool_count:
                raise SystemExit(
                    "pipeline-lease topology must not tie max_run_workers to eval card count; "
                    f"got max_run_workers={max_run_workers!r}, eval_pool_count={eval_pool_count!r}"
                )
            if max_postrun_workers < max_run_workers:
                raise SystemExit(
                    "pipeline-lease topology should not serialize postrun by eval card count; "
                    f"max_postrun_workers must be >= max_run_workers, got "
                    f"{max_postrun_workers!r} < {max_run_workers!r}"
                )
            if node["inference"] != {"engine": "sglang", "base_url": expected_base_url}:
                raise SystemExit(f"topology inference mismatch: {node['inference']!r}")
            if node["inference"]["base_url"].endswith("/v1"):
                raise SystemExit("topology inference base_url must not include /v1")

        print(f"task_request=ok op={op}")
        print("topology=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
