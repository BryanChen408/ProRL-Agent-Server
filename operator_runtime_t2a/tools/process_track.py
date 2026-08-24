#!/usr/bin/env python3
"""process_track — 算子生成 agent 的过程事件采集(process reward 数据源,agent 侧).

设计见 dev_docs/dev_04_process_reward_design.md / dev_05_process_reward_implementation.md。

每次固定评测入口(ascendc_eval_pipeline.sh)在 write_metrics() 落盘 metrics.json 之后
调用 `record-eval`:从 metrics.json 原样读出 success/ast_check_ok/correctness_ok/
error_type/speedup(与 reward 路径同源,逐字节一致 —— judge 侧 V3 校验依赖这一点),
推导 stage/substeps,追加进 process_info.json 的 events 数组,重算 summary,
原子写主副本 + 镜像副本($ARTIFACTS_DIR,直通宿主机 gateway session 目录)。

纪律(与 write_metrics 同款):
  - 采集端永远不能让固定入口失败 —— 一切异常吞掉,只向 stderr 打警告,exit 0;
  - 纯 stdlib,无第三方依赖(agent 容器 python 环境不可控);
  - judge 侧(AGENT_SIDE=0)不调用本脚本:judge 的评测是打分动作,不是被评分的过程。

命令:
  record-eval  --metrics METRICS_JSON --file PROCESS_INFO [--mirror PATH] [--op-name NAME]
  milestone    --name NAME [--status done] [--note NOTE] --file PROCESS_INFO [--mirror PATH]
               (遥测用,v1 不计分)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1

# 与 operator_reward.reward_from_metrics / judge 侧 validate 共用同一张推导表
# (两侧无依赖关系,靠单测锁定一致性;改这里必须同步改 operator_reward._terminal_from_booleans)。
def derive_stage_substeps(
    ast_ok: bool, corr_ok: bool, success: bool, error_type: str | None
) -> tuple[str, dict]:
    """write_metrics 四参数 -> (本次评测推进到的最深阶段, 四子步骤状态)。

    stage 取值: ast / compile / verify / benchmark / done,单调递进;
    substeps 取值: pass / fail / skip(前面挂了没轮到)。
    """
    if not ast_ok:
        return "ast", {"ast": "fail", "compile": "skip", "verify": "skip", "benchmark": "skip"}
    if not corr_ok:
        if str(error_type or "") == "ascendc_compile_failed":
            return "compile", {"ast": "pass", "compile": "fail", "verify": "skip", "benchmark": "skip"}
        return "verify", {"ast": "pass", "compile": "pass", "verify": "fail", "benchmark": "skip"}
    if not success:
        return "benchmark", {"ast": "pass", "compile": "pass", "verify": "pass", "benchmark": "fail"}
    return "done", {"ast": "pass", "compile": "pass", "verify": "pass", "benchmark": "pass"}


def _finite_nonneg(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return v


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key) or default)
    except (TypeError, ValueError):
        return default


def _load(path: Path) -> dict:
    """读旧文件;损坏则备份后从空开始(采集端绝不失败)。"""
    empty = {"schema_version": SCHEMA_VERSION, "events": [], "milestones": []}
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            data.setdefault("milestones", [])
            return data
    except Exception as exc:  # noqa: BLE001
        print(f"[process_track] WARN: {path} unreadable ({exc!r}), restarting", file=sys.stderr)
    try:
        backup = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        path.replace(backup)
    except Exception:  # noqa: BLE001
        pass
    return empty


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_all(main: Path, mirror: Path | None, data: dict) -> None:
    _atomic_write(main, data)
    if mirror is not None and mirror != main:
        try:
            _atomic_write(mirror, data)
        except Exception as exc:  # noqa: BLE001
            print(f"[process_track] WARN: mirror write failed ({exc!r})", file=sys.stderr)


def _summarize(events: list[dict]) -> dict:
    first_pass: dict[str, int | None] = {"ast": None, "compile": None, "verify": None, "benchmark": None}
    for ev in events:
        for name, state in (ev.get("substeps") or {}).items():
            if name in first_pass and state == "pass" and first_pass[name] is None:
                first_pass[name] = ev.get("global_step")
    success_steps = [e.get("global_step") for e in events if e.get("status") == "pass"]
    speedups = [
        (e.get("global_step"), _finite_nonneg(e.get("speedup_vs_torch")))
        for e in events
    ]
    speedups = [(s, v) for s, v in speedups if v is not None]
    best_step, best_speedup = max(speedups, key=lambda kv: kv[1]) if speedups else (None, None)
    # 与 operator_reward._consecutive_repeat_max 同口径:只数 fail 连击,
    # 连续 pass(optimization 刷 speedup)是有效尝试,不算打转。
    repeat_max, run, prev_key = 0, 0, None
    for ev in events:
        if ev.get("status") == "pass":
            run, prev_key = 0, None
            continue
        key = (ev.get("stage"), ev.get("error_type"))
        run = run + 1 if key == prev_key else 1
        prev_key = key
        repeat_max = max(repeat_max, run)
    gen = sum(1 for e in events if e.get("phase") == "generation")
    opt = sum(1 for e in events if e.get("phase") == "optimization")
    return {
        "eval_calls": len(events),
        "gen_calls": gen,
        "opt_calls": opt,
        "first_pass": first_pass,
        "first_success_step": min(success_steps) if success_steps else None,
        "best_speedup": best_speedup,
        "best_speedup_step": best_step,
        "consecutive_repeat_max": repeat_max,
        "entered_optimization": opt > 0,
    }


def cmd_record_eval(args: argparse.Namespace) -> None:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    ast_ok = bool(metrics.get("ast_check_ok", False))
    corr_ok = bool(metrics.get("correctness_ok", False))
    success = bool(metrics.get("success", False))
    error_type = metrics.get("error_type") or None
    stage, substeps = derive_stage_substeps(ast_ok, corr_ok, success, error_type)
    speedup = _finite_nonneg((metrics.get("perf_data") or {}).get("speedup_vs_torch"))

    path = Path(args.file)
    data = _load(path)
    events = data["events"]

    # phase 自包含推导(不依赖 pipeline 预算计数器 —— 它在脚本尾才 +1,此处读会差一):
    # 首个 pass 事件之前都算 generation,之后算 optimization,语义与 pipeline 一致。
    prior_success = any(e.get("status") == "pass" for e in events)
    phase = "optimization" if prior_success else "generation"
    phase_step = sum(1 for e in events if e.get("phase") == phase) + 1
    phase_limit = (
        _env_int("POLAR_OPT_PIPELINE_MAX", 3)
        if phase == "optimization"
        else _env_int("POLAR_GEN_PIPELINE_MAX", 6)
    )

    events.append(
        {
            "global_step": len(events) + 1,
            "kind": "eval",
            "phase": phase,
            "phase_step": phase_step,
            "phase_limit": phase_limit,
            "stage": stage,
            "status": "pass" if success else "fail",
            "error_type": error_type,
            "substeps": substeps,
            "speedup_vs_torch": speedup,
            "ts_unix": time.time(),
        }
    )
    data["op_name"] = data.get("op_name") or args.op_name or metrics.get("op_name")
    data["session_id"] = data.get("session_id") or os.environ.get("SESSION_ID") or None
    data["summary"] = _summarize(events)
    mirror = Path(args.mirror) if args.mirror else None
    _write_all(path, mirror, data)
    print(
        f"[process_track] step={events[-1]['global_step']} phase={phase} "
        f"({phase_step}/{phase_limit}) stage={stage} status={events[-1]['status']}"
    )


def cmd_milestone(args: argparse.Namespace) -> None:
    path = Path(args.file)
    data = _load(path)
    data["milestones"].append(
        {
            "global_step": len(data["events"]),
            "name": args.name,
            "status": args.status,
            "note": args.note or None,
            "ts_unix": time.time(),
        }
    )
    mirror = Path(args.mirror) if args.mirror else None
    _write_all(path, mirror, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="process event tracker (agent side)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("record-eval", help="append one eval event from metrics.json")
    p_eval.add_argument("--metrics", required=True)
    p_eval.add_argument("--file", required=True)
    p_eval.add_argument("--mirror", default="")
    p_eval.add_argument("--op-name", default="")

    p_ms = sub.add_parser("milestone", help="append a self-reported milestone (telemetry only)")
    p_ms.add_argument("--name", required=True)
    p_ms.add_argument("--status", default="done")
    p_ms.add_argument("--note", default="")
    p_ms.add_argument("--file", required=True)
    p_ms.add_argument("--mirror", default="")

    args = parser.parse_args()
    try:
        if args.cmd == "record-eval":
            cmd_record_eval(args)
        else:
            cmd_milestone(args)
    except Exception as exc:  # noqa: BLE001 — 采集端绝不拖累固定入口
        print(f"[process_track] WARN: {args.cmd} failed ({exc!r})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
