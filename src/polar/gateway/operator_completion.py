"""Operator completion decisions reconstructed outside the agent filesystem.

Use the same captured call history as the budget watcher, but never promote a
cached/local task_state verdict into completion authority. This shares the
existing harness-history trust boundary; it is not a new kernel correctness judge.
"""
from __future__ import annotations

import math
import shlex
from pathlib import PurePosixPath
from typing import Any

from polar.trajectory.builder.attempt_spans import is_pipeline_invocation, parse_verdict
from polar.trajectory.evaluator.operator_reward import INFRA_ERROR_TYPES
from polar.trajectory.pipeline_budget import analyze_budget


def completion_state(
    session_id: str, record: dict[str, Any], *, workdir: str,
    generation_max: int, optimization_max: int, perf_target: float, op_name: str,
) -> dict[str, Any]:
    if type(generation_max) is not int or generation_max < 0:
        raise ValueError("generation_max must be a nonnegative integer")
    if type(optimization_max) is not int or optimization_max < 0:
        raise ValueError("optimization_max must be a nonnegative integer")
    if isinstance(perf_target, bool) or not math.isfinite(perf_target) or perf_target <= 0:
        raise ValueError("perf_target must be finite and positive")
    if not op_name:
        raise ValueError("operator name is required")
    budget = analyze_budget(session_id, record)
    best = None
    generation_used = 0
    optimization_used = 0
    last_metrics = None
    pending = False
    for call in budget.pipeline_calls:
        # Restrict the shared command recognizer to this runtime's actual tools
        # mount; a copied /tmp/tools script is not the fixed evaluation entry.
        if not is_pipeline_invocation(call.command):
            continue
        command = call.command.replace("\\\r\n", " ").replace("\\\n", " ")
        lexer = shlex.shlex(command.replace("\n", ";"), posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
        scripts = [t for t in tokens if t.endswith("/ascendc_eval_pipeline.sh")]
        if len(scripts) != 1:
            continue
        # The shared recognizer permits only literal absolute cd prefixes.
        # Resolve relative scripts from the last cd; absolute scripts ignore it.
        cwd = workdir
        for i in range(tokens.index(scripts[0])):
            if tokens[i] == "cd" and (i == 0 or tokens[i - 1] in {"&&", ";"}):
                cwd = tokens[i + 1]
        if PurePosixPath(cwd, scripts[0]) != PurePosixPath(workdir, "tools/ascendc_eval_pipeline.sh"):
            continue
        if "--op_name" not in tokens or tokens[tokens.index("--op_name") + 1] != op_name:
            continue
        if call.cached:
            continue  # metrics.json can be edited before a cache hit.
        if best is not None:
            optimization_used += 1
        else:
            generation_used += 1
        metrics = parse_verdict(call.result)
        pending = metrics is None
        if metrics is None:
            continue
        last_metrics = metrics
        if metrics.get("success") and metrics.get("correctness_ok"):
            speedup = metrics["perf_data"]["speedup_vs_torch"]
            if math.isfinite(speedup) and speedup > 0:
                best = max(best or 0.0, speedup)

    target_met = best is not None and best >= perf_target
    exhausted = best is not None and optimization_used >= optimization_max
    generation_exhausted = best is None and generation_used >= generation_max
    infra = (last_metrics or {}).get("error_type") in INFRA_ERROR_TYPES
    complete = not pending and (target_met or exhausted)
    # A failed generation must use its remaining budget. Missing/unfinished
    # evidence cannot release the gate, even when the last call reached the cap.
    allowed = (not pending and last_metrics is not None
               and (complete or infra or generation_exhausted))
    reason = (
        "pipeline_evidence_pending" if pending or last_metrics is None else
        "target_met" if target_met else "budget_exhausted" if exhausted else
        "infra" if infra else "generation_budget_exhausted" if generation_exhausted else
        "pending_generation" if best is None else "pending_optimization"
    )
    remaining = max(0, optimization_max - optimization_used)
    generation_remaining = max(0, generation_max - generation_used)
    phase = "generation" if best is None else "optimization"
    return {
        "source": "gateway_completion_history",
        "session_id": session_id,
        "operator_valid": best is not None,
        "task_complete": complete,
        "stop_allowed": allowed,
        "completion_reason": reason,
        "perf_data": {"speedup_vs_torch": best},
        "next_step": {
            "phase": phase,
            "generation_budget": generation_max,
            "generation_used": generation_used,
            "generation_remaining": generation_remaining,
            "perf_target_speedup": perf_target, "target_met": target_met,
            "optimization_budget": optimization_max,
            "optimization_used": optimization_used,
            "optimization_remaining": remaining,
        },
        "reason": (
            f"Gateway 完成检查: {reason}; best={best}x, target={perf_target}x, "
            f"{phase}_remaining={generation_remaining if best is None else remaining}。"
            "本地 task_state/metrics/budget 修改不影响此判定。"
            + ("继续按固定入口结果诊断并实质修改源码，不要重复提交总结。" if not allowed else "")
        ),
    }
