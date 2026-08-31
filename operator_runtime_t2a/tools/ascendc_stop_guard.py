#!/usr/bin/env python3
"""Claude Code Stop hook for unfinished AscendC optimization.

The hook is intentionally read-only: the fixed evaluation pipeline owns task state;
this script only prevents an end turn while that state says useful budget remains.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# Keep aligned with operator_reward.INFRA_ERROR_TYPES.  The Stop hook runs inside the
# isolated operator container and must not import the trainer package.
_INFRA_ERROR_TYPES = frozenset(
    {
        "task_missing",
        "input_load_failed",
        "judge_container_failed",
        "judge_metrics_unreadable",
        "judge_no_metrics",
        "npu_runtime_unavailable",
        "submission_fetch_failed",
        "profiler_unavailable",
        "judge_classification_failed",
    }
)


def _hook_input() -> dict[str, Any]:
    try:
        value = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _project_dir(payload: dict[str, Any]) -> Path:
    configured = os.environ.get("CLAUDE_PROJECT_DIR")
    if configured:
        return Path(configured)
    cwd = payload.get("cwd")
    return Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_metrics(project_dir: Path) -> dict[str, Any] | None:
    return _load_json(project_dir / "judge_out" / "metrics.json")


def _load_task_state(project_dir: Path) -> dict[str, Any] | None:
    return _load_json(project_dir / "judge_out" / "task_state.json")


def _block_message(
    state: dict[str, Any], metrics: dict[str, Any] | None = None
) -> str | None:
    # INFRA means the evaluator could not establish operator behaviour.  Blocking here would
    # make the agent edit a good kernel or spin on unavailable hardware; let Polar retry it.
    current_error = str((metrics or {}).get("error_type") or "")
    if current_error in _INFRA_ERROR_TYPES:
        return None
    next_step = state.get("next_step")
    if not isinstance(next_step, dict):
        return None
    remaining = _positive_int(next_step.get("optimization_remaining"))
    if not (
        state.get("operator_valid") is True
        and state.get("task_complete") is False
        and next_step.get("target_met") is False
        and remaining is not None
    ):
        return None

    perf = state.get("perf_data")
    speedup = perf.get("speedup_vs_torch") if isinstance(perf, dict) else None
    target = next_step.get("perf_target_speedup")
    action = next_step.get("action")
    detail = (
        f"当前 speedup={speedup}x，目标={target}x，optimization 还剩 {remaining} 次。"
    )
    if metrics is not None and metrics.get("success") is False and current_error:
        detail += (
            f" 当前优化候选失败(error_type={current_error})；先按固定入口给出的错误分类"
            "直接 Read 对应文档并修复或回退，再重新评测。历史 .best 不受影响。"
        )
    elif isinstance(action, str) and action.strip():
        detail += f" 下一步：{action.strip()}"
    return (
        "固定入口已确认 operator_valid=true，但 task_complete=false；目标未达时不能结束会话。"
        + detail
    )


def main() -> int:
    payload = _hook_input()
    project_dir = _project_dir(payload)
    metrics = _load_metrics(project_dir)
    # task_state survives a broken optimization candidate overwriting metrics.json.  Old
    # workdirs without it retain the original metrics-only behaviour.
    state = _load_task_state(project_dir) or metrics
    message = _block_message(state, metrics) if state is not None else None
    if message:
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": message,
                    "systemMessage": "继续优化并重新运行固定入口；只有 task_complete=true 才能结束。",
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
