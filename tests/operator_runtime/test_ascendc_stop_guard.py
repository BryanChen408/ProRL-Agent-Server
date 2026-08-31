"""AscendC task_complete Stop hook and project-settings wiring."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "operator_runtime_t2a" / "tools" / "ascendc_stop_guard.py"
PREPARE = ROOT / "operator_runtime_t2a" / "runtime" / "prepare_operator_workdir.py"


def _run_guard(
    tmp_path: Path,
    metrics: dict | None,
    *,
    task_state: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    if metrics is not None:
        judge_out = tmp_path / "judge_out"
        judge_out.mkdir(parents=True, exist_ok=True)
        (judge_out / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
        )
    if task_state is not None:
        judge_out = tmp_path / "judge_out"
        judge_out.mkdir(parents=True, exist_ok=True)
        (judge_out / "task_state.json").write_text(
            json.dumps(task_state, ensure_ascii=False), encoding="utf-8"
        )
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps({"stop_hook_active": False, "cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _metrics(*, task_complete: bool, target_met: bool, remaining: int) -> dict:
    return {
        "success": True,
        "operator_valid": True,
        "task_complete": task_complete,
        "completion_reason": "target_met" if target_met else "pending_optimization",
        "perf_data": {"speedup_vs_torch": 0.859},
        "next_step": {
            "perf_target_speedup": 1.1,
            "target_met": target_met,
            "optimization_remaining": remaining,
            "action": "profile, edit, and re-run",
        },
    }


def test_guard_blocks_end_while_target_unmet_and_budget_remains(tmp_path):
    proc = _run_guard(
        tmp_path,
        _metrics(task_complete=False, target_met=False, remaining=3),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "block"
    assert "task_complete=false" in payload["reason"]
    assert "还剩 3 次" in payload["reason"]


def test_guard_keeps_blocking_after_a_previous_stop_rejection(tmp_path):
    """stop_hook_active 不能绕过状态门禁；只有重新评测改变 metrics 才能放行。"""
    metrics = _metrics(task_complete=False, target_met=False, remaining=2)
    judge_out = tmp_path / "judge_out"
    judge_out.mkdir()
    (judge_out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps({"stop_hook_active": True}),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert json.loads(proc.stdout)["decision"] == "block"


def test_guard_keeps_pending_best_after_current_optimization_candidate_fails(tmp_path):
    """当前失败 metrics 不能抹掉历史正确实现尚有优化预算这一持久状态。"""
    state = _metrics(task_complete=False, target_met=False, remaining=2)
    failed_candidate = {
        "success": False,
        "operator_valid": False,
        "error_type": "ascendc_compile_failed",
    }
    proc = _run_guard(tmp_path, failed_candidate, task_state=state)
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "block"
    assert "ascendc_compile_failed" in payload["reason"]
    assert "历史 .best 不受影响" in payload["reason"]


def test_guard_allows_infra_even_when_a_valid_best_has_pending_budget(tmp_path):
    """INFRA 应交给 Polar retry，不能把 agent 卡在硬件/评测环境故障上。"""
    state = _metrics(task_complete=False, target_met=False, remaining=2)
    infra = {
        "success": False,
        "operator_valid": False,
        "error_type": "profiler_unavailable",
    }
    proc = _run_guard(tmp_path, infra, task_state=state)
    assert proc.stdout == ""


def test_extra_error_class_metadata_cannot_bypass_pending_budget(tmp_path):
    """历史/外部分类字段不能覆盖 task_state 的未完成状态。"""
    state = _metrics(task_complete=False, target_met=False, remaining=2)
    current = {
        "success": False,
        "operator_valid": False,
        "error_type": "ascendc_compile_failed",
        "error_class": "C",
    }
    proc = _run_guard(tmp_path, current, task_state=state)
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "block"
    assert "task_complete=false" in payload["reason"]


def test_guard_allows_target_met_budget_exhausted_or_missing_state(tmp_path):
    cases = (
        _metrics(task_complete=True, target_met=True, remaining=3),
        _metrics(task_complete=True, target_met=False, remaining=0),
        None,
    )
    for index, metrics in enumerate(cases):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        proc = _run_guard(case_dir, metrics)
        assert proc.returncode == 0
        assert proc.stdout == "", metrics


def test_guard_fails_open_on_incomplete_or_malformed_control_state(tmp_path):
    for index, metrics in enumerate(
        (
            {"operator_valid": True, "task_complete": False},
            {
                "operator_valid": "true",
                "task_complete": False,
                "next_step": {"target_met": False, "optimization_remaining": 3},
            },
            {
                "operator_valid": True,
                "task_complete": False,
                "next_step": {"target_met": False, "optimization_remaining": 0},
            },
        )
    ):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        proc = _run_guard(case_dir, metrics)
        assert proc.returncode == 0
        assert proc.stdout == "", metrics


def _load_prepare():
    spec = importlib.util.spec_from_file_location("prepare_stop_guard_test", PREPARE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_settings_merge_adds_stop_hook_once_and_preserves_existing_entries(tmp_path):
    prepare = _load_prepare()
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "hooks": {
                    "Stop": [
                        {
                            "matcher": "legacy",
                            "hooks": [{"type": "command", "command": "legacy-stop"}],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    prepare._write_claude_settings(
        tmp_path, ["review"], enable_stop_guard=True
    )
    prepare._write_claude_settings(
        tmp_path, ["review"], enable_stop_guard=True
    )

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["permissions"] == {"allow": ["Read"]}
    assert data["skillOverrides"]["review"] == "off"
    commands = [
        hook.get("command")
        for group in data["hooks"]["Stop"]
        for hook in group.get("hooks", [])
    ]
    assert commands.count(prepare._STOP_GUARD_COMMAND) == 1
    assert "legacy-stop" in commands
