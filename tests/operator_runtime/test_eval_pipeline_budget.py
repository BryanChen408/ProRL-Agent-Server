"""AscendC 固定入口的预算落盘 + 成功后的优化提示。

两个缺口的回归锁:

1. `pipeline_budget_status.json` 从来没落过盘。triton 侧 `pipeline_status_write()`
   把预算状态写进 `$ARTIFACTS_DIR`(gateway 侧 session 目录),ascendc 移植时整段漏了
   —— 实测 4 个 run / 695 个 session 里该文件出现 0 次,watcher 的
   `should_cancel_from_status` 分支对 ascendc 一直是空跑。

2. 正确性一过就收工。实测 179 个成功 session 只有 2 个进过 optimization 阶段,
   speedup 中位数 0.859x、58.8% 比 torch 慢 —— 而 phase 是在下一次调用开头才判定的,
   agent 只看到 "错误分类: 通过",永远不知道 optimization 阶段存在。现在固定入口必须把
   operator_valid 与 task_complete 分开，供 Stop hook 做机械门禁。

口径说明:这份状态文件和 workdir 里的 `.selfcheck` 计数器一样在 session bind mount 内,
agent 够得到 —— 它是第二信号 + 遥测。真正的强制层是 watcher 从 gateway completion 流里
数固定入口调用次数(`polar_pipeline_budget_watcher.analyze_budget`),那条链路 agent 改不到。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "operator_runtime_t2a" / "tools" / "ascendc_eval_pipeline.sh"
WATCHER = ROOT / "deploy" / "ascend_operator" / "tools" / "polar_pipeline_budget_watcher.py"


def _extract(func: str) -> str:
    """抓出真实脚本里的函数体(带 <<'PY' heredoc,不能简单按 ^} 截断)。"""
    lines = PIPELINE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"{func}() {{"))
    out, in_heredoc = [], False
    for line in lines[start:]:
        out.append(line)
        if "<<'PY'" in line:
            in_heredoc = True
        elif in_heredoc and line.strip() == "PY":
            in_heredoc = False
        elif not in_heredoc and line == "}" and len(out) > 1:
            return "\n".join(out)
    raise AssertionError(f"未找到 {func} 的结尾")


def _run(func: str, body_call: str, env: dict[str, str], cwd: Path) -> str:
    script = "\n".join(["set -uo pipefail", _extract(func), body_call])
    proc = subprocess.run(
        ["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _load_watcher():
    spec = importlib.util.spec_from_file_location("budget_watcher", WATCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _status_env(tmp_path: Path, **over: str) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "SESSION_ID": "sk-polar-test", "TASK_ID": "task-test", "OP_NAME": "op_test",
        "PIPELINE_PHASE": "generation", "PIPELINE_ATTEMPT": "3", "PIPELINE_LIMIT": "10",
        "PIPELINE_GEN_COUNT": "3", "PIPELINE_OPT_COUNT": "0", "PIPELINE_FIRST_SUCCESS": "0",
    }
    env.update(over)
    return env


def test_status_file_lands_in_artifacts_dir(tmp_path):
    """预算状态必须落到 $ARTIFACTS_DIR —— 这是 ascendc 移植时漏掉的整段。"""
    _run("pipeline_status_write", "pipeline_status_write", _status_env(tmp_path), tmp_path)
    path = tmp_path / "artifacts" / "pipeline_budget_status.json"
    assert path.exists(), "pipeline_budget_status.json 没落盘"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phase"] == "generation"
    assert data["attempt"] == 3 and data["limit"] == 10
    assert data["gen_count"] == 3 and data["opt_count"] == 0
    assert data["first_success"] is False
    assert data["limit_exhausted"] is False
    assert data["session_id"] == "sk-polar-test"


def test_status_file_absent_when_artifacts_dir_unset(tmp_path):
    """gateway 之外(本地手跑)不该炸,静默跳过即可。"""
    env = _status_env(tmp_path)
    env.pop("ARTIFACTS_DIR")
    _run("pipeline_status_write", "pipeline_status_write", env, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_status_agrees_with_watcher_cancel_decision(tmp_path):
    """写出来的状态要能被 watcher 直接消费 —— 字段名/口径对齐,不是各写各的。"""
    watcher = _load_watcher()
    within = _status_env(tmp_path, PIPELINE_ATTEMPT="10", PIPELINE_GEN_COUNT="10")
    _run("pipeline_status_write", "pipeline_status_write", within, tmp_path)
    data = json.loads((tmp_path / "artifacts" / "pipeline_budget_status.json").read_text())
    cancel, _ = watcher.should_cancel_from_status(data)
    assert cancel is False, "attempt == limit 只是提示 agent 收尾,还不该被 watcher 砍"

    over = _status_env(tmp_path, PIPELINE_ATTEMPT="11", PIPELINE_GEN_COUNT="11")
    _run("pipeline_status_write", "pipeline_status_write", over, tmp_path)
    data = json.loads((tmp_path / "artifacts" / "pipeline_budget_status.json").read_text())
    assert data["limit_exhausted"] is True
    cancel, reason = watcher.should_cancel_from_status(data)
    assert cancel is True and "11>10" in reason


def _prompt_env(**over: str) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "AGENT_SIDE": "1", "PERF_TARGET": "1.1",
        "PIPELINE_OPT_MAX": "4", "PIPELINE_OPT_COUNT": "0",
        "OUT_DIR": ".",   # _run 的 cwd=tmp_path,配合 _seed_metrics 使用
        "TASK_STATE_FILE": "task_state.json",
    }
    env.update(over)
    return env


_METRICS = {
    "schema_version": 2, "op_name": "op_x", "success": True,
    "ast_check_ok": True, "correctness_ok": True,
    "perf_data": {"framework_latency_ms": 1.0, "impl_latency_ms": 2.0, "speedup_vs_torch": 0.859},
    "error": None, "error_type": None, "error_file": None,
    "error_bytes": 0, "error_sha256": None, "error_truncated": False,
}


def _seed_metrics(tmp_path: Path) -> Path:
    f = tmp_path / "metrics.json"
    f.write_text(json.dumps(_METRICS, ensure_ascii=False), encoding="utf-8")
    return f


def test_prompt_says_not_met_below_target(tmp_path):
    """0.86x < 1.1x:必须明说未达标 + 还剩几次,否则 agent 看到"通过"就收工。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.859"',
               _prompt_env(), tmp_path)
    assert "未达标" in out and "不要结束任务" in out
    assert "0.859x" in out and "1.1x" in out
    assert "剩 4 次" in out
    assert "Read .claude/skills/ops-profiling/SKILL.md" in out


def test_prompt_stops_optimization_above_target(tmp_path):
    """1.34x ≥ 1.1x:达标后停止，避免继续消耗上板预算。"""
    f = _seed_metrics(tmp_path)
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "1.34"',
               _prompt_env(), tmp_path)
    assert "已达标" in out and "停止性能迭代" in out
    assert "进入 optimization" not in out
    metrics = json.loads(f.read_text(encoding="utf-8"))
    assert metrics["operator_valid"] is True
    assert metrics["task_complete"] is True
    assert metrics["completion_reason"] == "target_met"
    assert metrics["next_step"]["phase_next"] == "complete"


def test_budget_exhaustion_marks_task_complete_without_claiming_target_met(tmp_path):
    """优化预算用完要明确放行，但不能谎称性能达标。"""
    f = _seed_metrics(tmp_path)
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.5"',
               _prompt_env(PIPELINE_OPT_COUNT="4"), tmp_path)
    assert "task_complete=true" in out and "budget_exhausted" in out
    assert "预算已耗尽" in out
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["operator_valid"] is True
    assert d["task_complete"] is True
    assert d["completion_reason"] == "budget_exhausted"
    assert d["next_step"]["target_met"] is False
    assert d["next_step"]["optimization_remaining"] == 0


def test_prompt_silent_on_judge_side(tmp_path):
    """judge 侧(AGENT_SIDE=0)不该往评测输出里掺 agent 提示。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.5"',
               _prompt_env(AGENT_SIDE="0"), tmp_path)
    assert out.strip() == ""


def test_prompt_is_also_written_into_metrics_json(tmp_path):
    """指引必须同时落进 metrics.json,不能只走 stdout。

    评测常顶穿 Bash 超时被自动转后台,stdout 改写进临时文件、agent 只能轮询,还会撞上
    harness 的 "Wasted call — file unchanged" 护栏 —— 实测 run 133937 的 45 个会话里只有
    9 个收到过这段话(20%)。而 agent 拿不到 stdout 时恰恰是去读 metrics.json 补的:4 个
    「正确性过了、speedup 低于目标线、预算没用完就收工」的会话里有 2 个就是这么拿到
    speedup 的 —— 它们知道没达标,只是没人告诉它们还有优化预算。
    """
    f = _seed_metrics(tmp_path)
    _run("emit_optimization_prompt", 'emit_optimization_prompt "0.859"', _prompt_env(), tmp_path)
    d = json.loads(f.read_text(encoding="utf-8"))
    ns = d["next_step"]
    assert d["operator_valid"] is True
    assert d["task_complete"] is False
    assert d["completion_reason"] == "pending_optimization"
    assert ns["target_met"] is False
    assert ns["perf_target_speedup"] == 1.1
    assert ns["phase_next"] == "optimization"
    assert ns["optimization_remaining"] == 4
    assert "不要结束任务" in ns["action"]
    # 原有键一个都不能动:judge 侧的 reward 只认这几个
    for k in ("success", "error_type", "perf_data", "ast_check_ok", "correctness_ok"):
        assert d[k] == _METRICS[k]
    state = json.loads((tmp_path / "task_state.json").read_text(encoding="utf-8"))
    assert state["operator_valid"] is True
    assert state["task_complete"] is False
    assert state["next_step"]["optimization_remaining"] == 4


def test_persistent_task_state_consumes_budget_even_if_next_candidate_fails(tmp_path):
    """持久状态在新 metrics 被失败结果覆盖前就同步预算，不能永远停在旧 remaining。"""
    state = {
        "schema_version": 1,
        "operator_valid": True,
        "task_complete": False,
        "completion_reason": "pending_optimization",
        "perf_data": {"speedup_vs_torch": 0.859},
        "next_step": {
            "target_met": False,
            "optimization_budget": 4,
            "optimization_used": 0,
            "optimization_remaining": 4,
        },
    }
    path = tmp_path / "task_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    env = _prompt_env(
        PIPELINE_PHASE="optimization",
        PIPELINE_OPT_COUNT="2",
        TASK_STATE_FILE=str(path),
    )
    _run("sync_task_state_budget", "sync_task_state_budget", env, tmp_path)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["task_complete"] is False
    assert updated["next_step"]["optimization_used"] == 2
    assert updated["next_step"]["optimization_remaining"] == 2

    env["PIPELINE_OPT_COUNT"] = "4"
    _run("sync_task_state_budget", "sync_task_state_budget", env, tmp_path)
    exhausted = json.loads(path.read_text(encoding="utf-8"))
    assert exhausted["task_complete"] is True
    assert exhausted["completion_reason"] == "budget_exhausted"
    assert exhausted["next_step"]["optimization_remaining"] == 0


def test_judge_metrics_remain_byte_semantically_unchanged(tmp_path):
    """task 状态只写 agent 侧，judge/reward 的 metrics schema 不掺控制字段。"""
    f = _seed_metrics(tmp_path)
    _run("emit_optimization_prompt", 'emit_optimization_prompt "0.5"',
         _prompt_env(AGENT_SIDE="0"), tmp_path)
    assert json.loads(f.read_text(encoding="utf-8")) == _METRICS


def test_prompt_survives_missing_metrics_json(tmp_path):
    """metrics.json 不在时只留 stdout —— 这段是附加指引,不该让固定入口失败。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.859"',
               _prompt_env(), tmp_path)          # 没有 _seed_metrics
    assert "不要结束任务" in out
    assert not (tmp_path / "metrics.json").exists()


def test_perf_target_default_matches_claude_md():
    """三处目标线必须同一个数:pipeline / CLAUDE.md / 任务 prompt。"""
    assert 'PERF_TARGET="${POLAR_PERF_TARGET:-1.1}"' in PIPELINE.read_text(encoding="utf-8")
    assert 'cp -f "$PERF_JSON" "$OUT_DIR/performance.json"' in PIPELINE.read_text(encoding="utf-8")
    claude = (ROOT / "operator_runtime_t2a" / "CLAUDE.md").read_text(encoding="utf-8")
    assert "加速比 **≥ 1.1x** PyTorch reference → 达标" in claude
    assert "0.6x PyTorch reference" not in claude
    tasks = (ROOT / "deploy" / "ascend_operator" / "gen_ascendc_tasks.py").read_text(encoding="utf-8")
    assert "1.1x the PyTorch reference" in tasks


def test_success_output_no_longer_aliases_operator_valid_to_task_completion():
    """agent-facing 成功路径不能再出现会诱导 end_turn 的 `done — success=true`。"""
    script = PIPELINE.read_text(encoding="utf-8")
    assert '[ascendc-eval] done — success=true' not in script
    assert "operator_valid=true task_complete=false" in script


def test_only_t2a_profile_disables_skill_tool():
    """直接 Read 是 polar t2a 的局部策略，不能改变其他 profile 的工具能力。"""
    t2a = (ROOT / "deploy" / "ascend_operator" / "profile.t2a.yaml").read_text(
        encoding="utf-8"
    )
    assert 'allowed_tools: "Bash Read Edit Write Grep Glob"' in t2a
    assert 'allowed_tools: "Bash Read Edit Write Grep Glob Skill"' not in t2a
    assert 'disallowed_tools: "Skill ' in t2a
    for name in ("profile.yaml", "profile.ascendc.yaml", "profile.legacy.yaml"):
        other = (ROOT / "deploy" / "ascend_operator" / name).read_text(encoding="utf-8")
        assert 'allowed_tools: "Bash Read Edit Write Grep Glob"' not in other


def test_over_limit_gate_precedes_all_evaluation_work():
    script = PIPELINE.read_text(encoding="utf-8")
    gate = '"$PIPELINE_ATTEMPT" -gt "$PIPELINE_LIMIT"'

    assert gate in script
    assert script.index("budget already exhausted; skip evaluation work") < script.index("# Step0")


def test_only_in_budget_candidate_is_packed_for_best_comparison():
    """limit+1 次的候选不能在早退前偷偷参与 best 比较。"""
    script = PIPELINE.read_text(encoding="utf-8")
    start = script.index(
        "  pipeline_status_write\n  sync_task_state_budget\n  echo \"[pipeline-budget]"
    )
    end = script.index("\nfi\n\n_on_exit()", start)
    setup_block = script[start:end]

    guard = "if ! pipeline_over_limit; then"
    assert guard in setup_block
    assert setup_block.index(guard) < setup_block.index("    pack_best")


def test_new_evaluation_discards_previous_source_results_after_cache_and_budget_gates():
    """新源码开跑前清理旧结果；未改源码的缓存和超限收尾仍可先返回。"""
    script = PIPELINE.read_text(encoding="utf-8")
    cache_gate = "cached evaluation"
    budget_gate = "budget already exhausted; skip evaluation work"
    stale_clear = 'rm -f "$OUT_DIR/metrics.json"'

    assert script.index(cache_gate) < script.index(stale_clear)
    assert script.index(budget_gate) < script.index(stale_clear)
    assert script.index(stale_clear) < script.index("# Step0")


def test_over_limit_exit_does_not_attach_stale_metrics_to_current_source(tmp_path):
    """N+1 次未执行评测，不能用第 N 次的 metrics 认证当前源码。"""
    metrics = {
        "correctness_ok": True,
        "perf_data": {"speedup_vs_torch": 1.2},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    capture = tmp_path / "pack_args.txt"
    script = "\n".join([
        "set -uo pipefail",
        'pack_best() { printf "%s|%s\\n" "${1:-}" "${2:-}" >> "$CAPTURE"; }',
        _extract("pipeline_over_limit"),
        _extract("pipeline_at_limit"),
        _extract("_on_exit"),
        "AGENT_SIDE=1",
        f'OUT_DIR="{tmp_path}"',
        f'STATE_DIR="{state_dir}"',
        f'CAPTURE="{capture}"',
        'OP_NAME="op_test"',
        'CUR_HASH="new-source-hash"',
        'PIPELINE_PHASE="generation"',
        'PIPELINE_ATTEMPT=4',
        'PIPELINE_LIMIT=3',
        "_on_exit",
    ])
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert not capture.exists(), "超限退出仍然用旧 metrics 调用了 pack_best"
    assert "LIMIT_EXHAUSTED" in proc.stdout
