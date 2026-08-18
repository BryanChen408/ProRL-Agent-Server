"""AscendC 固定入口的预算落盘 + 成功后的优化提示。

两个缺口的回归锁:

1. `pipeline_budget_status.json` 从来没落过盘。triton 侧 `pipeline_status_write()`
   把预算状态写进 `$ARTIFACTS_DIR`(gateway 侧 session 目录),ascendc 移植时整段漏了
   —— 实测 4 个 run / 695 个 session 里该文件出现 0 次,watcher 的
   `should_cancel_from_status` 分支对 ascendc 一直是空跑。

2. 正确性一过就收工。实测 179 个成功 session 只有 2 个进过 optimization 阶段,
   speedup 中位数 0.859x、58.8% 比 torch 慢 —— 而 phase 是在下一次调用开头才判定的,
   agent 只看到 "错误分类: 通过",永远不知道 optimization 阶段存在。

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
    }
    env.update(over)
    return env


def test_prompt_says_not_met_below_target(tmp_path):
    """0.86x < 1.1x:必须明说未达标 + 还剩几次,否则 agent 看到"通过"就收工。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.859"',
               _prompt_env(), tmp_path)
    assert "未达标" in out and "不要结束任务" in out
    assert "0.859x" in out and "1.1x" in out
    assert "剩 4 次" in out


def test_prompt_still_invites_optimization_above_target(tmp_path):
    """1.34x ≥ 1.1x:达标了也要说清楚还能继续 —— 加速比没有上限。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "1.34"',
               _prompt_env(), tmp_path)
    assert "已达标" in out and "未达标" not in out
    assert "加速比越高得分越高" in out


def test_prompt_silent_when_optimization_budget_gone(tmp_path):
    """优化预算用完就闭嘴,不要劝一个已经没预算的 session 继续跑。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.5"',
               _prompt_env(PIPELINE_OPT_COUNT="4"), tmp_path)
    assert out.strip() == ""


def test_prompt_silent_on_judge_side(tmp_path):
    """judge 侧(AGENT_SIDE=0)不该往评测输出里掺 agent 提示。"""
    out = _run("emit_optimization_prompt", 'emit_optimization_prompt "0.5"',
               _prompt_env(AGENT_SIDE="0"), tmp_path)
    assert out.strip() == ""


def test_perf_target_default_matches_claude_md():
    """三处目标线必须同一个数:pipeline / CLAUDE.md / 任务 prompt。"""
    assert 'PERF_TARGET="${POLAR_PERF_TARGET:-1.1}"' in PIPELINE.read_text(encoding="utf-8")
    claude = (ROOT / "operator_runtime_t2a" / "CLAUDE.md").read_text(encoding="utf-8")
    assert "加速比 **≥ 1.1x** PyTorch reference → 达标" in claude
    assert "0.6x PyTorch reference" not in claude
    tasks = (ROOT / "deploy" / "ascend_operator" / "gen_ascendc_tasks.py").read_text(encoding="utf-8")
    assert "1.1x the PyTorch reference" in tasks
