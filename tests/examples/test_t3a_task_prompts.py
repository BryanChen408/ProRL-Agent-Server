import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONVERT = runpy.run_path(str(ROOT / "deploy/ascend_operator/t3a/convert_task_prompts.py"))["convert"]


@pytest.mark.parametrize("case_mode", ["full", "simple"])
def test_t3a_changes_only_prompt(case_mode):
    original = {"prompt": "old T2A", "label": 1, "metadata": {"op_name": "test_Sum", "uid": "keep"}}
    converted = CONVERT(original, case_mode=case_mode)
    assert original["prompt"] == "old T2A"
    assert {k: v for k, v in converted.items() if k != "prompt"} == {
        k: v for k, v in original.items() if k != "prompt"
    }
    text = converted["prompt"][0]["content"]
    assert "tilelang2ascendc-kernel-generator" in text
    assert "input/test_Sum.py" in text
    assert "ascendc_eval_pipeline.sh" not in text
    assert "input/ reference files are immutable" in text
    if case_mode == "simple":
        assert "retaining all 5 cases unchanged" in text
        assert "simple describes the case set, not the operator's development route" in text
        assert "do not simplify it again" in text
    else:
        assert "Phase 2 may simplify only that working copy" in text
    assert "Phase 6 is disabled" in text
    assert "go directly from Phase 5 to Phase 7" in text
    assert "separate judge validates candidates against all provided input cases" in text
    assert "full-case validation" not in text
    assert "Device selection belongs to the lease executor" in text
    assert "developer subagent must initialize" in text
    assert "Do not run the standalone" in text
    assert "resume that developer" in text
    assert "no count limit" in text


def test_t3a_rejects_unsafe_operator_name():
    with pytest.raises(ValueError):
        CONVERT({"metadata": {"op_name": "../bad"}})


def test_t3a_dispatch_references_match_registration():
    runtime = ROOT / "operator_runtime_t3a"
    agent = (runtime / "agents/tilelang2ascendc-kernel-generator.md").read_text()
    assert "name: tilelang2ascendc-kernel-generator" in agent
    assert '设置环境变量 `ASCEND_RT_VISIBLE_DEVICES=${npu}`' not in agent
    for relative in ("CLAUDE.md", "workflows/development-guide.md", "workflows/task-prompts.md",
                     "hooks/session-start-tilelang2ascendc-ops-generator"):
        text = (runtime / relative).read_text()
        assert "ascend-kernel-developer" not in text
        assert "tilelang2ascendc-kernel-generator" in text


def test_simple_cli_preserves_rows_and_refuses_overwrite(tmp_path):
    source, destination = tmp_path / "source.jsonl", tmp_path / "simple.jsonl"
    original = {"prompt": "old", "label": "keep", "metadata": {"op_name": "test_Add"}}
    source.write_text(json.dumps(original) + "\n")
    command = [sys.executable, str(ROOT / "deploy/ascend_operator/t3a/convert_task_prompts.py"),
               str(source), str(destination), "--case-mode", "simple"]
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads(destination.read_text()) == CONVERT(original, case_mode="simple")
    before = destination.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert destination.read_bytes() == before
    assert json.loads(source.read_text()) == original


def test_runtime_prompt_patch_is_reproducible_and_idempotent(tmp_path):
    relatives = ("workflows/task-prompts.md", "agents/tilelang2ascendc-kernel-generator.md",
                 "CLAUDE.md", "workflows/development-guide.md")
    copied = (*relatives, "skills/tilelang2ascend-operator-project-init/SKILL.md")
    for relative in copied:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "operator_runtime_t3a" / relative, target)
    script = (ROOT / "deploy/ascend_operator/build_t3a_replica.sh").read_text()
    patch = script.split("<<'PROMPTPY'\n", 1)[1].split("\nPROMPTPY", 1)[0]
    subprocess.run([sys.executable, "-", str(tmp_path)], input=patch, text=True, check=True)
    for relative in copied:
        assert (tmp_path / relative).read_bytes() == (ROOT / "operator_runtime_t3a" / relative).read_bytes()
    prompt = (tmp_path / relatives[0]).read_text()
    assert "设置 ASCEND_RT_VISIBLE_DEVICES={npu}" not in prompt
    assert "--compare" not in prompt
    assert "--quick" in prompt
    for relative in relatives:
        text = (tmp_path / relative).read_text()
        assert "本 RL 流程取消 Phase 6" in text
        assert "Phase 5 后直接进入 Phase 7" in text
        assert "独立 judge 使用 input/ 中的完整任务用例验收候选" in text
        assert all("本 RL 流程取消 Phase 6" in line
                   for line in text.splitlines() if "Phase 6" in line)
    agent = (tmp_path / relatives[1]).read_text()
    assert "不再执行下面的精简操作" in agent
    assert "通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量设置" not in agent
    assert "cp /opt/workspace/agent_workdir/.claude/skills/" in agent
    init = (tmp_path / copied[-1]).read_text()
    assert "mkdir -p {output_dir}/kernel/op_host" in init
    assert "所有算子统一生成在 `csrc/ops`" not in init
