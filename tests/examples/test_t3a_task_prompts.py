import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONVERT = runpy.run_path(str(ROOT / "deploy/ascend_operator/t3a/convert_task_prompts.py"))["convert"]


def test_t3a_changes_only_prompt():
    original = {"prompt": "old T2A", "label": 1, "metadata": {"op_name": "test_Sum", "uid": "keep"}}
    converted = CONVERT(original)
    assert original["prompt"] == "old T2A"
    assert {k: v for k, v in converted.items() if k != "prompt"} == {
        k: v for k, v in original.items() if k != "prompt"
    }
    text = converted["prompt"][0]["content"]
    assert "tilelang2ascendc-kernel-generator" in text
    assert "input/test_Sum.py" in text
    assert "ascendc_eval_pipeline.sh" not in text
    assert "input/ reference files are immutable" in text
    assert "Phase 2 may simplify only that working copy" in text
    assert "Phase 6 must restore the full cases" in text
    assert "Device selection belongs to the lease executor" in text


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
