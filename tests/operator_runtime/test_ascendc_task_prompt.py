from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "deploy" / "ascend_operator" / "gen_ascendc_tasks.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("polar_gen_ascendc_tasks", GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_uses_pre_generated_skeleton_and_delegates_routing_to_claude_md():
    module = _load_generator()

    text = module._instruction("npukernelbench_level1_2_SwiGLU", "input/op.py")

    assert "Complete the pre-generated project directory" in text
    assert "class Model + get_input_groups/get_init_inputs" in text
    assert "Reuse those files as the only project skeleton" in text
    assert "Follow `./CLAUDE.md` as the sole workflow and judging contract" in text
    assert "/opt/workspace/agent_workdir/tools/ascendc_eval_pipeline.sh" in text
    assert ".claude/skills/*/scripts/" in text
    assert ".claude/skills/ascendc-*/scripts/" not in text
    assert ".claude/skills/tilelang2ascend-*/scripts/" not in text


def test_prompt_does_not_embed_obsolete_or_foreign_skill_routes():
    module = _load_generator()

    text = module._instruction("op_test", "input/op_test.py")

    forbidden = (
        "ops-direct-invoke",
        "case-simplifier",
        "code-gen",
        "ascendc-tilelang-designer",
        "ascendc-translator",
        "Architect design",
        "Developer implement",
        "Reviewer review",
    )
    for token in forbidden:
        assert token not in text
