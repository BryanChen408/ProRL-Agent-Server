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
    assert "get_inputs() or get_input_groups()" in text
    assert "get_init_inputs() when defined" in text
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


def test_prompt_keeps_design_before_first_evaluation_and_reference_as_authority():
    text = _load_generator()._instruction("cudallm_op", "input/cudallm_op.py")
    assert "placeholders, not the operator specification" in text
    assert "including __init__, forward and relevant helpers" in text
    assert "Skipping TileLang does not skip" in text
    assert "first implementation is ready" in text
    assert "metrics do not exist yet" in text
    assert "collect its result rather than launching another" in text
    assert "SOC_VERSION from the environment" in text
    assert "910B2C" not in text


def test_t2a_system_prompt_allows_brief_design_without_premature_evaluation():
    import yaml

    profile = yaml.safe_load((ROOT / "deploy/ascend_operator/profile.t2a.yaml").read_text())
    prompt = profile["operator"]["agent"]["append_system_prompt"]
    assert "两者都没拿到就先重跑固定入口" not in prompt
    assert "拿不准的地方先写出代码跑出真实报错" not in prompt
    assert "简短设计核对" in prompt
    assert "不要因为没有 metrics 就评测未实现的骨架" in prompt
