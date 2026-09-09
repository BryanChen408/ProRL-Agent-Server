from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "ascend_operator" / "gen_op_assets.py"
REFRESH_SCRIPT = ROOT / "deploy" / "ascend_operator" / "tools" / "refresh_operator_task_prompts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_gen_op_assets", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_refresh_module():
    spec = importlib.util.spec_from_file_location("polar_refresh_operator_task_prompts", REFRESH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_op_name_allows_safe_basename():
    module = _load_module()

    assert module._validate_op_name("kernelbench_l1_19_19_ReLU", row=0) == "kernelbench_l1_19_19_ReLU"
    assert module._validate_op_name("op.1-fast", row=1) == "op.1-fast"


@pytest.mark.parametrize("name", ["../bad", "bad/name", "bad name", "$(touch pwned)", "", ".", ".hidden", "-flag"])
def test_validate_op_name_rejects_unsafe_names(name):
    module = _load_module()

    with pytest.raises(ValueError):
        module._validate_op_name(name, row=3)


def test_as_dict_accepts_json_and_python_literal_dict():
    module = _load_module()

    assert module._as_dict({"op_name": "x"}) == {"op_name": "x"}
    assert module._as_dict(json.dumps({"op_name": "x"})) == {"op_name": "x"}
    assert module._as_dict("{'op_name': 'x'}") == {"op_name": "x"}


def test_instruction_delegates_workflow_to_claude_md():
    module = _load_module()

    text = module._instruction("kernelbench_l1_19_19_ReLU")

    assert "kernelbench_l1_19_19_ReLU" in text
    assert "input/kernelbench_l1_19_19_ReLU.py" in text
    assert "./CLAUDE.md" in text
    assert "output/submission" not in text
    assert "tools/triton_eval_pipeline.sh" not in text
    assert "src/kernelbench_l1_19_19_ReLU.py" not in text
    assert "--json" not in text
    assert "triton-op-designer once" not in text
    assert "Phase 2 has 5 pipeline attempts" not in text
    assert "canonical pipeline" not in text
    assert "Do NOT edit anything under tools/" not in text


def test_instruction_legacy_uses_lightweight_fixed_pipeline_prompt():
    module = _load_module()

    text = module._instruction("kernelbench_l1_19_19_ReLU", workflow="legacy")

    assert "src/kernelbench_l1_19_19_ReLU.py" in text
    assert "output/submission/kernelbench_l1_19_19_ReLU_impl.py" in text
    assert "bash tools/triton_eval_pipeline.sh" in text
    assert "--op_name kernelbench_l1_19_19_ReLU" in text
    assert "--task src/kernelbench_l1_19_19_ReLU.py" in text
    assert "Use this fixed validation entry to judge pass/fail" in text
    assert "Small read-only probes" in text
    assert "Do not modify task files, tools, verifier scripts" in text
    assert "custom tests, torch.allclose, probe output" in text
    assert "第一次 Write/Edit/MultiEdit" not in text
    assert "禁止第二次 Write/Edit/MultiEdit" not in text
    assert "input/kernelbench_l1_19_19_ReLU.py" not in text


def test_build_assets_limit_writes_jsonl_and_task_file(monkeypatch, tmp_path: Path):
    module = _load_module()

    rows = [
        {
            "extra_info": {
                "op_name": "safe_op",
                "task_code": "class Model:\n    pass\n",
                "kernelbench_level": 1,
                "ignored_large_field": "drop-me",
            }
        },
        {
            "extra_info": {
                "op_name": "second_op",
                "task_code": "class Model:\n    pass\n",
            }
        },
    ]

    class _Iloc:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, item):
            if isinstance(item, slice):
                return _Frame(self._data[item])
            return self._data[item]

    class _Frame:
        def __init__(self, data):
            self._data = data
            self.iloc = _Iloc(data)

        def __len__(self):
            return len(self._data)

    monkeypatch.setitem(
        __import__("sys").modules,
        "pandas",
        SimpleNamespace(read_parquet=lambda path: _Frame(rows)),
    )

    emitted = module.build_assets(
        parquet=tmp_path / "input.parquet",
        out_dir=tmp_path / "assets",
        limit=1,
    )

    assert emitted == 1
    task = tmp_path / "assets" / "op_tasks" / "safe_op.py"
    assert task.read_text() == "class Model:\n    pass\n"
    jsonl = tmp_path / "assets" / "operator_tasks.jsonl"
    records = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["label"] == "safe_op"
    assert records[0]["metadata"] == {
        "op_name": "safe_op",
        "kernelbench_level": 1,
    }


def test_build_assets_rejects_duplicate_op_names(monkeypatch, tmp_path: Path):
    module = _load_module()

    rows = [
        {"extra_info": {"op_name": "dup_op", "task_code": "class Model: pass\n"}},
        {"extra_info": {"op_name": "dup_op", "task_code": "class Model: pass\n"}},
    ]

    class _Iloc:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, item):
            return self._data[item]

    class _Frame:
        def __init__(self, data):
            self._data = data
            self.iloc = _Iloc(data)

        def __len__(self):
            return len(self._data)

    monkeypatch.setitem(
        __import__("sys").modules,
        "pandas",
        SimpleNamespace(read_parquet=lambda path: _Frame(rows)),
    )

    with pytest.raises(ValueError, match="duplicate op_name"):
        module.build_assets(parquet=tmp_path / "input.parquet", out_dir=tmp_path / "assets")


def test_refresh_operator_task_prompts_rewrites_existing_jsonl(tmp_path: Path):
    refresh = _load_refresh_module()

    path = tmp_path / "operator_tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "old prompt"}],
                "label": "safe_op",
                "metadata": {"op_name": "safe_op"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = refresh.refresh_prompts(path, backup=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    prompt = rows[0]["prompt"][0]["content"]

    assert result["rows"] == 1
    assert result["changed"] == 1
    assert result["backup"]
    assert Path(result["backup"]).is_file()
    assert "input/safe_op.py" in prompt
    assert "./CLAUDE.md" in prompt
    assert "tools/triton_eval_pipeline.sh" not in prompt
    assert "output/submission" not in prompt
    assert "--json" not in prompt


def test_refresh_operator_task_prompts_can_emit_legacy_prompt(tmp_path: Path):
    refresh = _load_refresh_module()

    path = tmp_path / "operator_tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "old prompt"}],
                "label": "safe_op",
                "metadata": {"op_name": "safe_op"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = refresh.refresh_prompts(path, backup=False, workflow="legacy")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    prompt = rows[0]["prompt"][0]["content"]

    assert result["workflow"] == "legacy"
    assert result["changed"] == 1
    assert "src/safe_op.py" in prompt
    assert "output/submission/safe_op_impl.py" in prompt
    assert "tools/triton_eval_pipeline.sh" in prompt
    assert "Small read-only probes" in prompt
    assert "第一次 Write/Edit/MultiEdit" not in prompt


def test_refresh_t2a_only_changes_prompt_content_and_is_idempotent(tmp_path: Path):
    refresh = _load_refresh_module()
    path = tmp_path / "tasks.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "old", "extra": "keep"}],
         "label": f"label-{op}", "metadata": {"op_name": op, "operator_backend": "ascendc"},
         "extra": [1, {"keep": True}]}
        for op in ("second", "first")
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path.chmod(0o640)
    before = path.read_bytes()
    result = refresh.refresh_prompts(path, workflow="t2a")
    assert path.stat().st_mode & 0o777 == 0o640
    assert result["changed"] == 2
    assert Path(result["backup"]).read_bytes() == before
    after = [json.loads(line) for line in path.read_text().splitlines()]
    for original, updated in zip(rows, after):
        text = updated["prompt"][0]["content"]
        assert f"input/{original['metadata']['op_name']}.py" in text
        assert "pre-generated project directory" in text
        assert "ops-direct-invoke" not in text
        updated["prompt"][0]["content"] = original["prompt"][0]["content"]
        assert updated == original
    assert refresh.refresh_prompts(path, backup=False, workflow="t2a")["changed"] == 0


def test_refresh_t2a_rejects_mixed_backend_without_partial_write(tmp_path: Path):
    refresh = _load_refresh_module()
    path = tmp_path / "tasks.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "old"}],
         "metadata": {"op_name": "op", "operator_backend": backend}}
        for backend in ("ascendc", "triton")
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="requires metadata.operator_backend=ascendc"):
        refresh.refresh_prompts(path, backup=False, workflow="t2a")
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
