from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "ascend" / "polar_dockerruntime_e2e" / "gen_op_assets.py"
REFRESH_SCRIPT = ROOT / "examples" / "ascend" / "polar_dockerruntime_e2e" / "tools" / "refresh_operator_task_prompts.py"


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


def test_instruction_names_reference_and_submission_paths():
    module = _load_module()

    text = module._instruction("kernelbench_l1_19_19_ReLU")

    assert "src/kernelbench_l1_19_19_ReLU.py" in text
    assert "output/submission/kernelbench_l1_19_19_ReLU_impl.py" in text
    assert "bash tools/triton_eval_pipeline.sh --op_name kernelbench_l1_19_19_ReLU" in text
    assert "--impl output/submission/kernelbench_l1_19_19_ReLU_impl.py" in text
    assert "--task src/kernelbench_l1_19_19_ReLU.py" in text
    assert "--out_dir judge_out" in text
    assert "第一次 Write/Edit/MultiEdit" in text
    assert "禁止第二次 Write/Edit/MultiEdit" in text
    assert "禁止总结、点评、改写 reference 文档" in text
    assert "--json" not in text
    assert "triton-op-designer once" not in text
    assert "Phase 2 has 5 pipeline attempts" not in text
    assert "canonical pipeline" not in text
    assert "Do NOT edit anything under tools/" not in text


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
    assert "bash tools/triton_eval_pipeline.sh --op_name safe_op" in prompt
    assert "--out_dir judge_out" in prompt
    assert "--json" not in prompt
