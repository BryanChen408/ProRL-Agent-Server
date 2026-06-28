from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "ascend_operator" / "prepare_readonly_tools.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_prepare_readonly_tools", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_readonly_tools_filters_test_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "triton_eval_pipeline.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (source / "env.sh").write_text("export X=1\n", encoding="utf-8")
    (source / "npu_lease_exec.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (source / "fixtures").mkdir()
    (source / "fixtures" / "secret.txt").write_text("fixture\n", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests" / "test_tool.py").write_text("pass\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "x.pyc").write_bytes(b"pyc")

    dest = tmp_path / "readonly_tools"
    module.publish_readonly_tools(source, dest)

    assert (dest / "triton_eval_pipeline.sh").is_file()
    assert (dest / "env.sh").is_file()
    assert (dest / "npu_lease_exec.py").is_file()
    assert not (dest / "fixtures").exists()
    assert not (dest / "tests").exists()
    assert not (dest / "__pycache__").exists()
    assert (dest / "triton_eval_pipeline.sh").stat().st_mode & 0o111
    assert (dest / "npu_lease_exec.py").stat().st_mode & 0o111
