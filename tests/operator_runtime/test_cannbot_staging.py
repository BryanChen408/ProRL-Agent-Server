from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STAGING_PATH = ROOT / "operator_runtime" / "cannbot" / "runtime" / "stage_verifier_inputs.py"


def load_staging():
    spec = importlib.util.spec_from_file_location("cannbot_stage_verifier_inputs", STAGING_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_verifier_inputs_uses_cannbot_native_names(tmp_path):
    staging = load_staging()
    task = tmp_path / "input" / "demo.py"
    impl = tmp_path / "output" / "generated_code.py"
    task.parent.mkdir()
    impl.parent.mkdir()
    task.write_text("class Model: pass\n", encoding="utf-8")
    impl.write_text("class ModelNew: pass\n", encoding="utf-8")

    result = staging.stage_verifier_inputs(
        op_name="demo",
        task_path=task,
        impl_path=impl,
        verify_dir=tmp_path / "verify",
    )

    assert Path(result["task"]).name == "demo_torch.py"
    assert Path(result["impl"]).name == "demo_triton_ascend_impl.py"
    assert (tmp_path / "verify" / "demo_torch.py").read_text(encoding="utf-8") == "class Model: pass\n"
    assert (tmp_path / "verify" / "demo_triton_ascend_impl.py").read_text(encoding="utf-8") == "class ModelNew: pass\n"


def test_stage_verifier_inputs_copies_sidecar_under_renamed_task_module(tmp_path):
    staging = load_staging()
    task = tmp_path / "input" / "demo.py"
    impl = tmp_path / "impl.py"
    task.parent.mkdir()
    task.write_text("TASK\n", encoding="utf-8")
    task.with_suffix(".json").write_text('{"cases": []}\n', encoding="utf-8")
    impl.write_text("IMPL\n", encoding="utf-8")

    result = staging.stage_verifier_inputs(
        op_name="demo",
        task_path=task,
        impl_path=impl,
        verify_dir=tmp_path / "verify",
    )

    assert Path(result["sidecar"]).name == "demo_torch.json"
    assert (tmp_path / "verify" / "demo_torch.json").read_text(encoding="utf-8") == '{"cases": []}\n'


def test_stage_verifier_inputs_supports_custom_impl_name(tmp_path):
    staging = load_staging()
    task = tmp_path / "demo.py"
    impl = tmp_path / "optimized.py"
    task.write_text("TASK\n", encoding="utf-8")
    impl.write_text("IMPL\n", encoding="utf-8")

    result = staging.stage_verifier_inputs(
        op_name="demo",
        task_path=task,
        impl_path=impl,
        verify_dir=tmp_path / "verify",
        triton_impl_name="triton_optimized",
    )

    assert Path(result["impl"]).name == "demo_triton_optimized.py"
    assert (tmp_path / "verify" / "demo_triton_optimized.py").read_text(encoding="utf-8") == "IMPL\n"


def test_stage_verifier_inputs_rejects_unsafe_names(tmp_path):
    staging = load_staging()
    task = tmp_path / "demo.py"
    impl = tmp_path / "impl.py"
    task.write_text("TASK\n", encoding="utf-8")
    impl.write_text("IMPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe op_name"):
        staging.stage_verifier_inputs(
            op_name="../demo",
            task_path=task,
            impl_path=impl,
            verify_dir=tmp_path / "verify",
        )

    with pytest.raises(ValueError, match="unsafe op_name"):
        staging.stage_verifier_inputs(
            op_name="demo",
            task_path=task,
            impl_path=impl,
            verify_dir=tmp_path / "verify",
            triton_impl_name="../impl",
        )


def test_stage_verifier_inputs_requires_task_and_impl(tmp_path):
    staging = load_staging()

    with pytest.raises(FileNotFoundError, match="required file missing"):
        staging.stage_verifier_inputs(
            op_name="demo",
            task_path=tmp_path / "missing_task.py",
            impl_path=tmp_path / "missing_impl.py",
            verify_dir=tmp_path / "verify",
        )
