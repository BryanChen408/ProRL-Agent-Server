from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = (
    ROOT
    / "operator_runtime"
    / "cannbot"
    / "skills"
    / "triton-op-verifier"
    / "scripts"
    / "_polar_runtime.py"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location("cannbot_polar_runtime", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase_from_impl_defaults_to_generation_or_optimization():
    runtime = load_runtime()

    assert runtime.phase_from_impl("triton_ascend_impl") == "generation"
    assert runtime.phase_from_impl("triton_optimized") == "optimization"
    assert runtime.phase_from_impl("triton_ascend_impl", "optimization") == "optimization"


def test_verify_budget_allows_limit_and_blocks_next_attempt(tmp_path, monkeypatch):
    runtime = load_runtime()
    monkeypatch.setenv("POLAR_BUDGET_DIR", str(tmp_path))
    monkeypatch.setenv("POLAR_GEN_PIPELINE_MAX", "1")
    monkeypatch.setenv("SESSION_ID", "s1")

    status = runtime.consume_verify_budget("generation", op_name="op")

    assert status["attempt"] == 1
    assert status["limit"] == 1
    assert status["limit_exhausted"] is False

    with pytest.raises(runtime.BudgetExceeded):
        runtime.consume_verify_budget("generation", op_name="op")

    status_path = tmp_path / "pipeline_budget_status.json"
    status_text = status_path.read_text(encoding="utf-8")
    assert '"attempt": 2' in status_text
    assert '"limit_exhausted": true' in status_text


def test_npu_lease_sets_and_restores_visible_devices(tmp_path, monkeypatch):
    runtime = load_runtime()
    monkeypatch.setenv("POLAR_NPU_LEASE_POOL", "8,9")
    monkeypatch.setenv("POLAR_NPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")

    with runtime.npu_lease("generation", work_dir=tmp_path) as lease:
        assert lease is not None
        assert lease.device_id == "8"
        assert lease.lock_path.name == "npu8.lock"
        assert runtime.os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "8"

    assert runtime.os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert (tmp_path / "npu_lease_status.generation.json").is_file()
