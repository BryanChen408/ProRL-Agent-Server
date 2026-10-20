"""Golden precision must not leak into candidates or CPU fallback execution."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[2] / (
        "operator_runtime_t2a/skills/tilelang2ascend-translator/scripts/verification_ascendc.py"
    )
    spec = importlib.util.spec_from_file_location("conv_precision_verifier", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("outcome", ["success", "cpu_fallback", "both_fail", "candidate_fails"])
def test_reference_precision_is_scoped(verifier, monkeypatch, initial, outcome):
    conv = SimpleNamespace(allow_hf32=initial)
    matmul = SimpleNamespace(allow_hf32=True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(conv=conv, matmul=matmul), raising=False)
    # Exercise the real reference/candidate orchestration without accelerator hardware.
    monkeypatch.setattr(verifier, "_move_to_device", lambda value, device: value)
    seen = []

    class Reference(torch.nn.Module):
        def forward(self, x):
            seen.append(("reference", conv.allow_hf32))
            if outcome == "both_fail" or (outcome == "cpu_fallback" and len(seen) == 1):
                raise RuntimeError("reference failure")
            return x + 1

    class Candidate(torch.nn.Module):
        def forward(self, x):
            seen.append(("candidate", conv.allow_hf32))
            if outcome == "candidate_fails":
                raise RuntimeError("candidate failure")
            return x + 1

    def execute():
        return verifier._execute_models(
            Reference(), Candidate(), [[torch.tensor([1.0])]], SimpleNamespace(type="npu")
        )

    if outcome in ("both_fail", "candidate_fails"):
        message = "failed on both NPU and CPU" if outcome == "both_fail" else "candidate failure"
        with pytest.raises(RuntimeError, match=message):
            execute()
    else:
        ref, cand, _, fallback = execute()
        assert torch.equal(ref[0], cand[0])
        assert bool(fallback) == (outcome == "cpu_fallback")

    expected = [("reference", False)]
    if outcome in ("cpu_fallback", "both_fail"):
        expected.append(("reference", initial))
    if outcome != "both_fail":
        expected.append(("candidate", initial))
    assert seen == expected
    assert conv.allow_hf32 is initial
    assert matmul.allow_hf32 is True


def test_cpu_reference_does_not_touch_npu_precision(verifier, monkeypatch):
    class ForbiddenConv:
        @property
        def allow_hf32(self):
            pytest.fail("CPU evaluation must not access NPU precision settings")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(conv=ForbiddenConv()), raising=False)
    result = verifier._execute_models(
        torch.nn.Identity(), torch.nn.Identity(), [[torch.ones(2)]], torch.device("cpu")
    )
    assert torch.equal(result[0][0], result[1][0])
    assert result[3] is None
