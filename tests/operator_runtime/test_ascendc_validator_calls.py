from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / (
    "operator_runtime_t2a/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py"
)
spec = importlib.util.spec_from_file_location("ascendc_validator_calls", SCRIPT)
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)

PREFIX = 'import torch\nimport torch.nn as nn\ntorch.ops.load_library("probe.so")\n'


@pytest.mark.parametrize("source", [
    """class ModelNew(nn.Module):
    def forward(self, x):
        return torch.ops.npu.probe(x)
""",
    """class ModelNew(nn.Module):
    def helper(self, x):
        return torch.ops.npu.probe(x)
    def forward(self, x):
        return self.helper(x)
""",
    """def inner(x):
    return torch.ops.npu.probe(x)
def outer(x):
    return inner(x)
class ModelNew(nn.Module):
    def forward(self, x):
        return outer(x)
""",
    """kernel = torch.ops.npu.probe
class ModelNew(nn.Module):
    def forward(self, x):
        return kernel(x)
""",
    """class ModelNew(nn.Module):
    def __init__(self):
        self.kernel = torch.ops.npu.probe
    def forward(self, x):
        return self.kernel(x)
""",
    """class ModelNew(nn.Module):
    def forward(self, xs):
        n = sum(x.numel() for x in xs)
        return torch.ops.npu.probe(xs, n)
""",
    """class ModelNew(nn.Module):
    def forward(self, x):
        return torch.ops.npu.probe(x, sum(x.shape))
""",
    """class ModelNew(nn.Module):
    def forward(self, x):
        def helper(y):
            return torch.ops.npu.probe(y)
        return helper(x)
""",
    """class ModelNew(nn.Module):
    def forward(self, x):
        kernel = torch.ops.npu.probe
        return kernel(x)
""",
    """import probe_ext
class ModelNew(nn.Module):
    def helper(self, x):
        return probe_ext.run(x)
    def forward(self, x):
        return self.helper(x)
""",
])
def test_accept_kernel_wrappers_and_metadata(source):
    result = validator.validate(PREFIX + source)
    assert result["valid"], result


@pytest.mark.parametrize("source,regression_type", [
    ("""def helper(x):
    return torch.sigmoid(torch.ops.npu.probe(x))
class ModelNew(nn.Module):
    def forward(self, x):
        return helper(x)
""", 3),
    ("""class ModelNew(nn.Module):
    def inner(self, x):
        return torch.ops.npu.probe(x).sum()
    def helper(self, x):
        return self.inner(x)
    def forward(self, x):
        return self.helper(x)
""", 3),
    ("""class ModelNew(nn.Module):
    def forward(self, xs):
        return torch.ops.npu.probe(sum(xs))
""", 3),
    ("""class Other:
    def helper(self, x):
        return torch.ops.npu.probe(x)
class ModelNew(nn.Module):
    def helper(self, x):
        return x
    def forward(self, x):
        return self.helper(x)
""", 2),
    ("""def helper(x):
    return torch.ops.npu.probe(x)
class ModelNew(nn.Module):
    def forward(self, x):
        return x.helper()
""", 2),
    ("""class ModelNew(nn.Module):
    def forward(self, x):
        def unused(y):
            return torch.ops.npu.probe(y)
        return x
""", 2),
    ("""def helper(x):
    return helper(x)
class ModelNew(nn.Module):
    def forward(self, x):
        return helper(x)
""", 2),
    ("""kernel = torch.ops.npu.probe
kernel = torch.sigmoid
class ModelNew(nn.Module):
    def forward(self, x):
        return kernel(x)
""", 2),
    ("""def helper(x):
    return torch.ops.npu.probe(x)
class ModelNew(nn.Module):
    def forward(self, x):
        return self.activation(helper(x))
""", 3),
    ("""def helper(xs):
    return torch.ops.npu.probe(sum(xs))
class ModelNew(nn.Module):
    def forward(self, xs):
        return helper(xs)
""", 3),
    ("""def helper(x):
    y = torch.ops.npu.probe(x)
    for i in range(x.numel()):
        y[i] = x[i] + x[i] * x[i] - x[i] / (x[i] + 1)
    return y
class ModelNew(nn.Module):
    def forward(self, x):
        return helper(x)
""", 4),
    ("""kernel = torch.ops.npu.probe
class ModelNew(nn.Module):
    def forward(self, kernel, x):
        return kernel(x)
""", 2),
    ("""class ModelNew(nn.Module):
    def forward(self, x):
        return torch.ops.npu.probe((lambda y: torch.sigmoid(y))(x))
""", 3),
    ("""class ModelNew(nn.Module):
    def forward(self, x):
        def unused(y=torch.sigmoid(x)):
            return y
        return torch.ops.npu.probe(x)
""", 3),
])
def test_reject_hidden_compute_and_unrelated_kernel_calls(source, regression_type):
    result = validator.validate(PREFIX + source)
    assert not result["valid"], result
    assert result["regression_type"] == regression_type, result


def test_validator_cli_remains_self_contained(tmp_path):
    script = tmp_path / SCRIPT.name
    script.write_bytes(SCRIPT.read_bytes())
    candidate = tmp_path / "model_new_ascendc.py"
    candidate.write_text(PREFIX + """class ModelNew(nn.Module):
    def helper(self, x):
        return torch.ops.npu.probe(x)
    def forward(self, x):
        return self.helper(x)
""")
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(candidate), "--json"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["valid"]
