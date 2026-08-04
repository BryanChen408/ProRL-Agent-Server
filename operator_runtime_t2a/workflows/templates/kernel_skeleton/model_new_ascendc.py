import glob as _glob
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "{op_name}_ext*")

# register.cpp 用 TORCH_LIBRARY(无 PYBIND11_MODULE),裸 import 会因缺 PyInit 失败;
# 真正注册靠 fallback 的 torch.ops.load_library(...)。这段加载逻辑原样保留,别改。
try:
    import {op_name}_ext  # noqa: F401
except ImportError:
    if _LIB_PATTERN not in "".join(sys.path):
        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按你的算子改:参数、个数、返回值。必须真调 torch.ops.npu.{op_name}
        # (AST 退化检测会查),不要加 plain-torch 兜底。
        return torch.ops.npu.{op_name}(x)
