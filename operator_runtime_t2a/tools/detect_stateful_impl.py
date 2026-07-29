#!/usr/bin/env python3
"""用法: detect_stateful_impl.py <task_dir>   退出码 0=通过 1=不通过 2=跳过"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_class(mod, name):
    cls = getattr(mod, name, None)
    if cls is None:
        raise AttributeError(f"{name} not found in {mod.__file__}")
    return cls


def _input_groups(mod):
    if hasattr(mod, "get_input_groups"):
        groups = mod.get_input_groups()
        if groups:
            return list(groups)
    if hasattr(mod, "get_inputs"):
        one = mod.get_inputs()
        if one:
            return [one]
    return []


def _to_device(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def _same(a, b) -> bool:
    """两个输出是否逐位相同(含嵌套结构)。"""
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: detect_stateful_impl.py <task_dir>", file=sys.stderr)
        return 2
    task_dir = Path(argv[0]).resolve()
    import torch
    try:
        import torch_npu  # noqa: F401
        device = "npu" if torch.npu.is_available() else "cpu"
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_mod = _load(task_dir / "model.py", "_sd_ref")
    cand_mod = _load(task_dir / "model_new_ascendc.py", "_sd_cand")
    groups = _input_groups(ref_mod)
    if len(groups) < 2:
        print("[stateful-detect] SKIP: 只有 1 组用例,无法做变输入探测")
        return 2

    init_inputs = list(ref_mod.get_init_inputs()) if hasattr(ref_mod, "get_init_inputs") else []
    g0, g1 = groups[0], groups[1]

    with torch.no_grad():
        ref = _find_class(ref_mod, "Model")(*init_inputs).to(device).eval()
        r0 = ref(*_to_device(g0, device))
        r1 = ref(*_to_device(g1, device))
        if _same(r0, r1):
            print("[stateful-detect] SKIP: golden 对这两组输入本就产生相同输出,判据不适用")
            return 2

        cand = _find_class(cand_mod, "ModelNew")(*init_inputs).to(device).eval()
        c0 = cand(*_to_device(g0, device))
        c1 = cand(*_to_device(g1, device))

    if _same(c0, c1):
        print(
            "[stateful-detect] FAIL: ModelNew 对两组不同输入返回了完全相同的输出,"
            "而 golden 的输出不同 —— 实现没有随输入计算。"
        )
        return 1
    print("[stateful-detect] PASS: 输出随输入变化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
