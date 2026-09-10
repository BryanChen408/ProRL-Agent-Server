#!/usr/bin/env python3
"""用法: detect_stateful_impl.py <task_dir>   退出码 0=通过 1=检测命中 2=跳过 3=检测异常"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _same(a, b) -> bool:
    """两个输出是否逐位相同(含嵌套结构)。"""
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
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

    utility_paths = [
        Path(__file__).resolve().with_name("input_contract.py"),
        task_dir.parent / ".claude/skills/ops-profiling/scripts/msprof_perf_summary.py",
        Path(__file__).resolve().parents[1] / "skills/ops-profiling/scripts/msprof_perf_summary.py",
        Path(__file__).resolve().parents[1] / ".claude/skills/ops-profiling/scripts/msprof_perf_summary.py",
    ]
    utility_path = next((p for p in utility_paths if p.is_file()), None)
    if utility_path is None:
        raise FileNotFoundError("canonical ops-profiling input utilities missing")
    utils = _load(utility_path, "_sd_input_utils")
    sys.path.insert(0, str(task_dir))
    sys.path.insert(0, str(task_dir / "kernel/build"))
    utils._seed_model(0, device)
    ref_mod = _load(task_dir / "model.py", "_sd_ref")
    cand_mod = _load(task_dir / "model_new_ascendc.py", "_sd_cand")
    groups = utils._resolve_input_groups(ref_mod)
    if len(groups) < 2:
        print("[stateful-detect] SKIP: 只有 1 组用例,无法做变输入探测")
        return 2

    utils._seed_model(0, device)
    init_inputs = ref_mod.get_init_inputs() if hasattr(ref_mod, "get_init_inputs") else []
    g0, g1 = groups[0], groups[1]

    def run(model, case, run_device=device):
        inputs = utils._move(utils._clone(case), run_device)
        args, kwargs = utils._bind_case(ref_mod.Model, inputs)
        return utils._clone(model(*args, **kwargs))

    with torch.no_grad():
        utils._seed_model(0, device)
        ref = utils._find_cls(ref_mod, "Model")(*utils._clone(init_inputs)).to(device).eval()
        try:
            r0, r1 = run(ref, g0), run(ref, g1)
        except Exception as exc:
            if device == "cpu":
                raise
            # Match verification_ascendc: unsupported reference operations may
            # run on CPU; the candidate must still execute on the leased NPU.
            print(f"[stateful-detect] reference fallback to CPU: {type(exc).__name__}: {exc}")
            ref = ref.to("cpu")
            r0, r1 = run(ref, g0, "cpu"), run(ref, g1, "cpu")
        if _same(r0, r1):
            print("[stateful-detect] SKIP: golden 对这两组输入本就产生相同输出,判据不适用")
            return 2

        utils._seed_model(0, device)
        cand = utils._find_cls(cand_mod, "ModelNew")(*utils._clone(init_inputs)).to(device).eval()
        c0, c1 = run(cand, g0), run(cand, g1)

    if _same(c0, c1):
        print(
            "[stateful-detect] FAIL: ModelNew 对两组不同输入返回了完全相同的输出,"
            "而 golden 的输出不同 —— 实现没有随输入计算。"
        )
        return 1
    print("[stateful-detect] PASS: 输出随输入变化")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        print("[stateful-detect] ERROR: 检测未完成", file=sys.stderr)
        traceback.print_exc()
        code = 3
    raise SystemExit(code)
