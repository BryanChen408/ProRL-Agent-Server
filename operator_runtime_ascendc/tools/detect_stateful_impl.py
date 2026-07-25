#!/usr/bin/env python3
"""C 方案:探测 ModelNew 是否"不真算" —— 缓存/常量输出/有状态实现。

【为什么需要】performance.py 的测速是:对**同一组输入**连调 1+warmup+repeats(默认 56)次取平均。
它隐含假设"每次调用都真的算一遍" —— 对真人写的代码永远成立,但被测对象是一个正在被 reward
优化的模型时不成立。只要写成:

    def forward(self, x, dim):
        if self._cache is None:
            self._cache = torch.ops.npu.xxx(x, dim)   # 只有第 1 次真算
        return self._cache                             # 后面 55 次直接返回

对拍(只调一次)照过、退化检测(确实调了 torch.ops.npu)照过,而平均延迟趋近 0 →
speedup 可达几百 → reward 0.75+0.25*tanh(ln speedup) ≈ 1.0(满分)。
写一个真正快 2 倍的 kernel 才 0.90 —— 于是 GRPO 会收敛到"写个能编过的 kernel + 缓存结果"。

【判据】喂两组**不同**输入,输出必须跟着变。
  - 先用 golden Model 验证这两组输入本该产生不同输出(否则该算子对这两组恰好同结果,跳过);
  - 再看 ModelNew:两组输入 → 输出相同 = 不真算 → FAIL。
  射程覆盖 self._cache / 全局缓存 / lru_cache / 返回常量 / 忽略输入。

用法: python3 detect_stateful_impl.py <task_dir>     退出码 0=通过 1=判定不真算 2=无法判定(跳过)
"""
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
            "[stateful-detect] FAIL: ModelNew 对两组不同输入返回了**完全相同**的输出,"
            "而 golden 的输出是不同的 —— 说明实现没有真正随输入计算"
            "(缓存 / 返回常量 / 忽略输入)。性能测量对同一输入连调数十次,这类实现会测出虚高 speedup。"
        )
        return 1
    print("[stateful-detect] PASS: 输出随输入变化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
