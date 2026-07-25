#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent

# Import shared utility functions from canonical performance.py.
# Canonical source: ascendc-performance-analyzer/script/performance.py
_PERF_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "ascendc-performance-analyzer" / "script"
)
if str(_PERF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS))

from performance import (  # noqa: E402
    _load_module,
    _find_model_class,
    _clone_value,
    _move_to_device,
    _get_device,
)


# ---------------------------------------------------------------------------
# 精度对比标准（参考 精度对比方法.md）
# ---------------------------------------------------------------------------
# dtype_str -> Threshold (MERE 通过阈值)
PRECISION_THRESHOLDS = {
    "float16": 2 ** -10,       # ≈ 9.77e-4
    "bfloat16": 2 ** -7,       # ≈ 7.81e-3
    "float32": 2 ** -13,       # ≈ 1.22e-4
    "float64": 2 ** -30,       # ≈ 9.31e-10
    "complex64": 2 ** -13,     # 实部/虚部各为 float32
    "complex128": 2 ** -30,    # 实部/虚部各为 float64
    "hifloat32": 2 ** -11,     # ≈ 4.88e-4
    "float8_e4m3": 2 ** -3,    # ≈ 0.125
    "float8_e5m2": 2 ** -2,    # ≈ 0.25
}

# 量化整数输出的 LSB tolerance：dynamic quant / smooth quant 等算子的
# int8/int16 输出，在 NPU 实现侧通常经过 fp32→fp16→int 的中转 cast，
# 与 PyTorch CPU 全 fp32 .round() 比较时会出现 ±1 LSB 噪声。
# 这里允许每个元素最多差 1 个 LSB（相当于浮点的 1 ulp 容忍）。
# int32 / int64 / bool 不带 round 语义（通常是索引、计数、mask），
# 仍走严格 torch.equal。
INT_LSB_TOLERANCE = {
    torch.int8: 1,
    torch.int16: 1,
}


def _compute_mere(actual: torch.Tensor, golden: torch.Tensor, threshold: float, eps: float = 1e-7) -> float:
    """计算平均相对误差 (Mean Relative Error).

    MERE = mean(|actual - golden| / (|golden| + eps))

    对于绝对误差已经小于阈值的元素，相对误差直接视为 0，
    避免因 golden 值极小导致相对误差被不合理放大。
    """
    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + eps)
    # 绝对误差已在阈值内的元素，不计入相对误差
    rel = torch.where(diff < threshold * 1e-3, 0.0, rel)
    if rel.numel() == 0:
        return 0.0
    return float(rel.mean().item())


def _compute_mare(actual: torch.Tensor, golden: torch.Tensor, threshold: float, eps: float = 1e-7) -> float:
    """计算最大相对误差 (Max Relative Error).

    MARE = max(|actual - golden| / (|golden| + eps))

    对于绝对误差已经小于阈值的元素，相对误差直接视为 0，
    避免因 golden 值极小导致相对误差被不合理放大。
    """
    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + eps)
    rel = torch.where(diff < threshold * 1e-3, 0.0, rel)
    if rel.numel() == 0:
        return 0.0
    return float(rel.max().item())


def _get_threshold_for_tensor(t: torch.Tensor) -> float:
    """根据张量 dtype 获取 MERE 阈值."""
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
    }
    # 安全获取可能不存在的 dtype（取决于 PyTorch 版本）
    _hifloat32 = getattr(torch, "hifloat32", None)
    if _hifloat32 is not None:
        dtype_map[_hifloat32] = "hifloat32"
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if _float8_e4m3fn is not None:
        dtype_map[_float8_e4m3fn] = "float8_e4m3"
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    if _float8_e5m2 is not None:
        dtype_map[_float8_e5m2] = "float8_e5m2"

    dtype_str = dtype_map.get(t.dtype)
    if dtype_str is not None:
        return PRECISION_THRESHOLDS.get(dtype_str, 1e-2)

    # 其他浮点类型回退到 float32 阈值
    if torch.is_floating_point(t):
        return PRECISION_THRESHOLDS.get("float32", 1e-2)

    # 复数类型回退到对应浮点阈值
    if t.is_complex():
        return PRECISION_THRESHOLDS.get("float32", 1e-2)

    # 非浮点类型使用宽松阈值
    return 1e-2


def _check_precision_mere_mare(actual: torch.Tensor, golden: torch.Tensor) -> tuple:
    """根据《精度对比方法.md》判定数值精度是否通过.

    通过标准:
        MERE < Threshold 且 MARE < 10 * Threshold

    Returns:
        (passed: bool, mere: float, mare: float, threshold: float, mare_threshold: float)
    """
    threshold = _get_threshold_for_tensor(golden)
    mare_threshold = 10 * threshold

    # NaN 处理：两者都为 NaN 的位置视为匹配并过滤；仅一方为 NaN 直接判定失败
    actual_nan = torch.isnan(actual)
    golden_nan = torch.isnan(golden)
    if (actual_nan ^ golden_nan).any():
        return False, float('inf'), float('inf'), threshold, mare_threshold

    # Inf 处理：两者都为 Inf 且同号的位置视为匹配并过滤；仅一方为 Inf 或符号不同直接判定失败
    actual_inf = torch.isinf(actual)
    golden_inf = torch.isinf(golden)
    if (actual_inf ^ golden_inf).any():
        return False, float('inf'), float('inf'), threshold, mare_threshold
    if actual_inf.any():
        actual_sign = torch.sign(actual[actual_inf])
        golden_sign = torch.sign(golden[actual_inf])
        if not torch.equal(actual_sign, golden_sign):
            return False, float('inf'), float('inf'), threshold, mare_threshold

    valid_mask = ~(actual_nan | actual_inf)
    if valid_mask.any():
        actual_valid = actual[valid_mask]
        golden_valid = golden[valid_mask]
    else:
        # 所有元素均为双 NaN 或双 Inf
        return True, 0.0, 0.0, threshold, mare_threshold

    mere = _compute_mere(actual_valid, golden_valid, threshold)
    mare = _compute_mare(actual_valid, golden_valid, threshold)

    passed = mere < threshold and mare < mare_threshold
    return passed, mere, mare, threshold, mare_threshold


def _normalize_output(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_normalize_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_output(item) for key, item in value.items()}
    return value


def _contains_int8_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.int8
    if isinstance(value, list):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_int8_tensor(item) for item in value.values())
    return False


def _find_first_mismatch(lhs, rhs, mismatch_mask):
    """Return a string describing the first mismatched element, or empty if no mismatch."""
    if not mismatch_mask.numel():
        return ""
    mismatch_count = mismatch_mask.sum().item()
    if not mismatch_count:
        return ""
    first_linear_idx = int(torch.nonzero(mismatch_mask.reshape(-1), as_tuple=False)[0].item())
    if lhs.ndim == 0:
        first_index = ()
        lhs_val = lhs.item()
        rhs_val = rhs.item()
    else:
        rem = first_linear_idx
        first_index = [0] * lhs.ndim
        for d in range(lhs.ndim - 1, -1, -1):
            first_index[d] = rem % lhs.shape[d]
            rem //= lhs.shape[d]
        first_index = tuple(first_index)
        lhs_val = lhs[first_index].item()
        rhs_val = rhs[first_index].item()
    return f", first_mismatch(index={first_index}, ref={lhs_val}, cand={rhs_val})"


def _complex_diff_summary(lhs, rhs):
    passed_r, mere_r, mare_r, thr_r, mthr_r = _check_precision_mere_mare(rhs.real, lhs.real)
    passed_i, mere_i, mare_i, thr_i, mthr_i = _check_precision_mere_mare(rhs.imag, lhs.imag)
    lhs_fp = torch.view_as_real(lhs).to(torch.float32)
    rhs_fp = torch.view_as_real(rhs).to(torch.float32)
    diff = (lhs_fp - rhs_fp).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
        f"MERE(real={mere_r:.6g}, imag={mere_i:.6g}), "
        f"MARE(real={mare_r:.6g}, imag={mare_i:.6g}), "
        f"threshold={thr_r:.6g}, mare_threshold={mthr_r:.6g}, "
        f"passed_real={passed_r}, passed_imag={passed_i}"
    )


def _float_diff_summary(lhs, rhs):
    lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
    rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
    diff = (lhs_fp - rhs_fp).abs()
    both_inf_mask = torch.isinf(lhs_fp) & torch.isinf(rhs_fp) & (torch.sign(lhs_fp) == torch.sign(rhs_fp))
    diff[both_inf_mask] = 0.0
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    passed, mere, mare, threshold, mare_threshold = _check_precision_mere_mare(rhs, lhs)
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
        f"MERE={mere:.6g}, MARE={mare:.6g}, "
        f"threshold={threshold:.6g}, mare_threshold={mare_threshold:.6g}, "
        f"passed={passed}"
    )


def _int_diff_summary(lhs, rhs, total):
    lhs_i32 = lhs.to(torch.int32)
    rhs_i32 = rhs.to(torch.int32)
    delta = rhs_i32 - lhs_i32
    abs_diff = delta.abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() else 0
    mean_abs = abs_diff.float().mean().item() if abs_diff.numel() else 0.0
    mismatch_mask = delta != 0
    mismatch_count = mismatch_mask.sum().item() if delta.numel() else 0
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    cand_gt_ref = ((delta > 0) & mismatch_mask).sum().item() if delta.numel() else 0
    cand_lt_ref = ((delta < 0) & mismatch_mask).sum().item() if delta.numel() else 0
    first_mismatch = _find_first_mismatch(lhs, rhs, mismatch_mask) if mismatch_count else ""
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"unequal_elements={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}, "
        f"max_abs_diff={max_abs}, mean_abs_diff={mean_abs:.6g}, "
        f"cand_gt_ref={cand_gt_ref}, cand_lt_ref={cand_lt_ref}"
        f"{first_mismatch}"
    )


def _tensor_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.shape != rhs.shape:
        return f"shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    if lhs.is_complex() or rhs.is_complex():
        return _complex_diff_summary(lhs, rhs)

    if torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
        return _float_diff_summary(lhs, rhs)

    return _int_diff_summary(lhs, rhs, lhs.numel())


def _compare_tensors(lhs, rhs, path):
    """Compare two tensors using MERE/MARE precision standards."""
    if lhs.shape != rhs.shape:
        return False, f"{path}: shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    needs_numeric_check = (
        torch.is_floating_point(lhs) or torch.is_floating_point(rhs)
        or lhs.is_complex() or rhs.is_complex()
    )
    if not needs_numeric_check:
        tol = INT_LSB_TOLERANCE.get(lhs.dtype) if lhs.dtype == rhs.dtype else None
        if tol is not None:
            diff = (rhs.to(torch.int32) - lhs.to(torch.int32)).abs()
            max_abs = diff.max().item() if diff.numel() else 0
            if max_abs <= tol:
                return True, (
                    f"{path}: matched within ±{tol} LSB "
                    f"(max_abs_diff={max_abs}, dtype={lhs.dtype})"
                )
            return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"
        if torch.equal(lhs, rhs):
            return True, f"{path}: matched"
        return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"

    if lhs.is_complex() or rhs.is_complex():
        real_passed, real_mere, real_mare, real_thr, real_mthr = _check_precision_mere_mare(rhs.real, lhs.real)
        imag_passed, imag_mere, imag_mare, imag_thr, imag_mthr = _check_precision_mere_mare(rhs.imag, lhs.imag)
        passed = real_passed and imag_passed
        if passed:
            return True, (
                f"{path}: matched, "
                f"MERE(real={real_mere:.6g}, imag={imag_mere:.6g}), "
                f"MARE(real={real_mare:.6g}, imag={imag_mare:.6g}), "
                f"threshold={real_thr:.6g}, mare_threshold={real_mthr:.6g}"
            )
        return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"

    passed, mere, mare, threshold, mare_threshold = _check_precision_mere_mare(rhs, lhs)
    if passed:
        return True, (
            f"{path}: matched, "
            f"MERE={mere:.6g}, MARE={mare:.6g}, "
            f"threshold={threshold:.6g}, mare_threshold={mare_threshold:.6g}"
        )
    return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"


def _compare_structured(lhs, rhs, compare_leaf, path: str = "output"):
    """Generic structural comparison. Delegates leaf values to `compare_leaf`."""
    if type(lhs) is not type(rhs):
        return False, f"{path}: type mismatch: ref={type(lhs).__name__}, cand={type(rhs).__name__}"

    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return False, f"{path}: list length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_structured(a, b, compare_leaf, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return False, f"{path}: tuple length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_structured(a, b, compare_leaf, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, dict):
        if lhs.keys() != rhs.keys():
            return False, f"{path}: dict keys mismatch: ref={sorted(lhs.keys())}, cand={sorted(rhs.keys())}"
        for key in lhs:
            ok, message = _compare_structured(lhs[key], rhs[key], compare_leaf, f"{path}.{key}")
            if not ok:
                return False, message
        return True, f"{path}: matched"

    return compare_leaf(lhs, rhs, path)


def _compare_values(lhs, rhs, path: str = "output"):
    """Recursively compare values using MERE/MARE precision for Tensors."""

    def _compare_leaf(a, b, p):
        if isinstance(a, torch.Tensor):
            return _compare_tensors(a, b, p)
        if a == b:
            return True, f"{p}: matched"
        return False, f"{p}: value mismatch: ref={a}, cand={b}"

    return _compare_structured(lhs, rhs, _compare_leaf, path)


def _resolve_task_dir(op: str, workdir: Path = WORKDIR) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = workdir / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _format_tensor_summary(tensor: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _summarize_value(value, name: str):
    if isinstance(value, torch.Tensor):
        return [f"{name}: {_format_tensor_summary(value)}"]
    if isinstance(value, list):
        lines = [f"{name}: list[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, tuple):
        lines = [f"{name}: tuple[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, dict):
        lines = [f"{name}: dict[{len(value)}]"]
        for key, item in value.items():
            lines.extend(_summarize_value(item, f"{name}.{key}"))
        return lines
    return [f"{name}: {type(value).__name__}({value})"]


def _get_input_groups(module):
    # Prefer get_input_groups(); fall back to get_inputs() wrapped in a list.
    if hasattr(module, "get_input_groups"):
        input_groups = module.get_input_groups()
        if not isinstance(input_groups, list) or not input_groups:
            raise ValueError("get_input_groups() must return a non-empty list")
        return input_groups

    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("get_inputs() must return a non-empty list")
        return [inputs]

    raise AttributeError(f"Neither get_input_groups() nor get_inputs() found in {module.__file__}")


def _make_verification_report(op):
    return {
        "op": op,
        "ok": False,
        "device": str(_get_device()),
        "task_dir": "",
        "reference": "",
        "candidate": "",
        "kernel_build_dir": "",
        "inputs": [],
        "comparisons": [],
        "comparison": "",
        "error": "",
    }


def _setup_paths(kernel_build_dir):
    inserted_paths = []
    paths_to_add = [str(WORKDIR)]
    if kernel_build_dir.is_dir():
        paths_to_add.append(str(kernel_build_dir))
    else:
        import warnings
        warnings.warn(
            f"{kernel_build_dir} not found; assuming model_new_ascendc.py handles its own import path.",
            UserWarning, stacklevel=2,
        )
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted_paths.append(p)
    return inserted_paths


def _execute_models(ref_model, cand_model, input_groups, device):
    """Run both models on all input groups, returning normalized outputs and summaries."""
    ref_outputs = []
    cand_outputs = []
    input_summaries = []
    for index, inputs in enumerate(input_groups):
        ref_inputs = _move_to_device(_clone_value(inputs), device)
        cand_inputs = _move_to_device(_clone_value(inputs), device)
        input_summaries.extend(_summarize_value(ref_inputs, f"inputs[{index}]"))

        with torch.no_grad():
            ref_out = ref_model(*ref_inputs)
            cand_out = cand_model(*cand_inputs)

        if hasattr(ref_model, "postprocess_output"):
            ref_out = ref_model.postprocess_output(ref_out, inputs)
            cand_out = ref_model.postprocess_output(cand_out, inputs)

        ref_outputs.append(_normalize_output(ref_out))
        cand_outputs.append(_normalize_output(cand_out))

    return ref_outputs, cand_outputs, input_summaries


def _run_comparisons(ref_model, cand_model, input_groups, device):
    all_ok = True
    comparisons = []
    ref_outputs, cand_outputs, input_summaries = _execute_models(
        ref_model, cand_model, input_groups, device)
    for index, (ref_out, cand_out) in enumerate(zip(ref_outputs, cand_outputs)):
        ok, comparison = _compare_values(ref_out, cand_out, path=f"output[{index}]")
        comparisons.append(f"case[{index}]: {comparison}")
        all_ok = all_ok and ok
    return all_ok, comparisons, input_summaries


def _run_verification(op: str):
    report = _make_verification_report(op)

    task_dir = _resolve_task_dir(op)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_ascendc.py"
    kernel_build_dir = task_dir / "kernel" / "build"
    report["task_dir"] = str(task_dir)
    report["reference"] = str(ref_path)
    report["candidate"] = str(cand_path)
    report["kernel_build_dir"] = str(kernel_build_dir)

    if not ref_path.is_file():
        report["error"] = f"missing reference model: {ref_path}"
        return report
    if not cand_path.is_file():
        report["error"] = f"missing candidate model: {cand_path}"
        return report

    inserted_paths = _setup_paths(kernel_build_dir)
    try:
        ref_module = _load_module(ref_path, f"{op}_ref_model")
        cand_module = _load_module(cand_path, f"{op}_ascendc_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)
        if hasattr(cand_module, "get_init_inputs"):
            init_inputs = cand_module.get_init_inputs()
        else:
            init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
        input_groups = _get_input_groups(ref_module)
        device = _get_device()

        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
        cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

        all_ok, comparisons, input_summaries = _run_comparisons(
            ref_model, cand_model, input_groups, device)

        report["inputs"] = input_summaries
        report["comparisons"] = comparisons
        report["comparison"] = "\n".join(comparisons)
        report["ok"] = all_ok
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERIFICATION_ASCENDC_DEBUG") == "1":
            raise
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        for p in inserted_paths:
            if p in sys.path:
                sys.path.remove(p)


def verify(op: str) -> bool:
    return _run_verification(op)["ok"]


def _print_report(report, title="AscendC Verification Report",
                  extra_header_lines=None, debug_env_var="VERIFICATION_ASCENDC_DEBUG"):
    status = "PASS" if report["ok"] else "FAIL"
    lines = []
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(f"Status    : {status}")
    lines.append(f"Operator  : {report['op']}")
    lines.append(f"Device    : {report['device']}")
    lines.append(f"Task Dir  : {report['task_dir']}")
    lines.append(f"Reference : {report['reference']}")
    lines.append(f"Candidate : {report['candidate']}")
    if extra_header_lines:
        lines.extend(extra_header_lines)

    if report["inputs"]:
        lines.append("-" * 72)
        lines.append("Inputs")
        lines.append("-" * 72)
        for line in report["inputs"]:
            lines.append(line)

    lines.append("-" * 72)
    lines.append("Comparison")
    lines.append("-" * 72)
    if report["comparison"]:
        lines.append(report["comparison"])
    elif report["error"]:
        lines.append(report["error"])
    else:
        lines.append("No comparison information available")

    if report["error"] and os.environ.get(debug_env_var) == "1":
        lines.append("-" * 72)
        lines.append("Traceback")
        lines.append("-" * 72)
        lines.append(report.get("traceback", ""))

    lines.append("-" * 72)
    lines.append(f"Result: {'pass' if report['ok'] else 'fail'}")

    logger.info("\n".join(lines))


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python .claude/skills/ascendc-translator/scripts/verification_ascendc.py <op>")
        logger.error("Result: fail")
        raise SystemExit(1)

    report = _run_verification(sys.argv[1])
    _print_report(report,
                  extra_header_lines=[f"Kernel    : {report['kernel_build_dir']}"])
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
