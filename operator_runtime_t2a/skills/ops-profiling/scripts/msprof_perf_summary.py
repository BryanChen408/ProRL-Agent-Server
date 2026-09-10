#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
# msprof 解析 & 归档 & 对比测试脚本（统一入口）
#
# 支持三种模式：
#   1. 标准模式 (默认): 解析 PROF_GROUP，生成 summary.txt 并归档 CSV
#      python3 msprof_perf_summary.py <PROF_GROUP_dir> <ops_dir>
#
#   2. 对比模式 (--compare): 对算子目录做 model.py vs model_new_ascendc.py 对比测试
#      python3 msprof_perf_summary.py --compare --output-dir <op_dir> [--warm-up=N] [--device=N]
#
#   3. 批量模式 (--batch): 扫描多个算子目录，汇总批量报告
#      python3 msprof_perf_summary.py --batch <base_dir> [--output-md <path>] [--output-json <path>]
#
# 归档位置 (与 perf_summary.py 保持一致):
#     <ops_dir>/docs/perf/round_NNN/
#         op_summary_<Metric>.csv    (7 份)
#         task_time.csv              (若存在)
#         op_statistic.csv           (若存在)
#         summary.txt                (合并后的统计摘要)
# ----------------------------------------------------------------------------------------------------------

import argparse
import csv
import importlib.util
import inspect
import json
import logging
import glob
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

PERF_TARGET_SPEEDUP = 1.1

METRICS = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
]


# ============================================================================
# 通用工具函数
# ============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().rstrip("\t ")
    if s in ("", "N/A", "NA", "-"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    return int(safe_float(val, default))


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


# ============================================================================
# 标准模式：解析 PROF_GROUP
# ============================================================================

def find_op_summary(prof_metric_dir: str) -> Optional[str]:
    pattern = os.path.join(prof_metric_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv")
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[-1] if hits else None


def pick_target_row(rows: List[Dict[str, str]], target_name: Optional[str]) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    if target_name:
        for r in rows:
            if r.get("Op Name", "").strip() == target_name:
                return r
    ai_core_rows = []
    for r in rows:
        if "AI_CORE" in r.get("Task Type", "") \
                or "AIV" in r.get("Task Type", "") \
                or "MIX" in r.get("Task Type", ""):
            ai_core_rows.append(r)
    candidates = ai_core_rows or rows
    return max(candidates, key=lambda r: safe_float(r.get("Task Duration(us)")))


def _merge_row_values(merged: Dict[str, Any], row: Dict[str, str]) -> None:
    for k, v in row.items():
        if k in (None, ""):
            continue
        if k not in merged:
            merged[k] = v
        else:
            old = merged[k]
            if (old in (None, "", "N/A", "NA")) and v not in (None, "", "N/A", "NA"):
                merged[k] = v
            elif safe_float(old) == 0 and safe_float(v) != 0:
                merged[k] = v


def merge_metric_rows(group_dir: str, target_name: Optional[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged["_metric_sources"] = {}
    merged["_missing_metrics"] = []

    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            merged["_missing_metrics"].append(metric)
            continue
        csv_path = find_op_summary(prof_metric_dir)
        if not csv_path:
            merged["_missing_metrics"].append(metric)
            continue
        rows = read_csv_rows(csv_path)
        row = pick_target_row(rows, target_name)
        if not row:
            merged["_missing_metrics"].append(metric)
            continue
        merged["_metric_sources"][metric] = csv_path
        _merge_row_values(merged, row)
    return merged


def load_per_core_cycles(group_dir: str) -> List[Tuple[int, int]]:
    candidates = glob.glob(os.path.join(group_dir, "PROF_Sample", "PROF_*", "device_0", "sqlite", "aicore.db"))
    if not candidates:
        return []
    db = sorted(candidates)[-1]
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        rows = list(cur.execute(
            "SELECT coreid, SUM(task_cyc) FROM AICoreOriginalData WHERE task_cyc>0 GROUP BY coreid ORDER BY coreid"
        ))
        conn.close()
        return [(int(cid), int(cyc)) for cid, cyc in rows if cid is not None]
    except sqlite3.Error:
        return []


def per_core_balance_section(merged: Dict[str, Any], group_dir: str) -> List[str]:
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return []
    aicore_time_us = safe_float(merged.get("aicore_time(us)"))
    if aicore_time_us <= 0:
        return []
    max_cyc = max(c for _, c in core_rows)
    if max_cyc <= 0:
        return []
    ns_per_cyc = aicore_time_us * 1000.0 / max_cyc
    freq_ghz = 1.0 / ns_per_cyc
    times = [(cid, cyc * ns_per_cyc / 1000.0) for cid, cyc in core_rows]
    t_values = [t for _, t in times]
    t_min = min(t_values)
    t_max = max(t_values)
    t_avg = statistics.mean(t_values)
    spread_pct = (t_max - t_min) / t_max * 100.0 if t_max > 0 else 0.0

    if spread_pct < 10:
        verdict = "达标 (<10%)"
    elif spread_pct < 30:
        verdict = "警告 (10~30%)"
    else:
        verdict = "严重问题 (>30%)"

    lines = ["", "--- 逐核负载均衡 (sample-based aicore.db) ---"]
    lines.append(f"  有效核数: {len(times)}  | 主频推算: {freq_ghz:.3f} GHz ({ns_per_cyc:.4f} ns/cycle)")
    lines.append(f"  min={t_min:.3f}us  avg={t_avg:.3f}us  max={t_max:.3f}us")
    lines.append(f"  (max-min)/max = {spread_pct:.2f}%  ->  {verdict}")

    sorted_desc = sorted(times, key=lambda x: -x[1])
    slow_top = sorted_desc[:3]
    fast_top = sorted_desc[-3:][::-1]
    lines.append("  Top-3 慢核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in slow_top))
    lines.append("  Top-3 快核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in fast_top))

    sorted_by_id = sorted(times, key=lambda x: x[0])
    if len(sorted_by_id) >= 4:
        mid = len(sorted_by_id) // 2
        g1 = [t for _, t in sorted_by_id[:mid]]
        g2 = [t for _, t in sorted_by_id[mid:]]
        g1_avg = statistics.mean(g1)
        g2_avg = statistics.mean(g2)
        gap = abs(g1_avg - g2_avg) / max(g1_avg, g2_avg) * 100.0
        if gap >= 2.0:
            lines.append(
                f"  [提示] 前半段 core 均值 {g1_avg:.2f}us vs 后半段 {g2_avg:.2f}us，"
                f"差距 {gap:.2f}%"
            )
            lines.append(
                f"         疑似两簇 (NUMA / L2 slice) 负载偏斜，"
                f"建议尝试 block swat / 尾轮均衡策略。"
            )
    return lines


def archive_per_core_csv(group_dir: str, round_dir: str) -> Optional[str]:
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return None
    out = os.path.join(round_dir, "per_core_cycles.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coreid", "task_cycles"])
        for cid, cyc in core_rows:
            w.writerow([cid, cyc])
    return out


def archive_csvs(group_dir: str, round_dir: str) -> List[str]:
    os.makedirs(round_dir, exist_ok=True)
    copied = []
    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            continue
        op_csv = find_op_summary(prof_metric_dir)
        if op_csv:
            dst = os.path.join(round_dir, f"op_summary_{metric}.csv")
            shutil.copy2(op_csv, dst)
            copied.append(os.path.basename(dst))
        mso_dir = os.path.dirname(op_csv) if op_csv else None
        if mso_dir:
            _copy_extra_csvs(mso_dir, metric, round_dir, copied)
    return copied


def _copy_extra_csvs(mso_dir: str, metric: str, round_dir: str, copied: List[str]) -> None:
    for extra in ("op_statistic_", "task_time_", "api_statistic_"):
        for f in sorted(glob.glob(os.path.join(mso_dir, f"{extra}*.csv"))):
            name = f"{extra.rstrip('_')}_{metric}.csv"
            dst = os.path.join(round_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied.append(os.path.basename(dst))
            break


def fmt_ratio(val: Any, width: int = 6) -> str:
    v = safe_float(val) * 100.0
    return f"{v:>{width}.2f}%"


def fmt_float(val: Any, width: int = 10, prec: int = 2) -> str:
    return f"{safe_float(val):>{width}.{prec}f}"


def _add_memory_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _mem_keys = [
        "aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)",
        "aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)",
        "aic_l1_read_bw(GB/s)", "aic_l1_write_bw(GB/s)",
        "aiv_ub_read_bw(GB/s)", "aiv_ub_write_bw(GB/s)",
    ]
    has_mem = any(safe_float(merged.get(k)) > 0 for k in _mem_keys)
    if not has_mem:
        return
    lines.append("")
    lines.append("--- Memory 带宽 (aic-metrics=Memory) ---")
    mem_rows = [
        ("aic main_mem read", "aic_main_mem_read_bw(GB/s)"),
        ("aic main_mem write", "aic_main_mem_write_bw(GB/s)"),
        ("aiv main_mem read", "aiv_main_mem_read_bw(GB/s)"),
        ("aiv main_mem write", "aiv_main_mem_write_bw(GB/s)"),
        ("aic L1 read", "aic_l1_read_bw(GB/s)"),
        ("aic L1 write", "aic_l1_write_bw(GB/s)"),
        ("aiv UB read", "aiv_ub_read_bw(GB/s)"),
        ("aiv UB write", "aiv_ub_write_bw(GB/s)"),
    ]
    for label, key in mem_rows:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_l0_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _l0_keys = [
        "aic_l0a_read_bw(GB/s)", "aic_l0a_write_bw(GB/s)",
        "aic_l0b_read_bw(GB/s)", "aic_l0b_write_bw(GB/s)",
        "aic_l0c_read_bw_cube(GB/s)", "aic_l0c_write_bw_cube(GB/s)",
    ]
    has_l0 = any(safe_float(merged.get(k)) > 0 for k in _l0_keys)
    if not has_l0:
        return
    lines.append("")
    lines.append("--- MemoryL0 ---")
    for label, key in [
        ("L0A read", "aic_l0a_read_bw(GB/s)"),
        ("L0A write", "aic_l0a_write_bw(GB/s)"),
        ("L0B read", "aic_l0b_read_bw(GB/s)"),
        ("L0B write", "aic_l0b_write_bw(GB/s)"),
        ("L0C read (cube)", "aic_l0c_read_bw_cube(GB/s)"),
        ("L0C write (cube)", "aic_l0c_write_bw_cube(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_ub_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _ub_keys = [
        "aiv_ub_read_bw_vector(GB/s)", "aiv_ub_write_bw_vector(GB/s)",
        "aiv_ub_read_bw_scalar(GB/s)", "aiv_ub_write_bw_scalar(GB/s)",
        "aic_ub_read_bw_scalar(GB/s)", "aic_ub_write_bw_scalar(GB/s)",
        "aiv_fixp2ub_write_bw(GB/s)", "aic_fixp2ub_write_bw(GB/s)",
    ]
    has_ub = any(safe_float(merged.get(k)) > 0 for k in _ub_keys)
    if not has_ub:
        return
    lines.append("")
    lines.append("--- MemoryUB ---")
    for label, key in [
        ("UB read (vector)", "aiv_ub_read_bw_vector(GB/s)"),
        ("UB write (vector)", "aiv_ub_write_bw_vector(GB/s)"),
        ("UB read (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
        ("UB write (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ("aic UB read (scalar)", "aic_ub_read_bw_scalar(GB/s)"),
        ("aic UB write (scalar)", "aic_ub_write_bw_scalar(GB/s)"),
        ("aiv fixp2ub write", "aiv_fixp2ub_write_bw(GB/s)"),
        ("aic fixp2ub write", "aic_fixp2ub_write_bw(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_l2cache_section(lines: List[str], merged: Dict[str, Any]) -> None:
    l2_fields_aic = [
        ("aic read hit", "aic_read_local_l2_hit"),
        ("aic read miss", "aic_read_local_l2_miss"),
        ("aic read victim", "aic_read_local_l2_victim"),
        ("aic write hit", "aic_write_local_l2_hit"),
        ("aic write miss", "aic_write_local_l2_miss"),
        ("aic write victim", "aic_write_local_l2_victim"),
    ]
    l2_fields_aiv = [
        ("aiv read hit", "aiv_read_local_l2_hit"),
        ("aiv read miss", "aiv_read_local_l2_miss"),
        ("aiv read victim", "aiv_read_local_l2_victim"),
        ("aiv write hit", "aiv_write_local_l2_hit"),
        ("aiv write miss", "aiv_write_local_l2_miss"),
        ("aiv write victim", "aiv_write_local_l2_victim"),
    ]
    l2_has = any(safe_float(merged.get(k)) > 0 for _, k in l2_fields_aic + l2_fields_aiv)
    if not l2_has:
        return
    lines.append("")
    lines.append("--- L2Cache ---")

    def _emit(group_label, fields):
        hit = safe_float(merged.get(fields[0][1]))
        miss = safe_float(merged.get(fields[1][1]))
        total = hit + miss
        if total > 0:
            rate = hit / total * 100.0
            lines.append(f"  {group_label} read: hit={int(hit)} miss={int(miss)} hit_rate={rate:.2f}%")
        whit = safe_float(merged.get(fields[3][1]))
        wmiss = safe_float(merged.get(fields[4][1]))
        wtotal = whit + wmiss
        if wtotal > 0:
            rate = whit / wtotal * 100.0
            lines.append(f"  {group_label} write: hit={int(whit)} miss={int(wmiss)} hit_rate={rate:.2f}%")

    _emit("aic", l2_fields_aic)
    _emit("aiv", l2_fields_aiv)


def _add_rc_section(lines: List[str], merged: Dict[str, Any]) -> None:
    rc_fields = [
        ("vec_bank_cflt", "aiv_vec_bank_cflt_ratio"),
        ("vec_resc_cflt", "aiv_vec_resc_cflt_ratio"),
    ]
    if not any(safe_float(merged.get(k)) > 0 for _, k in rc_fields):
        return
    lines.append("")
    lines.append("--- ResourceConflict ---")
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for label, key in rc_fields]
    lines.append("  " + " | ".join(parts))


def _add_arith_section(lines: List[str], merged: Dict[str, Any]) -> None:
    arith_fields = [
        ("mac_fp16", "aic_mac_fp16_ratio"),
        ("mac_int8", "aic_mac_int8_ratio"),
    ]
    has_arith_fields = any(
        safe_float(merged.get(k)) > 0 for _, k in arith_fields
    )
    arith_has = has_arith_fields or safe_float(merged.get("aic_cube_fops")) > 0
    if not arith_has:
        return
    lines.append("")
    lines.append("--- ArithmeticUtilization ---")
    parts = []
    for label, key in arith_fields:
        v = safe_float(merged.get(key))
        if v > 0:
            parts.append(f"{label}={fmt_ratio(merged.get(key))}")
    fops = safe_float(merged.get("aic_cube_fops"))
    if fops > 0:
        parts.append(f"cube_fops={fops:.0f}")
    if parts:
        lines.append("  " + " | ".join(parts))


def _add_basic_info(lines: List[str], merged: Dict[str, Any]) -> Tuple[float, float, float]:
    op_name = merged.get("Op Name", "unknown")
    op_type = merged.get("OP Type", "unknown")
    task_type = merged.get("Task Type", "")
    duration = safe_float(merged.get("Task Duration(us)"))
    block_dim = safe_int(merged.get("Block Num", 0))
    mix_block = safe_int(merged.get("Mix Block Num", 0))
    aicore_time = safe_float(merged.get("aicore_time(us)"))
    aiv_time = safe_float(merged.get("aiv_time(us)"))

    lines.append("=== 上板性能统计摘要 (msprof) ===")
    lines.append(f"Op: {op_name}")
    lines.append(
        f"Type: {op_type} | TaskType: {task_type} | Duration: {duration}us"
        f" | BlockDim: {block_dim} (mix={mix_block})"
    )
    if merged.get("_missing_metrics"):
        lines.append(f"[WARN] 缺失指标: {', '.join(merged['_missing_metrics'])}")
    lines.append("")
    lines.append("[注] msprof 的 op_summary 是 per-op 聚合值（不含逐核 min/avg/max）；")
    lines.append("     如需逐核数据请改用 msprof op (需要 msopprof 二进制)。")
    return duration, aicore_time, aiv_time


def _add_pipe_ratios(lines: List[str], merged: Dict[str, Any], aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    aic_cube_like = max(safe_float(merged.get("aic_mac_ratio")), safe_float(merged.get("aic_mte2_ratio")))
    aiv_vec_like = safe_float(merged.get("aiv_vec_ratio"))
    prefix = "aic" if aic_cube_like >= aiv_vec_like else "aiv"
    lines.append(f"--- Pipe ratios (主导核 = {prefix}) ---")
    lines.append(f"  aicore_time: {aicore_time:.3f}us | aiv_time: {aiv_time:.3f}us")

    aic_fields = [
        ("aic_mac_ratio", "mac"),
        ("aic_cube_ratio", "cube"),
        ("aic_mte1_ratio", "mte1"),
        ("aic_mte2_ratio", "mte2"),
        ("aic_mte3_ratio", "mte3"),
        ("aic_fixpipe_ratio", "fixpipe"),
        ("aic_scalar_ratio", "scalar"),
        ("aic_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aic_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aic: " + " | ".join(parts))

    aiv_fields = [
        ("aiv_vec_ratio", "vec"),
        ("aiv_scalar_ratio", "scalar"),
        ("aiv_mte2_ratio", "mte2"),
        ("aiv_mte3_ratio", "mte3"),
        ("aiv_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aiv_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aiv: " + " | ".join(parts))

    util = safe_float(merged.get("cube_utilization(%)"))
    if util > 0:
        lines.append(f"  cube_utilization: {util:.2f}%")


def _add_overhead(lines: List[str], duration: float, aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    core_time_max = max(aicore_time, aiv_time)
    overhead = max(0.0, duration - core_time_max)
    overhead_pct = (overhead / duration * 100.0) if duration > 0 else 0.0
    lines.append("--- 头开销 ---")
    lines.append(
        f"  Task Duration: {duration}us | 核最长耗时: {core_time_max:.3f}us"
        f" | 头开销: {overhead:.3f}us ({overhead_pct:.1f}%)"
    )


def _add_footer(lines: List[str], merged: Dict[str, Any], round_dir: str, group_dir: str) -> None:
    lines.append("")
    lines.append("--- 原始数据位置 ---")
    lines.append(f"  归档 CSV : {round_dir}/")
    lines.append(f"  原始 PROF: {group_dir}/")
    lines.append("  按 aic-metrics 拆分的 op_summary_<Metric>.csv 均已复制到归档目录，")
    lines.append("  如需逐列查看可直接 Read。")
    if merged.get("_metric_sources"):
        lines.append("")
        lines.append("--- Metric 来源 ---")
        for m in METRICS:
            src = merged["_metric_sources"].get(m)
            if src:
                lines.append(f"  {m:<22s} <- {src}")
            else:
                lines.append(f"  {m:<22s} <MISSING>")


def generate_summary(merged: Dict[str, Any], round_dir: str, group_dir: str) -> str:
    lines: List[str] = []
    duration, aicore_time, aiv_time = _add_basic_info(lines, merged)
    _add_pipe_ratios(lines, merged, aicore_time, aiv_time)
    _add_overhead(lines, duration, aicore_time, aiv_time)
    _add_memory_section(lines, merged)
    _add_memory_l0_section(lines, merged)
    _add_memory_ub_section(lines, merged)
    _add_l2cache_section(lines, merged)
    _add_rc_section(lines, merged)
    _add_arith_section(lines, merged)
    lines.extend(per_core_balance_section(merged, group_dir))
    _add_footer(lines, merged, round_dir, group_dir)
    return "\n".join(lines)


# ============================================================================
# 对比模式：model.py vs model_new_ascendc.py
# ============================================================================

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _find_cls(module, preferred: str):
    import torch.nn as nn
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError(f"no nn.Module subclass found in {module.__file__}")


def _move(v, d):
    import torch
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, Mapping):
        return {key: _move(value, d) for key, value in v.items()}
    if isinstance(v, list):
        return [_move(x, d) for x in v]
    if isinstance(v, tuple):
        return tuple(_move(x, d) for x in v)
    return v


def _clone(v):
    """Deep clone tensors nested in mappings, lists, or tuples."""
    import torch
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, Mapping):
        return {key: _clone(value) for key, value in v.items()}
    if isinstance(v, list):
        return [_clone(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone(x) for x in v)
    return v


def _resolve_input_groups(module):
    """Return model input cases without conflating a case with its arguments.

    ``get_input_groups()`` returns multiple cases.  CUDA-LLM's ``get_inputs()``
    returns the arguments of exactly one case, so it must be wrapped once rather
    than indexed as if each argument were a separate case.
    """
    if hasattr(module, "get_input_groups"):
        groups = module.get_input_groups()
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError("get_input_groups() must return a non-empty list or tuple")
        return list(groups)
    if hasattr(module, "get_inputs"):
        return [module.get_inputs()]
    module_path = getattr(module, "__file__", repr(module))
    raise AttributeError(
        f"Neither get_input_groups() nor get_inputs() found in {module_path}"
    )


def _forward_signature(model_or_class):
    """Return a callable signature for a bound model or an nn.Module class."""
    target = getattr(model_or_class, "forward", model_or_class)
    signature = inspect.signature(target)
    parameters = list(signature.parameters.values())
    if inspect.isclass(model_or_class) and parameters and parameters[0].name in ("self", "cls"):
        signature = signature.replace(parameters=parameters[1:])
    return signature


def _bind_case(model_or_class, case):
    """Bind one dataset case to ``forward`` and return ``(args, kwargs)``.

    Mapping cases bind by parameter name.  Sequence cases retain the historical
    positional contract, with any values after the declared positional slots
    bound to keyword-only parameters in declaration order.  This covers both
    CUDA-LLM's positional ``get_inputs()`` and NPUKernelBench Level-4 providers
    that flatten positional and keyword-only values into one sequence.

    The function validates the call before execution.  It deliberately never
    catches a ``TypeError`` raised by the model body, because that is a genuine
    model failure rather than evidence that another binding strategy is needed.
    """
    signature = _forward_signature(model_or_class)
    if isinstance(case, Mapping):
        args = ()
        kwargs = dict(case)
        signature.bind(*args, **kwargs)
        return args, kwargs

    if isinstance(case, (list, tuple)):
        values = list(case)
    else:
        values = [case]

    parameters = list(signature.parameters.values())
    positional = [
        parameter for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    keyword_only = [
        parameter for parameter in parameters
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )

    if has_varargs and keyword_only and len(values) > len(positional):
        raise TypeError(
            "flat input case is ambiguous for forward(*args, keyword-only...); "
            "return a mapping from the input provider"
        )

    if has_varargs:
        args = tuple(values)
        kwargs = {}
    else:
        args = tuple(values[:len(positional)])
        remaining = values[len(positional):]
        if len(remaining) > len(keyword_only):
            raise TypeError(
                f"input case has {len(values)} values but forward accepts at most "
                f"{len(positional) + len(keyword_only)}"
            )
        kwargs = {
            parameter.name: value
            for parameter, value in zip(keyword_only, remaining)
        }

    signature.bind(*args, **kwargs)
    return args, kwargs


def _invoke_model(model, case):
    """Invoke a model with a case using the shared dataset binding contract."""
    args, kwargs = _bind_case(model, case)
    return model(*args, **kwargs)


def _read_jsonl_file(path: Path):
    """读取 JSONL 文件，返回解析后的 cases 列表。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _load_cases_jsonl(out_dir: Path):
    patterns = ["*_perf_cases.jsonl", "*.jsonl"]
    for pattern in patterns:
        jsonl_files = sorted(out_dir.glob(pattern))
        if jsonl_files:
            path = jsonl_files[0]
            cases = _read_jsonl_file(path)
            return cases, f"jsonl:{path.name}"
    return [], None


def _load_cases_json(out_dir: Path):
    """从输出目录中查找并解析 JSON 文件。

    找到第一个非备份、非排除的 JSON 文件，优先按标准 JSON 文档解析（单对象或数组），
    失败时回退到 JSONL 逐行解析。
    """
    json_files = sorted(out_dir.glob("*.json"))
    json_path = None
    for f in json_files:
        if not f.name.endswith(".bak") and f.name not in ("performance.json", "perf_report.json"):
            json_path = f
            break
    if not json_path:
        return [], None

    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 优先按标准 JSON 文档解析（单对象或数组），失败时回退到 JSONL 逐行解析
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            cases = parsed
        elif isinstance(parsed, dict):
            cases = [parsed]
        else:
            raise ValueError("Unexpected JSON root type")
    except ValueError:
        cases = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases, f"json:{json_path.name}"


def _extract_shape_dtype_from_jsonl(case):
    if not case:
        return "?", "?"
    inputs = case.get("inputs", [])
    if not inputs:
        return "?", "?"
    for inp in inputs:
        if inp.get("type") == "tensor":
            return str(inp.get("shape", "?")), inp.get("dtype", "?")
    return str(inputs[0].get("shape", "?")), inputs[0].get("dtype", "?")


def _case_has_empty_tensor(case):
    """判断 case 中是否包含 0 元素张量（空 tensor）。"""
    if not case:
        return False
    if "_provider_has_empty_tensor" in case:
        return bool(case["_provider_has_empty_tensor"])
    for inp in case.get("inputs", []):
        if inp.get("type") == "tensor":
            shape = inp.get("shape", [])
            # shape=[] is a scalar tensor (one element), while shape=None is an
            # omitted optional tensor.  Only an explicit zero dimension is empty.
            if isinstance(shape, (list, tuple)) and any(size == 0 for size in shape):
                return True
    return False


def _value_has_empty_tensor(value):
    """Inspect a provider value recursively; provider data is authoritative."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, Mapping):
        return any(_value_has_empty_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_has_empty_tensor(item) for item in value)
    return False


def _jsonl_scalar_value(inp):
    """Resolve an attr/scalar while preserving its declared Python semantics."""
    dtype_str = str(inp.get("dtype", "")).lower()
    if "value" in inp:
        value = inp.get("value")
    else:
        value = None
        range_values = inp.get("range_values")
        if isinstance(range_values, (int, float, bool, complex, str, list, tuple)):
            if isinstance(range_values, list) and range_values and dtype_str not in ("list", "tuple"):
                value = range_values[0]
            else:
                value = range_values
        elif isinstance(range_values, dict):
            mean = range_values.get("mean")
            if isinstance(mean, list) and mean:
                value = mean[0]

    if value is None:
        if "value" in inp:
            return None
        if dtype_str == "bool":
            return True
        if dtype_str.startswith(("int", "uint")):
            return 1
        if dtype_str in ("str", "string"):
            return ""
        if dtype_str == "tuple":
            return ()
        if dtype_str == "list":
            return []
        return 1.0

    if dtype_str == "tuple":
        return tuple(value)
    if dtype_str == "list":
        return list(value)
    if dtype_str in ("str", "string", "dtype") or isinstance(value, str):
        return str(value)
    if dtype_str == "bool":
        return bool(value)
    if dtype_str.startswith("complex"):
        return complex(value)
    if dtype_str.startswith(("int", "uint")):
        return int(value)
    if isinstance(value, (list, tuple, dict, bool)):
        return value
    return float(value)


_JSONL_DTYPE_MAP = {
    "fp16": "torch.float16", "float16": "torch.float16", "half": "torch.float16",
    "fp32": "torch.float32", "float32": "torch.float32", "float": "torch.float32",
    "fp64": "torch.float64", "float64": "torch.float64",
    "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    "int8": "torch.int8", "int16": "torch.int16", "int32": "torch.int32",
    "int": "torch.int32", "int64": "torch.int64", "long": "torch.int64",
    "uint8": "torch.uint8", "uint16": "torch.uint16", "uint32": "torch.uint32",
    "uint64": "torch.uint64", "bool": "torch.bool",
    "complex64": "torch.complex64", "complex128": "torch.complex128",
}

_JSONL_DTYPE_ATTRIBUTE_MAP = {
    **_JSONL_DTYPE_MAP,
    # PyTorch exposes packed int4 through the quint4x2 dtype object.
    "int4": "torch.quint4x2",
}


def _jsonl_tensor_code(shape, dtype_str: str) -> str:
    """根据 KernelBench 的 dtype/shape 生成构造 tensor 的代码。"""
    dtype_str = str(dtype_str).lower()
    dtype = _JSONL_DTYPE_MAP.get(dtype_str, "torch.float32")
    shape_expr = repr(tuple(shape))

    if dtype_str == "bool":
        return f"torch.randint(0, 2, {shape_expr}, dtype={dtype})"
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        # 先以 int64 生成再转换到目标类型，避免 torch.randint 不支持 uint/低精度 int
        return f"torch.randint(-100, 100, {shape_expr}, dtype=torch.int64).to({dtype})"
    if dtype_str.startswith("complex"):
        return f"torch.randn({shape_expr}, dtype={dtype})"
    # float/half/bf16
    return f"torch.randn({shape_expr}, dtype={dtype})"


def _jsonl_value_code(inp):
    """Return deterministic Python source for one JSON input descriptor."""
    typ = inp.get("type", "tensor")
    if typ == "tensor":
        dtype_str = str(inp.get("dtype", "float16")).lower()
        if inp.get("shape") is None and "value" not in inp:
            return "None"
        if "value" in inp:
            value = inp.get("value")
            if value is None:
                return "None"
            dtype = _JSONL_DTYPE_MAP.get(dtype_str, "torch.float32")
            return f"torch.tensor({value!r}, dtype={dtype})"
        return _jsonl_tensor_code(inp.get("shape", []), dtype_str)
    if typ == "tensor_list":
        values = []
        for tensor_info in inp.get("value", []):
            if "value" in tensor_info:
                dtype_str = str(tensor_info.get("dtype", "float16")).lower()
                dtype = _JSONL_DTYPE_MAP.get(
                    dtype_str, "torch.float32"
                )
                values.append(f"torch.tensor({tensor_info.get('value')!r}, dtype={dtype})")
            else:
                values.append(_jsonl_tensor_code(
                    tensor_info.get("shape", []), tensor_info.get("dtype", "float16")
                ))
        return "[" + ", ".join(values) + "]"
    if typ in ("attr", "scalar"):
        value = _jsonl_scalar_value(inp)
        if str(inp.get("dtype", "")).lower() == "dtype" and isinstance(value, str):
            normalized = value.removeprefix("torch.")
            dtype_expr = _JSONL_DTYPE_ATTRIBUTE_MAP.get(normalized)
            if dtype_expr is None:
                raise ValueError(f"unsupported torch dtype attribute: {value}")
            return dtype_expr
        return repr(value)
    if "value" in inp:
        return repr(inp.get("value"))
    raise ValueError(f"unsupported JSON input descriptor type: {typ}")


def _serialize_jsonl_inputs(case):
    """Serialize JSON fallback inputs without losing names or Python types."""
    descriptors = case.get("inputs", [])
    names = [descriptor.get("name") for descriptor in descriptors]
    use_mapping = bool(descriptors) and all(names) and len(set(names)) == len(names)
    lines = ["fallback_case = {}" if use_mapping else "fallback_case = []"]
    for descriptor in descriptors:
        value_code = _jsonl_value_code(descriptor)
        if use_mapping:
            lines.append(f"fallback_case[{descriptor['name']!r}] = {value_code}")
        else:
            lines.append(f"fallback_case.append({value_code})")
    return "\n".join(lines)


@dataclass
class _WrapperConfig:
    """封装 _generate_wrapper_script 的参数。"""
    out_dir: Path
    case_idx: int
    impl: str
    seed: int
    device_id: int
    warmup: int
    jsonl_case: Optional[Dict[str, Any]] = None
    case_cache_path: Optional[Path] = None
    repeats: int = 1


_WRAPPER_SCRIPT_TEMPLATE = """\
#!/usr/bin/env python3
import importlib.util
import inspect
import os
import sys
import torch
from collections.abc import Mapping
from pathlib import Path

out_dir = Path("{out_dir}")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "{device_id}"
sys.path.insert(0, str(out_dir / "kernel" / "build"))
sys.path.insert(0, str(out_dir))

torch.manual_seed({seed})
try:
    torch.npu.manual_seed_all({seed})
except Exception:
    pass

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

mod = _load(out_dir / "{model_file}", "prof_mod")
cls = getattr(mod, "{cls_name}")

{inputs_code}

device = torch.device("npu")
def _move(v):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, Mapping):
        return {{key: _move(value) for key, value in v.items()}}
    if isinstance(v, list):
        return [_move(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_move(x) for x in v)
    return v

{binding_code}

input_case = _move(input_case)
_call_args, _call_kwargs = _bind_case(_contract_cls, input_case)

# ---- model construction (init 参数与对拍脚本 verification_ascendc.py 同一约定:扁平展开) ----
# 能走到测速的模型都已被对拍用 cls(*get_init_inputs()) 成功建过,照抄该约定即对所有
# 可测速算子兼容。init 来源先试实现模块(model_new 一般没有 get_init_inputs),没有再
# 回落 model.py —— 构造参数属于任务定义,不属于实现。
_init_src = mod if hasattr(mod, "get_init_inputs") else _ref_mod
_init_vals = _init_src.get_init_inputs() if hasattr(_init_src, "get_init_inputs") else []
model = cls(*_init_vals).to(device).eval()

def _one_iter():
    with torch.no_grad():
        _ = model(*_call_args, **_call_kwargs)
    torch.npu.synchronize()

# 预热和正式测试共用一个 Python/NPU 进程。msprof 会采到两者，解析器只保留
# 后 repeats 次正式测试，从而避免“外部进程预热、正式进程仍是冷启动”的问题。
for _ in range({warmup}):
    _one_iter()

for _ in range({repeats}):
    _one_iter()
"""


_WRAPPER_BINDING_CODE = """\
def _forward_signature(model_or_class):
    target = getattr(model_or_class, "forward", model_or_class)
    signature = inspect.signature(target)
    parameters = list(signature.parameters.values())
    if inspect.isclass(model_or_class) and parameters and parameters[0].name in ("self", "cls"):
        signature = signature.replace(parameters=parameters[1:])
    return signature

def _bind_case(model_or_class, case):
    signature = _forward_signature(model_or_class)
    if isinstance(case, Mapping):
        args, kwargs = (), dict(case)
        signature.bind(*args, **kwargs)
        return args, kwargs
    values = list(case) if isinstance(case, (list, tuple)) else [case]
    parameters = list(signature.parameters.values())
    positional = [p for p in parameters if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    keyword_only = [p for p in parameters if p.kind == inspect.Parameter.KEYWORD_ONLY]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
    if has_varargs and keyword_only and len(values) > len(positional):
        raise TypeError(
            "flat input case is ambiguous for forward(*args, keyword-only...); "
            "return a mapping from the input provider")
    if has_varargs:
        args, kwargs = tuple(values), {}
    else:
        args = tuple(values[:len(positional)])
        remaining = values[len(positional):]
        if len(remaining) > len(keyword_only):
            raise TypeError(
                "input case has %d values but forward accepts at most %d" %
                (len(values), len(positional) + len(keyword_only)))
        kwargs = {p.name: value for p, value in zip(keyword_only, remaining)}
    signature.bind(*args, **kwargs)
    return args, kwargs
"""


def _build_wrapper_script_content(cfg, model_file, cls_name, inputs_code):
    """Build the wrapper script string from components."""
    return _WRAPPER_SCRIPT_TEMPLATE.format(
        out_dir=cfg.out_dir,
        device_id=cfg.device_id,
        case_idx=cfg.case_idx,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        seed=cfg.seed,
        model_file=model_file,
        cls_name=cls_name,
        inputs_code=inputs_code,
        binding_code=_WRAPPER_BINDING_CODE,
    )


def _provider_inputs_code(case_idx, has_fallback):
    fallback = "input_case = fallback_case" if has_fallback else (
        'raise AttributeError("model.py must provide get_inputs() or get_input_groups()")'
    )
    return f'''\
_ref_mod = _load(out_dir / "model.py", "ref_for_inputs")
_contract_cls = getattr(_ref_mod, "Model")
if hasattr(_ref_mod, "get_input_groups"):
    _input_groups = _ref_mod.get_input_groups()
    if not isinstance(_input_groups, (list, tuple)) or not _input_groups:
        raise ValueError("get_input_groups() must return a non-empty list or tuple")
    input_case = _input_groups[{case_idx}]
elif hasattr(_ref_mod, "get_inputs"):
    if {case_idx} != 0:
        raise IndexError("get_inputs() defines exactly one input case")
    input_case = _ref_mod.get_inputs()
else:
    {fallback}
'''


def _cached_inputs_code(case_cache_path: Path):
    """Load one pipeline-scoped case instead of re-running its provider."""
    return f'''\
_ref_mod = _load(out_dir / "model.py", "ref_for_inputs")
_contract_cls = getattr(_ref_mod, "Model")
input_case = torch.load({str(case_cache_path)!r}, map_location="cpu")
'''


def _generate_wrapper_script(cfg: _WrapperConfig):
    if cfg.impl == "reference":
        model_file = "model.py"
        cls_name = "Model"
    else:
        model_file = "model_new_ascendc.py"
        cls_name = "ModelNew"

    if cfg.case_cache_path is not None:
        inputs_code = _cached_inputs_code(cfg.case_cache_path)
    else:
        provider_code = _provider_inputs_code(cfg.case_idx, cfg.jsonl_case is not None)
        if cfg.jsonl_case is not None:
            inputs_code = _serialize_jsonl_inputs(cfg.jsonl_case) + "\n\n" + provider_code
        else:
            inputs_code = provider_code
    return _build_wrapper_script_content(cfg, model_file, cls_name, inputs_code)


def _find_msprof_script():
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "msprof_profile_run.sh"
    if candidate.exists():
        return str(candidate)
    return "msprof_profile_run.sh"


def _save_app_output(output_dir: str, stdout: str, stderr: str) -> str:
    """Persist profiler/app output so wrapper failures remain diagnosable."""
    log_path = os.path.join(output_dir, "app_output.log")
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as output:
            output.write("=== stdout ===\n")
            output.write(stdout or "")
            output.write("\n=== stderr ===\n")
            output.write(stderr or "")
    except OSError:
        pass
    return log_path


def _extract_app_crash(stdout: str, stderr: str):
    """Detect Python app failures that some msprof versions report with rc=0."""
    output = (stdout or "") + "\n" + (stderr or "")
    if "Traceback" not in output and "An exception has occurred in process App" not in output:
        return None
    exception_lines = re.findall(
        r"^(\w[\w.]*(?:Error|Exception|Interrupt)\b[^\n]*)", output, re.MULTILINE
    )
    if exception_lines:
        return exception_lines[-1].strip()[:200]
    return "app raised an exception during profiling"


def _run_msprof_standard(wrapper_script: str, output_dir: str, device_id: int, warmup: int = 3):
    """调用 msprof_profile_run.sh 进行完整采集（7 组 aic-metrics + sample-based）。

    注意：不设置 timeout，与原来 kernel_perf.py 行为一致（默认无限等待）。
    8 轮采集（7 metrics + 1 sample）耗时较长，由调用方控制整体超时。
    """
    wrapper_path = os.path.join(output_dir, "_wrapper.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script)

    cmd = [
        "bash", _find_msprof_script(),
        f"--warm-up={warmup}",
        f"--output={output_dir}",
        "--",
        sys.executable, wrapper_path
    ]
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    try:
        os.remove(wrapper_path)
    except OSError:
        pass

    app_log = _save_app_output(output_dir, result.stdout, result.stderr)
    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}\n(app log: {app_log})"

    app_crash = _extract_app_crash(result.stdout, result.stderr)
    if app_crash:
        return None, f"profiled app crashed: {app_crash} (app log: {app_log})"

    prof_dirs = sorted(Path(output_dir).glob("PROF_GROUP_*"))
    if not prof_dirs:
        return None, f"no PROF_GROUP directory found (app log: {app_log})"
    return str(prof_dirs[-1]), None


def _run_msprof_quick(wrapper_script: str, output_dir: str, device_id: int):
    """快速模式：只采集 1 轮（不采集 7 个 aic-metrics，只获取 kernel 时间）。

    直接调用 msprof 命令（不通过 msprof_profile_run.sh，避免循环调用）。
    使用 msprof --task-time=on --ascendcl=on，不设置 --aic-metrics。

    warmup 和正式测试都在同一个被采集进程中执行，由解析器丢弃 warmup 轮次。
    """
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()

    # Measurement: wrapper 在同一个进程内先预热再正式测试。
    wrapper_path = os.path.join(output_dir, "_wrapper.py")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script)
    cmd = [
        "msprof",
        f"--output={output_dir}",
        "--task-time=on",
        "--ascendcl=on",
        sys.executable, wrapper_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    try:
        os.remove(wrapper_path)
    except OSError:
        pass

    app_log = _save_app_output(output_dir, result.stdout, result.stderr)
    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}\n(app log: {app_log})"

    app_crash = _extract_app_crash(result.stdout, result.stderr)
    if app_crash:
        return None, f"profiled app crashed: {app_crash} (app log: {app_log})"

    prof_dirs = sorted(Path(output_dir).glob("PROF_*"))
    if not prof_dirs:
        return None, f"no PROF directory found (app log: {app_log})"
    return str(prof_dirs[-1]), None


def _extract_compute_row(r: dict):
    """从 op_summary 行提取计算算子的 (duration, op_name)。

    非计算行、非法值或非正耗时返回 None。
    """
    task_type = r.get("Task Type", "")
    if "AI_CORE" not in task_type and "AIV" not in task_type and "MIX" not in task_type:
        return None
    try:
        duration = float(r.get("Task Duration(us)", 0) or 0)
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration, r.get("Op Name", "unknown")


def _parse_msprof_duration(prof_group_dir: str):
    csv_pattern = os.path.join(prof_group_dir, "PROF_*/PROF_*/mindstudio_profiler_output/op_summary_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        return None, None, "no op_summary csv found"

    try:
        with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return None, None, f"read csv error: {e}"

    if not rows:
        return None, None, "empty csv"

    # 汇总所有实际计算算子的 Task Duration；参考实现可能包含多个 PyTorch native kernel，
    # AscendC 实现也可能拆分为多个 kernel，因此采用累加而非取最大。
    compute_rows = [cr for cr in (_extract_compute_row(r) for r in rows) if cr is not None]

    if compute_rows:
        total_duration = sum(d for d, _ in compute_rows)
        op_name = compute_rows[0][1] if len(compute_rows) == 1 else "multiple_kernels"
        return total_duration, op_name, None

    return None, None, "no compute rows found"


_META_KERNEL_TYPES = (
    "PROFILING_ENABLE", "PROFILING_DISABLE", "TASK_TIMEOUT_SET", "EVENT_RECORD", ""
)


def _collect_task_time_kernels(rows):
    """Collect positive non-metadata device tasks ordered by device start time."""
    kernels = []
    for row in rows:
        if row.get("kernel_type", "") in _META_KERNEL_TYPES:
            continue
        try:
            duration = float(row.get("task_time(us)", "") or 0)
            start = float((row.get("task_start(us)", "") or "0").strip())
        except ValueError:
            continue
        if duration > 0:
            kernels.append((start, row.get("kernel_name", "unknown"), duration))
    kernels.sort(key=lambda item: item[0])
    return kernels


def _split_task_time_runs(rows, n_runs: int):
    """Split a repeated kernel sequence into ``n_runs`` identical iterations."""
    kernels = _collect_task_time_kernels(rows)
    if not kernels or n_runs < 1 or len(kernels) % n_runs:
        return None
    kernels_per_run = len(kernels) // n_runs
    expected_names = [item[1] for item in kernels[:kernels_per_run]]
    runs = []
    for run_idx in range(n_runs):
        chunk = kernels[run_idx * kernels_per_run:(run_idx + 1) * kernels_per_run]
        if [item[1] for item in chunk] != expected_names:
            return None
        runs.append([item[2] for item in chunk])
    return runs


def _split_kernels_by_position(kernels, warmup: int, repeats: int):
    """Fallback split when repeated kernel names are not perfectly identical."""
    if not kernels:
        return None
    n_total = max(1, warmup + repeats)
    n_kernels = len(kernels)
    if n_kernels % n_total == 0:
        kernels_per_run = n_kernels // n_total
        active = kernels[warmup * kernels_per_run:]
        return sum(item[2] for item in active) / max(1, repeats)

    split_idx = max(0, int(n_kernels * warmup / n_total))
    active = kernels[split_idx:]
    if not active:
        return sum(item[2] for item in kernels) / n_total
    per_run_kernel_count = n_kernels / n_total
    return sum(item[2] for item in active) / (len(active) / per_run_kernel_count)


def _parse_msprof_duration_quick(prof_group_dir: str, warmup: int = 0, repeats: int = 1):
    """快速模式解析：从 PROF 目录中查找 task_time.csv 或 api_statistic.csv 并提取时间。

    msprof 采集窗口包含 warmup + repeats 次同进程迭代。按重复的 kernel 序列
    拆分后丢弃 warmup，只对 repeats 次正式测试求平均。
    """
    # 1. 尝试从 task_time.csv 提取
    task_time_pattern = os.path.join(prof_group_dir, "mindstudio_profiler_output/task_time_*.csv")
    task_time_files = sorted(glob.glob(task_time_pattern))
    if task_time_files:
        try:
            with open(task_time_files[0], "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            runs = _split_task_time_runs(rows, max(0, warmup) + max(1, repeats))
            if runs is not None:
                active = runs[max(0, warmup):]
                per_run = [sum(run) for run in active]
                if per_run:
                    return sum(per_run) / len(per_run), "multiple_kernels", None
            kernels = _collect_task_time_kernels(rows)
            duration = _split_kernels_by_position(
                kernels, max(0, warmup), max(1, repeats)
            )
            if duration is not None:
                kernel_names = {item[1] for item in kernels}
                kernel_name = next(iter(kernel_names)) if len(kernel_names) == 1 else "multiple_kernels"
                return duration, kernel_name, None
        except Exception:
            pass

    # 2. 尝试从 api_statistic.csv 提取（launch 行）
    api_pattern = os.path.join(prof_group_dir, "mindstudio_profiler_output/api_statistic_*.csv")
    api_files = sorted(glob.glob(api_pattern))
    if api_files:
        try:
            with open(api_files[0], "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # 查找 Level=node, API Name=launch 的行
            for r in rows:
                level = r.get("Level", "")
                api_name = r.get("API Name", "")
                if level == "node" and api_name == "launch":
                    time_us = r.get("Time(us)", "")
                    if time_us:
                        try:
                            duration = float(time_us)
                            if duration > 0:
                                return duration, "launch", None
                        except ValueError:
                            continue
        except Exception:
            return None, None, "no task_time or api_statistic csv found"

    return None, None, "no task_time or api_statistic csv found"


def pick_idle_npu(default=0):
    try:
        p = subprocess.run(["/usr/local/bin/npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return default
    except Exception:
        return default

    devices = {}
    cur = None
    head_re = re.compile(r"^\|\s+(\d+)\s+\S+\s+\|\s+\w+\s+\|")
    bus_re = re.compile(
        r"^\|\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?"
    )
    for line in p.stdout.splitlines():
        m2 = bus_re.match(line)
        if m2 and cur is not None:
            aicore = int(m2.group(1))
            mem_used = int(m2.group(2))
            mem_total = max(int(m2.group(3)), 1)
            hbm_used = int(m2.group(4)) if m2.group(4) else 0
            hbm_total = max(int(m2.group(5)), 1) if m2.group(5) else 1
            mem_ratio = max(mem_used / mem_total, hbm_used / hbm_total)
            devices[cur] = (aicore, mem_ratio)
            cur = None
            continue
        m1 = head_re.match(line)
        if m1:
            cur = int(m1.group(1))

    if not devices:
        return default
    best_id, _ = min(devices.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return best_id


def _add_compare_header(lines, report):
    """Add the header section to the compare markdown report."""
    lines.append("# 性能评估结果")
    lines.append("")
    lines.append(f"- **Operator**: {report['task']}")
    lines.append(f"- **Device**: npu:{report['device_id']} (source={report['device_select_source']})")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeats**: {report['repeats']}")
    lines.append(f"- **Seed**: {report['seed']}")
    lines.append(f"- **Timing method**: {report['timing_method']}")
    lines.append("")


def _add_per_case_table(lines, report):
    """Add the per-case comparison table."""
    if not report.get("per_case"):
        return
    lines.append("## 性能对比")
    lines.append("")
    lines.append("| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |")
    lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
    for case in report["per_case"]:
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref = case.get("ref_us")
        asc = case.get("asc_us")
        sp = case.get("speedup")
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        asc_str = f"{asc:.2f}" if asc is not None else "N/A"
        sp_str = f"{sp:.3f}" if sp is not None else "N/A"
        lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
    lines.append("")


def _add_summary_section(lines, report):
    """Add the summary and dtype tables."""
    if report.get("geomean_speedup") is None:
        return
    lines.append("## 全量汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| ---- | -- |")
    lines.append(f"| 用例数 | {report['n_cases_total']} |")
    lines.append(f"| 平均加速比（>1 表示自定义算子更快） | {report['mean_speedup']:.3f} |")
    if report.get("geomean_ref_us") is not None:
        lines.append(f"| 标杆几何平均耗时 (us) | {report['geomean_ref_us']:.2f} |")
    if report.get("geomean_asc_us") is not None:
        lines.append(f"| 自定义算子几何平均耗时 (us) | {report['geomean_asc_us']:.2f} |")
    better = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] > 1)
    worse = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] < 1)
    lines.append(f"| 自定义算子更优（比值>1） | {better} |")
    lines.append(f"| 标杆更优（比值<1） | {worse} |")
    lines.append("")

    dtype_groups = {}
    for case in report.get("per_case", []):
        dtype = case.get("dtype", "?")
        sp = case.get("speedup")
        if sp is not None:
            dtype_groups.setdefault(dtype, []).append(sp)
    if dtype_groups:
        lines.append("### 按数据类型汇总")
        lines.append("")
        lines.append("| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |")
        lines.append("| ----- | ------ | ------------------- | ------------- | -------- |")
        for dtype, sps in sorted(dtype_groups.items()):
            mean_sp = statistics.mean(sps)
            better = sum(1 for sp in sps if sp > 1)
            worse = sum(1 for sp in sps if sp < 1)
            lines.append(f"| {dtype} | {len(sps)} | {mean_sp:.3f} | {better} | {worse} |")
        lines.append("")


def _add_analysis_sections(lines, report):
    """Add the short analysis and deep bottleneck analysis sections."""
    lines.append("## 简短分析")
    lines.append("")
    if report.get("mean_speedup") is not None:
        if report["mean_speedup"] > 1:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 大于 1，自定义算子整体有优势。")
        else:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 小于 1，标杆路径整体更优。")
    lines.append("- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。")
    lines.append("")

    lines.append("## 深度瓶颈分析")
    lines.append("")
    lines.append(
        "如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），"
        "可运行："
    )
    lines.append("```bash")
    lines.append(
        f"python3 ${{SKILL_PATH}}/scripts/msprof_perf_summary.py "
        f"{report.get('prof_group_dir', './PROF_GROUP_*')} {report['task']}"
    )
    lines.append("```")
    lines.append("")
    lines.append("或参考 `ops-profiling/references/optimization_quickref.md` 获取优化建议。")
    lines.append("")


def _report_compare_to_markdown(report: Dict[str, Any]) -> str:
    lines = []
    _add_compare_header(lines, report)
    _add_per_case_table(lines, report)
    _add_summary_section(lines, report)
    _add_analysis_sections(lines, report)
    return "\n".join(lines)


def _report_compare_to_text(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"Kernel-level Performance (msprof): {report['task']}  "
                 f"(warmup={report['warmup']}, repeats={report['repeats']}, seed={report['seed']})")
    lines.append("=" * 100)
    lines.append(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Ref(us)':>12} {'Asc(us)':>12} {'Speedup':>10}")
    lines.append("-" * 100)

    for case in report.get("per_case", []):
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref_us = case.get("ref_us")
        asc_us = case.get("asc_us")
        sp = case.get("speedup")
        if ref_us is not None and asc_us is not None and sp is not None:
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            ref_str = f"{ref_us:.2f}" if ref_us is not None else "N/A"
            asc_str = f"{asc_us:.2f}" if asc_us is not None else "N/A"
            ref_err = case.get("ref_error", "")
            asc_err = case.get("asc_error", "")
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} "
                         f"{ref_str:>12} {asc_str:>12} "
                         f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

    lines.append("-" * 100)
    if report.get("geomean_speedup") is not None:
        lines.append("--- Speedup ---")
        lines.append(f"  Geomean : {report['geomean_speedup']:.2f}x  ← 主指标")
        lines.append(f"  Mean    : {report['mean_speedup']:.2f}x")
        lines.append(f"  Median  : {report['median_speedup']:.2f}x")
        lines.append(f"  Min/Max : {report['min_speedup']:.2f}x / {report['max_speedup']:.2f}x")
        lines.append(f"  Valid   : {report['n_cases_valid']}/{report['n_cases_total']}")
    if report.get("mean_ref_us") is not None:
        lines.append("--- Task Duration (us) ---")
        lines.append(
            f"  Ref  mean/median/geomean/total : {report['mean_ref_us']:.2f} / "
            f"{report['median_ref_us']:.2f} / {report['geomean_ref_us']:.2f} / "
            f"{report['total_ref_us']:.2f}"
        )
        lines.append(
            f"  Asc  mean/median/geomean/total : {report['mean_asc_us']:.2f} / "
            f"{report['median_asc_us']:.2f} / {report['geomean_asc_us']:.2f} / "
            f"{report['total_asc_us']:.2f}"
        )
        lines.append(f"  Total speedup (Σref/Σasc) : {report['total_speedup']:.2f}x")
    lines.append("=" * 100)

    return "\n".join(lines)


def _select_device_id(args):
    """Select NPU device from CLI arg, env var, or auto-detect."""
    if args.device is not None:
        return args.device, "cli"
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return int(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[0]), "env"
    return pick_idle_npu(default=0), "auto"


@dataclass
class _MeasureInput:
    """封装 _measure_one_impl 的测量参数。

    Args:
        out_dir: 算子输出目录（Path）
        case_idx: case 索引号
        impl: 实现类型，\"reference\" 或 \"ascendc\"
        args: argparse.Namespace（需含 retry, seed, warmup 属性）
        device_id: NPU 设备 ID
        jsonl_case: JSONL case 字典，可选
    """
    out_dir: Path
    case_idx: int
    impl: str
    args: argparse.Namespace
    device_id: int
    jsonl_case: Optional[Dict[str, Any]] = None
    case_cache_path: Optional[Path] = None


def _measure_one_impl(mi: _MeasureInput):
    """Measure one implementation (reference/ascendc) with retries.

    Returns (duration_us, error, prof_dir).
    """
    prof_dir = None
    impl_abbr = "ref" if mi.impl == "reference" else "asc"
    for _ in range(1 + mi.args.retry):
        wrapper = _generate_wrapper_script(_WrapperConfig(
            mi.out_dir, mi.case_idx, mi.impl, mi.args.seed, mi.device_id,
            mi.args.warmup, mi.jsonl_case, mi.case_cache_path))
        tmpdir = f"/tmp/msprof_{impl_abbr}_{mi.out_dir.name}_c{mi.case_idx}"
        prof_dir, err = _run_msprof_standard(wrapper, tmpdir, mi.device_id, mi.args.warmup)
        if prof_dir:
            duration, _op_name, parse_err = _parse_msprof_duration(prof_dir)
            if duration is not None:
                return duration, None, prof_dir
            err = parse_err
        time.sleep(0.5)
    return None, err, prof_dir


def _measure_one_impl_quick(mi: _MeasureInput):
    """快速模式：Measure one implementation with retries and repeats.

    --retry 用于解析失败重试；--repeats 控制 wrapper 内 timed iteration 次数。
    wrapper 在同一个被采集进程内做 warmup，再做 repeats 次正式测试；解析器
    丢弃前 warmup 轮次，避免重复初始化 Python/NPU。

    Returns (duration_us, error, prof_dir).
    """
    prof_dir = None
    impl_abbr = "ref" if mi.impl == "reference" else "asc"
    repeats = max(1, getattr(mi.args, "repeats", 1))
    warmup = max(0, getattr(mi.args, "warmup", 0))

    for _ in range(1 + mi.args.retry):
        wrapper = _generate_wrapper_script(_WrapperConfig(
            mi.out_dir, mi.case_idx, mi.impl, mi.args.seed, mi.device_id,
            warmup, mi.jsonl_case, mi.case_cache_path, repeats))
        tmpdir = f"/tmp/msprof_quick_{impl_abbr}_{mi.out_dir.name}_c{mi.case_idx}"
        _cleanup_prof_dirs(tmpdir)
        prof_dir, err = _run_msprof_quick(wrapper, tmpdir, mi.device_id)
        if not prof_dir:
            continue

        duration, _op_name, parse_err = _parse_msprof_duration_quick(
            prof_dir, warmup, repeats)
        if duration is None:
            err = parse_err
            continue

        return duration, None, prof_dir

    return None, err, prof_dir


def _generate_grouped_wrapper_script(out_dir: Path, blocks: list, seed: int,
                                     device_id: int, warmup: int, repeats: int,
                                     manifest_path: Path) -> str:
    """Generate one profiled process for every case and both implementations.

    Two ``torch.npu.Event`` records bracket each timed block. msprof exports those
    records as ``EVENT_RECORD`` rows, which lets the parent split one task-time CSV
    back into per-case/per-implementation durations without changing the timing
    metric.
    """
    block_specs = [
        {
            "case": int(case_idx),
            "impl": impl,
            "case_path": str(case_path),
        }
        for case_idx, impl, case_path in blocks
    ]
    return f'''\
#!/usr/bin/env python3
import importlib.util
import inspect
import json
import os
import sys
import torch
from collections.abc import Mapping
from pathlib import Path

out_dir = Path({str(out_dir)!r})
manifest_path = Path({str(manifest_path)!r})
blocks = {block_specs!r}
seed = {int(seed)}
warmup = {max(0, int(warmup))}
repeats = {max(1, int(repeats))}
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = {str(device_id)!r}
sys.path.insert(0, str(out_dir / "kernel" / "build"))
sys.path.insert(0, str(out_dir))

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {{key: _move(item, device) for key, item in value.items()}}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value

{_WRAPPER_BINDING_CODE}

device = torch.device("npu")
contract_mod = _load(out_dir / "model.py", "grouped_contract")
contract_cls = getattr(contract_mod, "Model")
results = []

def _save_manifest():
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_path.write_text(json.dumps({{"blocks": results}}, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(manifest_path)

for block_no, block in enumerate(blocks):
    entry = {{
        "case": block["case"], "impl": block["impl"],
        "ok": False, "has_markers": False, "error": None,
    }}
    try:
        torch.manual_seed(seed)
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass

        impl = block["impl"]
        model_file = "model.py" if impl == "reference" else "model_new_ascendc.py"
        cls_name = "Model" if impl == "reference" else "ModelNew"
        module = _load(out_dir / model_file, f"grouped_{{impl}}_{{block_no}}")
        cls = getattr(module, cls_name)
        input_case = torch.load(block["case_path"], map_location="cpu")
        input_case = _move(input_case, device)
        call_args, call_kwargs = _bind_case(contract_cls, input_case)

        init_src = module if hasattr(module, "get_init_inputs") else contract_mod
        init_vals = init_src.get_init_inputs() if hasattr(init_src, "get_init_inputs") else []
        model = cls(*init_vals).to(device).eval()

        def _one_iter():
            with torch.no_grad():
                model(*call_args, **call_kwargs)
            torch.npu.synchronize()

        for _ in range(warmup):
            _one_iter()

        start_event = torch.npu.Event()
        end_event = torch.npu.Event()
        start_event.record()
        timed_error = None
        try:
            for _ in range(repeats):
                _one_iter()
        except Exception as exc:
            timed_error = exc
        finally:
            end_event.record()
            end_event.synchronize()
            entry["has_markers"] = True

        if timed_error is not None:
            raise timed_error
        entry["ok"] = True
    except Exception as exc:
        entry["error"] = f"{{type(exc).__name__}}: {{exc}}"[:300]
    results.append(entry)
    _save_manifest()
'''


def _load_grouped_manifest(manifest_path: Path):
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            return blocks, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"invalid grouped manifest: {exc}"
    return None, "invalid grouped manifest: missing blocks list"


def _parse_msprof_grouped(prof_dir: str, manifest_path: Path, repeats: int):
    """Split one msprof task-time CSV by paired NPU event markers."""
    manifest, manifest_error = _load_grouped_manifest(manifest_path)
    if manifest is None:
        return None, manifest_error

    task_time_files = sorted(glob.glob(os.path.join(
        prof_dir, "mindstudio_profiler_output", "task_time_*.csv"
    )))
    if not task_time_files:
        return None, "no task_time csv found"
    try:
        with open(task_time_files[0], "r", encoding="utf-8", errors="replace") as stream:
            rows = list(csv.DictReader(stream))
        rows.sort(key=lambda row: safe_float(row.get("task_start(us)")))
    except (OSError, ValueError, TypeError) as exc:
        return None, f"read task_time csv error: {exc}"

    marked_blocks = [block for block in manifest if block.get("has_markers")]
    marker_positions = [
        idx for idx, row in enumerate(rows) if row.get("kernel_type") == "EVENT_RECORD"
    ]
    expected_markers = 2 * len(marked_blocks)
    if len(marker_positions) != expected_markers:
        return None, (
            f"grouped marker mismatch: expected {expected_markers}, "
            f"found {len(marker_positions)}"
        )

    measurements = {}
    marker_idx = 0
    for block in manifest:
        key = (int(block["case"]), str(block["impl"]))
        if not block.get("has_markers"):
            measurements[key] = {"duration_us": None, "error": block.get("error")}
            continue

        start_pos = marker_positions[marker_idx]
        end_pos = marker_positions[marker_idx + 1]
        marker_idx += 2
        if end_pos <= start_pos:
            return None, f"invalid grouped marker order for case {key[0]} {key[1]}"
        kernels = _collect_task_time_kernels(rows[start_pos + 1:end_pos])
        if block.get("ok") and kernels:
            measurements[key] = {
                "duration_us": sum(item[2] for item in kernels) / max(1, repeats),
                "error": None,
            }
        else:
            measurements[key] = {
                "duration_us": None,
                "error": block.get("error") or "no timed device task found",
            }
    return measurements, None


def _measure_all_impls_grouped(out_dir: Path, blocks: list, args, device_id: int):
    """Profile all supplied blocks in one Python/NPU process and one msprof run."""
    tmpdir = f"/tmp/msprof_quick_grouped_{out_dir.name}"
    manifest_path = Path(tmpdir) / "grouped_manifest.json"
    last_error = None
    for _ in range(1 + args.retry):
        _cleanup_prof_dirs(tmpdir)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        wrapper = _generate_grouped_wrapper_script(
            out_dir, blocks, args.seed, device_id, args.warmup, args.repeats,
            manifest_path,
        )
        prof_dir, error = _run_msprof_quick(wrapper, tmpdir, device_id)
        if prof_dir:
            measurements, parse_error = _parse_msprof_grouped(
                prof_dir, manifest_path, max(1, args.repeats)
            )
            if measurements is not None:
                return measurements, None, prof_dir
            error = parse_error
        last_error = error
        time.sleep(0.5)
    return None, last_error, None


@dataclass
class _CompareSummaryInput:
    """封装 _compute_compare_summary 的汇总计算参数。

    Args:
        out_dir: 算子输出目录（Path）
        rows: 逐 case 的测量结果列表
        speedups: 有效 speedup 值列表
        ref_times: 参考实现耗时列表（us）
        asc_times: AscendC 实现耗时列表（us）
        n_cases: case 总数
        args: argparse.Namespace（需含 warmup, repeats, seed 属性）
        device_id: NPU 设备 ID
        device_src: 设备来源描述（cli/env/auto）
    """
    out_dir: Path
    rows: list
    speedups: list
    ref_times: list
    asc_times: list
    n_cases: int
    args: argparse.Namespace
    device_id: int
    device_src: str


def _compute_compare_summary(csi: _CompareSummaryInput):
    """Compute the summary statistics dict for compare mode."""
    speedup_stats = _compute_speedup_stats(csi.speedups)
    timing_stats = _compute_timing_stats(csi.ref_times, csi.asc_times)
    geomean_speedup = speedup_stats["geomean_speedup"]
    return {
        "task": csi.out_dir.name,
        "task_dir": str(csi.out_dir),
        "n_cases_total": csi.n_cases,
        **speedup_stats,
        **timing_stats,
        "perf_target_speedup": PERF_TARGET_SPEEDUP,
        "target_met": (
            geomean_speedup is not None
            and geomean_speedup >= PERF_TARGET_SPEEDUP
        ),
        "warmup": csi.args.warmup,
        "repeats": csi.args.repeats,
        "seed": csi.args.seed,
        "device_id": csi.device_id,
        "device_select_source": csi.device_src,
        "timing_method": "msprof.op_summary.Task_Duration",
        "per_case": csi.rows,
    }


def _compute_speedup_stats(speedups: list) -> dict:
    """Compute speedup statistics from a list of speedup values.

    Returns a dict with n_cases_valid and geomean/mean/median/min/max speedup.
    """
    if not speedups:
        return {
            "n_cases_valid": 0,
            "geomean_speedup": None,
            "mean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
        }
    return {
        "n_cases_valid": len(speedups),
        "geomean_speedup": statistics.geometric_mean(speedups),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
    }


def _compute_timing_stats(ref_times: list, asc_times: list) -> dict:
    """Compute timing statistics for reference and AscendC implementations.

    Returns a dict with mean/median/geomean/total for ref and asc, plus total_speedup.
    """
    if ref_times:
        ref_stats = {
            "mean_ref_us": statistics.mean(ref_times),
            "median_ref_us": statistics.median(ref_times),
            "geomean_ref_us": statistics.geometric_mean(ref_times),
            "total_ref_us": sum(ref_times),
        }
    else:
        ref_stats = {"mean_ref_us": None, "median_ref_us": None,
                     "geomean_ref_us": None, "total_ref_us": None}

    if asc_times:
        asc_stats = {
            "mean_asc_us": statistics.mean(asc_times),
            "median_asc_us": statistics.median(asc_times),
            "geomean_asc_us": statistics.geometric_mean(asc_times),
            "total_asc_us": sum(asc_times),
        }
    else:
        asc_stats = {"mean_asc_us": None, "median_asc_us": None,
                     "geomean_asc_us": None, "total_asc_us": None}

    total_speedup = None
    if ref_times and asc_times:
        asc_sum = sum(asc_times)
        if asc_sum > 0:
            total_speedup = sum(ref_times) / asc_sum

    return {**ref_stats, **asc_stats, "total_speedup": total_speedup}


def _log_and_save_compare_reports(summary, out_dir, speedups, n_cases):
    """Log summary results and save JSON/log/Markdown reports."""
    LOGGER.info("-" * 100)
    if speedups:
        LOGGER.info("--- Speedup ---")
        LOGGER.info(f"  Geomean : {summary['geomean_speedup']:.2f}x  ← 主指标")
        LOGGER.info(f"  Mean    : {summary['mean_speedup']:.2f}x")
        LOGGER.info(f"  Median  : {summary['median_speedup']:.2f}x")
        LOGGER.info(f"  Min/Max : {summary['min_speedup']:.2f}x / {summary['max_speedup']:.2f}x")
        LOGGER.info(f"  Valid   : {len(speedups)}/{n_cases}")
    if summary.get("mean_ref_us") is not None:
        LOGGER.info("--- Task Duration (us) ---")
        LOGGER.info(
            f"  Ref  mean/median/geomean/total : {summary['mean_ref_us']:.2f} / "
            f"{summary['median_ref_us']:.2f} / {summary['geomean_ref_us']:.2f} / "
            f"{summary['total_ref_us']:.2f}"
        )
        LOGGER.info(
            f"  Asc  mean/median/geomean/total : {summary['mean_asc_us']:.2f} / "
            f"{summary['median_asc_us']:.2f} / {summary['geomean_asc_us']:.2f} / "
            f"{summary['total_asc_us']:.2f}"
        )
        LOGGER.info(f"  Total speedup (Σref/Σasc) : {summary['total_speedup']:.2f}x")
    LOGGER.info("=" * 100)

    # 保存 JSON 报告
    json_path = out_dir / "performance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"\n[INFO] JSON report saved to: {json_path}")

    # 保存打屏日志
    log_path = out_dir / "performance.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_report_compare_to_text(summary))
    LOGGER.info(f"[INFO] Console report saved to: {log_path}")

    # 保存 Markdown 报告
    md_path = out_dir / "perf_report.md"
    md = _report_compare_to_markdown(summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    LOGGER.info(f"[INFO] Markdown report saved to: {md_path}")


def _load_cases_from_model(out_dir):
    """数据集算子没有 case 文件(jsonl/json)时,从 model.py 枚举 case 数。

    benchmark 的 case 枚举默认从文件读;但 RL 数据集算子只有 get_inputs/get_input_groups,
    提交目录里没有 jsonl/json → 不 fallback 就会 n_cases=0,一行都不跑(空表 → benchmark_failed)。
    这里回退到 model.py:get_input_groups 返回多组直接用其个数;get_inputs 是单组 → 1 个 case。
    返回 [None]*n 占位 —— None 让 wrapper 走 get_inputs/get_input_groups fallback 取真实输入。"""
    model_path = Path(out_dir) / "model.py"
    if not model_path.is_file():
        return [], None
    try:
        ref_mod = _load_module(str(model_path), "ref_for_cases")
    except Exception:
        return [], None
    try:
        n = len(_resolve_input_groups(ref_mod))
    except Exception:
        return [], None
    return [None] * n, "model:get_inputs/get_input_groups"


def _load_case_metadata(out_dir):
    """Load optional JSON metadata without letting it define provider inputs."""
    try:
        metadata_cases, metadata_source = _load_cases_jsonl(out_dir)
        if not metadata_cases:
            metadata_cases, metadata_source = _load_cases_json(out_dir)
        return metadata_cases, metadata_source
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return [], None


def _load_compare_cases(out_dir):
    """Load case metadata while treating model.py's provider as authoritative.

    The JSON files bundled with NPUKernelBench are useful metadata, but the
    provider owns valid values and exact case count.  CUDA-LLM normally has no
    JSON at all; an agent-created JSON must therefore not replace its single
    ``get_inputs()`` case or alter the benchmark contract.
    """
    model_cases, model_source = _load_cases_from_model(out_dir)
    metadata_cases, metadata_source = _load_case_metadata(out_dir)

    if model_cases:
        if metadata_cases and len(metadata_cases) == len(model_cases):
            return metadata_cases, f"{model_source}+{metadata_source}:metadata"
        return model_cases, model_source
    return metadata_cases, metadata_source


def _materialize_compare_cases(out_dir: Path, cache_dir: Path, seed: int):
    """Materialize provider cases once for one compare/quick pipeline.

    Reference and AscendC wrappers load separate tensor objects from the same
    serialized case, so they see identical values without sharing mutations.
    The caller owns ``cache_dir`` and removes it when the pipeline finishes;
    inputs are never reused across separate evaluation attempts.

    If a provider returns an object that ``torch.save`` cannot serialize, fall
    back to the historical per-wrapper provider call rather than rejecting an
    otherwise valid task.
    """
    import torch

    model_path = out_dir / "model.py"
    if not model_path.is_file():
        cases, source = _load_compare_cases(out_dir)
        return cases, [None] * len(cases), source

    out_dir_str = str(out_dir)
    added_to_path = out_dir_str not in sys.path
    if added_to_path:
        sys.path.insert(0, out_dir_str)
    try:
        ref_mod = _load_module(str(model_path), "ref_for_case_materialization")
    finally:
        if added_to_path:
            sys.path.remove(out_dir_str)
    if not hasattr(ref_mod, "get_input_groups") and not hasattr(ref_mod, "get_inputs"):
        cases, source = _load_compare_cases(out_dir)
        return cases, [None] * len(cases), source

    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed_all(seed)
    except Exception:
        pass
    provider_cases = _resolve_input_groups(ref_mod)
    contract_cls = getattr(ref_mod, "Model")
    for case in provider_cases:
        _bind_case(contract_cls, case)

    metadata_cases, metadata_source = _load_case_metadata(out_dir)
    if metadata_cases and len(metadata_cases) == len(provider_cases):
        display_cases = [dict(case) if isinstance(case, Mapping) else {} for case in metadata_cases]
        source = f"model:get_inputs/get_input_groups+{metadata_source}:metadata+pipeline-cache"
    else:
        display_cases = [{} for _ in provider_cases]
        source = "model:get_inputs/get_input_groups+pipeline-cache"

    for display_case, provider_case in zip(display_cases, provider_cases):
        display_case["_provider_has_empty_tensor"] = _value_has_empty_tensor(provider_case)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = []
    try:
        for idx, case in enumerate(provider_cases):
            cache_path = cache_dir / f"case_{idx:03d}.pt"
            torch.save(_move(case, torch.device("cpu")), cache_path)
            cache_paths.append(cache_path)
    except (OSError, TypeError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "Provider case cache unavailable (%s); falling back to per-wrapper input generation",
            exc,
        )
        for cache_path in cache_paths:
            cache_path.unlink(missing_ok=True)
        return display_cases, [None] * len(provider_cases), source.replace(
            "+pipeline-cache", "+provider-fallback"
        )

    return display_cases, cache_paths, source


def _log_compare_header(out_dir, args):
    """Log the compare mode header."""
    LOGGER.info("=" * 100)
    LOGGER.info(f"Kernel-level Performance (msprof): {out_dir.name}  "
          f"(warmup={args.warmup}, repeats={args.repeats}, seed={args.seed})")
    LOGGER.info("=" * 100)
    LOGGER.info(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Ref(us)':>12} {'Asc(us)':>12} {'Speedup':>10}")
    LOGGER.info("-" * 100)


def _cleanup_prof_dirs(*dirs):
    """Remove profiling directories if they exist."""
    for d in dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def _run_compare_loop(out_dir, cases, case_cache_paths, n_cases, args, device_id):
    """Run the measurement loop over all cases. Returns rows and stats."""
    rows, speedups, ref_times, asc_times = [], [], [], []

    for idx in range(n_cases):
        shape, dtype = _extract_shape_dtype_from_jsonl(cases[idx]) if cases else ("?", "?")
        jsonl_case = cases[idx] if cases else None
        case_cache_path = case_cache_paths[idx] if case_cache_paths else None

        if _case_has_empty_tensor(jsonl_case):
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'--':>12} {'--':>12} "
                  f"{'skip':>10}  (empty tensor)")
            rows.append({
                "case": idx, "shape": shape, "dtype": dtype,
                "ref_us": None, "asc_us": None, "speedup": None,
                "ref_error": None, "asc_error": None,
                "skipped": "empty_tensor",
            })
            continue

        ref_mi = _MeasureInput(
            out_dir, idx, "reference", args, device_id, jsonl_case, case_cache_path
        )
        asc_mi = _MeasureInput(
            out_dir, idx, "ascendc", args, device_id, jsonl_case, case_cache_path
        )
        ref_us, ref_err, ref_prof_dir = _measure_one_impl(ref_mi)
        asc_us, asc_err, asc_prof_dir = _measure_one_impl(asc_mi)

        if ref_us is not None and asc_us is not None and asc_us > 0:
            sp = ref_us / asc_us
            speedups.append(sp)
            ref_times.append(ref_us)
            asc_times.append(asc_us)
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'N/A' if ref_us is None else f'{ref_us:.2f}':>12} "
                  f"{'N/A' if asc_us is None else f'{asc_us:.2f}':>12} "
                  f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

        rows.append({
            "case": idx, "shape": shape, "dtype": dtype,
            "ref_us": ref_us, "asc_us": asc_us,
            "speedup": (ref_us / asc_us) if (ref_us and asc_us and asc_us > 0) else None,
            "ref_error": ref_err,
            "asc_error": asc_err,
            "ref_prof_dir": ref_prof_dir,
            "asc_prof_dir": asc_prof_dir,
        })

        if not args.keep_prof:
            _cleanup_prof_dirs(ref_prof_dir, asc_prof_dir)

    return rows, speedups, ref_times, asc_times


def _run_quick_loop(out_dir, cases, case_cache_paths, n_cases, args, device_id):
    """快速模式：Run the measurement loop over all cases (只跑 1 轮 msprof).

    Returns rows and stats.
    """
    rows, speedups, ref_times, asc_times = [], [], [], []

    for idx in range(n_cases):
        shape, dtype = _extract_shape_dtype_from_jsonl(cases[idx]) if cases else ("?", "?")
        jsonl_case = cases[idx] if cases else None
        case_cache_path = case_cache_paths[idx] if case_cache_paths else None

        if _case_has_empty_tensor(jsonl_case):
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'--':>12} {'--':>12} "
                  f"{'skip':>10}  (empty tensor)")
            rows.append({
                "case": idx, "shape": shape, "dtype": dtype,
                "ref_us": None, "asc_us": None, "speedup": None,
                "ref_error": None, "asc_error": None,
                "skipped": "empty_tensor",
            })
            continue

        ref_mi = _MeasureInput(
            out_dir, idx, "reference", args, device_id, jsonl_case, case_cache_path
        )
        asc_mi = _MeasureInput(
            out_dir, idx, "ascendc", args, device_id, jsonl_case, case_cache_path
        )
        ref_us, ref_err, ref_prof_dir = _measure_one_impl_quick(ref_mi)
        asc_us, asc_err, asc_prof_dir = _measure_one_impl_quick(asc_mi)

        if ref_us is not None and asc_us is not None and asc_us > 0:
            sp = ref_us / asc_us
            speedups.append(sp)
            ref_times.append(ref_us)
            asc_times.append(asc_us)
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'N/A' if ref_us is None else f'{ref_us:.2f}':>12} "
                  f"{'N/A' if asc_us is None else f'{asc_us:.2f}':>12} "
                  f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

        rows.append({
            "case": idx, "shape": shape, "dtype": dtype,
            "ref_us": ref_us, "asc_us": asc_us,
            "speedup": (ref_us / asc_us) if (ref_us and asc_us and asc_us > 0) else None,
            "ref_error": ref_err,
            "asc_error": asc_err,
            "ref_prof_dir": ref_prof_dir,
            "asc_prof_dir": asc_prof_dir,
        })

        if not args.keep_prof:
            _cleanup_prof_dirs(ref_prof_dir, asc_prof_dir)

    return rows, speedups, ref_times, asc_times


def _run_quick_grouped_loop(out_dir, cases, case_cache_paths, n_cases, args, device_id):
    """Run every non-empty case and both implementations in one msprof process."""
    blocks = []
    for idx in range(n_cases):
        jsonl_case = cases[idx] if cases else None
        if _case_has_empty_tensor(jsonl_case):
            continue
        case_path = case_cache_paths[idx] if case_cache_paths else None
        if case_path is None:
            return None, "grouped mode requires materialized provider inputs", None
        blocks.extend(((idx, "reference", case_path), (idx, "ascendc", case_path)))

    if not blocks:
        measurements, prof_dir = {}, None
    else:
        measurements, error, prof_dir = _measure_all_impls_grouped(
            out_dir, blocks, args, device_id
        )
        if measurements is None:
            return None, error, prof_dir
        failed = [
            f"case {case_idx} {impl}: {measurements.get((case_idx, impl), {}).get('error')}"
            for case_idx, impl, _case_path in blocks
            if measurements.get((case_idx, impl), {}).get("duration_us") is None
        ]
        if failed:
            if prof_dir and not args.keep_prof:
                _cleanup_prof_dirs(prof_dir)
            return None, "; ".join(failed[:3]), prof_dir

    rows, speedups, ref_times, asc_times = [], [], [], []
    for idx in range(n_cases):
        shape, dtype = _extract_shape_dtype_from_jsonl(cases[idx]) if cases else ("?", "?")
        jsonl_case = cases[idx] if cases else None
        if _case_has_empty_tensor(jsonl_case):
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                        f"{'--':>12} {'--':>12} {'skip':>10}  (empty tensor)")
            rows.append({
                "case": idx, "shape": shape, "dtype": dtype,
                "ref_us": None, "asc_us": None, "speedup": None,
                "ref_error": None, "asc_error": None, "skipped": "empty_tensor",
            })
            continue

        ref_result = measurements.get((idx, "reference"), {})
        asc_result = measurements.get((idx, "ascendc"), {})
        ref_us, ref_err = ref_result.get("duration_us"), ref_result.get("error")
        asc_us, asc_err = asc_result.get("duration_us"), asc_result.get("error")
        speedup = None
        if ref_us is not None and asc_us is not None and asc_us > 0:
            speedup = ref_us / asc_us
            speedups.append(speedup)
            ref_times.append(ref_us)
            asc_times.append(asc_us)
            LOGGER.info(
                f"{idx:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} "
                f"{asc_us:>12.2f} {speedup:>9.3f}x"
            )
        else:
            LOGGER.info(
                f"{idx:<5} {shape:<35} {dtype:<10} "
                f"{'N/A' if ref_us is None else f'{ref_us:.2f}':>12} "
                f"{'N/A' if asc_us is None else f'{asc_us:.2f}':>12} "
                f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})"
            )
        rows.append({
            "case": idx, "shape": shape, "dtype": dtype,
            "ref_us": ref_us, "asc_us": asc_us, "speedup": speedup,
            "ref_error": ref_err, "asc_error": asc_err,
            "ref_prof_dir": prof_dir, "asc_prof_dir": prof_dir,
        })

    if prof_dir and not args.keep_prof:
        _cleanup_prof_dirs(prof_dir)
    return (rows, speedups, ref_times, asc_times), None, prof_dir


def run_compare_mode(args):
    """执行对比模式：model.py vs model_new_ascendc.py"""
    out_dir = Path(args.output_dir).resolve()

    device_id, device_src = _select_device_id(args)
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")

    with tempfile.TemporaryDirectory(prefix="polar_perf_cases_") as cache_root:
        cases, case_cache_paths, case_source = _materialize_compare_cases(
            out_dir, Path(cache_root), args.seed
        )
        n_cases = len(cases)
        if case_source:
            LOGGER.info(f"[INFO] Loaded {n_cases} cases from {case_source}")

        _log_compare_header(out_dir, args)
        rows, speedups, ref_times, asc_times = _run_compare_loop(
            out_dir, cases, case_cache_paths, n_cases, args, device_id
        )

    csi = _CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, n_cases, args, device_id, device_src)
    summary = _compute_compare_summary(csi)
    _log_and_save_compare_reports(summary, out_dir, speedups, n_cases)


def run_quick_mode(args):
    """执行快速模式：默认把全部 case 合并到一次 msprof 采集中。"""
    out_dir = Path(args.output_dir).resolve()

    device_id, device_src = _select_device_id(args)
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")
    LOGGER.info("[INFO] Quick mode: grouped profiling (one msprof for all cases)")

    with tempfile.TemporaryDirectory(prefix="polar_perf_cases_") as cache_root:
        cases, case_cache_paths, case_source = _materialize_compare_cases(
            out_dir, Path(cache_root), args.seed
        )
        n_cases = len(cases)
        if case_source:
            LOGGER.info(f"[INFO] Loaded {n_cases} cases from {case_source}")

        _log_compare_header(out_dir, args)
        requested_engine = getattr(args, "quick_engine", "grouped")
        if requested_engine == "per-case":
            grouped_result, grouped_error = None, "per-case engine requested"
        else:
            grouped_result, grouped_error, _prof_dir = _run_quick_grouped_loop(
                out_dir, cases, case_cache_paths, n_cases, args, device_id
            )
        if grouped_result is None:
            if requested_engine == "per-case":
                LOGGER.info("[INFO] Quick mode: per-case msprof explicitly requested")
            else:
                LOGGER.warning(
                    "Grouped quick profiling unavailable (%s); falling back to per-case msprof",
                    grouped_error,
                )
            rows, speedups, ref_times, asc_times = _run_quick_loop(
                out_dir, cases, case_cache_paths, n_cases, args, device_id
            )
            profiling_engine = "per_case_fallback"
            msprof_invocations = 2 * sum(
                not _case_has_empty_tensor(case) for case in cases
            )
        else:
            rows, speedups, ref_times, asc_times = grouped_result
            profiling_engine = "grouped_event_markers"
            msprof_invocations = 1 if any(
                not _case_has_empty_tensor(case) for case in cases
            ) else 0

    csi = _CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, n_cases, args, device_id, device_src)
    summary = _compute_compare_summary(csi)
    summary["timing_method"] = "msprof.quick.Task_Duration"
    summary["profiling_mode"] = "quick"
    summary["profiling_engine"] = profiling_engine
    summary["msprof_invocations"] = msprof_invocations
    _log_and_save_compare_reports(summary, out_dir, speedups, n_cases)


# ============================================================================
# 批量模式：扫描多个算子目录，汇总报告
# ============================================================================

def _is_valid_table_row(line: str) -> bool:
    return line.startswith('|') and 'Level' not in line and '---' not in line and len(line) > 5


def _extract_trace_table_rows(trace_file_path: str) -> List[str]:
    if not os.path.exists(trace_file_path):
        return []
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    start_idx = content.find('## 汇总表报告')
    if start_idx == -1:
        return []
    section_content = content[start_idx:]
    lines = section_content.split('\n')
    valid_rows = []
    for line in lines:
        line = line.strip()
        if _is_valid_table_row(line):
            valid_rows.append(_normalize_trace_table_row(line))
    return valid_rows


def _normalize_trace_table_row(line: str) -> str:
    """Normalize current and historical trace rows to the single 1.1x target column.

    Historical traces have two performance columns (0.6x and 0.8x).  Recompute the
    new target result from the recorded speedup so batch reports remain aligned.
    """
    columns = [column.strip() for column in line.strip().strip("|").split("|")]
    if len(columns) < 11:
        return line
    raw_speedup = columns[8].lower().removesuffix("x").strip()
    try:
        speedup = float(raw_speedup)
    except ValueError:
        speedup = None
    target_met = "是" if speedup is not None and speedup >= PERF_TARGET_SPEEDUP else "否"
    normalized = columns[:11] + [target_met]
    return "| " + " | ".join(normalized) + " |"


def _load_performance_json(op_dir: Path) -> Optional[Dict[str, Any]]:
    perf_json = op_dir / "performance.json"
    if perf_json.exists():
        try:
            with open(perf_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load performance.json from %s: %s", op_dir, e)
    return None


def _build_batch_md_summary_table(op_results):
    """Build the batch summary table markdown lines."""
    md_lines = []
    if op_results:
        md_lines.append("## 性能汇总")
        md_lines.append("")
        md_lines.append(
            "| 算子名称 | 用例数 | 有效用例 | 几何平均加速比 | 平均加速比 | 达标(≥1.1x) |"
        )
        md_lines.append("| -------- | ------ | -------- | -------------- | ---------- | ---- |")
        for op in op_results:
            data = op["data"]
            name = op["name"]
            n_total = data.get("n_cases_total", 0)
            n_valid = data.get("n_cases_valid", 0)
            geo = data.get("geomean_speedup")
            mean = data.get("mean_speedup")
            geo_str = f"{geo:.3f}" if geo is not None else "N/A"
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            status = (
                "✅" if geo is not None and geo >= PERF_TARGET_SPEEDUP
                else "⚠️" if geo is not None else "❌"
            )
            md_lines.append(f"| {name} | {n_total} | {n_valid} | {geo_str} | {mean_str} | {status} |")
        md_lines.append("")
    return md_lines


def _build_batch_md_per_op_details(op_results):
    """Build per-operator detail tables in markdown."""
    md_lines = []
    for op in op_results:
        data = op["data"]
        name = op["name"]
        md_lines.append(f"## {name}")
        md_lines.append("")
        if data.get("per_case"):
            md_lines.append("| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |")
            md_lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
            for case in data["per_case"]:
                shape = case.get("shape", "?")
                dtype = case.get("dtype", "?")
                ref = case.get("ref_us")
                asc = case.get("asc_us")
                sp = case.get("speedup")
                ref_str = f"{ref:.2f}" if ref is not None else "N/A"
                asc_str = f"{asc:.2f}" if asc is not None else "N/A"
                sp_str = f"{sp:.3f}" if sp is not None else "N/A"
                md_lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
            md_lines.append("")
    return md_lines


def _build_batch_md_trace_section(trace_rows):
    """Build trace table section in markdown if rows exist."""
    if not trace_rows:
        return []
    md_lines = [
        "## Trace 汇总表",
        "",
        ("| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | "
         "PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | "
         "精度正确 | 性能达标(≥1.1x) |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    md_lines.extend(trace_rows)
    md_lines.append("")
    return md_lines


def _generate_batch_md_report(args, op_results, trace_rows, base_dir):
    """Generate and save the batch Markdown report."""
    md_lines = []
    md_lines.append("# 📊 算子批量性能汇总报告")
    md_lines.append("")
    md_lines.append(f"- **扫描目录**: {base_dir}")
    md_lines.append(f"- **算子总数**: {len(op_results)}")
    md_lines.append(f"- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    md_lines.extend(_build_batch_md_summary_table(op_results))
    md_lines.extend(_build_batch_md_per_op_details(op_results))
    md_lines.extend(_build_batch_md_trace_section(trace_rows))

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    LOGGER.info("Batch markdown report saved to: %s", md_path)


def _generate_batch_json_report(args, op_results, base_dir):
    """Generate and save the batch JSON summary."""
    batch_summary = {
        "base_dir": str(base_dir),
        "n_operators": len(op_results),
        "perf_target_speedup": PERF_TARGET_SPEEDUP,
        "operators": [
            {
                "name": op["name"],
                "n_cases_total": op["data"].get("n_cases_total", 0),
                "n_cases_valid": op["data"].get("n_cases_valid", 0),
                "geomean_speedup": op["data"].get("geomean_speedup"),
                "mean_speedup": op["data"].get("mean_speedup"),
                "mean_ref_us": op["data"].get("mean_ref_us"),
                "mean_asc_us": op["data"].get("mean_asc_us"),
                "target_met": (
                    op["data"].get("geomean_speedup") is not None
                    and op["data"].get("geomean_speedup") >= PERF_TARGET_SPEEDUP
                ),
            }
            for op in op_results
        ],
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("Batch JSON summary saved to: %s", json_path)


def run_batch_mode(args):
    """执行批量模式：扫描 base_dir 下所有子目录，汇总性能报告。"""
    base_dir = Path(args.batch).resolve()

    if not base_dir.is_dir():
        raise ValueError("'%s' is not a directory." % base_dir)

    # 收集所有子目录的 performance.json
    op_results = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        perf_data = _load_performance_json(subdir)
        if perf_data:
            op_results.append({
                "name": subdir.name,
                "data": perf_data,
                "dir": subdir,
            })

    # 同时收集 trace.md 中的表格行（兼容旧 batch_report.py 功能）
    trace_rows = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        trace_file = subdir / "trace.md"
        if trace_file.exists():
            rows = _extract_trace_table_rows(str(trace_file))
            trace_rows.extend(rows)

    LOGGER.info("Found %d operators with performance.json in %s", len(op_results), base_dir)

    if args.output_md:
        _generate_batch_md_report(args, op_results, trace_rows, base_dir)

    if args.output_json:
        _generate_batch_json_report(args, op_results, base_dir)


# ============================================================================
# 主入口
# ============================================================================

def _run_standard_mode(args):
    """Execute standard mode: parse PROF_GROUP and generate summary."""
    group_dir = os.path.abspath(args.prof_group_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(group_dir):
        raise ValueError("'%s' is not a directory." % group_dir)
    if not os.path.isdir(ops_dir):
        raise ValueError("'%s' is not a directory." % ops_dir)

    merged = merge_metric_rows(group_dir, args.op_name)
    if not merged.get("Op Name"):
        raise ValueError("no op_summary_*.csv rows discovered under %s" % group_dir)

    perf_dir = os.path.join(ops_dir, "docs", "perf")
    round_dir = os.path.join(perf_dir, args.round_name) if args.round_name else find_next_round(perf_dir)

    copied = archive_csvs(group_dir, round_dir)
    pc_csv = archive_per_core_csv(group_dir, round_dir)
    if pc_csv:
        copied.append(os.path.basename(pc_csv))
    LOGGER.info("Archived %d CSV files to: %s", len(copied), round_dir)

    summary = generate_summary(merged, round_dir, group_dir)
    summary_path = os.path.join(round_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    LOGGER.info("Summary written to: %s", summary_path)
    LOGGER.info("\n%s", summary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="msprof 解析 & 归档 & 对比测试脚本（统一入口）")

    # 标准模式参数
    parser.add_argument("prof_group_dir", nargs="?", help="PROF_GROUP_<timestamp> directory")
    parser.add_argument("ops_dir", nargs="?", help="Operator directory")
    parser.add_argument("--op-name", default=None, help="Exact Op Name to pick in op_summary.csv")
    parser.add_argument("--round-name", default=None, help="Override round directory name")

    # 对比模式参数
    parser.add_argument("--compare", action="store_true", help="启用对比模式（8 轮采集：7 metrics + sample）")
    parser.add_argument("--quick", action="store_true", help="启用快速模式（1 轮采集：只获取 kernel 时间，不采集 7 个 aic-metrics）")
    parser.add_argument(
        "--quick-engine", choices=("grouped", "per-case"), default="grouped",
        help="快速模式采集引擎：grouped 将全部 case 合并为一次 msprof；per-case 用于回退",
    )
    parser.add_argument("--output-dir", dest="output_dir", help="算子输出目录（对比模式/快速模式）")
    parser.add_argument("--warmup", type=int, default=3, help="msprof warmup 次数")
    parser.add_argument("--repeats", type=int, default=1, help="重复采集次数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--retry", type=int, default=2, help="单 case 解析失败重试次数")
    parser.add_argument("--device", type=int, default=None, help="NPU 设备 id")
    parser.add_argument("--keep-prof", action="store_true", help="保留 msprof 原始 PROF 目录")

    # 批量模式参数
    parser.add_argument("--batch", metavar="BASE_DIR", help="启用批量模式，指定根目录")
    parser.add_argument("--output-md", help="批量模式 Markdown 输出路径")
    parser.add_argument("--output-json", help="批量模式 JSON 输出路径")

    args = parser.parse_args()

    # 模式路由
    if args.compare:
        if not args.output_dir:
            parser.error("--compare 模式必须指定 --output-dir")
        run_compare_mode(args)
    elif args.quick:
        if not args.output_dir:
            parser.error("--quick 模式必须指定 --output-dir")
        run_quick_mode(args)
    elif args.batch:
        try:
            run_batch_mode(args)
        except ValueError as e:
            LOGGER.error("%s", e)
            sys.exit(1)
    else:
        if not args.prof_group_dir or not args.ops_dir:
            parser.error("标准模式需要 prof_group_dir 和 ops_dir 参数")
        try:
            _run_standard_mode(args)
        except ValueError as e:
            LOGGER.error("%s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
