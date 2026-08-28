#!/usr/bin/env python3
"""Parse persisted VIME/Polar logs and plot comparable training/rollout curves."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONFIG_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*)\s+\.{2,}\s+(.*?)\s*$")
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
TRAIN_RE = re.compile(r"\bstep (\d+): (\{.*\})$")
PERF_RE = re.compile(r"\bperf (\d+): (\{.*\})$")
ROLLOUT_RE = re.compile(r"\brollout (\d+): (\{.*\})$")
DURATION_RE = re.compile(r"Async rollout collected (\d+) groups in ([0-9.]+)s")
RUN_ID_RE = re.compile(r"train_qwen36_polar_(\d{8}-\d{6})\.log$")

CONFIG_KEYS = (
    "lr",
    "global_batch_size",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "rollout_max_active_sessions",
    "rollout_release_on_postrun",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "actor_num_gpus_per_node",
    "actor_num_nodes",
    "rollout_max_response_len",
    "rollout_max_off_policy_steps",
    "rollout_scheduler_mode",
    "polar_trajectory_pg_floor",
    "rollout_stop_token_ids",
    "vllm_speculative_config",
    "hf_checkpoint",
    "clip_grad",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "sequence_parallel",
    "offload_train",
    "offload_rollout",
)

RUN_CONTEXT: dict[str, dict[str, Any]] = {
    "20260820-221632": {
        "kind": "normal long run",
        "changes": [
            "VIME 8d91aa62: attempt credit + PG floor",
            "Polar 055ccc03/89a6ff6d: truncation penalty + attempt spans",
            "Polar 5c28fa2e: post-best masking",
        ],
    },
    "20260821-192814": {
        "kind": "normal long run; reward scale changed",
        "changes": [
            "Polar aec0a5d0: empty-truncation merge / no chain split",
            "Polar 2c05dc08: independent post-best switch (default still on)",
            "Polar 7441b966: CoT/reasoning enters training by default",
            "Polar 84481b85: process reward enabled by default",
        ],
    },
    "20260824-114318": {
        "kind": "MTP abnormal run",
        "changes": [
            "MTP speculative decoding, 3 draft tokens",
            "Checkpoint changed to base Qwen3.6-35B-A3B",
            "LR changed to 1e-6",
            "VIME cee071ca: clear trainer cache after weight update",
        ],
    },
    "20260824-211918": {
        "kind": "normal long run",
        "changes": [
            "MTP disabled; agentical-4t checkpoint restored",
            "Polar 5af3e04a: process reward in Ascend operator path",
            "VIME 0aab9283: KV-free synchronous update window",
            "VIME b2f3a15f: hybrid vLLM memory util restored to 0.85",
        ],
    },
    "20260825-214539": {
        "kind": "2x2 smoke test; no useful GRPO variance",
        "changes": [
            "VIME 4a6c5e51: Qwen shared attention-spec pollution fix",
            "Polar e804bf57: CPU evaluation/timeout/profile adjustments",
            "28 configured rollout GPUs; train/rollout offload enabled",
            "B4 (2 groups x 2 samples), max stale=1",
        ],
    },
    "20260826-102857": {
        "kind": "2x2 smoke test; no useful GRPO variance",
        "changes": [
            "Same Qwen attention-spec fix and CPU evaluation profile",
            "28 configured rollout GPUs; train/rollout offload enabled",
            "B4 (2 groups x 2 samples), max stale=0",
        ],
    },
    "20260826-130952": {
        "kind": "MTP eager abnormal run",
        "changes": [
            "MTP speculative decoding, 3 draft tokens + enforce_eager",
            "Checkpoint changed to base Qwen3.6-35B-A3B",
            "B32 (4 groups x 8 samples), max stale=0",
            "Started before the 15:17 fail-closed commits",
        ],
    },
    "20260826-173048": {
        "kind": "current comparable synchronous run",
        "changes": [
            "VIME d9b3a3b8 + Polar b8901037: fail-closed weight boundary",
            "VIME bc79852b: canonical MindSpeed actor init/repatch order",
            "VIME ca07ae09: native CP full-attention mapping",
            "Polar 94476486: postrun workers 8 -> 16",
            "B64 (8x8), 24 rollout GPUs, max stale=0, MTP off",
        ],
    },
}


def scalar(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def literal_dict(text: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def timestamp(line: str) -> str | None:
    match = TIMESTAMP_RE.search(line)
    return match.group(1) if match else None


def parse_log(path: Path) -> dict[str, Any]:
    match = RUN_ID_RE.search(path.name)
    run_id = match.group(1) if match else path.stem
    config: dict[str, Any] = {}
    train: dict[int, dict[str, Any]] = {}
    perf: dict[int, dict[str, Any]] = {}
    rollout: dict[int, dict[str, Any]] = {}
    durations: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line.rstrip())
            cfg = CONFIG_RE.match(line)
            if cfg and cfg.group(1) not in config:
                config[cfg.group(1)] = scalar(cfg.group(2))

            duration_match = DURATION_RE.search(line)
            if duration_match:
                durations.append(
                    {
                        "index": len(durations),
                        "groups": int(duration_match.group(1)),
                        "seconds": float(duration_match.group(2)),
                        "timestamp": timestamp(line),
                    }
                )

            if "model.py:" in line and "'train/loss'" in line:
                metric_match = TRAIN_RE.search(line)
                if metric_match:
                    values = literal_dict(metric_match.group(2))
                    if values is not None:
                        values["timestamp"] = timestamp(line)
                        train[int(metric_match.group(1))] = values
            elif "rollout.py:" in line and "'perf/rollout_time'" in line:
                metric_match = PERF_RE.search(line)
                if metric_match:
                    values = literal_dict(metric_match.group(2))
                    if values is not None:
                        values["timestamp"] = timestamp(line)
                        perf[int(metric_match.group(1))] = values
            elif "data.py:" in line and "'rollout/raw_reward'" in line:
                metric_match = ROLLOUT_RE.search(line)
                if metric_match:
                    values = literal_dict(metric_match.group(2))
                    if values is not None:
                        values["timestamp"] = timestamp(line)
                        rollout[int(metric_match.group(1))] = values

    return {
        "run_id": run_id,
        "path": str(path),
        "config": config,
        "train": train,
        "perf": perf,
        "rollout": rollout,
        "durations": durations,
    }


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = (
        "train/loss",
        "train/pg_loss",
        "train/entropy_loss",
        "train/grad_norm",
        "train/train_rollout_logprob_abs_diff",
        "train/ois",
        "train/tis",
        "train/tis_clipfrac",
        "train/tis_abs",
        "train/lr-pg_0",
        "train/global_batch_size",
        "rollout/raw_reward",
        "rollout/response_lengths",
        "rollout/truncated",
        "polar/reward_mean",
        "polar/reward_mean_completed",
        "polar/reward_std",
        "polar/rollout_success_rate",
        "polar/staleness/mean",
        "rollout_bench/output_throughput",
        "rollout_bench/total_token_throughput",
        "rollout_bench/tpot_mean_ms",
        "rollout_bench/ttft_mean_ms",
        "rollout/response_len/mean",
        "rollout/truncated_ratio",
        "perf/rollout_time",
        "perf/tokens_per_gpu_per_sec",
        "perf/effective_tokens_per_gpu_per_sec",
    )
    for run in runs:
        indices = sorted(set(run["train"]) | set(run["perf"]) | set(run["rollout"]))
        for index in indices:
            merged: dict[str, Any] = {}
            for section in (run["perf"].get(index), run["rollout"].get(index), run["train"].get(index)):
                if section:
                    merged.update(section)
            row: dict[str, Any] = {
                "run_id": run["run_id"],
                "log_path": run["path"],
                "step": index,
                "timestamp": (run["train"].get(index) or {}).get("timestamp")
                or (run["perf"].get(index) or {}).get("timestamp"),
            }
            row.update({key: merged.get(key) for key in keys})
            if index < len(run["durations"]):
                row["collected_groups"] = run["durations"][index]["groups"]
                row["rollout_collected_seconds"] = run["durations"][index]["seconds"]
            else:
                row["collected_groups"] = None
                row["rollout_collected_seconds"] = None
            rows.append(row)
    return rows


def config_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = {"run_id": run["run_id"], "log_path": run["path"], "train_steps": len(run["train"])}
        row.update({key: run["config"].get(key) for key in CONFIG_KEYS})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_label(run: dict[str, Any]) -> str:
    cfg = run["config"]
    batch = cfg.get("global_batch_size", "?")
    lr = cfg.get("lr", "?")
    mode = ", MTP" if cfg.get("vllm_speculative_config") else ""
    return f"{run['run_id'][4:]} (B{batch}, lr={lr}{mode})"


def plot_training(runs: list[dict[str, Any]], path: Path, title: str) -> None:
    specs = (
        ("polar/reward_mean", "Polar reward mean", False),
        ("polar/reward_mean_completed", "Completed-session reward", False),
        ("train/loss", "Training loss", False),
        ("train/grad_norm", "Gradient norm (log scale)", True),
        ("train/train_rollout_logprob_abs_diff", "Train-rollout |logprob diff|", True),
        ("train/tis", "Token importance sampling (TIS)", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for run_index, run in enumerate(runs):
        steps = sorted(run["train"])
        for axis, (key, _, log_scale) in zip(axes.flat, specs):
            source = run["train"] if key.startswith("train/") else run["perf"]
            xs, ys = [], []
            for step in steps:
                value = finite((source.get(step) or {}).get(key))
                if value is not None and (not log_scale or value > 0):
                    xs.append(step)
                    ys.append(value)
            if xs:
                axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, color=colors(run_index), label=run_label(run))
    for axis, (_, panel_title, log_scale) in zip(axes.flat, specs):
        axis.set_title(panel_title)
        axis.set_xlabel("Training step within run")
        axis.grid(True, alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(title, fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rollout(runs: list[dict[str, Any]], path: Path) -> None:
    specs = (
        ("perf/rollout_time", "Rollout collection time (hours)", 1 / 3600),
        ("rollout_bench/output_throughput", "Output throughput (tokens/s)", 1),
        ("rollout/response_len/mean", "Mean response length per terminal unit", 1),
        ("polar/rollout_success_rate", "Rollout success rate", 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for run_index, run in enumerate(runs):
        steps = sorted(run["perf"])
        for axis, (key, _, scale) in zip(axes.flat, specs):
            xs, ys = [], []
            for step in steps:
                value = finite((run["perf"].get(step) or {}).get(key))
                if value is not None:
                    xs.append(step)
                    ys.append(value * scale)
            if xs:
                axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, color=colors(run_index), label=run_label(run))
    for axis, (_, title, _) in zip(axes.flat, specs):
        axis.set_title(title)
        axis.set_xlabel("Rollout index within run")
        axis.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Rollout time, load and output quality", fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mean(values: list[Any]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return float(np.mean(numbers)) if numbers else None


def summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        cfg = run["config"]
        train_values = list(run["train"].values())
        perf_values = list(run["perf"].values())
        rollout_hours = (mean([value.get("perf/rollout_time") for value in perf_values]) or 0) / 3600 if perf_values else None
        samples_per_rollout = finite(cfg.get("rollout_batch_size"))
        samples_per_group = finite(cfg.get("n_samples_per_prompt"))
        sample_count = samples_per_rollout * samples_per_group if samples_per_rollout and samples_per_group else None
        rows.append(
            {
                "run_id": run["run_id"],
                "steps": len(train_values),
                "batch": cfg.get("global_batch_size"),
                "groups_x_samples": f"{cfg.get('rollout_batch_size')}x{cfg.get('n_samples_per_prompt')}",
                "lr": cfg.get("lr") or (train_values[0].get("train/lr-pg_0") if train_values else None),
                "mode": "MTP" if cfg.get("vllm_speculative_config") else "normal",
                "checkpoint": Path(str(cfg.get("hf_checkpoint", ""))).name,
                "rollout_gpus": cfg.get("rollout_num_gpus"),
                "active_sessions": cfg.get("rollout_max_active_sessions"),
                "reward_mean": mean([value.get("polar/reward_mean") for value in perf_values]),
                "reward_first": finite(perf_values[0].get("polar/reward_mean")) if perf_values else None,
                "reward_last": finite(perf_values[-1].get("polar/reward_mean")) if perf_values else None,
                "grad_norm_mean": mean([value.get("train/grad_norm") for value in train_values]),
                "grad_norm_max": max((finite(value.get("train/grad_norm")) or 0 for value in train_values), default=None),
                "logprob_diff_mean": mean([value.get("train/train_rollout_logprob_abs_diff") for value in train_values]),
                "rollout_hours_mean": rollout_hours,
                "samples_per_hour": sample_count / rollout_hours if sample_count and rollout_hours else None,
                "output_tok_s_mean": mean([value.get("rollout_bench/output_throughput") for value in perf_values]),
                "response_len_mean": mean([value.get("rollout/response_len/mean") for value in perf_values]),
                "success_rate_mean": mean([value.get("polar/rollout_success_rate") for value in perf_values]),
                "staleness_mean": mean([value.get("polar/staleness/mean") for value in perf_values]),
            }
        )
    return rows


def plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["run_id"][4:] for row in rows]
    specs = (
        ("reward_mean", "Mean reward"),
        ("grad_norm_mean", "Mean gradient norm (log scale)"),
        ("logprob_diff_mean", "Mean train-rollout |logprob diff|"),
        ("rollout_hours_mean", "Mean rollout time (hours)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, specs):
        values = [finite(row.get(key)) or 0 for row in rows]
        bars = axis.bar(labels, values, color=plt.get_cmap("tab10")(range(len(rows))))
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.25)
        if key == "grad_norm_mean" and all(value > 0 for value in values):
            axis.set_yscale("log")
    fig.suptitle("Run-level comparison (each bar is one persisted training run)", fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_individual_runs(runs: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        steps = sorted(run["perf"])
        context = RUN_CONTEXT.get(run["run_id"], {"kind": "", "changes": []})
        cfg = run["config"]
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = fig.add_gridspec(3, 1, height_ratios=(3.2, 2.5, 1.6))
        reward_axis = fig.add_subplot(grid[0])
        time_axis = fig.add_subplot(grid[1])
        note_axis = fig.add_subplot(grid[2])

        raw_reward = [finite((run["perf"].get(step) or {}).get("polar/reward_mean")) for step in steps]
        completed_reward = [
            finite((run["perf"].get(step) or {}).get("polar/reward_mean_completed")) for step in steps
        ]
        reward_axis.plot(steps, raw_reward, marker="o", linewidth=2.2, label="reward_mean")
        if any(value is not None for value in completed_reward):
            reward_axis.plot(
                steps,
                completed_reward,
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label="reward_mean_completed",
            )
        reward_axis.set_title(f"Reward — {run['run_id']}")
        reward_axis.set_xlabel("Rollout / training index")
        reward_axis.set_ylabel("Reward")
        reward_axis.grid(True, alpha=0.25)
        reward_axis.legend()

        rollout_hours = [
            (finite((run["perf"].get(step) or {}).get("perf/rollout_time")) or 0) / 3600 for step in steps
        ]
        time_axis.bar(steps, rollout_hours, color="#4C78A8", alpha=0.85)
        time_axis.plot(steps, rollout_hours, color="#1f3d5a", marker="o", linewidth=1.2)
        time_axis.set_title("Rollout collection wall time")
        time_axis.set_xlabel("Rollout index")
        time_axis.set_ylabel("Hours")
        time_axis.grid(True, axis="y", alpha=0.25)

        note_axis.axis("off")
        mode = "MTP" if cfg.get("vllm_speculative_config") else "normal"
        checkpoint = Path(str(cfg.get("hf_checkpoint", ""))).name
        config_line = (
            f"Observed config: {cfg.get('rollout_batch_size')}x{cfg.get('n_samples_per_prompt')} "
            f"(B{cfg.get('global_batch_size')}), lr={cfg.get('lr')}, rollout_gpus={cfg.get('rollout_num_gpus')}, "
            f"active={cfg.get('rollout_max_active_sessions')}, max_stale={cfg.get('rollout_max_off_policy_steps')}, "
            f"mode={mode}, checkpoint={checkpoint}"
        )
        notes = "\n".join(f"• {item}" for item in context.get("changes", []))
        note_axis.text(
            0,
            0.98,
            f"Run class: {context.get('kind', '')}\n{textwrap.fill(config_line, 130)}\nChanges present before this run:\n{notes}",
            va="top",
            ha="left",
            fontsize=10.5,
            linespacing=1.35,
        )
        fig.savefig(output_dir / f"{run['run_id']}_reward_rollout.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_individual_grad_norm(runs: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        steps = sorted(run["train"])
        values = [finite((run["train"].get(step) or {}).get("train/grad_norm")) for step in steps]
        valid = [(step, value) for step, value in zip(steps, values) if value is not None and value > 0]
        if not valid:
            continue

        context = RUN_CONTEXT.get(run["run_id"], {"kind": "", "changes": []})
        cfg = run["config"]
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=(4, 1.65))
        axis = fig.add_subplot(grid[0])
        note_axis = fig.add_subplot(grid[1])
        xs = [item[0] for item in valid]
        ys = [item[1] for item in valid]

        axis.plot(xs, ys, marker="o", linewidth=2.2, color="#D62728", label="train/grad_norm")
        clip_grad = finite(cfg.get("clip_grad"))
        if clip_grad and clip_grad > 0:
            axis.axhline(
                clip_grad,
                color="#555555",
                linestyle="--",
                linewidth=1.6,
                label=f"clip_grad={clip_grad:g}",
            )
        for step, value in valid:
            axis.annotate(f"{value:.4g}", (step, value), xytext=(0, 8), textcoords="offset points", ha="center")
        axis.set_yscale("log")
        axis.set_title(f"Gradient norm — {run['run_id']}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("grad_norm (log scale)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

        mode = "MTP" if cfg.get("vllm_speculative_config") else "normal"
        checkpoint = Path(str(cfg.get("hf_checkpoint", ""))).name
        note_axis.axis("off")
        note_axis.text(
            0,
            0.98,
            textwrap.fill(
                f"Run class: {context.get('kind', '')}. Observed config: "
                f"{cfg.get('rollout_batch_size')}x{cfg.get('n_samples_per_prompt')} "
                f"(B{cfg.get('global_batch_size')}), lr={cfg.get('lr')}, mode={mode}, "
                f"max_stale={cfg.get('rollout_max_off_policy_steps')}, checkpoint={checkpoint}",
                135,
            ),
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.35,
        )
        fig.savefig(output_dir / f"{run['run_id']}_grad_norm.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def change_timeline_markdown(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# 每次训练曲线对应的代码与配置时间线",
        "",
        "> 说明：旧训练日志没有记录 VIME/Polar commit hash。下表的参数来自日志，是直接证据；commit 是按仓库提交时间筛选出的“启动前已进入代码树”的改动。除非日志出现该版本独有的信息，不能把后者当成进程实际加载 commit 的绝对证明。",
        "",
    ]
    for run in runs:
        cfg = run["config"]
        context = RUN_CONTEXT.get(run["run_id"], {"kind": "", "changes": []})
        lines.extend(
            [
                f"## {run['run_id']}",
                "",
                f"- 类型：{context.get('kind', '')}",
                f"- 日志实参：`{cfg.get('rollout_batch_size')}×{cfg.get('n_samples_per_prompt')}`，"
                f"`global_batch={cfg.get('global_batch_size')}`，`lr={cfg.get('lr')}`，"
                f"`rollout_gpus={cfg.get('rollout_num_gpus')}`，"
                f"`max_stale={cfg.get('rollout_max_off_policy_steps')}`，"
                f"`MTP={'开' if cfg.get('vllm_speculative_config') else '关'}`，"
                f"`checkpoint={Path(str(cfg.get('hf_checkpoint', ''))).name}`。",
                "- 启动前相关改动：",
                "",
            ]
        )
        lines.extend(f"  - {item}" for item in context.get("changes", []))
        lines.extend(
            [
                "",
                f"- Reward/rollout：[{run['run_id']}_reward_rollout.png](per_run/{run['run_id']}_reward_rollout.png)",
                f"- Grad norm：[{run['run_id']}_grad_norm.png](per_run_grad_norm/{run['run_id']}_grad_norm.png)",
                "",
            ]
        )
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = (
        "run_id",
        "steps",
        "batch",
        "groups_x_samples",
        "lr",
        "mode",
        "checkpoint",
        "rollout_gpus",
        "active_sessions",
        "reward_mean",
        "grad_norm_mean",
        "logprob_diff_mean",
        "rollout_hours_mean",
        "samples_per_hour",
        "output_tok_s_mean",
        "response_len_mean",
        "success_rate_mean",
        "staleness_mean",
    )

    def display(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(display(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def analysis_report(rows: list[dict[str, Any]]) -> str:
    by_id = {row["run_id"]: row for row in rows}
    old = by_id["20260824-211918"]
    latest = by_id["20260826-173048"]
    throughput_gain = latest["output_tok_s_mean"] / old["output_tok_s_mean"]
    logprob_reduction = 1 - latest["logprob_diff_mean"] / old["logprob_diff_mean"]
    grad_reduction = 1 - latest["grad_norm_mean"] / old["grad_norm_mean"]
    return f"""# VIME / Polar 历史训练曲线分析（2026-08-27）

## 数据范围

- 扫描 `/mnt/pipeline-data/train_log/train_qwen36_polar_*.log`。
- 只纳入真正出现 `model.py ... train/loss` 的 8 份日志，共 40 个训练 step。
- `training_metrics.csv` 另保留 2 个完成 rollout、但未进入训练的轮次，所以有 42 行。
- 图中 `reward` 是日志记录的 Polar 标量奖励；不同奖励代码版本之间不能直接当成同一量尺。

## 能直接从日志确认的结论

1. 最新正式运行 `20260826-173048` 的 reward 为 `0.556 → 0.478 → 0.558 → 0.608 → 0.644 → 0.576`。线性趋势略向上，但只有 6 步且波动明显，尚不足以证明已经稳定收敛。
2. 与最近一组可比的正常长轨迹运行 `20260824-211918` 相比：
   - 平均 reward：`{old['reward_mean']:.3f} → {latest['reward_mean']:.3f}`，基本持平；这不是明显的退化。
   - `train-rollout |logprob diff|`：`{old['logprob_diff_mean']:.3f} → {latest['logprob_diff_mean']:.3f}`，下降 {logprob_reduction:.1%}。
   - `grad_norm`：`{old['grad_norm_mean']:.2f} → {latest['grad_norm_mean']:.3f}`，下降 {grad_reduction:.1%}。旧运行 `clip_grad=1.0`，所以旧运行几乎每步都会被裁剪；最新 6 步均低于 1，训练稳定性明显改善。
   - staleness：`{old['staleness_mean']:.3f} → {latest['staleness_mean']:.3f}`。最新训练 batch 本身没有旧权重样本。
3. 最新运行的配置指标只记录到 24 张 rollout GPU，不是 28：混合布局里 56 节点有一段卡因 HCCL 同域排他不进入 rollout。与 `20260824-211918` 的 12 张相比，平均输出吞吐 `507.7 → 1106 tok/s`，提高 {throughput_gain:.2f} 倍。单轮 rollout 仍从 `2.00h → 2.67h`，主要因为训练组从 `4×8=32` 翻到 `8×8=64`，并且长尾 session 决定 barrier 时间；不能只拿墙钟时间判断扩卡没收益。
4. `20260825-214539` 和 `20260826-102857` 是 `2×2 / batch=4` 功能验证。日志每轮都有 `rollout/zero_std/count_0.2=2`，即两个 group 内样本全是 0.2，组内相对优势为零。这两条平坦曲线没有有效 RL 学习信号，不能用于判断模型学习效果。
5. 两次 MTP 运行是 `20260824-114318` 和 `20260826-130952`，同时都换成基础 checkpoint `Qwen3.6-35B-A3B`；正常运行使用 `Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16`。MTP 两次的 reward 约 0.19/0.21、平均响应仅约 4.8k/4.3k，且 logprob 差为 0.898/0.231。它们是异常采样口径，不能解释成训练把模型训坏；也不能把问题单独归因于 MTP，因为 checkpoint 同时变了。

## 哪些改动解释了曲线断点

### 直接改变奖励量尺

- Polar `84481b85`（2026-08-21 18:09）加入 process reward，默认启用，单条可在 outcome reward 上做最多约 ±0.10 的塑形。它发生在 `0820` 与 `0821` 两次正式运行之间，因此 `reward_mean 0.461 → 0.582` 不能全解释为模型学习。
- Polar `7441b966` 把 reasoning 从默认 mask 改为默认进入训练。它影响 loss、梯度覆盖 token 和训练耗时，不直接改变 raw reward。
- attempt credit、PG floor=0.05、截断惩罚、post-best 在第一份日志前已经存在；每份日志的 PG floor 都是 0.05。`2c05dc08` 只是把 post-best 开关从 attempt credit 中拆开、默认仍开，因此它不是后续曲线突变的新原因。

### 直接改善训推一致性

- VIME `4a6c5e51` 修复 Qwen3.5 block spec 共享对象污染，避免全注意力层被错误替换。
- VIME `bc79852b` 把 MindSpeed `repatch(args)` 提前到 model-parallel 初始化前，并使用完整 MindSpeed 参数。
- VIME `ca07ae09` 强制全注意力使用 MindSpeed 按 CP 配置选择的原生映射。
- VIME `d9b3a3b8` + Polar `b8901037` 建立 fail-closed 权重边界；最新日志实际出现 `paused=True drained=True inflight=0`，并设置 `rollout_max_off_policy_steps=0`，旧组被明确 drop。

这些改动与 `logprob diff 0.155 → 0.033` 的方向完全一致，并且代码确实修改了训练 forward 和权重边界。不过当前没有一次“只切换单个 commit、其余完全相同”的 A/B，因此不能把 79% 的改善精确分摊给某一个 commit。

### 只影响优化步幅或吞吐，不足以单独解释 grad_norm

- lr 从 2e-6 降到 1e-6：会减小参数更新，但 `grad_norm` 是 optimizer step 前的梯度统计，lr 不会直接把同一 batch 的 grad_norm 降低两个数量级。
- batch 从 32 增到 64：会降低估计噪声，但也不足以单独解释 `30.95 → 0.301`。
- Polar `94476486` 把 postrun worker 8→16：只减少收尾排队，不改变 reward/loss 公式。

## 当前判断

- 最新曲线最重要的正向信号不是 reward 短期涨幅，而是 `staleness=0`、`logprob diff≈0.033`、`grad_norm≈0.30` 三者同时稳定；这说明进入训练的数据和训练 forward 比旧运行健康得多。
- reward 六步还不足以下学习已经显著提升的结论。建议保持当前正常 checkpoint、关闭 MTP，至少继续到 20–30 个有效 step，再看同一批重复算子的 paired reward；不要把 smoke/MTP 运行混进学习曲线比较。
- 当前长时间运行的主要剩余风险是 scheduler 拥有 group 无界堆积造成后续不再 admission；这与学习曲线本身无关，使用本次单独提供的 owned-group backpressure patch 处理。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("/mnt/pipeline-data/train_log"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/pipeline-data/train_curve_analysis_20260827"))
    args = parser.parse_args()

    candidates = sorted(args.log_dir.glob("train_qwen36_polar_*.log"))
    runs = [parse_log(path) for path in candidates]
    runs = [run for run in runs if run["train"]]
    if not runs:
        raise SystemExit("No logs containing train/loss were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_rows(runs)
    configs = config_rows(runs)
    summaries = summary_rows(runs)
    write_csv(args.output_dir / "training_metrics.csv", metrics)
    write_csv(args.output_dir / "run_configs.csv", configs)
    write_csv(args.output_dir / "run_summary.csv", summaries)
    plot_rollout(runs, args.output_dir / "all_rollout_curves.png")
    plot_summary(summaries, args.output_dir / "run_comparison.png")
    plot_individual_runs(runs, args.output_dir / "per_run")
    plot_individual_grad_norm(runs, args.output_dir / "per_run_grad_norm")
    (args.output_dir / "run_summary.md").write_text(markdown_table(summaries), encoding="utf-8")
    (args.output_dir / "analysis_report.md").write_text(analysis_report(summaries), encoding="utf-8")
    (args.output_dir / "run_change_timeline.md").write_text(change_timeline_markdown(runs), encoding="utf-8")
    (args.output_dir / "parsed_runs.json").write_text(
        json.dumps(
            runs,
            ensure_ascii=False,
            indent=2,
            default=lambda value: sorted(value) if isinstance(value, set) else repr(value),
        ),
        encoding="utf-8",
    )
    print(f"Parsed {len(runs)} runs / {len(metrics)} aligned steps into {args.output_dir}")


if __name__ == "__main__":
    main()
