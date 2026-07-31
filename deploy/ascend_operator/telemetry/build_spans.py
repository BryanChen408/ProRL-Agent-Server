#!/usr/bin/env python3
"""块④:从 agent 转录(每事件带 timestamp)重建 rollout 时间拆解 —— 无需改 agent。

每个 session 输出:
  - spans.jsonl        每段 span 一行(inference / tool:<name> / env_verify / wait_cpu)
  - rollout_spans.jsonl 每 session 汇总(inference_ms/tool_ms/verify_ms/wait_cpu_ms + infer_frac …)

推理段口径 = 上一条 user/tool_result → 下一条 assistant 的间隔(含排队+网络+生成,与 gateway latency 同口径);
工具段 = tool_use → 对应 tool_result 的间隔;跑 triton_eval_pipeline 的 Bash → env_verify;
wait_cpu = 墙钟 − 上面各段(= agent CPU + 各种等待)。
verify 段若旁边有 npu_lease_status.*.json(schema v2)则附 lease_wait/exec_seconds。

    python3 build_spans.py <run_dir> [--out <dir>]   # 默认写到 <run_dir>/telemetry_spans/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime


def _ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _iter_events(jsonl_path):
    for line in open(jsonl_path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = _ts(o.get("timestamp"))
        if t is not None:
            yield t, o


def _content_blocks(o):
    msg = o.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, list):
            return c
    return []


def _lease_refine(session_dir):
    """收集该 session 的 npu_lease_status.*.json → {phase: {wait_seconds, exec_seconds, card}}。"""
    out = {}
    for f in glob.glob(os.path.join(session_dir, "**", "npu_lease_status.*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        phase = os.path.basename(f).split(".")[1] if "." in os.path.basename(f) else "?"
        out[phase] = {"lease_wait_s": d.get("wait_seconds"),
                      "exec_s": d.get("exec_seconds"), "card": d.get("device_id")}
    return out


def process_session(transcript, session_id, session_dir):
    events = sorted(_iter_events(transcript), key=lambda x: x[0])
    if len(events) < 2:
        return None, []
    spans = []
    verify_tool_ids = set()   # tool_use_id 属于 triton_eval_pipeline
    tool_name_of = {}         # tool_use_id -> name
    wall_start, wall_end = events[0][0], events[-1][0]
    sid = 0

    def add(kind, a, b, detail=None):
        nonlocal sid
        if a is None or b is None or b <= a:
            return
        sid += 1
        spans.append({"session_id": session_id, "span_id": sid, "type": kind,
                      "start_ts": round(a, 3), "end_ts": round(b, 3),
                      "dur_ms": round((b - a) * 1000.0, 1), "detail": detail})

    # 预扫:登记每个 tool_use 的名字/是否 verify(供 tool_result 分类)
    for _t, o in events:
        for b in _content_blocks(o):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                tool_name_of[b.get("id")] = b.get("name", "?")
                if "triton_eval_pipeline" in json.dumps(b.get("input", {}))[:400]:
                    verify_tool_ids.add(b.get("id"))

    # 逐相邻 gap 归因(互不重叠、和=墙钟 → 占比必 ≤1)
    prev_t = events[0][0]
    for t, o in events[1:]:
        typ = o.get("type")
        blocks = _content_blocks(o)
        result_ids = [b.get("tool_use_id") for b in blocks
                      if isinstance(b, dict) and b.get("type") == "tool_result"]
        if result_ids:
            tu = result_ids[0]
            if tu in verify_tool_ids:
                add("env_verify", prev_t, t, "triton_eval_pipeline")
            else:
                add(f"tool:{tool_name_of.get(tu, '?')}", prev_t, t, tool_name_of.get(tu))
        elif typ == "assistant":
            add("inference", prev_t, t)
        else:
            add("wait_cpu", prev_t, t)
        prev_t = t

    lease = _lease_refine(session_dir)
    # 汇总(spans 由相邻 gap 构成,互不重叠、和≈墙钟)
    agg = {"inference": 0.0, "verify": 0.0, "tool": 0.0, "wait_cpu": 0.0}
    n_turns = n_tools = 0
    for s in spans:
        if s["type"] == "inference":
            agg["inference"] += s["dur_ms"]; n_turns += 1
        elif s["type"] == "env_verify":
            agg["verify"] += s["dur_ms"]
        elif s["type"].startswith("tool:"):
            agg["tool"] += s["dur_ms"]; n_tools += 1
        elif s["type"] == "wait_cpu":
            agg["wait_cpu"] += s["dur_ms"]
    wall_ms = (wall_end - wall_start) * 1000.0
    wait_cpu = agg["wait_cpu"]
    frac = lambda x: round(x / wall_ms, 4) if wall_ms > 0 else None
    summary = {
        "session_id": session_id, "wall_ms": round(wall_ms, 1),
        "inference_ms": round(agg["inference"], 1), "verify_ms": round(agg["verify"], 1),
        "tool_ms": round(agg["tool"], 1), "wait_cpu_ms": round(wait_cpu, 1),
        "n_turns": n_turns, "n_tool_calls": n_tools,
        "infer_frac": frac(agg["inference"]), "verify_frac": frac(agg["verify"]),
        "tool_frac": frac(agg["tool"]), "wait_cpu_frac": frac(wait_cpu),
        "lease": lease or None,
    }
    return summary, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run_dir, "telemetry_spans")
    os.makedirs(out_dir, exist_ok=True)

    session_dirs = glob.glob(os.path.join(args.run_dir, "polar_sessions", "**", "session-*"), recursive=True)
    session_dirs = [d for d in session_dirs if os.path.isdir(d)]
    n_ok = 0
    with open(os.path.join(out_dir, "spans.jsonl"), "w") as fs, \
         open(os.path.join(out_dir, "rollout_spans.jsonl"), "w") as fr:
        for sd in session_dirs:
            session_id = os.path.basename(sd).replace("session-", "")
            transcripts = glob.glob(os.path.join(sd, ".claude", "**", "*.jsonl"), recursive=True)
            if not transcripts:
                continue
            transcript = max(transcripts, key=lambda p: os.path.getsize(p))
            summary, spans = process_session(transcript, session_id, sd)
            if not summary:
                continue
            n_ok += 1
            fr.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for s in spans:
                fs.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"sessions with spans: {n_ok}/{len(session_dirs)} → {out_dir}/{{spans,rollout_spans}}.jsonl")


if __name__ == "__main__":
    main()
