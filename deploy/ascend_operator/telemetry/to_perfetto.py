#!/usr/bin/env python3
"""单 rollout 的时间瀑布 → Chrome Trace / Perfetto JSON(块④深挖可视化)。

  python3 to_perfetto.py <spans.jsonl> --session <session_id> -o trace.json
  # 打开 https://ui.perfetto.dev → Open trace file → 选 trace.json

每段 span 一个 "X"(complete)事件,按类型分轨(inference / env_verify / tool / wait_cpu),
时长即卡在验证/工具上的墙钟一目了然。
"""
from __future__ import annotations

import argparse
import json


TRACKS = {"inference": 1, "env_verify": 2, "wait_cpu": 4}


def track_of(span_type: str) -> tuple[int, str]:
    if span_type.startswith("tool:"):
        return 3, "tool"
    return TRACKS.get(span_type, 5), span_type.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spans")
    ap.add_argument("--session", required=True)
    ap.add_argument("-o", "--out", default="trace.json")
    args = ap.parse_args()

    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    spans = [s for s in spans if s.get("session_id") == args.session]
    if not spans:
        raise SystemExit(f"no spans for session {args.session}")

    t0 = min(s["start_ts"] for s in spans)
    events = [{
        "name": "session", "ph": "M", "pid": 1, "tid": 0,
        "args": {"name": "process_name", "session": args.session},
    }]
    for tid, label in {0: "rollout", 1: "inference", 2: "env_verify", 3: "tool", 4: "wait_cpu", 5: "other"}.items():
        events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
                       "args": {"name": label}})
    for s in spans:
        tid, cat = track_of(s["type"])
        events.append({
            "name": s["type"], "cat": cat, "ph": "X", "pid": 1, "tid": tid,
            "ts": round((s["start_ts"] - t0) * 1e6, 1),   # 微秒
            "dur": round(s["dur_ms"] * 1e3, 1),
            "args": {"detail": s.get("detail"), "dur_s": round(s["dur_ms"] / 1000.0, 2)},
        })
    json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, open(args.out, "w"))
    tot = sum(s["dur_ms"] for s in spans) / 1000.0
    print(f"{len(spans)} spans, {tot:.0f}s → {args.out}  (open in ui.perfetto.dev)")


if __name__ == "__main__":
    main()
