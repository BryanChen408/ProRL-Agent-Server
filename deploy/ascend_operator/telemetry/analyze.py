#!/usr/bin/env python3
"""Join gateway completion_metrics.jsonl + engine_metrics/*.jsonl → 块②/③报表。

现在就能跑(gateway 侧数据已有);engine 侧存在时自动补 TTFT/prefill/decode/prefix-cache。
纯标准库,无依赖。

    python3 analyze.py <run_dir> [--engine-dir <engine_metrics_dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict


def _dist(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "min": xs[0], "p50": q(.50), "p90": q(.90), "p99": q(.99),
            "max": xs[-1], "mean": round(st.mean(xs), 1)}


def _load_jsonl_dir(pattern):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        for line in open(f, errors="replace"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _load_gateway(run_dir):
    # gateway 事件:每 session 一个 completion_metrics.jsonl(每 completion 一行)
    return _load_jsonl_dir(os.path.join(run_dir, "**", "completion_metrics.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--engine-dir", default=None)
    args = ap.parse_args()

    gw = _load_gateway(args.run_dir)
    eng = _load_jsonl_dir(os.path.join(args.engine_dir or os.path.join(args.run_dir, "engine_metrics"), "*.jsonl"))
    eng_by_trace = {r.get("trace_id"): r for r in eng if r.get("trace_id")}

    print(f"gateway completions: {len(gw)} | engine rows: {len(eng)} "
          f"({'joined' if eng else 'engine side not deployed yet — gateway-only view'})")

    # ---- 块③:长度分布(gateway 口径 always;engine 口径若有)----
    print("\n### 长度分布(块③)###")
    print("  prefill(prompt_tokens):", _dist([r.get("prompt_tokens") for r in gw]))
    print("  decode(completion_tokens):", _dist([r.get("completion_tokens") for r in gw]))
    print("  training_sample(total_tokens):", _dist([r.get("total_tokens") for r in gw]))
    if eng:
        print("  [engine] num_prefill_tokens:", _dist([r.get("num_prefill_tokens") for r in eng]))
        print("  [engine] prefix_cache_hit_pct:", _dist([r.get("prefix_cache_hit_pct") for r in eng]))
        print("  [engine] ttft_ms:", _dist([r.get("ttft_ms") for r in eng]))
        print("  [engine] prefill_ms:", _dist([r.get("prefill_ms") for r in eng]))
        print("  [engine] decode_ms:", _dist([r.get("decode_ms") for r in eng]))
        print("  [engine] queue_ms:", _dist([r.get("queue_ms") for r in eng]))

    # ---- 块②:按 rollout_step 聚合 ----
    print("\n### 按 rollout_step(块②)###")
    by_step = defaultdict(list)
    for r in gw:
        by_step[r.get("rollout_step")].append(r)
    for step, rows in sorted(by_step.items(), key=lambda kv: str(kv[0]))[:20]:
        pre = _dist([r.get("prompt_tokens") for r in rows])
        dec = _dist([r.get("completion_tokens") for r in rows])
        print(f"  step={step}: reqs={len(rows)} "
              f"prefill_p50/p99={pre and (pre['p50'], pre['p99'])} "
              f"decode_p50/p99={dec and (dec['p50'], dec['p99'])}")

    # ---- 每 engine 均衡(engine_url;engine 侧有则用 engine_id)----
    print("\n### 每 engine 负载均衡 ###")
    key = "engine_id" if eng else "engine_url"
    src = eng if eng else gw
    cnt = Counter(r.get(key) for r in src)
    tok = defaultdict(int)
    for r in src:
        tok[r.get(key)] += int(r.get("completion_tokens") or r.get("num_generation_tokens") or 0)
    counts = [v for k, v in cnt.items() if k]
    cv = round(st.pstdev(counts) / st.mean(counts), 3) if len(counts) > 1 and st.mean(counts) else 0.0
    for k in sorted(cnt, key=lambda x: str(x)):
        print(f"  {key}={k}: requests={cnt[k]} gen_tokens={tok[k]}")
    print(f"  → 请求数 CV(越大越不均)= {cv}")

    # ---- goodput:finish_reason / abort ----
    print("\n### goodput 线索 ###")
    fr = Counter(r.get("finish_reason") for r in gw)
    print("  finish_reason:", dict(fr))
    if eng:
        ab = sum(1 for r in eng if r.get("aborted"))
        print(f"  engine aborted: {ab}/{len(eng)}")

    # ---- 块④:rollout 非推理占比(需先跑 build_spans.py)----
    spans = _load_jsonl_dir(os.path.join(args.run_dir, "telemetry_spans", "rollout_spans.jsonl"))
    if spans:
        print("\n### 块④:rollout 时间拆解(build_spans 输出)###")
        for k, label in (("infer_frac", "推理"), ("verify_frac", "环境验证"),
                         ("tool_frac", "工具"), ("wait_cpu_frac", "等待/CPU")):
            xs = [s[k] for s in spans if isinstance(s.get(k), (int, float))]
            print(f"  {label}占比: mean={round(st.mean(xs),3) if xs else None}  分布={_dist([100*x for x in xs])}")
        # 验证池 lease_wait(4 卡饱和)
        waits = [ph.get("lease_wait_s") for s in spans if isinstance(s.get("lease"), dict)
                 for ph in s["lease"].values() if isinstance(ph, dict) and ph.get("lease_wait_s") is not None]
        if waits:
            print("  验证池 lease_wait(s):", _dist(waits))

    # ---- T8:rollout_step goodput(ses.json:completed vs aborted)----
    ses = _load_jsonl_dir(os.path.join(args.run_dir, "rollout_results", "**", "ses_*.json"))
    if ses:
        print("\n### T8:rollout goodput(ses.json)###")
        stat = Counter(s.get("status") for s in ses)
        n = len(ses)
        comp = stat.get("COMPLETED", 0)
        print(f"  sessions={n}  status={dict(stat)}  goodput(completed/total)={round(comp/n,3) if n else None}")
        by_step = defaultdict(lambda: Counter())
        for s in ses:
            md = (s.get("trajectory", {}) or {}).get("metadata", {}) or {}
            by_step[md.get("rollout_step")][s.get("status")] += 1
        for step, c in sorted(by_step.items(), key=lambda kv: str(kv[0]))[:20]:
            tot = sum(c.values())
            print(f"    step={step}: {dict(c)} goodput={round(c.get('COMPLETED',0)/tot,3) if tot else None}")


if __name__ == "__main__":
    main()
