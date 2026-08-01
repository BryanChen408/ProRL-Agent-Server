#!/usr/bin/env python3
"""采集 T2 engine_state:抓每个引擎的 vllm 原生 /metrics → jsonl(与 npu_smi/engine_metrics 同工作流)。

这是原设计 Plane A 的 T2:引擎内部态与时序,来源 = vllm /metrics(V1 也有,每 engine 一份)。
TTFT/TPOT/吞吐/KV/队列/抢占/prefix 都在这里拿——不靠 per-request hook,不用重启。

用法:
  python3 vllm_metrics_poller.py --engines 80.5.25.119:15000 80.5.25.119:15002 80.5.25.119:15004 \
      --out /mnt/share/polar_engine_metrics/vllm_state --interval 5
  # 缺省 --engines:读 card_topology.yaml 的 engine_endpoints
每 interval 每引擎写一行:counter 原值 + histogram 的 _sum/_count(相邻两行差分即区间 avg TTFT/TPOT)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

# 要抓的 vllm 指标(与原设计 T2 字段对齐)。histogram 取 _sum/_count(差分算 avg);counter/gauge 取值。
_COUNTERS = ["vllm:prompt_tokens_total", "vllm:generation_tokens_total",
             "vllm:num_preemptions_total", "vllm:num_preemptions",
             "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
             "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
             "vllm:request_success_total"]
_GAUGES = ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
           "vllm:num_requests_running", "vllm:num_requests_waiting"]
_HISTS = ["vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
          "vllm:request_time_per_output_token_seconds", "vllm:e2e_request_latency_seconds"]

_LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)$')


def _load_endpoints(topo_path: Path) -> dict[str, str]:
    try:
        import yaml  # type: ignore
        d = yaml.safe_load(topo_path.read_text()) or {}
        ep = d.get("engine_endpoints") or {}
        return {k: v for k, v in ep.items()}
    except Exception:
        pass
    out, section = {}, None
    for line in topo_path.read_text().splitlines():
        s = line.strip()
        if s.endswith(":") and not s.startswith("-"):
            section = s[:-1].strip()
        elif section == "engine_endpoints" and ":" in s and not s.startswith("#"):
            k, _, v = s.partition(":")
            out[k.strip()] = v.strip()
    return out


def _scrape(url: str) -> dict[str, float]:
    """拉一次 /metrics,解析成 {metric_name(+bucketless): value}。histogram 累加各 label 的 _sum/_count。"""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return {"__error__": str(e)[:80]}
    agg: dict[str, float] = {}
    want_hist_sfx = tuple(h + s for h in _HISTS for s in ("_sum", "_count"))
    want_flat = set(_COUNTERS + _GAUGES)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, _lbl, val = m.group(1), m.group(2), m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if name in want_flat or name.endswith(want_hist_sfx):
            agg[name] = agg.get(name, 0.0) + v  # 跨 label(多 rank/多 finish_reason)求和
    return agg


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--engines", nargs="*", help="host:port 列表;缺省读 topology engine_endpoints")
    ap.add_argument("--topology", type=Path, default=here / "card_topology.yaml")
    ap.add_argument("--out", default=os.environ.get("POLAR_ENGINE_METRICS_DIR", "/mnt/share/polar_engine_metrics") + "/vllm_state")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if args.engines:
        engines = {f"engine-{hp.rsplit(':', 1)[-1]}": ("http://" + hp) for hp in args.engines}
    else:
        engines = {name: (url if url.startswith("http") else "http://" + url)
                   for name, url in _load_endpoints(args.topology).items()}
    if not engines:
        raise SystemExit("no engines: pass --engines host:port ... or fill topology engine_endpoints")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"vllm_metrics_poller: {list(engines)} → {out}/*.jsonl (interval {args.interval}s)")

    while True:
        ts = time.time()
        for eid, base in engines.items():
            row = {"recorded_at_unix": round(ts, 3), "engine_id": eid,
                   "endpoint": base, **_scrape(base.rstrip("/") + "/metrics")}
            with open(out / f"{eid}.jsonl", "a", buffering=1) as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.once:
            print("wrote one snapshot per engine"); return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
