#!/usr/bin/env python3
"""一轮 rollout 系统负载分析报告(T1-T8)。

窗口 = 一批已完成 session 的墙钟跨度(默认取最新 run 的全部完成 session;--window-sessions N 取最近 N 个)。
产出:<run>/perf_report/REPORT.md + source_data/(各表 jsonl 切到本窗口)。

维度:①NPU 利用率时序+不均(T1) ②引擎负载时序(T2,counter 差分只用 scrape_ok 行)
     ③块④时间拆解(T5/T7) ④goodput/reward/engine 均衡(T8)+验证(T6)。

用法:
  python3 perf_report.py                       # 最新 run,全部完成 session
  python3 perf_report.py --run <run_dir> --window-sessions 64
  python3 perf_report.py --metrics-dir /mnt/share/polar_engine_metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import time
from collections import Counter, defaultdict


def _load(p):
    try:
        return [json.loads(x) for x in open(p, errors="replace")]
    except Exception:
        return []


def _dist(xs, f=lambda x: x):
    xs = sorted(f(x) for x in xs if isinstance(x, (int, float)))
    if not xs:
        return "无"
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return f"n={n} min={xs[0]:.1f} p50={q(.5):.1f} p90={q(.9):.1f} max={xs[-1]:.1f} mean={st.mean(xs):.1f}"


def _t2_valid(rows):
    """只保留 scrape 成功的行:显式 scrape_ok=False 的丢;旧数据无该字段时按 num_requests_running 是否为 None 兜底。"""
    out = []
    for r in rows:
        if r.get("scrape_ok") is False:
            continue
        if "scrape_ok" not in r and r.get("vllm:num_requests_running") is None:
            continue
        out.append(r)
    return out


def build(run_dir, metrics_dir, window_n):
    der = os.path.join(run_dir, "telemetry_derived")
    out = os.path.join(run_dir, "perf_report")
    os.makedirs(out, exist_ok=True)

    # 窗口:完成 session 的时间跨度
    ses = []
    for f in glob.glob(f"{run_dir}/rollout_results/**/ses_*.json", recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        md = (s.get("trajectory", {}) or {}).get("metadata", {}) or {}
        tm = md.get("task_metadata") or {}
        run_ms = (s.get("timing", {}) or {}).get("run_ms") or 0
        end = os.path.getmtime(f)
        ses.append(dict(op=tm.get("op_name"), status=s.get("status"),
                        reward=(md.get("evaluation", {}) or {}).get("reward"),
                        run_ms=run_ms, start=end - run_ms / 1000.0, end=end))
    if not ses:
        raise SystemExit("无完成 session,无法出报告")
    ses.sort(key=lambda s: s["end"])
    if window_n:
        ses = ses[-window_n:]
    W0, W1 = min(s["start"] for s in ses), max(s["end"] for s in ses)
    inwin = lambda rows: [r for r in rows if W0 <= r.get("recorded_at_unix", 0) <= W1]

    R = ["# 一轮 rollout 系统负载分析报告\n",
         f"- **run**: {os.path.basename(run_dir.rstrip('/'))}",
         f"- **窗口**: {len(ses)} 个完成 session,墙钟 **{(W1 - W0) / 3600:.2f}h**"
         f"（{time.strftime('%m-%d %H:%M', time.localtime(W0))} → {time.strftime('%H:%M', time.localtime(W1))}）",
         f"- **数据源**: 同目录 source_data/（各表 jsonl 已切到本窗口）\n"]

    # 0 概览
    stc = Counter(s["status"] for s in ses)
    comp = stc.get("COMPLETED", 0)
    R += ["## 0. 概览",
          f"- 结局: {dict(stc)} | goodput={comp / len(ses):.2f}",
          f"- reward: {_dist([s['reward'] for s in ses])}",
          f"- 单 session 时长(min): {_dist([s['run_ms'] for s in ses], lambda x: x / 60000)}",
          f"- 覆盖算子: {len(set(s['op'] for s in ses))} 个\n"]

    # 1 T1 NPU
    R.append("## 1. NPU 利用率时序 + 不均（T1）")
    t1 = inwin(_load(f"{metrics_dir}/npu_state/npu_card.jsonl"))
    NB = 15 * 60
    buckets = defaultdict(lambda: defaultdict(list))
    for r in t1:
        if r.get("aicore_util_pct") is not None:
            buckets[int((r["recorded_at_unix"] - W0) // NB)][r["engine_id"]].append(r["aicore_util_pct"])
    engs = ["infer-0", "infer-1", "infer-2", "verify-pool"]
    R.append("每 15min aicore 均值(%):\n")
    R.append("| t(min) | " + " | ".join(engs) + " |")
    R.append("|" + "---|" * (len(engs) + 1))
    for b in sorted(buckets):
        row = [f"{b * 15}"] + [f"{st.mean(v):.0f}" if (v := buckets[b].get(e)) else "—" for e in engs]
        R.append("| " + " | ".join(row) + " |")
    emean = {e: st.mean([x for bb in buckets.values() for x in bb.get(e, [])] or [0]) for e in engs}
    im = [emean[e] for e in engs[:3]]
    R += [f"\n- 窗口均值: " + " ".join(f"{e}={emean[e]:.0f}%" for e in engs),
          f"- **engine 间不均 CV** = {st.pstdev(im) / st.mean(im):.3f}（推理三 engine）",
          f"- 推理卡 HBM(MB): {_dist([r.get('hbm_used_mb') for r in t1 if r['pool'] == 'inference' and r.get('hbm_used_mb')])}（满 65536）\n"]

    # 2 T2 引擎(只用 scrape_ok 行差分)
    R.append("## 2. 引擎负载时序（T2）")
    R.append("| engine | 区间gen_tok | 吞吐tok/s | 均TTFT | 均running | 均waiting | 均KV% | scrape失败行 |")
    R.append("|---|---|---|---|---|---|---|---|")
    for e in ["infer-0", "infer-1", "infer-2"]:
        allrows = inwin(_load(f"{metrics_dir}/vllm_state/{e}.jsonl"))
        rows = _t2_valid(allrows)
        nfail = len(allrows) - len(rows)
        if len(rows) < 2:
            R.append(f"| {e} | 数据不足 | | | | | | {nfail} |")
            continue
        dt = rows[-1]["recorded_at_unix"] - rows[0]["recorded_at_unix"]
        dg = rows[-1].get("vllm:generation_tokens_total", 0) - rows[0].get("vllm:generation_tokens_total", 0)
        dc = rows[-1].get("vllm:time_to_first_token_seconds_count", 0) - rows[0].get("vllm:time_to_first_token_seconds_count", 0)
        dss = rows[-1].get("vllm:time_to_first_token_seconds_sum", 0) - rows[0].get("vllm:time_to_first_token_seconds_sum", 0)
        run = st.mean([r.get("vllm:num_requests_running", 0) for r in rows])
        wait = st.mean([r.get("vllm:num_requests_waiting", 0) for r in rows])
        kv = st.mean([r.get("vllm:kv_cache_usage_perc", 0) * 100 for r in rows])
        R.append(f"| {e} | {int(dg)} | {dg / dt:.1f} | {dss / dc:.2f}s | {run:.1f} | {wait:.1f} | {kv:.1f}% | {nfail} |")
    R.append("")

    # 3 块④
    R.append("## 3. 块④ 时间拆解（T5/T7）")
    sp = _load(f"{der}/rollout_span.jsonl")
    if sp:
        for k, lab in [("infer_frac", "推理"), ("verify_frac", "验证"), ("tool_frac", "工具"), ("wait_cpu_frac", "等待/CPU")]:
            R.append(f"- {lab}占比: {_dist([s.get(k) for s in sp], lambda x: x * 100)} %")
        R.append(f"\n→ 推理外占比 ≈ {100 - st.mean([s.get('infer_frac', 0) for s in sp]) * 100:.0f}%\n")
    else:
        R.append("- (T5 暂无)\n")

    # 4 T8/T6
    R.append("## 4. goodput / reward / engine 均衡（T8）")
    for r in _load(f"{der}/rollout_step.jsonl"):
        b = r.get("engine_balance") or {}
        R += [f"- step={r['rollout_step']}: sessions={r['num_sessions']} goodput={r['goodput']} reward_p50={(r.get('reward_dist') or {}).get('p50')}",
              f"  - engine 请求份额: {b.get('requests_per_engine')}  CV={b.get('request_cv')}",
              f"  - engine gen_token 份额: {b.get('gen_tokens_per_engine')}"]
    vj = _load(f"{der}/verify_job.jsonl")
    if vj:
        R.append(f"- 验证(T6): {len(vj)}次 success={sum(1 for v in vj if v.get('success'))} error={dict(Counter(v.get('error_type') for v in vj))}")

    open(f"{out}/REPORT.md", "w").write("\n".join(R))

    # 源数据切片
    sd = f"{out}/source_data"
    os.makedirs(sd, exist_ok=True)
    def cut(src, dst):
        with open(dst, "w") as f:
            for r in inwin(_load(src)):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cut(f"{metrics_dir}/npu_state/npu_card.jsonl", f"{sd}/T1_npu_card.jsonl")
    for e in ["infer-0", "infer-1", "infer-2"]:
        cut(f"{metrics_dir}/vllm_state/{e}.jsonl", f"{sd}/T2_{e}.jsonl")
    cut(f"{metrics_dir}/host_state/host_proc.jsonl", f"{sd}/T3_host_proc.jsonl")
    for t in ["rollout_span", "verify_job", "rollout", "rollout_step"]:
        p = f"{der}/{t}.jsonl"
        if os.path.exists(p):
            open(f"{sd}/{t}.jsonl", "w").write(open(p).read())
    return f"{out}/REPORT.md", sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run 目录;缺省取 --runs-root 下最新")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
                    "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR", "/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0, help="取最近 N 个完成 session 为窗口;0=全部")
    args = ap.parse_args()
    run = args.run or max(glob.glob(f"{args.runs_root}/*/"), key=os.path.getmtime)
    rep, sd = build(run, args.metrics_dir, args.window_sessions)
    print(f"报告: {rep}\n源数据: {sd}/")
    print("=" * 50)
    print(open(rep).read())


if __name__ == "__main__":
    main()
