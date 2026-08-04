#!/usr/bin/env python3
"""专家级系统负载 + 瓶颈分析报告(T1-T8),带 ASCII 图 / 图例 / 数据溯源。

设计原则:每个结论都走三段式 —— 【数据来源】哪张表哪个字段 →【如何计算】→【结论依据】为什么。
图:sparkline 时序 / 直方图 / 水平条,纯文本零依赖,Markdown 与终端都可读。

用法: python3 perf_report_detailed.py [--run <dir>] [--window-sessions N] [--out-suffix _expert]
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, time
from collections import Counter, defaultdict
try:
    import perf_charts as CH
except Exception:
    CH = None
_PNG = bool(CH and CH.available())
_OUT = {"dir": None}  # 报告输出目录(charts 存这里)

# ============ ASCII 图元件(PNG 不可用时降级) ============
_SPARK = "▁▂▃▄▅▆▇█"
def spark(vals):
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs: return ""
    lo, hi = min(xs), max(xs)
    if hi == lo: return _SPARK[3] * len(xs)
    return "".join(_SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in xs)

def hbar(frac, width=30):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n) + f" {frac*100:.1f}%"

def histogram(vals, bins=10, width=40, unit=""):
    xs = sorted(v for v in vals if isinstance(v, (int, float)))
    if not xs: return ["  (无数据)"]
    lo, hi = xs[0], xs[-1]
    if hi == lo: hi = lo + 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in xs:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    mx = max(counts) or 1
    out = []
    for i, c in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        bar = "█" * int(round(c / mx * width))
        out.append(f"  [{a:>10.0f},{b:>10.0f}){unit:<4} |{bar:<{width}} {c}")
    return out

# ============ 数据/统计 helpers ============
def _load(p):
    try: return [json.loads(x) for x in open(p, errors="replace") if x.strip()]
    except Exception: return []

def _q(xs, p):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

def _stats(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs: return None
    n = len(xs); f = lambda p: xs[min(n - 1, int(p * n))]
    return dict(n=n, min=xs[0], p50=f(.5), p90=f(.9), p99=f(.99), max=xs[-1], mean=st.mean(xs))

def _fmt(s, sc=1.0, u=""):
    if not s: return "无数据"
    return (f"n={s['n']} · min={s['min']*sc:.1f} · p50={s['p50']*sc:.1f} · p90={s['p90']*sc:.1f} · "
            f"p99={s['p99']*sc:.1f} · max={s['max']*sc:.1f} · mean={s['mean']*sc:.1f}{u}")

def _cv(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.pstdev(xs) / st.mean(xs), 3) if len(xs) > 1 and st.mean(xs) else 0.0

def _valid_t2(rows):
    return [r for r in rows if r.get("scrape_ok") is not False and
            not ("scrape_ok" not in r and r.get("vllm:num_requests_running") is None)]

def _bucketize(rows, key, W0, W1, nb):
    step = (W1 - W0) / nb if nb else 1
    b = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None:
            b[int((r["recorded_at_unix"] - W0) // step) if step else 0].append(v)
    return [(round(i * step / 60), st.mean(b[i])) for i in sorted(b)], step

# ============ 报告主体 ============
def build(run_dir, metrics_dir, window_n, out_suffix):
    der = os.path.join(run_dir, "telemetry_derived")
    out = os.path.join(run_dir, "perf_report" + (out_suffix or "_expert"))
    os.makedirs(out, exist_ok=True)
    _OUT["dir"] = out
    R = []
    def P(*a): R.append(" ".join(str(x) for x in a))

    # ---- 窗口 ----
    ses = []
    for f in glob.glob(f"{run_dir}/rollout_results/**/ses_*.json", recursive=True):
        try: s = json.load(open(f))
        except Exception: continue
        md = (s.get("trajectory", {}) or {}).get("metadata", {}) or {}
        tm = md.get("task_metadata") or {}; t = s.get("timing", {}) or {}
        rm = t.get("run_ms") or 0; end = os.path.getmtime(f)
        ses.append(dict(op=tm.get("op_name"), status=s.get("status"),
                        reward=(md.get("evaluation", {}) or {}).get("reward"),
                        run_ms=rm, init_ms=t.get("init_ms"), postrun_ms=t.get("postrun_ms"),
                        step=md.get("rollout_step"), trace=md.get("trace_count"),
                        start=end - rm / 1000.0, end=end))
    if not ses: raise SystemExit("无完成 session")
    ses.sort(key=lambda s: s["end"])
    if window_n: ses = ses[-window_n:]
    W0, W1 = min(s["start"] for s in ses), max(s["end"] for s in ses)
    span_h = (W1 - W0) / 3600
    inwin = lambda rows: [r for r in rows if W0 <= r.get("recorded_at_unix", 0) <= W1]
    NB = max(6, min(24, int(span_h * 4)))

    t1 = inwin(_load(f"{metrics_dir}/npu_state/npu_card.jsonl"))
    t2 = {e: _valid_t2(inwin(_load(f"{metrics_dir}/vllm_state/{e}.jsonl"))) for e in ("infer-0","infer-1","infer-2")}
    t3 = inwin(_load(f"{metrics_dir}/host_state/host_proc.jsonl"))
    t4 = [r for f in glob.glob(f"{metrics_dir}/engine-*.jsonl") for r in _load(f)]
    t5 = _load(f"{der}/rollout_span.jsonl"); t6 = _load(f"{der}/verify_job.jsonl")
    t7 = _load(f"{der}/rollout.jsonl"); t8 = _load(f"{der}/rollout_step.jsonl")

    # ---- 抬头 + 方法论 ----
    P(f"# 系统负载与瓶颈分析报告 — {os.path.basename(run_dir.rstrip('/'))}\n")
    P(f"> 生成于 {time.strftime('%Y-%m-%d %H:%M')} · 窗口 **{len(ses)} 个完成 session** · "
      f"墙钟 **{span_h:.2f} h** ({time.strftime('%m-%d %H:%M', time.localtime(W0))} → {time.strftime('%H:%M', time.localtime(W1))})\n")
    P("## 0. 阅读方法 · 数据溯源")
    P("本报告每个结论遵循三段式:**【来源】**采集表+字段 → **【计算】**如何得出 → **【依据】**为何能下此结论。\n")
    P("| 采集层 | 表 | 采样 | 关键字段 | 落盘 |")
    P("|---|---|---|---|---|")
    P("| T1 NPU 卡 | npu_card | 5s/卡 | aicore_util_pct, hbm_used_mb, power_w | npu_state/npu_card.jsonl |")
    P("| T2 引擎态 | vllm /metrics | 5s/引擎 | kv_cache_usage, num_running/waiting, ttft/itl histogram, *_tokens_total | vllm_state/infer-*.jsonl |")
    P("| T3 主机 | /proc | 5s | mem_avail, cpu_pct, load, nfs, net | host_state/host_proc.jsonl |")
    P("| T4 逐请求 | vllm 内钩子 | 每请求 | num_prompt/generation_tokens, prefix_cache_hit_pct | engine-*.jsonl |")
    P("| T5 时间拆解 | 转录重建 | 每 session | infer/verify/tool/wait_frac | telemetry_derived/rollout_span.jsonl |")
    P("| T6 验证作业 | metrics.json | 每验证 | success, error_type, speedup | telemetry_derived/verify_job.jsonl |")
    P("| T7 rollout | ses.json | 每 session | status, reward, timing | telemetry_derived/rollout.jsonl |")
    P("| T8 step 聚合 | 派生 | 每 step | goodput, engine_balance(CV) | telemetry_derived/rollout_step.jsonl |")
    P(f"\n图表: {'PNG(charts/,matplotlib 生成)' if _PNG else 'ASCII 降级(未装 matplotlib)'}。图内标签为英文,正文中文对照。\n")
    P("拓扑:16 chip = **12 推理**(3 engine × TP4, infer-0/1/2)+ **4 验证**(npu_lease 池)。数据源见上表,均已切到本窗口。\n")
    P("---\n")

    _section_engine(P, t1, t2, W0, W1, NB, span_h)
    _section_length(P, t4)
    _section_agent(P, t5, t6, t7, ses)
    _section_multigrain(P, t8, ses)
    _section_host(P, t3)
    _section_summary(P, t1, t2, t5, ses, t6)

    open(f"{out}/REPORT.md", "w").write("\n".join(R))
    sd = f"{out}/source_data"; os.makedirs(sd, exist_ok=True)
    def cut(src, dst):
        with open(dst, "w") as f:
            for r in inwin(_load(src)): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cut(f"{metrics_dir}/npu_state/npu_card.jsonl", f"{sd}/T1_npu_card.jsonl")
    for e in ("infer-0","infer-1","infer-2"): cut(f"{metrics_dir}/vllm_state/{e}.jsonl", f"{sd}/T2_{e}.jsonl")
    cut(f"{metrics_dir}/host_state/host_proc.jsonl", f"{sd}/T3_host_proc.jsonl")
    for t in ("rollout_span","verify_job","rollout","rollout_step"):
        if os.path.exists(f"{der}/{t}.jsonl"): open(f"{sd}/{t}.jsonl","w").write(open(f"{der}/{t}.jsonl").read())
    return f"{out}/REPORT.md", sd

# ===== 1. 引擎/NPU 性能瓶颈 =====
def _section_engine(P, t1, t2, W0, W1, NB, span_h):
    P("## 1. 推理性能瓶颈(引擎 + NPU 算力)\n")
    # 1.1 引擎饱和度总表
    P("### 1.1 引擎延迟 / 吞吐 / 饱和(T2,每引擎)")
    P("**【来源】** `vllm_state/infer-{0,1,2}.jsonl`,vllm 原生 /metrics,5s/次。")
    P("**【计算】** 吞吐=Δgeneration_tokens_total/Δt;TTFT/ITL=Δ(_sum)/Δ(_count);KV/running/waiting=窗口均值;抢占=Δnum_preemptions_total。\n")
    P("| engine | 吞吐 tok/s | TTFT均值 | ITL ms/tok | KV均值% | KV峰值% | running均 | waiting均 | waiting峰 | 抢占次数 | prefix命中% |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {}
    for e, rows in t2.items():
        if len(rows) < 2: P(f"| {e} | 采样不足 | | | | | | | | | |"); continue
        dt = rows[-1]["recorded_at_unix"] - rows[0]["recorded_at_unix"]
        dg = rows[-1].get("vllm:generation_tokens_total",0)-rows[0].get("vllm:generation_tokens_total",0)
        ds = rows[-1].get("vllm:time_to_first_token_seconds_sum",0)-rows[0].get("vllm:time_to_first_token_seconds_sum",0)
        dc = rows[-1].get("vllm:time_to_first_token_seconds_count",0)-rows[0].get("vllm:time_to_first_token_seconds_count",0)
        dis = rows[-1].get("vllm:inter_token_latency_seconds_sum",0)-rows[0].get("vllm:inter_token_latency_seconds_sum",0)
        dic = rows[-1].get("vllm:inter_token_latency_seconds_count",0)-rows[0].get("vllm:inter_token_latency_seconds_count",0)
        kv = [r.get("vllm:kv_cache_usage_perc",0)*100 for r in rows]
        run = [r.get("vllm:num_requests_running",0) for r in rows]
        wt = [r.get("vllm:num_requests_waiting",0) for r in rows]
        pre = rows[-1].get("vllm:num_preemptions_total",0)-rows[0].get("vllm:num_preemptions_total",0)
        ph = rows[-1].get("vllm:prefix_cache_hits_total",0)-rows[0].get("vllm:prefix_cache_hits_total",0)
        pq = rows[-1].get("vllm:prefix_cache_queries_total",0)-rows[0].get("vllm:prefix_cache_queries_total",0)
        agg[e] = dict(tps=dg/dt if dt else 0, kv=st.mean(kv), run=st.mean(run), wait=st.mean(wt))
        P(f"| {e} | {dg/dt if dt else 0:.1f} | {ds/dc if dc else 0:.2f}s | {dis/dic*1000 if dic else 0:.1f} | "
          f"{st.mean(kv):.1f} | {max(kv):.0f} | {st.mean(run):.1f} | {st.mean(wt):.2f} | {max(wt):.0f} | {int(pre)} | {100*ph/pq if pq else 0:.1f} |")
    # 1.2 KV 时序图(饱和度可视化)
    P("\n### 1.2 KV cache 占用时序(饱和度)")
    P("**【来源】** T2 `vllm:kv_cache_usage_perc`。**【计算】** 每 ~%dmin 分桶取均值,画时序。" % round((W1-W0)/NB/60))
    kvser = {}
    for e, rows in t2.items():
        series, _ = _bucketize(rows, "vllm:kv_cache_usage_perc", W0, W1, NB)
        kvser[e] = [(t, v*100) for t, v in series]
    png = CH.lines_timeseries(_OUT["dir"], "kv_timeseries", kvser, "time (min)", "KV cache %",
                              "KV cache usage over time (per engine)") if _PNG else None
    if png: P(f"\n![KV时序]({png})\n")
    else:
        for e, s in kvser.items():
            vals=[v for _,v in s]; P(f"  {e} KV% `{spark(vals)}` {min(vals):.0f}→{max(vals):.0f}%" if vals else f"  {e} 无")
    # 1.3 判定
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    allwt = [r.get("vllm:num_requests_waiting",0) for rows in t2.values() for r in rows]
    allrun = [r.get("vllm:num_requests_running",0) for rows in t2.values() for r in rows]
    sat = st.mean(allkv) > 70 or st.mean(allwt) > 2
    P(f"\n**【依据 → 判定】** KV 均值 {st.mean(allkv):.1f}%(峰 {max(allkv):.0f}%),waiting 均值 {st.mean(allwt):.2f},running 均值 {st.mean(allrun):.1f}。")
    P(f"  - KV 是显存里放 KV cache 的占比;waiting 是排队请求数。**两者都低 ⇒ 引擎没被喂满**。")
    P(f"  - 结论:{'⚠️ 引擎接近饱和' if sat else '✅ **推理引擎远未吃满,推理算力不是瓶颈**(KV 仅 ~%.0f%%、几乎不排队)。RL rollout 并发还能大幅提高。' % st.mean(allkv)}")
    # 1.4 引擎均衡
    tps = {e: a["tps"] for e, a in agg.items()}
    if len(tps) > 1:
        cv = _cv(list(tps.values()))
        P(f"\n### 1.3 引擎间负载均衡")
        P(f"**【来源】** 上表吞吐列。**【计算】** CV=标准差/均值。**【依据】** 三引擎吞吐 {[f'{e}:{v:.0f}' for e,v in tps.items()]},**CV={cv}**。")
        P(f"  - {'⚠️ 不均(CV>0.1):某引擎被喂得少,LB/DP 分发不平。' if cv>0.1 else '✅ 均衡(CV<0.1)。'}")
    # 1.5 NPU 时序
    P("\n### 1.4 NPU aicore 利用率时序 + 池间/引擎间不均(T1)")
    P("**【来源】** `npu_card.jsonl` `aicore_util_pct`,16 chip 每 5s。**【计算】** 按 engine_id 聚合,每 ~%dmin 均值,画时序。" % round((W1-W0)/NB/60))
    aiser = {}
    for eng in ["infer-0","infer-1","infer-2","verify-pool"]:
        rows = [r for r in t1 if r.get("engine_id")==eng]
        series, _ = _bucketize(rows, "aicore_util_pct", W0, W1, NB)
        aiser[eng] = series
    png = CH.lines_timeseries(_OUT["dir"], "npu_aicore_timeseries", aiser, "time (min)", "aicore %",
                              "NPU aicore utilization over time (per pool/engine)") if _PNG else None
    if png: P(f"\n![NPU时序]({png})\n")
    else:
        for eng, s in aiser.items():
            vals=[v for _,v in s]
            if vals: P(f"  {eng:12s} `{spark(vals)}` 均{st.mean(vals):.0f}% 峰{max(vals):.0f}%")
    inf = [r.get("aicore_util_pct") for r in t1 if r["pool"]=="inference" and r.get("aicore_util_pct") is not None]
    ver = [r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    P(f"\n**【依据】** 推理池 aicore {_fmt(_stats(inf),1,'%')}")
    P(f"  - 验证池 aicore 均值 {st.mean(ver or [0]):.0f}%(峰 {max(ver or [0]):.0f}%)→ **4 卡验证池{'间歇打满、大部分时间空闲' if st.mean(ver or [0])<20 else '有持续负载'}**(算子编译+对拍才用卡)。")

# ===== 2. 长度瓶颈 =====
def _section_length(P, t4):
    P("\n---\n\n## 2. 长度瓶颈(prompt / decode / context)\n")
    P("**【来源】** `engine-*.jsonl`(T4 逐请求),字段 num_prompt_tokens / num_generation_tokens / prefix_cache_hit_pct。")
    pt = [r.get("num_prompt_tokens") for r in t4]; gt = [r.get("num_generation_tokens") for r in t4]
    ctx = [(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch = [r.get("prefix_cache_hit_pct") for r in t4]
    P(f"\n### 2.1 prompt 长度分布(prefill 压力)")
    P(f"**【计算】** 每请求 num_prompt_tokens 直方图。**【统计】** {_fmt(_stats(pt),1,' tok')}")
    png = CH.hist(_OUT["dir"], "prompt_len_hist", pt, "prompt tokens", "Prompt length distribution (T4 per-request)") if _PNG else None
    if png: P(f"\n![prompt长度]({png})\n")
    else:
        for l in histogram(pt, bins=8, unit="tok"): P(l)
    P(f"\n### 2.2 decode 长度分布(生成压力)")
    P(f"**【统计】** {_fmt(_stats(gt),1,' tok')}")
    png = CH.hist(_OUT["dir"], "decode_len_hist", gt, "generation tokens", "Decode length distribution (T4 per-request)") if _PNG else None
    if png: P(f"\n![decode长度]({png})\n")
    else:
        for l in histogram(gt, bins=8, unit="tok"): P(l)
    over = sum(1 for c in ctx if c > 240000)
    P(f"\n### 2.3 总 context 与上限压力")
    P(f"**【计算】** prompt+decode。**【统计】** {_fmt(_stats(ctx),1,' tok')} · 上限 262144。")
    png = CH.hist(_OUT["dir"], "context_hist", ctx, "total context tokens", "Total context vs 262144 limit", vline=262144, vline_label="262144 limit") if _PNG else None
    if png: P(f"\n![context分布]({png})\n")
    P(f"  - **逼近上限(>240k)请求: {over}/{len(ctx)}({100*over/max(1,len(ctx)):.1f}%)** {hbar(over/max(1,len(ctx)))}")
    P(f"  - **【依据】** {'⚠️ 有真实 context 触顶压力' if over/max(1,len(ctx))>0.02 else '✅ context 压力不大' }:prompt p50 已 {(_q(pt,.5) or 0):.0f} tok,长 prompt 是主体。")
    cs = _stats(ch)
    P(f"\n### 2.4 prefix cache 命中(prefill 复用)")
    P(f"**【统计】** {_fmt(cs,1,'%')} → 命中 {'高' if (cs and cs['mean']>80) else '一般'},长 prompt 的 prefill 大量走缓存,**这是引擎没被 prefill 压垮的原因**。")

# ===== 3. Agent 瓶颈 =====
def _section_agent(P, t5, t6, t7, ses):
    P("\n---\n\n## 3. Agent 瓶颈(时间拆解 / 轮数 / 验证 / 结局)\n")
    P("**【来源】** T5 rollout_span(转录重建时间段)+ T6 verify_job(metrics.json)+ T7/ses.json(结局)。")
    if t5:
        P("\n### 3.1 块④ rollout 时间拆解(一轮 rollout 里推理/验证/工具/等待占比)")
        P("**【来源】** T5 rollout_span 的 infer/verify/tool/wait_cpu_frac(转录相邻事件 gap 归因)。")
        P("**【计算】** 每 session 各段占墙钟比,再对 %d 个 session 取均值。" % len(t5))
        means = {}
        for k, cn in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
            vals = [s.get(k) for s in t5 if isinstance(s.get(k),(int,float))]
            means[cn] = st.mean(vals) if vals else 0
        png = CH.pie_time_breakdown(_OUT["dir"],
              {"Inference": means["推理"], "Verify": means["验证"], "Tool": means["工具"], "Wait/CPU": means["等待/CPU"]}
              ) if _PNG else None
        if png:
            P(f"\n![时间占比饼图]({png})\n")
            P("(饼图:一轮 rollout 墙钟时间构成。Inference=推理 · Verify=环境验证 · Tool=工具调用 · Wait/CPU=等待与CPU)")
        else:
            for cn, m in means.items(): P(f"  {cn:8s} {hbar(m)}")
        for k, cn in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
            vals = [s.get(k) for s in t5 if isinstance(s.get(k),(int,float))]
            P(f"  - {cn}: 均值 {means[cn]*100:.1f}% · p50 {(_q(vals,.5) or 0)*100:.0f}% · p90 {(_q(vals,.9) or 0)*100:.0f}%")
        noninf = 100 - means["推理"]*100
        P(f"\n  - **【依据】推理外开销 ≈ {noninf:.0f}%** = agentic RL 相比纯推理多付的代价(编译/对拍/工具/等待)。推理只占 ~{100-noninf:.0f}%。")
    P("\n### 3.2 结局 × 时长 × 轮数 相关性")
    P("**【计算】** 按 ses.status 分组,run_ms / trace_count 分位。")
    P("| status | 数量 | 占比 | run_ms p50(min) | run_ms p90(min) | trace_count p50 |")
    P("|---|---|---|---|---|---|")
    for stt in ("COMPLETED","ERROR","TIMEOUT","ABORTED"):
        g = [s for s in ses if s["status"]==stt]
        if not g: continue
        rm = [s["run_ms"] for s in g if s.get("run_ms")]; tc=[s.get("trace") for s in g if s.get("trace")]
        P(f"| {stt} | {len(g)} | {100*len(g)/len(ses):.0f}% | {(_q(rm,.5) or 0)/60000:.0f} | {(_q(rm,.9) or 0)/60000:.0f} | {_q(tc,.5) or '—'} |")
    er = [s for s in ses if s["status"]=="ERROR"]; co=[s for s in ses if s["status"]=="COMPLETED"]
    if er and co:
        e50=(_q([s['run_ms'] for s in er if s.get('run_ms')],.5) or 0)/60000
        c50=(_q([s['run_ms'] for s in co if s.get('run_ms')],.5) or 0)/60000
        P(f"\n  - **【依据】** ERROR p50={e50:.0f}min {'>' if e50>c50 else '<'} COMPLETED p50={c50:.0f}min → "
          f"{'失败的 session 反而更久(硬撑到耗尽预算/超时才放弃),是纯浪费' if e50>c50 else '失败快速返回'}。")
    if t6:
        et = Counter(v.get("error_type") for v in t6)
        P(f"\n### 3.3 验证结果分布(T6,{len(t6)} 次)")
        P("**【计算】** 每次 metrics.json 的 error_type 计数。")
        png = CH.bars_simple(_OUT["dir"], "verify_errtype", [(str(k),c) for k,c in et.most_common()],
                             "count", f"Verify error_type distribution (T6, n={len(t6)})") if _PNG else None
        if png: P(f"\n![验证结果]({png})\n")
        else:
            mx = max(et.values()) if et else 1
            for k, c in et.most_common(): P(f"  {str(k):24s} |{'█'*int(round(c/mx*30)):<30} {c}")
        top = et.most_common(1)[0] if et else (None,0)
        P(f"\n  - **【依据】** 主导失败 = `{top[0]}`({top[1]}/{len(t6)}={100*top[1]/len(t6):.0f}%)→ 这是拉低通过率的头号原因。")
        lw = [v.get("lease_wait_s") for v in t6 if v.get("lease_wait_s") is not None]
        P(f"  - 验证池排队 lease_wait: {(_fmt(_stats(lw),1,'s')) if lw else '无(运行时未带 lease 补丁,拿不到排队时长)'}")

# ===== 4. 多粒度 =====
def _section_multigrain(P, t8, ses):
    P("\n---\n\n## 4. 多粒度聚合(step / op)\n")
    P("### 4.1 per-step(T8:goodput + 引擎均衡)")
    P("**【来源】** rollout_step.jsonl。")
    for r in t8:
        b = r.get("engine_balance") or {}
        P(f"  - step={r['step'] if 'step' in r else r.get('rollout_step')}: sessions={r['num_sessions']} · goodput={r['goodput']} · "
          f"reward_p50={(r.get('reward_dist') or {}).get('p50')} · engine请求CV={b.get('request_cv')} · 份额={b.get('requests_per_engine')}")
    P("\n### 4.2 per-op(算子级:哪些算子难)")
    P("**【计算】** 按 op 分组,COMPLETED 比例(通过率)+ reward + 时长。图按通过率升序(红=难)。")
    by = defaultdict(list)
    for s in ses: by[s["op"]].append(s)
    ordered = sorted(by, key=lambda o: (sum(1 for s in by[o] if s["status"]=="COMPLETED")/len(by[o])))
    pairs = [(op, 100*sum(1 for s in by[op] if s["status"]=="COMPLETED")/len(by[op])) for op in ordered]
    png = CH.barh_ops(_OUT["dir"], "per_op_passrate", pairs, "COMPLETED %", "Per-operator pass rate (green≥80 / orange≥50 / red<50)") if _PNG else None
    if png: P(f"\n![per-op通过率]({png})\n")
    P("| op | n | COMPLETED | reward_p50 | run_min p50 |")
    P("|---|---|---|---|---|")
    for op in ordered:
        g = by[op]; comp=sum(1 for s in g if s["status"]=="COMPLETED")
        rw=[s["reward"] for s in g if isinstance(s.get("reward"),(int,float))]; rm=[s["run_ms"] for s in g if s.get("run_ms")]
        P(f"| {op} | {len(g)} | {comp}/{len(g)}({100*comp/len(g):.0f}%) | {_q(rw,.5) if rw else '—'} | {(_q(rm,.5) or 0)/60000:.0f} |")

# ===== 5. 主机 =====
def _section_host(P, t3):
    hs = [r for r in t3 if r.get("kind")=="host"]
    if not hs: return
    P("\n---\n\n## 5. 主机资源(T3)\n")
    P("**【来源】** host_proc.jsonl kind=host,/proc 采集。")
    mem = [r.get("mem_avail_mb") for r in hs]; cpu=[r.get("cpu_pct") for r in hs]; ld=[r.get("load1") for r in hs]
    P(f"  - 可用内存: {_fmt(_stats(mem),1/1024,'GB')}")
    P(f"  - CPU%: {_fmt(_stats(cpu),1,'%')} · load1: {_fmt(_stats(ld))}")
    P(f"  - **【依据】** 主机内存/CPU {'充裕' if (mem and st.mean(mem)/1024>200) else '偏紧'},非瓶颈。")

# ===== 6. 总结 =====
def _section_summary(P, t1, t2, t5, ses, t6):
    P("\n---\n\n## 6. 瓶颈总结(按影响排序)\n")
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    ver = [r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    rm = [s["run_ms"] for s in ses if s.get("run_ms")]
    noninf = 100 - st.mean([s.get('infer_frac',0) for s in t5])*100 if t5 else 0
    et = Counter(v.get("error_type") for v in t6) if t6 else Counter()
    items = []
    if et:
        top = et.most_common(1)[0]
        items.append(f"**通过率瓶颈 = `{top[0]}`**({100*top[1]/len(t6):.0f}% 的验证)—— 见 §3.3,这是把 goodput/reward 拉上去的第一杠杆。")
    if rm and _q(rm,.5)/60000 > 60:
        items.append(f"**单 session 太长**(p50 {_q(rm,.5)/60000:.0f}min)—— 见 §3.2,墙钟主要耗在 agent 多轮调试。")
    if allkv and st.mean(allkv) < 40:
        items.append(f"**推理算力过剩**(KV 仅 {st.mean(allkv):.0f}%)—— 见 §1.2,可提高 rollout 并发吃掉余量。")
    if noninf > 25:
        items.append(f"**agent 侧开销 {noninf:.0f}%**(推理外)—— 见 §3.1,验证/工具/等待是固有成本。")
    if ver and st.mean(ver) < 20:
        items.append(f"**验证池闲置**(4 卡均值 {st.mean(ver):.0f}%)—— 见 §1.4,资源利用不充分。")
    for i, it in enumerate(items, 1): P(f"{i}. {it}")
    P("\n> 每条结论的原始数据在 `source_data/`,可用 DuckDB/pandas 复核。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--out-suffix", default="_expert")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
        "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR","/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0)
    a = ap.parse_args()
    run = a.run or max(glob.glob(f"{a.runs_root}/*/"), key=os.path.getmtime)
    rep, sd = build(run, a.metrics_dir, a.window_sessions, a.out_suffix)
    print(f"报告: {rep}\n源数据: {sd}/")


if __name__ == "__main__":
    main()
