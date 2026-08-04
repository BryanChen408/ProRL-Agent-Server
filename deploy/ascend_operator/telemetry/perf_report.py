#!/usr/bin/env python3
"""一轮 rollout 系统负载 + 瓶颈全面分析(T1-T8,多粒度)。

三类瓶颈:①性能(推理引擎/NPU 算力、KV、排队、TTFT/TPOT/吞吐)②长度(prompt/decode/context
分布与压力、prefix)③Agent(块④时间拆解、轮数、预算取消、验证池、结局相关性)。
多粒度:per-request(T4)/per-session(T7)/per-step(T8)/per-op/per-engine。

产出:<run>/perf_report[_N]/REPORT.md + source_data/(各表切到窗口)。
用法:python3 perf_report.py [--run <dir>] [--window-sessions N] [--out-suffix _64]
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics as st, time
from collections import Counter, defaultdict

# ---------------- helpers ----------------
def _load(p):
    try: return [json.loads(x) for x in open(p, errors="replace") if x.strip()]
    except Exception: return []

def _q(xs, p):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

def _dist(xs, scale=1.0, unit=""):
    xs = sorted(x * scale for x in xs if isinstance(x, (int, float)))
    if not xs: return "无数据"
    n = len(xs); f = lambda p: xs[min(n - 1, int(p * n))]
    return (f"n={n} min={xs[0]:.1f} p50={f(.5):.1f} p90={f(.9):.1f} p99={f(.99):.1f} "
            f"max={xs[-1]:.1f} mean={st.mean(xs):.1f}{unit}")

def _cv(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.pstdev(xs) / st.mean(xs), 3) if len(xs) > 1 and st.mean(xs) else 0.0

def _t2_valid(rows):
    out = []
    for r in rows:
        if r.get("scrape_ok") is False: continue
        if "scrape_ok" not in r and r.get("vllm:num_requests_running") is None: continue
        out.append(r)
    return out

def _hist_quantile(rows, base, p):
    """从相邻两行 histogram 桶差分求分位(秒)。base 如 vllm:time_to_first_token_seconds。没桶则回 None。"""
    if len(rows) < 2: return None
    a, b = rows[0], rows[-1]
    buckets = []
    for k in b:
        if k.startswith(base + "_bucket@le="):
            le = k.split("@le=")[1]
            if le in ("+Inf", "inf"): continue
            try: buckets.append((float(le), b.get(k, 0) - a.get(k, 0)))
            except ValueError: pass
    if not buckets: return None
    buckets.sort()
    tot = buckets[-1][1]
    if tot <= 0: return None
    tgt = p * tot
    for le, c in buckets:
        if c >= tgt: return le
    return buckets[-1][0]

def _avg_delta(rows, s, c):
    if len(rows) < 2: return None
    ds = rows[-1].get(s, 0) - rows[0].get(s, 0); dc = rows[-1].get(c, 0) - rows[0].get(c, 0)
    return ds / dc if dc > 0 else None

# ---------------- main build ----------------
def build(run_dir, metrics_dir, window_n, out_suffix):
    der = os.path.join(run_dir, "telemetry_derived")
    out = os.path.join(run_dir, "perf_report" + (out_suffix or ""))
    os.makedirs(out, exist_ok=True)

    # 窗口 = 完成 session 时间跨度
    ses = []
    for f in glob.glob(f"{run_dir}/rollout_results/**/ses_*.json", recursive=True):
        try: s = json.load(open(f))
        except Exception: continue
        md = (s.get("trajectory", {}) or {}).get("metadata", {}) or {}
        tm = md.get("task_metadata") or {}
        t = s.get("timing", {}) or {}
        run_ms = t.get("run_ms") or 0
        end = os.path.getmtime(f)
        ses.append(dict(op=tm.get("op_name"), status=s.get("status"),
                        reward=(md.get("evaluation", {}) or {}).get("reward"),
                        run_ms=run_ms, init_ms=t.get("init_ms"), postrun_ms=t.get("postrun_ms"),
                        rollout_step=md.get("rollout_step"), trace_count=md.get("trace_count"),
                        start=end - run_ms / 1000.0, end=end))
    if not ses: raise SystemExit("无完成 session")
    ses.sort(key=lambda s: s["end"])
    if window_n: ses = ses[-window_n:]
    W0, W1 = min(s["start"] for s in ses), max(s["end"] for s in ses)
    span_h = (W1 - W0) / 3600
    inwin = lambda rows: [r for r in rows if W0 <= r.get("recorded_at_unix", 0) <= W1]

    R = []
    def P(*a): R.append(" ".join(str(x) for x in a))

    P("# 一轮 rollout 系统负载 + 瓶颈分析\n")
    P(f"- **run**: {os.path.basename(run_dir.rstrip('/'))} | **窗口**: {len(ses)} session, 墙钟 **{span_h:.2f}h** "
      f"({time.strftime('%m-%d %H:%M', time.localtime(W0))}→{time.strftime('%H:%M', time.localtime(W1))})")
    stc = Counter(s["status"] for s in ses)
    P(f"- **结局**: {dict(stc)} | goodput={stc.get('COMPLETED',0)/len(ses):.2f} | "
      f"reward {_dist([s['reward'] for s in ses])} | 覆盖算子 {len(set(s['op'] for s in ses))}\n")

    # 数据源
    t1 = inwin(_load(f"{metrics_dir}/npu_state/npu_card.jsonl"))
    t2 = {e: _t2_valid(inwin(_load(f"{metrics_dir}/vllm_state/{e}.jsonl"))) for e in ("infer-0","infer-1","infer-2")}
    t3 = inwin(_load(f"{metrics_dir}/host_state/host_proc.jsonl"))
    t4 = [r for f in glob.glob(f"{metrics_dir}/engine-*.jsonl") for r in _load(f)
          if isinstance(r.get("recorded_at"), str)]  # per-request(近似,时间用 recorded_at 字符串)
    t5 = _load(f"{der}/rollout_span.jsonl"); t6 = _load(f"{der}/verify_job.jsonl")
    t7 = _load(f"{der}/rollout.jsonl"); t8 = _load(f"{der}/rollout_step.jsonl")

    # =========================================================
    P("## 一、性能瓶颈(推理算力 / 引擎)")
    # T2 引擎:分位 + 时序趋势 + 饱和度
    P("### 引擎延迟/吞吐/饱和(T2,每引擎)")
    P("| eng | 吞吐tok/s | TTFTavg | TTFTp90 | ITLavg(ms/tok) | KVavg% | KVmax% | run_avg | wait_avg | wait_max | 抢占 | prefix% |")
    P("|"+"---|"*12)
    for e, rows in t2.items():
        if len(rows) < 2: P(f"| {e} | 数据不足 |"); continue
        dt = rows[-1]["recorded_at_unix"] - rows[0]["recorded_at_unix"]
        dg = rows[-1].get("vllm:generation_tokens_total",0) - rows[0].get("vllm:generation_tokens_total",0)
        ttft = _avg_delta(rows, "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count")
        ttft90 = _hist_quantile(rows, "vllm:time_to_first_token_seconds", .90)
        itl = _avg_delta(rows, "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count")
        kv = [r.get("vllm:kv_cache_usage_perc",0)*100 for r in rows]
        run = [r.get("vllm:num_requests_running",0) for r in rows]
        wait = [r.get("vllm:num_requests_waiting",0) for r in rows]
        preempt = rows[-1].get("vllm:num_preemptions_total",0) - rows[0].get("vllm:num_preemptions_total",0)
        ph = rows[-1].get("vllm:prefix_cache_hits_total",0)-rows[0].get("vllm:prefix_cache_hits_total",0)
        pq = rows[-1].get("vllm:prefix_cache_queries_total",0)-rows[0].get("vllm:prefix_cache_queries_total",0)
        P(f"| {e} | {dg/dt:.1f} | {(ttft or 0):.2f}s | {('%.2f'%ttft90+'s') if ttft90 else '—'} | "
          f"{(itl*1000 if itl else 0):.1f} | {st.mean(kv):.1f} | {max(kv):.1f} | {st.mean(run):.1f} | "
          f"{st.mean(wait):.2f} | {max(wait):.0f} | {int(preempt)} | {(100*ph/pq if pq else 0):.1f} |")
    # 饱和度判定
    allrun = [r.get("vllm:num_requests_running",0) for rows in t2.values() for r in rows]
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    allwait = [r.get("vllm:num_requests_waiting",0) for rows in t2.values() for r in rows]
    P(f"\n- **饱和度判定**: KV均值 {st.mean(allkv):.1f}%(max {max(allkv):.0f}%) · running均值 {st.mean(allrun):.1f} · "
      f"waiting均值 {st.mean(allwait):.2f}")
    verdict = ("引擎**吃满**(KV高+排队)" if st.mean(allkv)>70 or st.mean(allwait)>2
               else "引擎**远未吃满**(KV低+几乎不排队)→ 推理算力不是瓶颈,余量大")
    P(f"  → {verdict}")

    # T1 NPU 时序 + 不均
    P("\n### NPU 算力时序 + 不均(T1)")
    NB = max(1, int(span_h * 4)); step = (W1 - W0) / NB if NB else 1
    buck = defaultdict(lambda: defaultdict(list))
    for r in t1:
        if r.get("aicore_util_pct") is not None:
            buck[int((r["recorded_at_unix"]-W0)//step) if step else 0][r["engine_id"]].append(r["aicore_util_pct"])
    engs = ["infer-0","infer-1","infer-2","verify-pool"]
    P("每 ~%dmin aicore 均值(%%):" % round(step/60))
    P("| t(min) | "+" | ".join(engs)+" |"); P("|"+"---|"*(len(engs)+1))
    for b in sorted(buck)[:16]:
        P("| "+str(round(b*step/60))+" | "+" | ".join(f"{st.mean(v):.0f}" if (v:=buck[b].get(e)) else "—" for e in engs)+" |")
    inf = [r.get("aicore_util_pct") for r in t1 if r["pool"]=="inference" and r.get("aicore_util_pct") is not None]
    ver = [r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    em = {e: st.mean([x for bb in buck.values() for x in bb.get(e,[])] or [0]) for e in engs[:3]}
    P(f"\n- 推理池 aicore: {_dist(inf)} · **engine 间 CV={_cv(list(em.values()))}**(越小越均)")
    P(f"- 验证池 aicore: {_dist(ver)} → 4 卡验证池{'基本闲' if st.mean(ver or [0])<15 else '有负载'}")
    P(f"- 推理卡 HBM(MB): {_dist([r.get('hbm_used_mb') for r in t1 if r['pool']=='inference' and r.get('hbm_used_mb')])}(满 65536)")

    # =========================================================
    P("\n## 二、长度瓶颈(prompt / decode / context)")
    pt = [r.get("num_prompt_tokens") for r in t4]; gt = [r.get("num_generation_tokens") for r in t4]
    ctx = [(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch = [r.get("prefix_cache_hit_pct") for r in t4]
    P(f"- **prompt 长度**(per-request T4): {_dist(pt)} tok")
    P(f"- **decode 长度**: {_dist(gt)} tok")
    P(f"- **总 context**(prompt+decode): {_dist(ctx)} tok  ← 上限 262144")
    over = sum(1 for c in ctx if c > 240000); P(f"  - 逼近上限(>240k)的请求: {over}/{len(ctx)}")
    P(f"- **prefix cache 命中**: {_dist(ch)} %  → prefill 复用{'高' if st.mean([x for x in ch if isinstance(x,(int,float))] or [0])>80 else '一般'}")
    # per-op 长度
    by_op_len = defaultdict(list)
    for r in t4:
        if r.get("num_prompt_tokens"): by_op_len[str(r.get("session_id"))].append(r["num_prompt_tokens"])
    # session 训练样本长度(T7 trace / ses)
    P(f"- **session 轮数**(trace_count): {_dist([s.get('trace_count') for s in ses])}")

    # =========================================================
    P("\n## 三、Agent 瓶颈(块④时间 / 轮数 / 预算 / 结局)")
    if t5:
        for k, lab in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
            P(f"- {lab}占比: {_dist([s.get(k) for s in t5], 100, '%')}")
        P(f"  → **推理外开销 ≈ {100-st.mean([s.get('infer_frac',0) for s in t5])*100:.0f}%**(agentic RL 固有:验证+工具+等待)")
    # 结局相关性:时长/轮数 按 status
    P("\n### 结局相关性(时长/轮数 × status)")
    P("| status | 数量 | run_ms p50(min) | trace_count p50 |")
    P("|---|---|---|---|")
    for stt in ("COMPLETED","ERROR","TIMEOUT","ABORTED"):
        g = [s for s in ses if s["status"]==stt]
        if not g: continue
        rm = [s["run_ms"] for s in g if s.get("run_ms")]; tcnt=[s.get("trace_count") for s in g]
        P(f"| {stt} | {len(g)} | {(_q(rm,.5) or 0)/60000:.0f} | {_q([x for x in tcnt if x],.5) or '—'} |")
    # 验证池 T6
    if t6:
        et = Counter(v.get("error_type") for v in t6)
        lw = [v.get("lease_wait_s") for v in t6 if v.get("lease_wait_s") is not None]
        P(f"\n- **验证结果**(T6, {len(t6)}次): success={sum(1 for v in t6 if v.get('success'))} | error_type={dict(et)}")
        P(f"  - 验证池排队 lease_wait: {(_dist(lw)+' s') if lw else '无(运行时未带 lease 补丁)'}")

    # =========================================================
    P("\n## 四、多粒度聚合(per-step T8 / per-op)")
    for r in t8:
        b = r.get("engine_balance") or {}
        P(f"- **step={r['rollout_step']}**: sessions={r['num_sessions']} goodput={r['goodput']} "
          f"reward_p50={(r.get('reward_dist') or {}).get('p50')} · **engine请求CV={b.get('request_cv')}** "
          f"份额={b.get('requests_per_engine')}")
    # per-op:结局 + 时长
    P("\n### per-op(算子级)结局 + 时长")
    by_op = defaultdict(list)
    for s in ses: by_op[s["op"]].append(s)
    P("| op | n | COMPLETED | reward_p50 | run_min p50 |")
    P("|---|---|---|---|---|")
    for op in sorted(by_op, key=lambda o: -len(by_op[o]))[:20]:
        g = by_op[op]; comp = sum(1 for s in g if s["status"]=="COMPLETED")
        rw = [s["reward"] for s in g if isinstance(s.get("reward"),(int,float))]
        rm = [s["run_ms"] for s in g if s.get("run_ms")]
        P(f"| {op} | {len(g)} | {comp}/{len(g)} | {_q(rw,.5) if rw else '—'} | {(_q(rm,.5) or 0)/60000:.0f} |")

    # =========================================================
    P("\n## 五、瓶颈总结(一句话)")
    bn = []
    if 'verdict' in dir() and '未吃满' in verdict: bn.append("**推理算力不是瓶颈**(引擎余量大)")
    if t5 and (100-st.mean([s.get('infer_frac',0) for s in t5])*100) > 25:
        bn.append(f"**Agent 侧开销大**(推理外 {100-st.mean([s.get('infer_frac',0) for s in t5])*100:.0f}%)")
    rm_all = [s["run_ms"] for s in ses if s.get("run_ms")]
    if rm_all and _q(rm_all,.5)/60000 > 60: bn.append(f"**单 session 太长**(p50 {_q(rm_all,.5)/60000:.0f}min)")
    if ver and st.mean(ver)<15: bn.append("**验证池闲置**(4卡利用率低)")
    for i, b in enumerate(bn, 1): P(f"{i}. {b}")

    open(f"{out}/REPORT.md","w").write("\n".join(R))
    # 源数据切片
    sd = f"{out}/source_data"; os.makedirs(sd, exist_ok=True)
    def cut(src, dst):
        with open(dst,"w") as f:
            for r in inwin(_load(src)): f.write(json.dumps(r,ensure_ascii=False)+"\n")
    cut(f"{metrics_dir}/npu_state/npu_card.jsonl", f"{sd}/T1_npu_card.jsonl")
    for e in ("infer-0","infer-1","infer-2"): cut(f"{metrics_dir}/vllm_state/{e}.jsonl", f"{sd}/T2_{e}.jsonl")
    cut(f"{metrics_dir}/host_state/host_proc.jsonl", f"{sd}/T3_host_proc.jsonl")
    for t in ("rollout_span","verify_job","rollout","rollout_step"):
        if os.path.exists(f"{der}/{t}.jsonl"): open(f"{sd}/{t}.jsonl","w").write(open(f"{der}/{t}.jsonl").read())
    return f"{out}/REPORT.md", sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--out-suffix", default="")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
                    "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR","/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0)
    a = ap.parse_args()
    run = a.run or max(glob.glob(f"{a.runs_root}/*/"), key=os.path.getmtime)
    rep, sd = build(run, a.metrics_dir, a.window_sessions, a.out_suffix)
    print(f"报告: {rep}\n源数据: {sd}/\n"+"="*50); print(open(rep).read())


if __name__ == "__main__":
    main()
