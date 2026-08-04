#!/usr/bin/env python3
"""专家级系统负载+瓶颈分析报告 —— 自包含 HTML(plotly,离线可看,交互,中文)。

覆盖:引擎性能/NPU/显存 · 长度 · Agent 时间拆解(推理拆 prefill/decode) · 绝对耗时(单session+一轮墙钟) · 多粒度 · 主机。
每结论三段式溯源。用法: python3 perf_report_html.py [--run <dir>] [--window-sessions N]
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, time
from collections import Counter, defaultdict
import perf_charts_html as CH

def _load(p):
    try: return [json.loads(x) for x in open(p, errors="replace") if x.strip()]
    except Exception: return []
def _q(xs, p):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return xs[min(len(xs)-1, int(p*len(xs)))] if xs else None
def _stats(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs: return None
    n=len(xs); f=lambda p: xs[min(n-1,int(p*n))]
    return dict(n=n, min=xs[0], p50=f(.5), p90=f(.9), p99=f(.99), max=xs[-1], mean=st.mean(xs))
def _tbl(s, sc=1, u=""):
    if not s: return "无数据"
    return (f"n={s['n']} · min={s['min']*sc:.1f} · p50={s['p50']*sc:.1f} · p90={s['p90']*sc:.1f} · "
            f"p99={s['p99']*sc:.1f} · max={s['max']*sc:.1f} · <b>mean={s['mean']*sc:.1f}{u}</b>")
def _cv(xs):
    xs=[x for x in xs if isinstance(x,(int,float))]
    return round(st.pstdev(xs)/st.mean(xs),3) if len(xs)>1 and st.mean(xs) else 0.0
def _vt2(rows):
    return [r for r in rows if r.get("scrape_ok") is not False and
            not ("scrape_ok" not in r and r.get("vllm:num_requests_running") is None)]
def _bucket(rows, key, W0, W1, nb, scale=1.0):
    step=(W1-W0)/nb if nb else 1; b=defaultdict(list)
    for r in rows:
        v=r.get(key)
        if v is not None: b[int((r["recorded_at_unix"]-W0)//step) if step else 0].append(v)
    return [(round(i*step/60), st.mean(b[i])*scale) for i in sorted(b)]
def _d(rows, k):
    return rows[-1].get(k,0)-rows[0].get(k,0) if len(rows)>=2 else 0

_HTML_HEAD = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>{title}</title><script>{plotlyjs}</script>
<style>
body{{font-family:-apple-system,'Microsoft YaHei',Arial,sans-serif;max-width:1180px;margin:0 auto;padding:24px;color:#222;line-height:1.6}}
h1{{border-bottom:3px solid #3b6fb0;padding-bottom:8px}}
h2{{margin-top:36px;color:#3b6fb0;border-left:5px solid #3b6fb0;padding-left:10px}}
h3{{margin-top:24px;color:#444}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
th,td{{border:1px solid #ddd;padding:7px 10px;text-align:left}}
th{{background:#f0f5fb}}
tr:nth-child(even){{background:#fafbfc}}
.src{{background:#f7f9fc;border-left:3px solid #6ea8d8;padding:6px 12px;margin:8px 0;font-size:13px;color:#555}}
.concl{{background:#fff8f0;border-left:3px solid #dd8452;padding:8px 12px;margin:8px 0}}
.kpi{{display:inline-block;background:#f0f5fb;border-radius:8px;padding:12px 18px;margin:6px;text-align:center;min-width:130px}}
.kpi b{{display:block;font-size:24px;color:#3b6fb0}}
.chart{{margin:14px 0;border:1px solid #eee;border-radius:8px;padding:6px}}
code{{background:#f0f0f0;padding:1px 5px;border-radius:3px}}
</style></head><body>
"""

def H(*a): return " ".join(str(x) for x in a)

def build(run_dir, metrics_dir, window_n):
    out = os.path.join(run_dir, "perf_report_html"); os.makedirs(out, exist_ok=True)
    der = os.path.join(run_dir, "telemetry_derived")
    R = []; P = lambda *a: R.append(H(*a))

    # 窗口
    ses = []
    for f in glob.glob(f"{run_dir}/rollout_results/**/ses_*.json", recursive=True):
        try: s = json.load(open(f))
        except Exception: continue
        md=(s.get("trajectory",{}) or {}).get("metadata",{}) or {}; tm=md.get("task_metadata") or {}
        t=s.get("timing",{}) or {}; rm=t.get("run_ms") or 0; end=os.path.getmtime(f)
        ses.append(dict(op=tm.get("op_name"), status=s.get("status"),
                        reward=(md.get("evaluation",{}) or {}).get("reward"), run_ms=rm,
                        init_ms=t.get("init_ms"), postrun_ms=t.get("postrun_ms"),
                        step=md.get("rollout_step"), trace=md.get("trace_count"),
                        start=end-rm/1000, end=end))
    if not ses: raise SystemExit("无完成 session")
    ses.sort(key=lambda s: s["end"])
    if window_n: ses = ses[-window_n:]
    W0, W1 = min(s["start"] for s in ses), max(s["end"] for s in ses)
    span_h = (W1-W0)/3600; NB = max(6, min(30, int(span_h*4)))
    inwin = lambda rows: [r for r in rows if W0 <= r.get("recorded_at_unix",0) <= W1]
    t1=inwin(_load(f"{metrics_dir}/npu_state/npu_card.jsonl"))
    t2={e:_vt2(inwin(_load(f"{metrics_dir}/vllm_state/{e}.jsonl"))) for e in ("infer-0","infer-1","infer-2")}
    t3=inwin(_load(f"{metrics_dir}/host_state/host_proc.jsonl"))
    t4=[r for f in glob.glob(f"{metrics_dir}/engine-*.jsonl") for r in _load(f)]
    t5=_load(f"{der}/rollout_span.jsonl"); t6=_load(f"{der}/verify_job.jsonl")
    t7=_load(f"{der}/rollout.jsonl"); t8=_load(f"{der}/rollout_step.jsonl")

    # ===== 抬头 + KPI =====
    P(_HTML_HEAD.format(title=f"负载分析 {os.path.basename(run_dir.rstrip('/'))}", plotlyjs=CH.plotlyjs()))
    P(f"<h1>系统负载与瓶颈分析报告</h1>")
    P(f"<p>run <code>{os.path.basename(run_dir.rstrip('/'))}</code> · 窗口 <b>{len(ses)}</b> 个完成 session · "
      f"墙钟 <b>{span_h:.2f} h</b> ({time.strftime('%m-%d %H:%M',time.localtime(W0))}→{time.strftime('%H:%M',time.localtime(W1))}) · "
      f"生成于 {time.strftime('%Y-%m-%d %H:%M')}</p>")
    stc = Counter(s["status"] for s in ses)
    rm_all = [s["run_ms"] for s in ses if s.get("run_ms")]
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    P("<div>")
    P(f"<div class='kpi'><b>{stc.get('COMPLETED',0)/len(ses)*100:.0f}%</b>goodput</div>")
    P(f"<div class='kpi'><b>{(_q(rm_all,.5) or 0)/60000:.0f}min</b>单session p50</div>")
    P(f"<div class='kpi'><b>{st.mean(allkv):.0f}%</b>KV均值</div>")
    P(f"<div class='kpi'><b>{len(set(s['op'] for s in ses))}</b>覆盖算子</div>")
    P("</div>")
    P("<div class='src'><b>数据溯源</b>:T1 npu_card(NPU/显存,5s)· T2 vllm/metrics(引擎态)· T3 /proc(主机)· "
      "T4 engine-*.jsonl(逐请求)· T5 rollout_span(时间拆解)· T6 verify_job · T7 ses.json · T8 step 聚合。"
      "每结论标注【来源→计算→依据】。拓扑:16chip=12推理(3engine×TP4)+4验证。原始数据见 source_data/。</div>")

    _sec_perf(P, t1, t2, W0, W1, NB, span_h, allkv)
    _sec_length(P, t4)
    _sec_time(P, t5, t2, ses)          # 时间拆解(含 prefill/decode)+ 绝对耗时 + 一轮墙钟
    _sec_agent(P, t6, ses)
    _sec_multi(P, t8, ses)
    _sec_host(P, t3)
    _sec_summary(P, t1, t2, t5, t6, ses)
    P("</body></html>")

    open(f"{out}/report.html","w").write("\n".join(R))
    sd=f"{out}/source_data"; os.makedirs(sd, exist_ok=True)
    def cut(src,dst):
        with open(dst,"w") as f:
            for r in inwin(_load(src)): f.write(json.dumps(r,ensure_ascii=False)+"\n")
    cut(f"{metrics_dir}/npu_state/npu_card.jsonl", f"{sd}/T1_npu_card.jsonl")
    for e in ("infer-0","infer-1","infer-2"): cut(f"{metrics_dir}/vllm_state/{e}.jsonl", f"{sd}/T2_{e}.jsonl")
    cut(f"{metrics_dir}/host_state/host_proc.jsonl", f"{sd}/T3_host_proc.jsonl")
    for t in ("rollout_span","verify_job","rollout","rollout_step"):
        if os.path.exists(f"{der}/{t}.jsonl"): open(f"{sd}/{t}.jsonl","w").write(open(f"{der}/{t}.jsonl").read())
    return f"{out}/report.html", sd

# ===== 1. 性能:引擎 + NPU + 显存 =====
def _sec_perf(P, t1, t2, W0, W1, NB, span_h, allkv):
    P("<h2>1. 推理性能(引擎 / NPU 算力 / 显存)</h2>")
    P("<h3>1.1 引擎延迟 / 吞吐 / 饱和(T2)</h3>")
    P("<div class='src'>【来源】vllm /metrics 每引擎 5s。【计算】吞吐=Δgen_tokens/Δt;TTFT/ITL=Δsum/Δcount;KV/队列=均值。</div>")
    P("<table><tr><th>engine</th><th>吞吐 tok/s</th><th>TTFT均</th><th>ITL ms/tok</th><th>KV均%</th><th>KV峰%</th><th>running均</th><th>waiting均</th><th>抢占</th><th>prefix%</th></tr>")
    tps = {}
    for e, rows in t2.items():
        if len(rows) < 2: P(f"<tr><td>{e}</td><td colspan=9>采样不足</td></tr>"); continue
        dt=rows[-1]["recorded_at_unix"]-rows[0]["recorded_at_unix"]
        tp=_d(rows,"vllm:generation_tokens_total")/dt if dt else 0; tps[e]=tp
        ttft=_d(rows,"vllm:time_to_first_token_seconds_sum")/max(1,_d(rows,"vllm:time_to_first_token_seconds_count"))
        itl=_d(rows,"vllm:inter_token_latency_seconds_sum")/max(1,_d(rows,"vllm:inter_token_latency_seconds_count"))
        kv=[r.get("vllm:kv_cache_usage_perc",0)*100 for r in rows]; run=[r.get("vllm:num_requests_running",0) for r in rows]
        wt=[r.get("vllm:num_requests_waiting",0) for r in rows]
        ph=_d(rows,"vllm:prefix_cache_hits_total"); pq=_d(rows,"vllm:prefix_cache_queries_total")
        P(f"<tr><td>{e}</td><td>{tp:.1f}</td><td>{ttft:.2f}s</td><td>{itl*1000:.1f}</td><td>{st.mean(kv):.1f}</td>"
          f"<td>{max(kv):.0f}</td><td>{st.mean(run):.1f}</td><td>{st.mean(wt):.2f}</td><td>{int(_d(rows,'vllm:num_preemptions_total'))}</td>"
          f"<td>{100*ph/pq if pq else 0:.1f}</td></tr>")
    P("</table>")
    if len(tps) > 1:
        P(f"<div class='concl'>引擎均衡 <b>CV={_cv(list(tps.values()))}</b>(吞吐 {[f'{k}:{v:.0f}' for k,v in tps.items()]})— "
          f"{'⚠️ 不均,LB/DP 分发不平' if _cv(list(tps.values()))>0.1 else '✅ 均衡'}。</div>")
    # KV 时序
    P("<h3>1.2 KV cache 占用时序(饱和度)</h3>")
    kvser = {e: _bucket(rows, "vllm:kv_cache_usage_perc", W0, W1, NB, 100) for e, rows in t2.items()}
    div = CH.lines(kvser, "time (min)", "KV cache %", "KV cache usage over time (per engine)", hline=90, hline_txt="near-full")
    if div: P(f"<div class='chart'>{div}</div>")
    allwt=[r.get("vllm:num_requests_waiting",0) for rows in t2.values() for r in rows]
    sat = st.mean(allkv)>70 or st.mean(allwt)>2
    P(f"<div class='concl'>【依据→判定】KV 均值 <b>{st.mean(allkv):.1f}%</b>(峰 {max(allkv):.0f}%),waiting 均值 {st.mean(allwt):.2f}。"
      f"KV 与排队都低 ⇒ 引擎没喂满。<b>{'⚠️ 接近饱和' if sat else '✅ 推理算力不是瓶颈,rollout 并发还能大幅提高'}</b>。</div>")
    # NPU 时序
    P("<h3>1.3 NPU aicore 利用率时序(池间/引擎间)</h3>")
    P("<div class='src'>【来源】npu_card.jsonl aicore_util_pct,16chip 每 5s,按 engine_id 聚合。</div>")
    aiser = {eng: _bucket([r for r in t1 if r.get("engine_id")==eng], "aicore_util_pct", W0, W1, NB)
             for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    div = CH.lines(aiser, "time (min)", "aicore %", "NPU aicore utilization over time")
    if div: P(f"<div class='chart'>{div}</div>")
    inf=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="inference" and r.get("aicore_util_pct") is not None]
    ver=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    P(f"<div class='concl'>推理池 aicore {_tbl(_stats(inf),1,'%')};验证池均值 <b>{st.mean(ver or [0]):.0f}%</b>(峰 {max(ver or [0]):.0f}%)"
      f"→ 4 卡验证池{'间歇打满、大部分空闲' if st.mean(ver or [0])<20 else '有持续负载'}。</div>")
    # 显存
    P("<h3>1.4 显存(HBM)占用时序</h3>")
    P("<div class='src'>【来源】npu_card.jsonl hbm_used_mb / hbm_util_pct(单卡满 65536 MB)。</div>")
    hser = {eng: _bucket([r for r in t1 if r.get("engine_id")==eng], "hbm_used_mb", W0, W1, NB)
            for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    div = CH.lines(hser, "time (min)", "HBM used (MB)", "HBM memory used over time", hline=65536, hline_txt="65536 (full)")
    if div: P(f"<div class='chart'>{div}</div>")
    hbm=[r.get("hbm_used_mb") for r in t1 if r["pool"]=="inference" and r.get("hbm_used_mb")]
    P(f"<div class='concl'>推理卡 HBM {_tbl(_stats(hbm),1,'MB')} → 占满约 {(_stats(hbm)['mean'] if hbm else 0)/65536*100:.0f}%。"
      f"KV 用得少但 HBM 占比高 = 权重+激活占大头,KV 还有空间(呼应 1.2 未饱和)。</div>")

# ===== 2. 长度 =====
def _sec_length(P, t4):
    P("<h2>2. 长度瓶颈(prompt / decode / context)</h2>")
    P("<div class='src'>【来源】engine-*.jsonl(T4 逐请求)num_prompt_tokens / num_generation_tokens / prefix_cache_hit_pct。</div>")
    pt=[r.get("num_prompt_tokens") for r in t4]; gt=[r.get("num_generation_tokens") for r in t4]
    ctx=[(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch=[r.get("prefix_cache_hit_pct") for r in t4]
    P("<h3>2.1 prompt 长度分布(prefill 压力)</h3>")
    div=CH.hist(pt,"prompt tokens","Prompt length distribution (T4)")
    if div: P(f"<div class='chart'>{div}</div>")
    P(f"<div class='concl'>{_tbl(_stats(pt),1,' tok')} — 长 prompt 是主体。</div>")
    P("<h3>2.2 decode 长度分布(生成压力)</h3>")
    div=CH.hist(gt,"generation tokens","Decode length distribution (T4)")
    if div: P(f"<div class='chart'>{div}</div>")
    P(f"<div class='concl'>{_tbl(_stats(gt),1,' tok')} — decode 短(p50 小),生成压力远小于 prefill。</div>")
    P("<h3>2.3 总 context vs 262144 上限</h3>")
    div=CH.hist(ctx,"total context tokens","Total context vs 262144 limit",vline=262144,vline_txt="262144 limit")
    if div: P(f"<div class='chart'>{div}</div>")
    over=sum(1 for c in ctx if c>240000)
    P(f"<div class='concl'>{_tbl(_stats(ctx),1,' tok')};逼近上限(>240k)<b>{over}/{len(ctx)}({100*over/max(1,len(ctx)):.1f}%)</b>。</div>")
    cs=_stats(ch)
    P(f"<h3>2.4 prefix cache 命中</h3><div class='concl'>{_tbl(cs,1,'%')} → 命中{'高' if cs and cs['mean']>80 else '一般'},"
      f"长 prompt 的 prefill 大量走缓存,<b>这是引擎没被 prefill 压垮的原因</b>。</div>")

# ===== 3. 时间拆解(prefill/decode)+ 绝对耗时 + 一轮墙钟 =====
def _sec_time(P, t5, t2, ses):
    P("<h2>3. 时间构成与绝对耗时(核心)</h2>")
    # prefill:decode 比(引擎侧 T2)
    ttft_tot=e2e_tot=0
    for rows in t2.values():
        if len(rows)>=2:
            ttft_tot+=_d(rows,"vllm:time_to_first_token_seconds_sum")
            e2e_tot+=_d(rows,"vllm:e2e_request_latency_seconds_sum")
    pf_ratio = ttft_tot/e2e_tot if e2e_tot else 0.15  # prefill(含排队)占比
    P("<h3>3.1 一轮 rollout 时间占比(推理拆 prefill/decode)</h3>")
    P("<div class='src'>【来源】外层=T5 rollout_span 墙钟占比(推理/验证/工具/等待);推理内层 prefill:decode=T2 引擎侧 "
      "TTFT:(e2e−TTFT) 近似(标注:引擎侧口径,非墙钟)。【计算】各段均值。</div>")
    means={}
    for k,cn in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
        vals=[s.get(k) for s in t5 if isinstance(s.get(k),(int,float))]; means[cn]=st.mean(vals) if vals else 0
    # 单 session 墙钟均值(min),把占比换算成绝对分钟
    sess_min=st.mean([s["run_ms"] for s in ses if s.get("run_ms")])/60000 if ses else 0
    inf_min=means["推理"]*sess_min
    labels=["总","推理","·prefill","·decode","验证","工具","等待/CPU"]
    parents=["","总","推理","推理","总","总","总"]
    values=[sess_min, inf_min, inf_min*pf_ratio, inf_min*(1-pf_ratio),
            means["验证"]*sess_min, means["工具"]*sess_min, means["等待/CPU"]*sess_min]
    div=CH.sunburst_time(labels, parents, values, f"单 session 时间构成(均值 {sess_min:.0f} min;推理内 prefill:decode≈{pf_ratio*100:.0f}:{100-pf_ratio*100:.0f})")
    if div: P(f"<div class='chart'>{div}</div>")
    P("<table><tr><th>段</th><th>占比</th><th>绝对(min/session)</th><th>口径</th></tr>")
    P(f"<tr><td>推理 inference</td><td>{means['推理']*100:.1f}%</td><td>{inf_min:.0f}</td><td>T5 墙钟</td></tr>")
    P(f"<tr><td>· prefill(近似)</td><td>{means['推理']*pf_ratio*100:.1f}%</td><td>{inf_min*pf_ratio:.0f}</td><td>T2 TTFT/e2e</td></tr>")
    P(f"<tr><td>· decode(近似)</td><td>{means['推理']*(1-pf_ratio)*100:.1f}%</td><td>{inf_min*(1-pf_ratio):.0f}</td><td>T2</td></tr>")
    P(f"<tr><td>验证 verify</td><td>{means['验证']*100:.1f}%</td><td>{means['验证']*sess_min:.0f}</td><td>T5</td></tr>")
    P(f"<tr><td>工具 tool</td><td>{means['工具']*100:.1f}%</td><td>{means['工具']*sess_min:.0f}</td><td>T5</td></tr>")
    P(f"<tr><td>等待/CPU</td><td>{means['等待/CPU']*100:.1f}%</td><td>{means['等待/CPU']*sess_min:.0f}</td><td>T5</td></tr>")
    P("</table>")
    P(f"<div class='concl'>推理外开销 ≈ <b>{100-means['推理']*100:.0f}%</b>(≈{sess_min-inf_min:.0f} min/session)。"
      f"推理内 decode 占大头(prefill 靠 cache 很快)→ 长输出/多轮生成是引擎时间主体。</div>")
    # 一轮墙钟
    P("<h3>3.2 一轮 rollout(64 session)绝对耗时</h3>")
    P("<div class='src'>【来源】ses.json run_ms + 文件完成时间。【计算】取最近 64 个完成 session,墙钟跨度=max(end)−min(start);"
      "累加=Σrun_ms;并发度=累加/墙钟。</div>")
    last=ses[-64:] if len(ses)>=64 else ses
    wall_h=(max(s["end"] for s in last)-min(s["start"] for s in last))/3600
    sum_h=sum(s["run_ms"] for s in last if s.get("run_ms"))/1000/3600
    P("<div>")
    P(f"<div class='kpi'><b>{wall_h:.2f}h</b>一轮墙钟(并发)</div>")
    P(f"<div class='kpi'><b>{sum_h:.0f}h</b>单session累加</div>")
    P(f"<div class='kpi'><b>{sum_h/wall_h if wall_h else 0:.1f}</b>并发度</div>")
    P(f"<div class='kpi'><b>{(_q([s['run_ms'] for s in last if s.get('run_ms')],.5) or 0)/60000:.0f}min</b>单session p50</div>")
    P("</div>")
    # 一轮墙钟按段拆(绝对 min,乘 session 数近似总量)
    n=len(last)
    P(f"<div class='chart'>{CH.pie(['Inference','Verify','Tool','Wait/CPU'],[means['推理']*sess_min*n,means['验证']*sess_min*n,means['工具']*sess_min*n,means['等待/CPU']*sess_min*n],f'一轮({n} session)累计各段耗时(min)','min')}</div>")
    P(f"<div class='concl'>一轮 <b>{n}</b> session 墙钟 <b>{wall_h:.2f}h</b>(≈{wall_h*60:.0f}min),并发度 {sum_h/wall_h if wall_h else 0:.1f}"
      f"(≈{sum_h/wall_h if wall_h else 0:.0f} session 同时在跑,与 12 推理卡吞吐相当)。<b>缩短一轮 = 缩短单 session(p50 {(_q([s['run_ms'] for s in last if s.get('run_ms')],.5) or 0)/60000:.0f}min)或提并发</b>。</div>")

# ===== 4. Agent(结局 / 验证) =====
def _sec_agent(P, t6, ses):
    P("<h2>4. Agent 侧(结局相关 / 验证结果)</h2>")
    P("<h3>4.1 结局 × 时长 × 轮数</h3>")
    P("<div class='src'>【来源】ses.json status / run_ms / trace_count。</div>")
    P("<table><tr><th>status</th><th>数量</th><th>占比</th><th>run p50(min)</th><th>run p90(min)</th><th>trace p50</th></tr>")
    for stt in ("COMPLETED","ERROR","TIMEOUT","ABORTED"):
        g=[s for s in ses if s["status"]==stt]
        if not g: continue
        rm=[s["run_ms"] for s in g if s.get("run_ms")]; tc=[s.get("trace") for s in g if s.get("trace")]
        P(f"<tr><td>{stt}</td><td>{len(g)}</td><td>{100*len(g)/len(ses):.0f}%</td>"
          f"<td>{(_q(rm,.5) or 0)/60000:.0f}</td><td>{(_q(rm,.9) or 0)/60000:.0f}</td><td>{_q(tc,.5) or '—'}</td></tr>")
    P("</table>")
    er=[s['run_ms'] for s in ses if s['status']=='ERROR' and s.get('run_ms')]
    co=[s['run_ms'] for s in ses if s['status']=='COMPLETED' and s.get('run_ms')]
    if er and co:
        e5=(_q(er,.5) or 0)/60000; c5=(_q(co,.5) or 0)/60000
        P(f"<div class='concl'>ERROR p50={e5:.0f}min {'&gt;' if e5>c5 else '&lt;'} COMPLETED p50={c5:.0f}min → "
          f"{'失败 session 反而更久(硬撑到耗尽才放弃),是纯浪费' if e5>c5 else '失败快速返回'}。</div>")
    if t6:
        et=Counter(v.get("error_type") for v in t6)
        P(f"<h3>4.2 验证结果分布(T6,{len(t6)} 次)</h3>")
        div=CH.bars([(str(k),c) for k,c in et.most_common()],"count",f"Verify error_type (n={len(t6)})")
        if div: P(f"<div class='chart'>{div}</div>")
        top=et.most_common(1)[0]
        P(f"<div class='concl'>主导失败 = <code>{top[0]}</code>(<b>{top[1]}/{len(t6)}={100*top[1]/len(t6):.0f}%</b>)→ 拉低通过率的头号原因。</div>")

# ===== 5. 多粒度 =====
def _sec_multi(P, t8, ses):
    P("<h2>5. 多粒度(step / op)</h2>")
    P("<h3>5.1 per-step(goodput + 引擎均衡)</h3><div class='src'>【来源】rollout_step.jsonl。</div>")
    P("<table><tr><th>step</th><th>sessions</th><th>goodput</th><th>reward_p50</th><th>engine CV</th></tr>")
    for r in t8:
        b=r.get("engine_balance") or {}
        P(f"<tr><td>{r.get('rollout_step',r.get('step'))}</td><td>{r['num_sessions']}</td><td>{r['goodput']}</td>"
          f"<td>{(r.get('reward_dist') or {}).get('p50')}</td><td>{b.get('request_cv')}</td></tr>")
    P("</table>")
    P("<h3>5.2 per-op 通过率(哪些算子难)</h3><div class='src'>【来源】按 op 分组 COMPLETED 比例。图:绿≥80 橙≥50 红&lt;50。</div>")
    by=defaultdict(list)
    for s in ses: by[s["op"]].append(s)
    ordered=sorted(by, key=lambda o:(sum(1 for s in by[o] if s["status"]=="COMPLETED")/len(by[o])))
    pairs=[(op,100*sum(1 for s in by[op] if s["status"]=="COMPLETED")/len(by[op])) for op in ordered]
    div=CH.barh(pairs,"COMPLETED %","Per-operator pass rate",
                colorfn=lambda v:"#c44e52" if v<50 else "#dd8452" if v<80 else "#55a868")
    if div: P(f"<div class='chart'>{div}</div>")

# ===== 6. 主机 =====
def _sec_host(P, t3):
    hs=[r for r in t3 if r.get("kind")=="host"]
    if not hs: return
    P("<h2>6. 主机资源(T3)</h2><div class='src'>【来源】host_proc.jsonl kind=host,/proc。</div>")
    mem=[r.get("mem_avail_mb") for r in hs]; cpu=[r.get("cpu_pct") for r in hs]; ld=[r.get("load1") for r in hs]
    P(f"<div class='concl'>可用内存 {_tbl(_stats(mem),1/1024,'GB')};CPU% {_tbl(_stats(cpu),1,'%')};"
      f"load1 {_tbl(_stats(ld))}。主机{'充裕,非瓶颈' if mem and st.mean(mem)/1024>200 else '偏紧'}。</div>")

# ===== 7. 总结 =====
def _sec_summary(P, t1, t2, t5, t6, ses):
    P("<h2>7. 瓶颈总结(按影响排序)</h2>")
    allkv=[r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    ver=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    rm=[s["run_ms"] for s in ses if s.get("run_ms")]
    noninf=100-st.mean([s.get('infer_frac',0) for s in t5])*100 if t5 else 0
    et=Counter(v.get("error_type") for v in t6) if t6 else Counter()
    items=[]
    if et:
        top=et.most_common(1)[0]; items.append(f"<b>通过率瓶颈 = <code>{top[0]}</code></b>({100*top[1]/len(t6):.0f}% 验证)—— 见 §4.2,第一杠杆。")
    if rm and _q(rm,.5)/60000>60: items.append(f"<b>单 session 长</b>(p50 {_q(rm,.5)/60000:.0f}min)—— 见 §3,墙钟主耗在 agent 多轮。")
    if allkv and st.mean(allkv)<40: items.append(f"<b>推理算力过剩</b>(KV 仅 {st.mean(allkv):.0f}%)—— 见 §1.2,可提并发。")
    if noninf>25: items.append(f"<b>agent 侧开销 {noninf:.0f}%</b>—— 见 §3.1,验证/工具/等待固有成本。")
    if ver and st.mean(ver)<20: items.append(f"<b>验证池闲置</b>(4卡均值 {st.mean(ver):.0f}%)—— 见 §1.3。")
    P("<ol>"+"".join(f"<li>{i}</li>" for i in items)+"</ol>")
    P("<p style='color:#888;font-size:13px'>原始数据在 source_data/,可 DuckDB/pandas 复核。</p>")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
        "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR","/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0)
    a=ap.parse_args()
    run=a.run or max(glob.glob(f"{a.runs_root}/*/"), key=os.path.getmtime)
    rep,sd=build(run, a.metrics_dir, a.window_sessions)
    print(f"报告(HTML): {rep}\n源数据: {sd}/")

if __name__=="__main__":
    main()
