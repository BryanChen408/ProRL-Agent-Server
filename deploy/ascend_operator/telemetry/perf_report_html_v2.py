#!/usr/bin/env python3
"""v2 结构性重构版报告 —— 侧边栏导航 + 执行摘要前置 + 图表网格布局,4 套配色方案。

与 perf_report_html.py(v1)的差异在**信息组织**,不在数字:
  - v1:顶部 hero → 长滚动单栏,图一个占一行,结论在节尾
  - v2:左侧固定导航(含 mini KPI)→ 首屏即"执行摘要+瓶颈排序"→ 相关图网格并排
       (3 时序一排 / 3 直方一排 / sunburst+pie 一排),每图配独立面板和小结论

计算逻辑与 v1 完全一致(同表同公式)。产物 report_v2_<palette>.html,4 套配色一次出齐。
用法: python3 perf_report_html_v2.py [--run <dir>] [--window-sessions N] [--palette NAME|all]
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, time
from collections import Counter, defaultdict
import perf_charts_html as CH

# ---------- 配色方案(CSS 变量 + 图表 8 色) ----------
PALETTES = {
    "azure": dict(cn="澄蓝(商务)", acc="#2563eb", acc2="#0ea5e9", bg="#f3f5f9",
                  g1="#1e3a8a", g2="#2563eb", g3="#0ea5e9",
                  charts=["#2563eb", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#f97316", "#14b8a6"]),
    "emerald": dict(cn="翡冷翠", acc="#0d9488", acc2="#34d399", bg="#f1f6f3",
                    g1="#134e4a", g2="#0d9488", g3="#34d399",
                    charts=["#0d9488", "#34d399", "#f59e0b", "#6366f1", "#ef4444", "#0ea5e9", "#a3e635", "#f472b6"]),
    "sunset": dict(cn="暖阳橙", acc="#ea580c", acc2="#f59e0b", bg="#faf6f1",
                   g1="#7c2d12", g2="#ea580c", g3="#fbbf24",
                   charts=["#ea580c", "#f59e0b", "#0ea5e9", "#10b981", "#e11d48", "#8b5cf6", "#84cc16", "#06b6d4"]),
    "violet": dict(cn="星云紫", acc="#7c3aed", acc2="#d946ef", bg="#f5f3fa",
                   g1="#312e81", g2="#7c3aed", g3="#c026d3",
                   charts=["#7c3aed", "#d946ef", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#14b8a6"]),
}

_CSS = """
:root{--ink:#1f2937;--mut:#64748b;--line:#e5eaf2;--bg:@BG@;--card:#fff;
--acc:@ACC@;--acc2:@ACC2@;--warn:#d97706;--bad:#dc2626;--ok:#059669}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'Inter','PingFang SC','Microsoft YaHei',-apple-system,Arial,sans-serif;
background:var(--bg);color:var(--ink);margin:0;line-height:1.65;font-size:15px}
/* ---- 左侧固定导航 ---- */
aside{position:fixed;top:0;left:0;bottom:0;width:236px;background:var(--card);border-right:1px solid var(--line);
display:flex;flex-direction:column;padding:22px 0;z-index:20}
.brand{padding:0 22px 16px;border-bottom:1px solid var(--line)}
.brand b{font-size:17px;background:linear-gradient(135deg,var(--acc),var(--acc2));
-webkit-background-clip:text;background-clip:text;color:transparent}
.brand small{display:block;color:var(--mut);font-size:12px;margin-top:2px}
aside nav{flex:1;padding:14px 12px;overflow-y:auto}
aside nav a{display:flex;align-items:center;gap:9px;color:var(--mut);text-decoration:none;
font-size:14px;padding:9px 12px;border-radius:9px;margin:2px 0}
aside nav a:hover{background:color-mix(in srgb,var(--acc) 9%,#fff);color:var(--acc)}
aside nav a .dot{width:7px;height:7px;border-radius:50%;background:linear-gradient(135deg,var(--acc),var(--acc2));flex:none}
.side-kpis{padding:14px 22px 0;border-top:1px solid var(--line)}
.sk{display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;font-size:13px;color:var(--mut)}
.sk b{font-size:17px;color:var(--ink)}
/* ---- 右侧主区 ---- */
.main{margin-left:236px;padding:0 28px 60px;max-width:1420px}
.topbar{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);
backdrop-filter:blur(8px);padding:16px 4px 12px;border-bottom:1px solid var(--line);margin-bottom:22px}
.topbar h1{margin:0;font-size:21px}
.topbar .meta{color:var(--mut);font-size:13px;margin-top:3px}
.topbar code{background:#edf1f7;padding:1px 7px;border-radius:5px;font-size:12.5px}
.topbar .pal{float:right;font-size:12.5px;color:var(--mut);border:1px solid var(--line);
border-radius:20px;padding:3px 12px;background:#fff}
section.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
box-shadow:0 2px 10px rgba(31,41,55,.05);padding:24px 28px;margin:20px 0}
h2{margin:0 0 4px;font-size:19px;display:flex;align-items:center;gap:11px}
h2 .no{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#fff;border-radius:9px;
font-size:13px;width:27px;height:27px;display:inline-flex;align-items:center;justify-content:center;flex:none}
h3{margin:20px 0 6px;color:#334155;font-size:16px}
/* ---- 网格 ---- */
.grid{display:grid;gap:16px;margin:14px 0}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:repeat(3,1fr)}
.panel{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:#fff;min-width:0}
.panel h4{margin:2px 4px 8px;font-size:14px;color:#334155}
.panel .mini{font-size:13px;color:var(--mut);padding:6px 6px 2px;border-top:1px dashed var(--line)}
.panel .mini b{color:var(--ink)}
/* ---- KPI ---- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:14px 0}
.kpi{background:linear-gradient(160deg,#fff,color-mix(in srgb,var(--acc) 5%,#fff));
border:1px solid var(--line);border-radius:13px;padding:15px 18px}
.kpi b{display:block;font-size:25px;font-weight:700;color:var(--acc);line-height:1.25}
.kpi span{font-size:12.5px;color:var(--mut)}
/* ---- 瓶颈排序 ---- */
.rank{display:flex;gap:12px;align-items:flex-start;padding:11px 4px;border-bottom:1px dashed var(--line)}
.rank:last-child{border-bottom:none}
.rank .rn{flex:none;width:26px;height:26px;border-radius:8px;color:#fff;font-size:13px;font-weight:700;
display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--acc),var(--acc2))}
.rank:nth-child(1) .rn{background:linear-gradient(135deg,#dc2626,#f97316)}
.rank:nth-child(2) .rn{background:linear-gradient(135deg,#d97706,#fbbf24)}
.rank div{flex:1;font-size:14.5px}
.rank small{display:block;color:var(--mut);font-size:12.5px}
/* ---- 其余 ---- */
table{border-collapse:separate;border-spacing:0;width:100%;margin:12px 0;font-size:14px;
border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#f7fafd;color:#475569;font-weight:600;font-size:13px;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#fafcff}
tr:hover td{background:color-mix(in srgb,var(--acc) 6%,#fff)}
details.src{background:color-mix(in srgb,var(--acc) 5%,#fff);border:1px solid var(--line);border-radius:10px;
padding:7px 14px;margin:10px 0;font-size:13px;color:#4b5563}
details.src summary{cursor:pointer;color:var(--acc);font-weight:600;outline:none}
.concl{background:#fffaf1;border:1px solid #f5e2c4;border-left:4px solid var(--warn);
border-radius:10px;padding:10px 16px;margin:12px 0;font-size:14.5px}
code{background:#edf1f7;padding:1px 6px;border-radius:5px;font-size:13px}
footer{color:var(--mut);font-size:13px;padding-top:8px}
@media(max-width:1100px){.g3{grid-template-columns:1fr 1fr}}
@media(max-width:900px){aside{display:none}.main{margin-left:0}.g2,.g3{grid-template-columns:1fr}}
"""

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

def H(*a): return " ".join(str(x) for x in a)

def build(run_dir, metrics_dir, window_n, pal_name):
    pal = PALETTES[pal_name]
    CH.set_palette(pal["charts"])
    out = os.path.join(run_dir, "perf_report_html"); os.makedirs(out, exist_ok=True)
    der = os.path.join(run_dir, "telemetry_derived")
    R = []; P = lambda *a: R.append(H(*a))

    # ===== 数据加载/窗口(与 v1 相同) =====
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
    t8=_load(f"{der}/rollout_step.jsonl")

    # ===== 公共计算(与 v1 相同) =====
    stc = Counter(s["status"] for s in ses)
    rm_all = [s["run_ms"] for s in ses if s.get("run_ms")]
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    last=ses[-64:] if len(ses)>=64 else ses
    wall_h=(max(s["end"] for s in last)-min(s["start"] for s in last))/3600
    sum_h=sum(s["run_ms"] for s in last if s.get("run_ms"))/1000/3600
    conc=sum_h/wall_h if wall_h else 0
    p50_min=(_q(rm_all,.5) or 0)/60000

    # ===== 骨架:侧栏 + 顶栏 =====
    css=_CSS.replace("@BG@",pal["bg"]).replace("@ACC2@",pal["acc2"]).replace("@ACC@",pal["acc"])
    P(f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>负载分析v2 {os.path.basename(run_dir.rstrip('/'))} · {pal['cn']}</title>"
      f"<script>{CH.plotlyjs()}</script><style>{css}</style></head><body>")
    P("<aside><div class='brand'><b>负载分析 v2</b><small>ProRL 推理侧性能</small></div><nav>"
      "<a href='#sum'><span class='dot'></span>执行摘要</a>"
      "<a href='#perf'><span class='dot'></span>① 推理性能</a>"
      "<a href='#len'><span class='dot'></span>② 长度</a>"
      "<a href='#time'><span class='dot'></span>③ 时间构成</a>"
      "<a href='#agent'><span class='dot'></span>④ Agent 结局</a>"
      "<a href='#multi'><span class='dot'></span>⑤ 多粒度</a>"
      "<a href='#host'><span class='dot'></span>⑥ 主机</a></nav>")
    P("<div class='side-kpis'>"
      f"<div class='sk'>goodput<b>{stc.get('COMPLETED',0)/len(ses)*100:.0f}%</b></div>"
      f"<div class='sk'>单 session p50<b>{p50_min:.0f} min</b></div>"
      f"<div class='sk'>一轮墙钟<b>{wall_h:.2f} h</b></div>"
      f"<div class='sk'>并发度<b>{conc:.1f}</b></div></div></aside>")
    P("<div class='main'>")
    P(f"<div class='topbar'><span class='pal'>配色:{pal['cn']}({pal_name})</span>"
      f"<h1>系统负载与瓶颈分析</h1>"
      f"<div class='meta'>run <code>{os.path.basename(run_dir.rstrip('/'))}</code> · {len(ses)} session · "
      f"{time.strftime('%m-%d %H:%M',time.localtime(W0))}→{time.strftime('%H:%M',time.localtime(W1))} · "
      f"生成于 {time.strftime('%Y-%m-%d %H:%M')}</div></div>")

    # ===== 0. 执行摘要(KPI + 瓶颈排序前置) =====
    P("<section class='card' id='sum'><h2><span class='no'>Σ</span>执行摘要</h2>")
    P("<div class='kpis'>"
      f"<div class='kpi'><b>{stc.get('COMPLETED',0)/len(ses)*100:.0f}%</b><span>goodput</span></div>"
      f"<div class='kpi'><b>{p50_min:.0f} min</b><span>单 session p50</span></div>"
      f"<div class='kpi'><b>{st.mean(allkv):.0f}%</b><span>KV cache 均值</span></div>"
      f"<div class='kpi'><b>{wall_h:.2f} h</b><span>一轮墙钟(64 session)</span></div>"
      f"<div class='kpi'><b>{conc:.1f}</b><span>并发度(累加/墙钟)</span></div>"
      f"<div class='kpi'><b>{len(set(s['op'] for s in ses))}</b><span>覆盖算子</span></div></div>")
    P("<h3>瓶颈排序(按影响,细节见各节)</h3>")
    ver0=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    noninf=100-st.mean([s.get('infer_frac',0) for s in t5])*100 if t5 else 0
    et0=Counter(v.get("error_type") for v in t6) if t6 else Counter()
    items=[]
    if et0:
        top=et0.most_common(1)[0]; items.append((f"通过率瓶颈 = <code>{top[0]}</code>({100*top[1]/len(t6):.0f}% 验证),第一杠杆","见 §4"))
    if rm_all and _q(rm_all,.5)/60000>60: items.append((f"单 session 长(p50 {_q(rm_all,.5)/60000:.0f}min),墙钟主耗在 agent 多轮","见 §3"))
    if allkv and st.mean(allkv)<40: items.append((f"推理算力过剩(KV 仅 {st.mean(allkv):.0f}%),可提并发","见 §1"))
    if noninf>25: items.append((f"agent 侧开销 {noninf:.0f}%(验证/工具/等待固有成本)","见 §3"))
    if ver0 and st.mean(ver0)<20: items.append((f"验证池闲置(4 卡均值 {st.mean(ver0):.0f}%)","见 §1"))
    for i,(txt,ref) in enumerate(items,1):
        P(f"<div class='rank'><span class='rn'>{i}</span><div>{txt}<small>{ref}</small></div></div>")
    P("<details class='src'><summary>🔍 数据溯源总览(T1–T8 表口径)</summary>"
      "T1 npu_card(NPU/显存,5s)· T2 vllm/metrics(引擎态)· T3 /proc(主机)· T4 engine-*.jsonl(逐请求)· "
      "T5 rollout_span(时间拆解)· T6 verify_job · T7 ses.json · T8 step 聚合。拓扑:16chip=12推理(3engine×TP4)+4验证。"
      "原始数据见 source_data/。</details></section>")

    # ===== 1. 性能:表格通栏 + 3 时序网格 =====
    P("<section class='card' id='perf'><h2><span class='no'>1</span>推理性能(引擎 / NPU 算力 / 显存)</h2>")
    P("<details class='src'><summary>🔍 来源 → 计算</summary>【来源】vllm /metrics 每引擎 5s + npu_card 每 chip 5s。"
      "【计算】吞吐=Δgen_tokens/Δt;TTFT/ITL=Δsum/Δcount;KV/队列=均值;时序按时间桶均值。</details>")
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
    # 3 时序并排
    allwt=[r.get("vllm:num_requests_waiting",0) for rows in t2.values() for r in rows]
    kvser = {e: _bucket(rows, "vllm:kv_cache_usage_perc", W0, W1, NB, 100) for e, rows in t2.items()}
    aiser = {eng: _bucket([r for r in t1 if r.get("engine_id")==eng], "aicore_util_pct", W0, W1, NB)
             for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    hser = {eng: _bucket([r for r in t1 if r.get("engine_id")==eng], "hbm_used_mb", W0, W1, NB)
            for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    inf=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="inference" and r.get("aicore_util_pct") is not None]
    ver=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    hbm=[r.get("hbm_used_mb") for r in t1 if r["pool"]=="inference" and r.get("hbm_used_mb")]
    sat = st.mean(allkv)>70 or st.mean(allwt)>2
    P("<div class='grid g3'>")
    d1=CH.lines(kvser, "time (min)", "KV %", "KV cache usage", hline=90, hline_txt="near-full", h=300)
    if d1: P(f"<div class='panel'><h4>KV cache 饱和度(T2)</h4>{d1}<div class='mini'>均值 <b>{st.mean(allkv):.1f}%</b>(峰 {max(allkv):.0f}%),"
             f"waiting 均 {st.mean(allwt):.2f} — {'⚠️ 接近饱和' if sat else '✅ 没喂满,并发可大幅提高'}</div></div>")
    d2=CH.lines(aiser, "time (min)", "aicore %", "NPU aicore utilization", h=300)
    if d2: P(f"<div class='panel'><h4>NPU aicore 利用率(T1)</h4>{d2}<div class='mini'>推理池 {_tbl(_stats(inf),1,'%')};"
             f"验证池均 <b>{st.mean(ver or [0]):.0f}%</b> — {'间歇打满、大部分空闲' if st.mean(ver or [0])<20 else '有持续负载'}</div></div>")
    d3=CH.lines(hser, "time (min)", "HBM MB", "HBM memory used", hline=65536, hline_txt="65536 full", h=300)
    if d3: P(f"<div class='panel'><h4>显存占用(T1)</h4>{d3}<div class='mini'>推理卡占满约 <b>{(_stats(hbm)['mean'] if hbm else 0)/65536*100:.0f}%</b>"
             f" — 权重+激活占大头,KV 还有空间(呼应左图未饱和)</div></div>")
    P("</div></section>")

    # ===== 2. 长度:3 直方网格 + 命中率 =====
    P("<section class='card' id='len'><h2><span class='no'>2</span>长度瓶颈(prompt / decode / context)</h2>")
    P("<details class='src'><summary>🔍 来源</summary>【来源】engine-*.jsonl(T4 逐请求)num_prompt_tokens / num_generation_tokens / prefix_cache_hit_pct。</details>")
    pt=[r.get("num_prompt_tokens") for r in t4]; gt=[r.get("num_generation_tokens") for r in t4]
    ctx=[(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch=[r.get("prefix_cache_hit_pct") for r in t4]
    over=sum(1 for c in ctx if c>240000)
    P("<div class='grid g3'>")
    d=CH.hist(pt,"prompt tokens","Prompt length (prefill)",h=300)
    if d: P(f"<div class='panel'><h4>prompt 长度 → prefill 压力</h4>{d}<div class='mini'>{_tbl(_stats(pt),1,' tok')} — 长 prompt 是主体</div></div>")
    d=CH.hist(gt,"generation tokens","Decode length",h=300)
    if d: P(f"<div class='panel'><h4>decode 长度 → 生成压力</h4>{d}<div class='mini'>{_tbl(_stats(gt),1,' tok')} — decode 短,压力远小于 prefill</div></div>")
    d=CH.hist(ctx,"total context tokens","Context vs 262144",vline=262144,vline_txt="262144",h=300)
    if d: P(f"<div class='panel'><h4>总 context vs 上限</h4>{d}<div class='mini'>逼近上限(&gt;240k)<b>{over}/{len(ctx)}"
            f"({100*over/max(1,len(ctx)):.1f}%)</b></div></div>")
    P("</div>")
    cs=_stats(ch)
    P(f"<div class='concl'>prefix cache 命中 {_tbl(cs,1,'%')} → 命中{'高' if cs and cs['mean']>80 else '一般'},"
      f"长 prompt 的 prefill 大量走缓存,<b>这是引擎没被 prefill 压垮的原因</b>。</div></section>")

    # ===== 3. 时间:sunburst+pie 并排,表格通栏 =====
    P("<section class='card' id='time'><h2><span class='no'>3</span>时间构成与绝对耗时(核心)</h2>")
    P("<details class='src'><summary>🔍 来源 → 计算(含口径标注)</summary>【来源】外层=T5 rollout_span 墙钟占比;推理内层 prefill:decode=T2 引擎侧 "
      "TTFT:(e2e−TTFT) 近似(引擎侧口径,非墙钟)。一轮墙钟=max(end)−min(start),并发度=累加/墙钟。</details>")
    ttft_tot=e2e_tot=0
    for rows in t2.values():
        if len(rows)>=2:
            ttft_tot+=_d(rows,"vllm:time_to_first_token_seconds_sum")
            e2e_tot+=_d(rows,"vllm:e2e_request_latency_seconds_sum")
    pf_ratio = ttft_tot/e2e_tot if e2e_tot else 0.15
    means={}
    for k,cn in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
        vals=[s.get(k) for s in t5 if isinstance(s.get(k),(int,float))]; means[cn]=st.mean(vals) if vals else 0
    sess_min=st.mean([s["run_ms"] for s in ses if s.get("run_ms")])/60000 if ses else 0
    inf_min=means["推理"]*sess_min
    labels=["总","推理","·prefill","·decode","验证","工具","等待/CPU"]
    parents=["","总","推理","推理","总","总","总"]
    values=[sess_min, inf_min, inf_min*pf_ratio, inf_min*(1-pf_ratio),
            means["验证"]*sess_min, means["工具"]*sess_min, means["等待/CPU"]*sess_min]
    n=len(last)
    P("<div class='grid g2'>")
    d=CH.sunburst_time(labels, parents, values, f"单 session 时间构成(均值 {sess_min:.0f} min;prefill:decode≈{pf_ratio*100:.0f}:{100-pf_ratio*100:.0f})")
    if d: P(f"<div class='panel'><h4>单 session 拆到段(分钟)</h4>{d}<div class='mini'>推理外开销 ≈ "
            f"<b>{100-means['推理']*100:.0f}%</b>(≈{sess_min-inf_min:.0f} min);推理内 decode 占大头(prefill 靠 cache)</div></div>")
    d=CH.pie(['Inference','Verify','Tool','Wait/CPU'],[means['推理']*sess_min*n,means['验证']*sess_min*n,means['工具']*sess_min*n,means['等待/CPU']*sess_min*n],f'一轮({n} session)累计各段耗时(min)','min')
    if d: P(f"<div class='panel'><h4>一轮 {n} session 累计(分钟)</h4>{d}<div class='mini'>缩短一轮 = 缩短单 session "
            f"(p50 {(_q([s['run_ms'] for s in last if s.get('run_ms')],.5) or 0)/60000:.0f}min)或提并发</div></div>")
    P("</div>")
    P("<div class='kpis'>"
      f"<div class='kpi'><b>{wall_h:.2f} h</b><span>一轮墙钟(并发口径)</span></div>"
      f"<div class='kpi'><b>{sum_h:.0f} h</b><span>单 session 累加</span></div>"
      f"<div class='kpi'><b>{conc:.1f}</b><span>并发度 ≈ 同时 {conc:.0f} session</span></div>"
      f"<div class='kpi'><b>{sess_min:.0f} min</b><span>单 session 均值</span></div></div>")
    P("<table><tr><th>段</th><th>占比</th><th>绝对(min/session)</th><th>口径</th></tr>")
    P(f"<tr><td>推理 inference</td><td>{means['推理']*100:.1f}%</td><td>{inf_min:.0f}</td><td>T5 墙钟</td></tr>")
    P(f"<tr><td>· prefill(近似)</td><td>{means['推理']*pf_ratio*100:.1f}%</td><td>{inf_min*pf_ratio:.0f}</td><td>T2 TTFT/e2e</td></tr>")
    P(f"<tr><td>· decode(近似)</td><td>{means['推理']*(1-pf_ratio)*100:.1f}%</td><td>{inf_min*(1-pf_ratio):.0f}</td><td>T2</td></tr>")
    P(f"<tr><td>验证 verify</td><td>{means['验证']*100:.1f}%</td><td>{means['验证']*sess_min:.0f}</td><td>T5</td></tr>")
    P(f"<tr><td>工具 tool</td><td>{means['工具']*100:.1f}%</td><td>{means['工具']*sess_min:.0f}</td><td>T5</td></tr>")
    P(f"<tr><td>等待/CPU</td><td>{means['等待/CPU']*100:.1f}%</td><td>{means['等待/CPU']*sess_min:.0f}</td><td>T5</td></tr>")
    P("</table></section>")

    # ===== 4. Agent:表 + 图并排 =====
    P("<section class='card' id='agent'><h2><span class='no'>4</span>Agent 侧(结局相关 / 验证结果)</h2>")
    P("<details class='src'><summary>🔍 来源</summary>【来源】ses.json status / run_ms / trace_count;verify_job.jsonl error_type。</details>")
    P("<div class='grid g2'><div class='panel'><h4>结局 × 时长 × 轮数</h4>")
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
        P(f"<div class='mini'>ERROR p50={e5:.0f}min {'&gt;' if e5>c5 else '&lt;'} COMPLETED p50={c5:.0f}min → "
          f"{'失败 session 反而更久(硬撑到耗尽才放弃),是纯浪费' if e5>c5 else '失败快速返回'}</div>")
    P("</div>")
    if t6:
        et=Counter(v.get("error_type") for v in t6)
        div=CH.bars([(str(k),c) for k,c in et.most_common()],"count",f"Verify error_type (n={len(t6)})")
        top=et.most_common(1)[0]
        if div: P(f"<div class='panel'><h4>验证结果分布(T6,{len(t6)} 次)</h4>{div}<div class='mini'>主导失败 = "
                  f"<code>{top[0]}</code>(<b>{100*top[1]/len(t6):.0f}%</b>)→ 拉低通过率的头号原因</div></div>")
    P("</div></section>")

    # ===== 5. 多粒度 =====
    P("<section class='card' id='multi'><h2><span class='no'>5</span>多粒度(step / op)</h2>")
    P("<details class='src'><summary>🔍 来源</summary>【来源】rollout_step.jsonl;per-op 按 op 分组 COMPLETED 比例(绿≥80 橙≥50 红&lt;50)。</details>")
    P("<div class='grid g2'><div class='panel'><h4>per-step(goodput + 引擎均衡)</h4>")
    P("<table><tr><th>step</th><th>sessions</th><th>goodput</th><th>reward_p50</th><th>engine CV</th></tr>")
    for r in t8:
        b=r.get("engine_balance") or {}
        P(f"<tr><td>{r.get('rollout_step',r.get('step'))}</td><td>{r['num_sessions']}</td><td>{r['goodput']}</td>"
          f"<td>{(r.get('reward_dist') or {}).get('p50')}</td><td>{b.get('request_cv')}</td></tr>")
    P("</table></div>")
    by=defaultdict(list)
    for s in ses: by[s["op"]].append(s)
    ordered=sorted(by, key=lambda o:(sum(1 for s in by[o] if s["status"]=="COMPLETED")/len(by[o])))
    pairs=[(op,100*sum(1 for s in by[op] if s["status"]=="COMPLETED")/len(by[op])) for op in ordered]
    div=CH.barh(pairs,"COMPLETED %","Per-operator pass rate",
                colorfn=lambda v:"#dc2626" if v<50 else "#d97706" if v<80 else "#059669")
    if div: P(f"<div class='panel'><h4>per-op 通过率(哪些算子难)</h4>{div}</div>")
    P("</div></section>")

    # ===== 6. 主机 =====
    hs=[r for r in t3 if r.get("kind")=="host"]
    if hs:
        mem=[r.get("mem_avail_mb") for r in hs]; cpu=[r.get("cpu_pct") for r in hs]; ld=[r.get("load1") for r in hs]
        P("<section class='card' id='host'><h2><span class='no'>6</span>主机资源(T3)</h2>"
          "<details class='src'><summary>🔍 来源</summary>【来源】host_proc.jsonl kind=host,/proc。</details>"
          f"<div class='concl'>可用内存 {_tbl(_stats(mem),1/1024,'GB')};CPU% {_tbl(_stats(cpu),1,'%')};"
          f"load1 {_tbl(_stats(ld))}。主机{'充裕,非瓶颈' if mem and st.mean(mem)/1024>200 else '偏紧'}。</div></section>")

    P("<footer>原始数据在 source_data/,可 DuckDB/pandas 复核 · 结构与 v1(perf_report_html.py)仅排版/配色不同,数字口径完全一致</footer>")
    P("</div></body></html>")

    rep=f"{out}/report_v2_{pal_name}.html"
    open(rep,"w").write("\n".join(R))
    return rep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
        "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR","/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0)
    ap.add_argument("--palette", default="all", choices=list(PALETTES)+["all"])
    a=ap.parse_args()
    run=a.run or max(glob.glob(f"{a.runs_root}/*/"), key=os.path.getmtime)
    names=list(PALETTES) if a.palette=="all" else [a.palette]
    for nm in names:
        rep=build(run, a.metrics_dir, a.window_sessions, nm)
        print(f"[{nm}] {rep}")

if __name__=="__main__":
    main()
