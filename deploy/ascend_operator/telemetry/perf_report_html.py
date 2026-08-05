#!/usr/bin/env python3
"""专家级系统负载+瓶颈分析报告 —— 自包含 HTML(plotly,离线可看,交互,中文)。

版式:v1 页面组织(hero 页头 + KPI 大卡片 + sticky 目录 + 单栏卡片分节)+ 网格化图组
(§1 三时序并排 / §2 三直方并排 / §3 sunburst+pie 并排 / §4 表+图并排 / §5 主机三时序)。
支持 白天/黑夜 一键切换(右上角按钮,localStorage 记忆,plotly 图同步换肤)。
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script>/* 主题:防闪烁,渲染前先挂上 data-theme */
(function(){{var t=localStorage.getItem('perfTheme')||'light';
document.documentElement.setAttribute('data-theme',t);}})();</script>
<script>{plotlyjs}</script>
<style>
:root{{--ink:#1f2937;--mut:#64748b;--line:#e5eaf2;--bg:#f3f5f9;--card:#fff;
--acc:#2563eb;--acc2:#0ea5e9;--warn:#d97706;--bad:#dc2626;--ok:#059669;
--th:#f7fafd;--stripe:#fafcff;--hov:#f0f7ff;--srcbg:#f6f9ff;--srcbd:#dce9fb;
--conclbg:#fffaf1;--conclbd:#f5e2c4;--codebg:#edf1f7;
--plotbg:#ffffff;--plotgrid:#edf1f6;--plotline:#dbe2ec;--plotfont:#334155}}
[data-theme="dark"]{{--ink:#e2e8f0;--mut:#94a3b8;--line:#28334c;--bg:#0d1220;--card:#161d2e;
--th:#1d2740;--stripe:#1a2234;--hov:#1f2a44;--srcbg:#1a2340;--srcbd:#2c3a5e;
--conclbg:#26200f;--conclbd:#57491f;--codebg:#242e49;
--plotbg:#161d2e;--plotgrid:#232c42;--plotline:#334155;--plotfont:#cbd5e1}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:'Inter','PingFang SC','Microsoft YaHei',-apple-system,Arial,sans-serif;
background:var(--bg);color:var(--ink);margin:0;line-height:1.65;font-size:15px;transition:background .25s,color .25s}}
.hero{{background:linear-gradient(120deg,#1e3a8a 0%,#2563eb 55%,#0ea5e9 100%);color:#fff;padding:46px 0 64px}}
.hero .wrap,main,.toc .wrap,.kpis,footer{{max-width:1180px;margin-left:auto;margin-right:auto;padding-left:24px;padding-right:24px}}
.hero h1{{margin:0 0 10px;font-size:30px;font-weight:700;letter-spacing:.5px}}
.hero .meta{{opacity:.88;font-size:14px}}
.hero code{{background:rgba(255,255,255,.16);color:#fff;padding:2px 8px;border-radius:6px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap}}
.kpis.hero-kpis{{margin-top:-40px;position:relative;z-index:2}}
.kpis.flat{{margin:14px 0}}
.kpi{{flex:1;min-width:150px;background:linear-gradient(160deg,var(--card),color-mix(in srgb,var(--acc) 6%,var(--card)));
border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 22px rgba(30,58,138,.10);padding:16px 20px}}
.kpi b{{display:block;font-size:26px;font-weight:700;color:var(--acc);line-height:1.25}}
[data-theme="dark"] .kpi b{{color:#7aa7ff}}
.kpi span{{font-size:13px;color:var(--mut)}}
.toc{{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--card) 92%,transparent);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line);margin-top:20px}}
.toc .wrap{{display:flex;gap:4px;flex-wrap:wrap;padding-top:9px;padding-bottom:9px;align-items:center}}
.toc a{{color:var(--mut);text-decoration:none;font-size:14px;padding:6px 13px;border-radius:8px;white-space:nowrap}}
.toc a:hover{{background:color-mix(in srgb,var(--acc) 10%,transparent);color:var(--acc)}}
.themebtn{{margin-left:auto;cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--ink);
border-radius:20px;padding:5px 14px;font-size:13px;line-height:1.5}}
.themebtn:hover{{border-color:var(--acc)}}
main{{margin-top:8px;margin-bottom:60px}}
section.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;
box-shadow:0 2px 10px rgba(31,41,55,.05);padding:26px 30px;margin:24px 0}}
h2{{margin:0 0 4px;font-size:21px;color:var(--ink);display:flex;align-items:center;gap:11px}}
h2 .no{{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#fff;border-radius:9px;
font-size:14px;width:29px;height:29px;display:inline-flex;align-items:center;justify-content:center;flex:none}}
h3{{margin:24px 0 6px;color:var(--ink);font-size:16.5px}}
/* 网格图组 */
.grid{{display:grid;gap:16px;margin:14px 0}}
.g2{{grid-template-columns:1fr 1fr}}
.g3{{grid-template-columns:repeat(3,1fr)}}
.panel{{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:var(--card);min-width:0}}
.panel h4{{margin:2px 4px 8px;font-size:14px;color:var(--ink)}}
.panel .mini{{font-size:13px;color:var(--mut);padding:6px 6px 2px;border-top:1px dashed var(--line)}}
.panel .mini b{{color:var(--ink)}}
table{{border-collapse:separate;border-spacing:0;width:100%;margin:14px 0;font-size:14px;
border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th,td{{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line)}}
th{{background:var(--th);color:var(--mut);font-weight:600;font-size:13px;white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
tr:nth-child(even) td{{background:var(--stripe)}}
tr:hover td{{background:var(--hov)}}
details.src{{background:var(--srcbg);border:1px solid var(--srcbd);border-radius:10px;padding:8px 14px;
margin:10px 0;font-size:13px;color:var(--mut)}}
details.src summary{{cursor:pointer;color:var(--acc);font-weight:600;outline:none}}
[data-theme="dark"] details.src summary{{color:#7aa7ff}}
details.src[open] summary{{margin-bottom:4px}}
.concl{{background:var(--conclbg);border:1px solid var(--conclbd);border-left:4px solid var(--warn);
border-radius:10px;padding:10px 16px;margin:12px 0}}
.chart{{margin:6px 0}}
code{{background:var(--codebg);padding:1px 6px;border-radius:5px;font-size:13px}}
footer{{color:var(--mut);font-size:13px;padding-bottom:40px}}
@media(max-width:1100px){{.g3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:800px){{.g2,.g3{{grid-template-columns:1fr}}}}
</style></head><body>
"""

_THEME_JS = """<script>
function _plotlyTheme(t){
  var dark=(t==='dark');
  return {paper_bgcolor:dark?'#161d2e':'#ffffff',plot_bgcolor:dark?'#161d2e':'#ffffff',
    'font.color':dark?'#cbd5e1':'#334155',
    'xaxis.gridcolor':dark?'#232c42':'#edf1f6','yaxis.gridcolor':dark?'#232c42':'#edf1f6',
    'xaxis.linecolor':dark?'#334155':'#dbe2ec','yaxis.linecolor':dark?'#334155':'#dbe2ec',
    'xaxis.spikecolor':dark?'#64748b':'#94a3b8'};
}
function _applyTheme(t){
  document.documentElement.setAttribute('data-theme',t);
  localStorage.setItem('perfTheme',t);
  var b=document.getElementById('themebtn');
  if(b) b.textContent=(t==='dark')?'\\u2600\\uFE0F \\u767D\\u5929':'\\uD83C\\uDF19 \\u9ED1\\u591C';
  if(window.Plotly){document.querySelectorAll('.plotly-graph-div').forEach(function(g){
    try{Plotly.relayout(g,_plotlyTheme(t));}catch(e){}});}
}
function toggleTheme(){
  var cur=document.documentElement.getAttribute('data-theme')==='dark'?'dark':'light';
  _applyTheme(cur==='dark'?'light':'dark');
}
window.addEventListener('DOMContentLoaded',function(){
  var t=localStorage.getItem('perfTheme')||'light';_applyTheme(t);});
</script>"""

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

    # ===== 抬头 + KPI + 目录 =====
    P(_HTML_HEAD.format(title=f"负载分析 {os.path.basename(run_dir.rstrip('/'))}", plotlyjs=CH.plotlyjs()))
    P("<header class='hero'><div class='wrap'>")
    P(f"<h1>系统负载与瓶颈分析报告</h1>")
    P(f"<div class='meta'>run <code>{os.path.basename(run_dir.rstrip('/'))}</code> · 窗口 <b>{len(ses)}</b> 个完成 session · "
      f"墙钟 <b>{span_h:.2f} h</b> ({time.strftime('%m-%d %H:%M',time.localtime(W0))}→{time.strftime('%H:%M',time.localtime(W1))}) · "
      f"生成于 {time.strftime('%Y-%m-%d %H:%M')}</div>")
    P("</div></header>")
    stc = Counter(s["status"] for s in ses)
    rm_all = [s["run_ms"] for s in ses if s.get("run_ms")]
    allkv = [r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    P("<div class='kpis hero-kpis'>")
    P(f"<div class='kpi'><b>{stc.get('COMPLETED',0)/len(ses)*100:.0f}%</b><span>goodput</span></div>")
    P(f"<div class='kpi'><b>{(_q(rm_all,.5) or 0)/60000:.0f} min</b><span>单 session p50</span></div>")
    P(f"<div class='kpi'><b>{st.mean(allkv):.0f}%</b><span>KV cache 均值</span></div>")
    P(f"<div class='kpi'><b>{len(set(s['op'] for s in ses))}</b><span>覆盖算子</span></div>")
    P("</div>")
    P("<nav class='toc'><div class='wrap'>"
      "<a href='#sec1'>① 推理性能</a><a href='#sec2'>② 长度</a><a href='#sec3'>③ 时间构成</a>"
      "<a href='#sec4'>④ Agent 结局</a><a href='#sec5'>⑤ 主机</a>"
      "<a href='#sec6'>⑥ 瓶颈总结</a>"
      "<button id='themebtn' class='themebtn' onclick='toggleTheme()'>🌙 黑夜</button></div></nav>")
    P("<main>")
    P("<details class='src'><summary>🔍 数据溯源总览(T1–T8 表口径)</summary>"
      "<b>数据溯源</b>:T1 npu_card(NPU/显存,5s)· T2 vllm/metrics(引擎态)· T3 /proc(主机)· "
      "T4 engine-*.jsonl(逐请求)· T5 rollout_span(时间拆解)· T6 verify_job · T7 ses.json · T8 step 聚合。"
      "每结论标注【来源→计算→依据】。拓扑:16chip=12推理(3engine×TP4)+4验证。原始数据见 source_data/。</details>")

    _sec_perf(P, t1, t2, W0, W1, NB, allkv)
    _sec_length(P, t4)
    _sec_time(P, t5, t2, ses)
    _sec_agent(P, t6, ses)
    _sec_host(P, t3, W0, W1, NB)
    _sec_summary(P, t1, t2, t5, t6, ses)
    P("</main>")
    P(_THEME_JS)
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

# ===== 1. 性能:引擎表通栏 + KV/aicore/HBM 三时序网格 =====
def _sec_perf(P, t1, t2, W0, W1, NB, allkv):
    P("<section class='card' id='sec1'><h2><span class='no'>1</span>推理性能(引擎 / NPU 算力 / 显存)</h2>")
    P("<h3>1.1 引擎延迟 / 吞吐 / 饱和</h3>")
    P("<details class='src'><summary>🔍 来源 → 计算</summary>【来源】vllm /metrics 每引擎 5s;npu_card 每 chip 5s。"
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
          f"{'不均,LB/DP 分发不平' if _cv(list(tps.values()))>0.1 else '三引擎均衡'}。</div>")
    # 三时序并排
    P("<h3>1.2 饱和度三视角(KV / aicore / HBM)</h3>")
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
    d=CH.lines(kvser, "time (min)", "KV %", "KV cache usage", hline=90, hline_txt="near-full", h=300)
    if d: P(f"<div class='panel'><h4>KV cache 饱和度</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>均值 <b>{st.mean(allkv):.1f}%</b>(峰 {max(allkv):.0f}%),waiting 均 {st.mean(allwt):.2f} — "
            f"{'接近饱和,扩并发空间有限' if sat else '引擎没喂满,rollout 并发还能大幅提高'}</div></div>")
    d=CH.lines(aiser, "time (min)", "aicore %", "NPU aicore utilization", h=300)
    if d: P(f"<div class='panel'><h4>NPU aicore 利用率</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>推理池 {_tbl(_stats(inf),1,'%')};验证池均 <b>{st.mean(ver or [0]):.0f}%</b>(峰 {max(ver or [0]):.0f}%)— "
            f"{'4 卡验证池间歇打满、大部分空闲' if st.mean(ver or [0])<20 else '验证池有持续负载'}</div></div>")
    d=CH.lines(hser, "time (min)", "HBM MB", "HBM memory used", hline=65536, hline_txt="65536 full", h=300)
    if d: P(f"<div class='panel'><h4>显存占用(单卡满 65536MB)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>推理卡占满约 <b>{(_stats(hbm)['mean'] if hbm else 0)/65536*100:.0f}%</b> — "
            f"权重+激活占大头,KV 还有空间(呼应左图未饱和)</div></div>")
    P("</div></section>")

# ===== 2. 长度:三直方网格 =====
def _sec_length(P, t4):
    P("<section class='card' id='sec2'><h2><span class='no'>2</span>长度瓶颈(prompt / decode / context)</h2>")
    P("<details class='src'><summary>🔍 来源</summary>【来源】engine-*.jsonl(T4 逐请求)num_prompt_tokens / num_generation_tokens / prefix_cache_hit_pct。</details>")
    pt=[r.get("num_prompt_tokens") for r in t4]; gt=[r.get("num_generation_tokens") for r in t4]
    ctx=[(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch=[r.get("prefix_cache_hit_pct") for r in t4]
    over=sum(1 for c in ctx if c>240000)
    P("<div class='grid g3'>")
    d=CH.hist(pt,"prompt tokens","Prompt length (prefill)",h=300)
    if d: P(f"<div class='panel'><h4>prompt 长度 → prefill 压力</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(pt),1,' tok')} — 长 prompt 是主体</div></div>")
    d=CH.hist(gt,"generation tokens","Decode length",h=300)
    if d: P(f"<div class='panel'><h4>decode 长度 → 生成压力</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(gt),1,' tok')} — decode 短(p50 小),生成压力远小于 prefill</div></div>")
    d=CH.hist(ctx,"total context tokens","Context vs 262144",vline=262144,vline_txt="262144",h=300)
    if d: P(f"<div class='panel'><h4>总 context vs 上限</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(ctx),1,' tok')};逼近上限(&gt;240k)<b>{over}/{len(ctx)}"
            f"({100*over/max(1,len(ctx)):.1f}%)</b></div></div>")
    P("</div>")
    cs=_stats(ch)
    P(f"<div class='concl'>prefix cache 命中 {_tbl(cs,1,'%')} → 命中{'高' if cs and cs['mean']>80 else '一般'},"
      f"长 prompt 的 prefill 大量走缓存,<b>这是引擎没被 prefill 压垮的原因</b>。</div></section>")

# ===== 3. 时间:sunburst+pie 并排 + KPI + 表 =====
def _sec_time(P, t5, t2, ses):
    P("<section class='card' id='sec3'><h2><span class='no'>3</span>时间构成与绝对耗时(核心)</h2>")
    P("<details class='src'><summary>🔍 来源 → 计算(含口径标注)</summary>【来源】外层=T5 rollout_span 墙钟占比(推理/验证/工具/等待);"
      "推理内层 prefill:decode=T2 引擎侧 TTFT:(e2e−TTFT) 近似(引擎侧口径,非墙钟)。"
      "一轮墙钟=max(end)−min(start);累加=Σrun_ms;并发度=累加/墙钟。</details>")
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
    last=ses[-64:] if len(ses)>=64 else ses
    wall_h=(max(s["end"] for s in last)-min(s["start"] for s in last))/3600
    sum_h=sum(s["run_ms"] for s in last if s.get("run_ms"))/1000/3600
    conc=sum_h/wall_h if wall_h else 0
    n=len(last)
    P("<div class='grid g2'>")
    d=CH.sunburst_time(labels, parents, values, f"单 session 时间构成(均值 {sess_min:.0f} min;prefill:decode≈{pf_ratio*100:.0f}:{100-pf_ratio*100:.0f})")
    if d: P(f"<div class='panel'><h4>单 session 拆到段(分钟)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>推理内 prefill:decode≈<b>{pf_ratio*100:.0f}:{100-pf_ratio*100:.0f}</b>(引擎侧近似);"
            f"推理外开销 ≈ <b>{100-means['推理']*100:.0f}%</b>(≈{sess_min-inf_min:.0f} min/session);"
            f"decode 占大头(prefill 靠 cache 很快)→ 长输出/多轮生成是引擎时间主体</div></div>")
    d=CH.pie(['Inference','Verify','Tool','Wait/CPU'],[means['推理']*sess_min*n,means['验证']*sess_min*n,means['工具']*sess_min*n,means['等待/CPU']*sess_min*n],f'一轮({n} session)累计各段耗时(min)','min')
    if d: P(f"<div class='panel'><h4>一轮 {n} session 累计(分钟)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'><b>缩短一轮 = 缩短单 session 或提并发</b></div></div>")
    P("</div>")
    P("<div class='kpis flat'>")
    P(f"<div class='kpi'><b>{wall_h:.2f} h</b><span>一轮墙钟(并发口径)</span></div>")
    P(f"<div class='kpi'><b>{sum_h:.0f} h</b><span>单 session 累加</span></div>")
    P(f"<div class='kpi'><b>{conc:.1f}</b><span>并发度 ≈ 同时 {conc:.0f} session</span></div>")
    P(f"<div class='kpi'><b>{(_q([s['run_ms'] for s in last if s.get('run_ms')],.5) or 0)/60000:.0f} min</b><span>单 session p50</span></div>")
    P("</div>")
    P("<table><tr><th>段</th><th>占比</th><th>绝对(min/session)</th><th>口径</th></tr>")
    P(f"<tr><td>推理 inference</td><td>{means['推理']*100:.1f}%</td><td>{inf_min:.0f}</td><td>session 墙钟</td></tr>")
    P(f"<tr><td>· prefill(近似)</td><td>{means['推理']*pf_ratio*100:.1f}%</td><td>{inf_min*pf_ratio:.0f}</td><td>引擎侧 TTFT/e2e 近似</td></tr>")
    P(f"<tr><td>· decode(近似)</td><td>{means['推理']*(1-pf_ratio)*100:.1f}%</td><td>{inf_min*(1-pf_ratio):.0f}</td><td>引擎侧近似</td></tr>")
    P(f"<tr><td>验证 verify</td><td>{means['验证']*100:.1f}%</td><td>{means['验证']*sess_min:.0f}</td><td>session 墙钟</td></tr>")
    P(f"<tr><td>工具 tool</td><td>{means['工具']*100:.1f}%</td><td>{means['工具']*sess_min:.0f}</td><td>session 墙钟</td></tr>")
    P(f"<tr><td>等待/CPU</td><td>{means['等待/CPU']*100:.1f}%</td><td>{means['等待/CPU']*sess_min:.0f}</td><td>session 墙钟</td></tr>")
    P("</table>")
    P(f"<div class='concl'>一轮 <b>{n}</b> session 墙钟 <b>{wall_h:.2f}h</b>(≈{wall_h*60:.0f}min),并发度 {conc:.1f}"
      f"(≈{conc:.0f} session 同时在跑,与 12 推理卡吞吐相当)。</div></section>")

# ===== 4. Agent:表 + 图并排 =====
def _sec_agent(P, t6, ses):
    P("<section class='card' id='sec4'><h2><span class='no'>4</span>Agent 侧(结局相关 / 验证结果)</h2>")
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
        if div: P(f"<div class='panel'><h4>验证结果分布({len(t6)} 次)</h4><div class='chart'>{div}</div>"
                  f"<div class='mini'>主导失败 = <code>{top[0]}</code>(<b>{top[1]}/{len(t6)}={100*top[1]/len(t6):.0f}%</b>)"
                  f"→ 拉低通过率的头号原因</div></div>")
    P("</div></section>")

# ===== 5. 主机:三时序(mem/CPU/load1) =====
def _sec_host(P, t3, W0, W1, NB):
    hs=[r for r in t3 if r.get("kind")=="host"]
    if not hs: return
    P("<section class='card' id='sec5'><h2><span class='no'>5</span>主机资源</h2>")
    P("<details class='src'><summary>🔍 来源</summary>【来源】host_proc.jsonl kind=host,/proc,每 5s;时序按时间桶均值。</details>")
    mem=[r.get("mem_avail_mb") for r in hs]; cpu=[r.get("cpu_pct") for r in hs]; ld=[r.get("load1") for r in hs]
    P("<div class='grid g3'>")
    d=CH.lines({"host": _bucket(hs, "mem_avail_mb", W0, W1, NB, 1/1024)}, "time (min)", "GB", "Memory available", h=300)
    if d: P(f"<div class='panel'><h4>可用内存(GB)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(mem),1/1024,'GB')} — 主机{'充裕,非瓶颈' if mem and st.mean(mem)/1024>200 else '偏紧'}</div></div>")
    d=CH.lines({"host": _bucket(hs, "cpu_pct", W0, W1, NB)}, "time (min)", "CPU %", "CPU utilization", h=300)
    if d: P(f"<div class='panel'><h4>CPU 利用率(%)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(cpu),1,'%')}</div></div>")
    d=CH.lines({"host": _bucket(hs, "load1", W0, W1, NB)}, "time (min)", "load1", "Load average (1m)", h=300)
    if d: P(f"<div class='panel'><h4>load1(1 分钟负载)</h4><div class='chart'>{d}</div>"
            f"<div class='mini'>{_tbl(_stats(ld))}</div></div>")
    P("</div></section>")

# ===== 6. 总结 =====
def _sec_summary(P, t1, t2, t5, t6, ses):
    P("<section class='card' id='sec6'><h2><span class='no'>6</span>瓶颈总结(按影响排序)</h2>")
    allkv=[r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    ver=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    rm=[s["run_ms"] for s in ses if s.get("run_ms")]
    noninf=100-st.mean([s.get('infer_frac',0) for s in t5])*100 if t5 else 0
    et=Counter(v.get("error_type") for v in t6) if t6 else Counter()
    items=[]
    if et:
        top=et.most_common(1)[0]; items.append(f"<b>通过率瓶颈 = <code>{top[0]}</code></b>({100*top[1]/len(t6):.0f}% 验证)—— 见 §4,第一杠杆。")
    if rm and _q(rm,.5)/60000>60: items.append(f"<b>单 session 长</b>(p50 {_q(rm,.5)/60000:.0f}min)—— 见 §3,墙钟主耗在 agent 多轮。")
    if allkv and st.mean(allkv)<40: items.append(f"<b>推理算力过剩</b>(KV 仅 {st.mean(allkv):.0f}%)—— 见 §1,可提并发。")
    if noninf>25: items.append(f"<b>agent 侧开销 {noninf:.0f}%</b>—— 见 §3,验证/工具/等待固有成本。")
    if ver and st.mean(ver)<20: items.append(f"<b>验证池闲置</b>(4卡均值 {st.mean(ver):.0f}%)—— 见 §1。")
    P("<ol>"+"".join(f"<li>{i}</li>" for i in items)+"</ol>")
    P("</section>")
    P("<footer>原始数据在 source_data/,可 DuckDB/pandas 复核。报告由 perf_report_html.py 生成。</footer>")

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
