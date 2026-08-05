#!/usr/bin/env python3
"""一页 PPT(HTML 单页幻灯,1920×1080)—— 系统性能负载汇报版。

内容规划(见 GUIDE 对话定稿):KPI 全保留;瓶颈排序 5 条全保留;sunburst 时间拆解保留;
引擎表压到 6 列;长度/主机图压成底条分位数;溯源压成底条一句口径。
数据计算复用 perf_report_html 的函数,与 report.html 数字严格一致。
用法: python3 perf_slide_html.py [--run <dir>] [--window-sessions N]
产物: <run_dir>/perf_report_html/slide.html
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, time
from collections import Counter, defaultdict
import perf_charts_html as CH
from perf_report_html import _load, _q, _stats, _cv, _vt2, _bucket, _d

CSS = """
*{box-sizing:border-box;margin:0}
body{background:#525659;display:flex;align-items:center;justify-content:center;min-height:100vh;
font-family:'Inter','PingFang SC','Microsoft YaHei',-apple-system,Arial,sans-serif;overflow:hidden}
.slide{width:1920px;height:1080px;flex:none;background:#f3f5f9;color:#1f2937;display:flex;flex-direction:column;
transform-origin:center center;box-shadow:0 12px 60px rgba(0,0,0,.4)}
/* 头部 */
.hd{background:linear-gradient(120deg,#1e3a8a,#2563eb 55%,#0ea5e9);color:#fff;padding:22px 40px 0;height:118px;flex:none}
.hd h1{font-size:30px;letter-spacing:.5px}
.hd .meta{opacity:.88;font-size:14.5px;margin-top:6px}
.hd code{background:rgba(255,255,255,.16);padding:1px 8px;border-radius:6px}
/* KPI 条 */
.kpis{display:flex;gap:16px;padding:0 40px;margin-top:-30px;flex:none;height:92px}
.kpi{flex:1;background:linear-gradient(160deg,#fff,#f4f8ff);border:1px solid #e5eaf2;border-radius:14px;
box-shadow:0 8px 22px rgba(30,58,138,.10);padding:12px 20px}
.kpi b{display:block;font-size:30px;color:#2563eb;line-height:1.2}
.kpi span{font-size:13.5px;color:#64748b}
/* 中栏 */
.mid{flex:1;display:flex;gap:18px;padding:16px 40px 0;min-height:0}
.col{background:#fff;border:1px solid #e5eaf2;border-radius:14px;padding:16px 20px;box-shadow:0 2px 10px rgba(31,41,55,.05);
display:flex;flex-direction:column;min-width:0}
.col h3{font-size:17px;color:#0f172a;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.col h3 .tag{background:linear-gradient(135deg,#2563eb,#0ea5e9);color:#fff;border-radius:7px;
font-size:12px;padding:2px 8px;font-weight:600}
.cA{width:600px;flex:none}.cB{flex:1}.cC{width:560px;flex:none}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14.5px;border:1px solid #e5eaf2;border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #e5eaf2}
th{background:#f7fafd;color:#475569;font-size:13px;font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#fafcff}
.note{font-size:13.5px;color:#64748b;margin-top:8px}.note b{color:#1f2937}
.wall{display:flex;gap:10px;margin-top:8px}
.wall div{flex:1;background:#f6f9ff;border:1px solid #dce9fb;border-radius:10px;padding:8px 12px;text-align:center}
.wall b{display:block;font-size:21px;color:#2563eb}
.wall span{font-size:12px;color:#64748b}
/* 瓶颈榜 */
.rank{display:flex;gap:11px;align-items:flex-start;padding:9px 2px;border-bottom:1px dashed #e5eaf2;font-size:15px}
.rank:last-child{border-bottom:none}
.rank .rn{flex:none;width:25px;height:25px;border-radius:7px;color:#fff;font-size:13px;font-weight:700;
display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#2563eb,#0ea5e9)}
.rank:nth-child(1) .rn{background:linear-gradient(135deg,#dc2626,#f97316)}
.rank:nth-child(2) .rn{background:linear-gradient(135deg,#d97706,#fbbf24)}
/* 底两栏 */
.row{flex:none;padding:14px 40px 0}
.tris{display:flex;gap:18px}
.tri{flex:1;background:#fff;border:1px solid #e5eaf2;border-radius:14px;padding:10px 14px;min-width:0;
box-shadow:0 2px 10px rgba(31,41,55,.05)}
.tri h4{font-size:14px;color:#334155;margin-bottom:2px}
.tri .st{font-size:13px;color:#64748b}.tri .st b{color:#1f2937}
.foot{display:flex;gap:26px;align-items:center;background:#fff;border:1px solid #e5eaf2;border-radius:14px;
margin:14px 40px 18px;padding:12px 22px;font-size:14px;color:#475569}
.foot b{color:#0f172a}
.foot .sep{color:#cbd5e1}
.foot .src{margin-left:auto;font-size:12.5px;color:#94a3b8;white-space:nowrap}
"""

def _mini_lines(series, ylabel, hline=None, hline_txt="", h=178):
    """紧凑时序(幻灯片 D 区用):小字号、窄边距、单系列隐图例。"""
    if not CH.available(): return ""
    fig = CH.go.Figure()
    for i, (lab, pts) in enumerate(series.items()):
        if not pts: continue
        fig.add_trace(CH.go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                                    name=lab, line=dict(width=2, color=CH.C[i % len(CH.C)])))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#ef4444", line_width=1,
                      annotation_text=hline_txt, annotation_font=dict(size=10, color="#ef4444"))
    fig.update_layout(template="plotly_white", height=h,
        margin=dict(l=44, r=10, t=28, b=22),
        font=dict(size=10.5, family="Inter, 'PingFang SC', 'Microsoft YaHei', Arial", color="#475569"),
        paper_bgcolor="#fff", plot_bgcolor="#fff", hovermode="x unified",
        xaxis=dict(title=None, gridcolor="#edf1f6", zeroline=False),
        yaxis=dict(title=dict(text=ylabel, font=dict(size=10.5)), gridcolor="#edf1f6", zeroline=False),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        showlegend=sum(1 for p in series.values() if p) > 1)
    return CH.pio.to_html(fig, include_plotlyjs=False, full_html=False, config=CH._CONFIG)

def build(run_dir, metrics_dir, window_n):
    out = os.path.join(run_dir, "perf_report_html"); os.makedirs(out, exist_ok=True)
    der = os.path.join(run_dir, "telemetry_derived")
    R = []; P = lambda *a: R.append(" ".join(str(x) for x in a))

    # ===== 数据加载(与 report 相同) =====
    ses = []
    for f in glob.glob(f"{run_dir}/rollout_results/**/ses_*.json", recursive=True):
        try: s = json.load(open(f))
        except Exception: continue
        md=(s.get("trajectory",{}) or {}).get("metadata",{}) or {}; tm=md.get("task_metadata") or {}
        t=s.get("timing",{}) or {}; rm=t.get("run_ms") or 0; end=os.path.getmtime(f)
        ses.append(dict(op=tm.get("op_name"), status=s.get("status"), run_ms=rm,
                        trace=md.get("trace_count"), start=end-rm/1000, end=end))
    if not ses: raise SystemExit("无完成 session")
    ses.sort(key=lambda s: s["end"])
    if window_n: ses = ses[-window_n:]
    W0, W1 = min(s["start"] for s in ses), max(s["end"] for s in ses)
    span_h=(W1-W0)/3600; NB=max(6, min(30, int(span_h*4)))
    inwin = lambda rows: [r for r in rows if W0 <= r.get("recorded_at_unix",0) <= W1]
    t1=inwin(_load(f"{metrics_dir}/npu_state/npu_card.jsonl"))
    t2={e:_vt2(inwin(_load(f"{metrics_dir}/vllm_state/{e}.jsonl"))) for e in ("infer-0","infer-1","infer-2")}
    t3=inwin(_load(f"{metrics_dir}/host_state/host_proc.jsonl"))
    t4=[r for f in glob.glob(f"{metrics_dir}/engine-*.jsonl") for r in _load(f)]
    t5=_load(f"{der}/rollout_span.jsonl"); t6=_load(f"{der}/verify_job.jsonl")

    # ===== 计算(与 report 相同口径) =====
    stc=Counter(s["status"] for s in ses)
    rm_all=[s["run_ms"] for s in ses if s.get("run_ms")]
    allkv=[r.get("vllm:kv_cache_usage_perc",0)*100 for rows in t2.values() for r in rows]
    allwt=[r.get("vllm:num_requests_waiting",0) for rows in t2.values() for r in rows]
    last=ses[-64:] if len(ses)>=64 else ses
    wall_h=(max(s["end"] for s in last)-min(s["start"] for s in last))/3600
    sum_h=sum(s["run_ms"] for s in last if s.get("run_ms"))/1000/3600
    conc=sum_h/wall_h if wall_h else 0
    last_p50=(_q([s["run_ms"] for s in last if s.get("run_ms")],.5) or 0)/60000
    # 引擎表
    eng_rows=[]; tps={}
    for e, rows in t2.items():
        if len(rows)<2: continue
        dt=rows[-1]["recorded_at_unix"]-rows[0]["recorded_at_unix"]
        tp=_d(rows,"vllm:generation_tokens_total")/dt if dt else 0; tps[e]=tp
        ttft=_d(rows,"vllm:time_to_first_token_seconds_sum")/max(1,_d(rows,"vllm:time_to_first_token_seconds_count"))
        itl=_d(rows,"vllm:inter_token_latency_seconds_sum")/max(1,_d(rows,"vllm:inter_token_latency_seconds_count"))
        kv=[r.get("vllm:kv_cache_usage_perc",0)*100 for r in rows]
        wt=[r.get("vllm:num_requests_waiting",0) for r in rows]
        ph=_d(rows,"vllm:prefix_cache_hits_total"); pq=_d(rows,"vllm:prefix_cache_queries_total")
        eng_rows.append((e,tp,ttft,itl*1000,st.mean(kv),st.mean(wt),100*ph/pq if pq else 0))
    # 时间拆解
    ttft_tot=e2e_tot=0
    for rows in t2.values():
        if len(rows)>=2:
            ttft_tot+=_d(rows,"vllm:time_to_first_token_seconds_sum")
            e2e_tot+=_d(rows,"vllm:e2e_request_latency_seconds_sum")
    pf_ratio=ttft_tot/e2e_tot if e2e_tot else 0.15
    means={}
    for k,cn in [("infer_frac","推理"),("verify_frac","验证"),("tool_frac","工具"),("wait_cpu_frac","等待/CPU")]:
        vals=[s.get(k) for s in t5 if isinstance(s.get(k),(int,float))]; means[cn]=st.mean(vals) if vals else 0
    sess_min=st.mean(rm_all)/60000 if rm_all else 0
    inf_min=means["推理"]*sess_min
    # 长度底条
    pt=[r.get("num_prompt_tokens") for r in t4]; gt=[r.get("num_generation_tokens") for r in t4]
    ctx=[(r.get("num_prompt_tokens") or 0)+(r.get("num_generation_tokens") or 0) for r in t4 if r.get("num_prompt_tokens")]
    ch=[r.get("prefix_cache_hit_pct") for r in t4]
    over=sum(1 for c in ctx if c>240000)
    # 主机
    hs=[r for r in t3 if r.get("kind")=="host"]
    mem=[r.get("mem_avail_mb") for r in hs]; cpu=[r.get("cpu_pct") for r in hs]; ld=[r.get("load1") for r in hs]
    # 瓶颈
    ver=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="verify" and r.get("aicore_util_pct") is not None]
    noninf=100-st.mean([s.get('infer_frac',0) for s in t5])*100 if t5 else 0
    et=Counter(v.get("error_type") for v in t6) if t6 else Counter()
    items=[]
    if et:
        top=et.most_common(1)[0]; items.append(f"通过率瓶颈 = <b>{top[0]}</b>({100*top[1]/len(t6):.0f}% 验证失败)—— 第一杠杆")
    if rm_all and _q(rm_all,.5)/60000>60: items.append(f"单 session 长:p50 <b>{_q(rm_all,.5)/60000:.0f} min</b>,墙钟主耗在 agent 多轮")
    if allkv and st.mean(allkv)<40: items.append(f"推理算力过剩:KV 仅 <b>{st.mean(allkv):.0f}%</b>,rollout 并发可大幅提高")
    if noninf>25: items.append(f"agent 侧开销 <b>{noninf:.0f}%</b>(验证/工具/等待固有成本)")
    if ver and st.mean(ver)<20: items.append(f"验证池闲置:4 卡 aicore 均值 <b>{st.mean(ver):.0f}%</b>")

    # ===== HTML =====
    P(f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>系统负载一页</title>"
      f"<script>{CH.plotlyjs()}</script><style>{CSS}</style></head><body><div class='slide'>")
    # 头
    P(f"<div class='hd'><h1>ProRL 推理侧系统负载分析</h1>"
      f"<div class='meta'>run <code>{os.path.basename(run_dir.rstrip('/'))}</code> · {len(ses)} session · "
      f"{time.strftime('%m-%d %H:%M',time.localtime(W0))}→{time.strftime('%H:%M',time.localtime(W1))}"
      f"({span_h:.1f}h)· 拓扑:16 NPU = 12 推理(3 引擎×TP4)+ 4 验证</div></div>")
    # KPI
    P("<div class='kpis'>"
      f"<div class='kpi'><b>{stc.get('COMPLETED',0)/len(ses)*100:.0f}%</b><span>goodput</span></div>"
      f"<div class='kpi'><b>{(_q(rm_all,.5) or 0)/60000:.0f} min</b><span>单 session p50</span></div>"
      f"<div class='kpi'><b>{st.mean(allkv):.0f}%</b><span>KV cache 均值</span></div>"
      f"<div class='kpi'><b>{wall_h:.2f} h</b><span>一轮墙钟(64 session)</span></div>"
      f"<div class='kpi'><b>{conc:.1f}</b><span>并发度(≈12 推理卡)</span></div>"
      f"<div class='kpi'><b>{len(set(s['op'] for s in ses))}</b><span>覆盖算子</span></div></div>")
    # 中栏
    P("<div class='mid'>")
    # A 引擎
    P("<div class='col cA'><h3><span class='tag'>A</span>引擎性能(3 × TP4)</h3>")
    P("<table><tr><th>引擎</th><th>吞吐<br>tok/s</th><th>TTFT<br>均s</th><th>ITL<br>ms/tok</th>"
      "<th>KV均<br>%</th><th>wait<br>均</th><th>prefix<br>%</th></tr>")
    for e,tp,ttft,itl,kv,wt,pf in eng_rows:
        P(f"<tr><td>{e}</td><td><b>{tp:.0f}</b></td><td>{ttft:.2f}</td><td>{itl:.1f}</td>"
          f"<td>{kv:.0f}</td><td>{wt:.2f}</td><td>{pf:.0f}</td></tr>")
    P("</table>")
    if len(tps)>1:
        cv=_cv(list(tps.values()))
        P(f"<div class='note'>引擎均衡 <b>CV={cv}</b> — {'不均,LB/DP 分发不平' if cv>0.1 else '三引擎均衡'};"
          f"waiting≈0 且 KV 低 → <b>引擎没喂满</b></div>")
    P("<div class='note'>prefix 命中高 → 长 prompt 的 prefill 大量走缓存,<b>引擎未被 prefill 压垮</b></div>")
    P("</div>")
    # B 时间
    P("<div class='col cB'><h3><span class='tag'>B</span>时间构成(单 session 均值,拆到 prefill/decode)</h3>")
    labels=["总","推理","·prefill","·decode","验证","工具","等待/CPU"]
    parents=["","总","推理","推理","总","总","总"]
    values=[sess_min, inf_min, inf_min*pf_ratio, inf_min*(1-pf_ratio),
            means["验证"]*sess_min, means["工具"]*sess_min, means["等待/CPU"]*sess_min]
    d=CH.sunburst_time(labels, parents, values, None, h=318)
    if d: P(d)
    P(f"<div class='note'>推理内 prefill:decode≈<b>{pf_ratio*100:.0f}:{100-pf_ratio*100:.0f}</b>(引擎侧近似);"
      f"推理外开销 ≈ <b>{100-means['推理']*100:.0f}%</b>(≈{sess_min-inf_min:.0f} min/session)</div>")
    P(f"<div class='wall'><div><b>{wall_h:.2f}h</b><span>一轮墙钟</span></div>"
      f"<div><b>{sum_h:.0f}h</b><span>单 session 累加</span></div>"
      f"<div><b>{conc:.1f}</b><span>并发度</span></div>"
      f"<div><b>{last_p50:.0f}min</b><span>p50(近64)</span></div></div>")
    P("</div>")
    # C 瓶颈
    P("<div class='col cC'><h3><span class='tag'>C</span>瓶颈排序(按影响)</h3>")
    for i,txt in enumerate(items,1):
        P(f"<div class='rank'><span class='rn'>{i}</span><div>{txt}</div></div>")
    P(f"<div class='note'>缩短一轮 = 缩短单 session(p50 {last_p50:.0f}min)<b>或</b>提并发(算力有余量)</div>")
    P("</div></div>")
    # D 三时序
    P("<div class='row'><div class='tris'>")
    kvser={e:_bucket(rows,"vllm:kv_cache_usage_perc",W0,W1,NB,100) for e,rows in t2.items()}
    aiser={eng:_bucket([r for r in t1 if r.get("engine_id")==eng],"aicore_util_pct",W0,W1,NB)
           for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    hser={eng:_bucket([r for r in t1 if r.get("engine_id")==eng],"hbm_used_mb",W0,W1,NB)
          for eng in ["infer-0","infer-1","infer-2","verify-pool"]}
    inf=[r.get("aicore_util_pct") for r in t1 if r["pool"]=="inference" and r.get("aicore_util_pct") is not None]
    hbm=[r.get("hbm_used_mb") for r in t1 if r["pool"]=="inference" and r.get("hbm_used_mb")]
    sat=st.mean(allkv)>70 or st.mean(allwt)>2
    d=_mini_lines(kvser,"KV %",hline=90,hline_txt="near-full")
    P(f"<div class='tri'><h4>KV cache 饱和度</h4>{d}"
      f"<div class='st'>均值 <b>{st.mean(allkv):.1f}%</b>(峰 {max(allkv):.0f}%)— {'接近饱和' if sat else '未饱和,可扩并发'}</div></div>")
    d=_mini_lines(aiser,"aicore %")
    P(f"<div class='tri'><h4>NPU aicore 利用率(推理 3 引擎 + 验证池)</h4>{d}"
      f"<div class='st'>推理池均 <b>{st.mean(inf or [0]):.0f}%</b> · 验证池均 <b>{st.mean(ver or [0]):.0f}%</b>(间歇打满)</div></div>")
    d=_mini_lines(hser,"HBM MB",hline=65536,hline_txt="full")
    P(f"<div class='tri'><h4>显存占用(单卡满 65536MB)</h4>{d}"
      f"<div class='st'>推理卡占满约 <b>{(_stats(hbm)['mean'] if hbm else 0)/65536*100:.0f}%</b> — 权重+激活占大头,KV 有空间</div></div>")
    P("</div></div>")
    # E 底条
    P("<div class='foot'>"
      f"<span>长度:prompt p50 <b>{(_q(pt,.5) or 0)/1000:.0f}k</b> tok · decode p50 <b>{_q(gt,.5) or 0:.0f}</b> tok · "
      f"context&gt;240k 占比 <b>{100*over/max(1,len(ctx)):.1f}%</b> · prefix 命中 p50 <b>{_q(ch,.5) or 0:.0f}%</b></span>"
      f"<span class='sep'>|</span>"
      f"<span>主机:可用内存均 <b>{st.mean(mem)/1024 if mem else 0:.0f} GB</b> · CPU 均 <b>{st.mean(cpu) if cpu else 0:.0f}%</b> · "
      f"load1 均 <b>{st.mean(ld) if ld else 0:.1f}</b> — 充裕,非瓶颈</span>"
      f"<span class='src'>口径:卡负载 5s · 引擎 metrics 差分 · 逐请求日志 · session 时间拆解(source_data/ 可复核)</span>"
      f"</div>")
    P("</div>")
    P("<script>function fit(){var s=document.querySelector('.slide');"
      "var sc=Math.min(window.innerWidth/1920,window.innerHeight/1080);"
      "s.style.transform='scale('+sc+')';}"
      "window.addEventListener('resize',fit);fit();</script>")
    P("</body></html>")
    rep=f"{out}/slide.html"
    open(rep,"w").write("\n".join(R))
    return rep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT",
        "/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/output/ascend_operator/runs"))
    ap.add_argument("--metrics-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR","/mnt/share/polar_engine_metrics"))
    ap.add_argument("--window-sessions", type=int, default=0)
    a=ap.parse_args()
    run=a.run or max(glob.glob(f"{a.runs_root}/*/"), key=os.path.getmtime)
    print(f"幻灯(HTML): {build(run, a.metrics_dir, a.window_sessions)}")

if __name__=="__main__":
    main()
