#!/usr/bin/env python3
"""plotly HTML 图表:各图返回 <div>(不含 plotly.js),由报告头部统一内联一次。

风格:现代浅色仪表盘 —— 统一配色、细网格、左对齐小标题、白色分隔。中文可直接用(浏览器字体)。
"""
from __future__ import annotations
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    _OK = True
except Exception:
    _OK = False

# 现代仪表盘配色(蓝→青→琥珀→绿→红→紫→橙→青绿)
C = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#f97316", "#14b8a6"]

def set_palette(colors):
    """切换全局图表配色(v2 多方案对比用)。colors: 8 色 list。"""
    global C
    C = list(colors)
_GRID = "#edf1f6"
_LAYOUT = dict(template="plotly_white",
               font=dict(size=13, family="Inter, 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif", color="#334155"),
               margin=dict(l=56, r=24, t=44, b=44),
               paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
               hoverlabel=dict(bgcolor="#1e293b", font_color="#fff", font_size=13, bordercolor="#1e293b"),
               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=12)))
_CONFIG = {"displaylogo": False, "responsive": True,
           "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

def available(): return _OK
def plotlyjs(): return get_plotlyjs() if _OK else ""

def _div(fig, h=380):
    # 图内标题一律省略:外层 panel 的 h4 已承担标题职责,顶部空间留给图例,避免重叠
    fig.update_layout(**_LAYOUT, height=h, title=None)
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, linecolor="#dbe2ec",
                     showspikes=True, spikecolor="#94a3b8", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, linecolor="#dbe2ec")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config=_CONFIG)

def sunburst_time(labels, parents, values, title, h=460):
    """双层时间占比:推理→prefill/decode + 验证/工具/等待。values 为分钟。"""
    if not _OK: return None
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total",
                                 marker=dict(colors=C[:len(labels)], line=dict(color="#ffffff", width=2)),
                                 textinfo="label+percent root",
                                 hovertemplate="%{label}<br>%{value:.1f} min<br>%{percentRoot:.1%}<extra></extra>"))
    fig.update_layout(title=title)
    return _div(fig, h)

def pie(labels, values, title, unit="min"):
    if not _OK: return None
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.55,
                           marker=dict(colors=C[:len(labels)], line=dict(color="#ffffff", width=2)),
                           textinfo="label+percent",
                           hovertemplate="%{label}<br>%{value:.1f} "+unit+"<br>%{percent}<extra></extra>"))
    fig.update_layout(title=title)
    return _div(fig, 420)

def lines(series, xlabel, ylabel, title, hline=None, hline_txt="", h=380):
    """series={label:[(x,y),...]}"""
    if not _OK: return None
    fig = go.Figure()
    for i, (lab, pts) in enumerate(series.items()):
        if not pts: continue
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                                 name=lab, line=dict(width=2.4, color=C[i % len(C)]),
                                 hovertemplate="%{y:.1f}<extra>"+lab+"</extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#ef4444", line_width=1.5,
                      annotation_text=hline_txt, annotation_font_color="#ef4444")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode="x unified",
                      showlegend=sum(1 for p in series.values() if p) > 1)
    return _div(fig, h)

def hist(vals, xlabel, title, vline=None, vline_txt="", nbins=50, h=380):
    if not _OK: return None
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs: return None
    fig = go.Figure(go.Histogram(x=xs, nbinsx=nbins,
                                 marker=dict(color="#2563eb", opacity=0.82, line=dict(color="#ffffff", width=0.6))))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color="#ef4444", line_width=1.5,
                      annotation_text=vline_txt, annotation_font_color="#ef4444")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="count", bargap=.05)
    return _div(fig, h)

def barh(pairs, xlabel, title, colorfn=None):
    """pairs=[(label,val)] 已排序;colorfn(val)->color。"""
    if not _OK: return None
    if not pairs: return None
    labels = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    colors = [colorfn(v) for v in vals] if colorfn else [C[0]] * len(vals)
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker=dict(color=colors, opacity=0.88),
                           text=[f"{v:.0f}" for v in vals], textposition="auto",
                           textfont=dict(color="#0f172a")))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis=dict(autorange="reversed"))
    return _div(fig, max(360, len(pairs) * 26))

def bars(pairs, ylabel, title):
    if not _OK: return None
    if not pairs: return None
    fig = go.Figure(go.Bar(x=[str(p[0]) for p in pairs], y=[p[1] for p in pairs],
                           marker=dict(color=C[0], opacity=0.88),
                           text=[str(int(p[1])) for p in pairs], textposition="auto",
                           textfont=dict(color="#0f172a")))
    fig.update_layout(title=title, yaxis_title=ylabel)
    return _div(fig)

def grouped_bars(cats, groups, ylabel, title):
    """cats=x轴类别; groups={name:[vals按cats]}"""
    if not _OK: return None
    fig = go.Figure()
    for i, (name, vals) in enumerate(groups.items()):
        fig.add_trace(go.Bar(name=name, x=cats, y=vals, marker_color=C[i % len(C)],
                             text=[f"{v:.0f}" for v in vals], textposition="auto"))
    fig.update_layout(title=title, yaxis_title=ylabel, barmode="group")
    return _div(fig)
