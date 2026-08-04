#!/usr/bin/env python3
"""plotly HTML 图表:各图返回 <div>(不含 plotly.js),由报告头部统一内联一次。

风格:plotly_white 模板 + 统一配色。中文可直接用(浏览器字体)。
"""
from __future__ import annotations
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    _OK = True
except Exception:
    _OK = False

C = ["#3b6fb0", "#6ea8d8", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]
_LAYOUT = dict(template="plotly_white", font=dict(size=13, family="Arial, 'Microsoft YaHei', sans-serif"),
               margin=dict(l=60, r=30, t=50, b=50), title_font=dict(size=16))

def available(): return _OK
def plotlyjs(): return get_plotlyjs() if _OK else ""

def _div(fig, h=380):
    fig.update_layout(**_LAYOUT, height=h)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": True, "responsive": True})

def sunburst_time(labels, parents, values, title):
    """双层时间占比:推理→prefill/decode + 验证/工具/等待。values 为分钟。"""
    if not _OK: return None
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total",
                                 marker=dict(colors=C[:len(labels)]),
                                 textinfo="label+percent root",
                                 hovertemplate="%{label}<br>%{value:.1f} min<br>%{percentRoot:.1%}<extra></extra>"))
    fig.update_layout(title=title)
    return _div(fig, 460)

def pie(labels, values, title, unit="min"):
    if not _OK: return None
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.5, marker=dict(colors=C[:len(labels)]),
                           textinfo="label+percent",
                           hovertemplate="%{label}<br>%{value:.1f} "+unit+"<br>%{percent}<extra></extra>"))
    fig.update_layout(title=title)
    return _div(fig, 420)

def lines(series, xlabel, ylabel, title, hline=None, hline_txt=""):
    """series={label:[(x,y),...]}"""
    if not _OK: return None
    fig = go.Figure()
    for i, (lab, pts) in enumerate(series.items()):
        if not pts: continue
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                 name=lab, line=dict(width=2, color=C[i % len(C)]), marker=dict(size=4)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#c44e52", annotation_text=hline_txt)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode="x unified")
    return _div(fig)

def hist(vals, xlabel, title, vline=None, vline_txt="", nbins=50):
    if not _OK: return None
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs: return None
    fig = go.Figure(go.Histogram(x=xs, nbinsx=nbins, marker_color="#3b6fb0"))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color="#c44e52", annotation_text=vline_txt)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="count", bargap=.02)
    return _div(fig)

def barh(pairs, xlabel, title, colorfn=None):
    """pairs=[(label,val)] 已排序;colorfn(val)->color。"""
    if not _OK: return None
    if not pairs: return None
    labels = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    colors = [colorfn(v) for v in vals] if colorfn else ["#3b6fb0"] * len(vals)
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.0f}" for v in vals], textposition="auto"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis=dict(autorange="reversed"))
    return _div(fig, max(360, len(pairs) * 26))

def bars(pairs, ylabel, title):
    if not _OK: return None
    if not pairs: return None
    fig = go.Figure(go.Bar(x=[str(p[0]) for p in pairs], y=[p[1] for p in pairs],
                           marker_color="#3b6fb0", text=[str(int(p[1])) for p in pairs], textposition="auto"))
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
