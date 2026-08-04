#!/usr/bin/env python3
"""报告用 PNG 图表(matplotlib)。缺库时函数返回 None,报告侧自动降级为文字。

所有函数把图存到 out_dir/charts/<name>.png,返回相对路径 'charts/<name>.png' 供 md 用 ![](...) 引用。
"""
from __future__ import annotations
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    _OK = True
    # 中文字体:有则用,无则英文标签(避免豆腐块)
    _CJK = None
    for _f in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf"):
        pass
    for _cand in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Source Han Sans SC"):
        try:
            font_manager.findfont(_cand, fallback_to_default=False); _CJK = _cand; break
        except Exception:
            continue
    if _CJK:
        plt.rcParams["font.sans-serif"] = [_CJK]; plt.rcParams["axes.unicode_minus"] = False
except Exception:
    _OK = False
    _CJK = None

_HAS_CJK = bool(globals().get("_CJK"))

def available():
    return _OK

def _lbl(cn, en):
    """有中文字体用中文,否则英文。"""
    return cn if _HAS_CJK else en

def _prep(out_dir):
    d = os.path.join(out_dir, "charts"); os.makedirs(d, exist_ok=True); return d

def pie_time_breakdown(out_dir, fracs: dict, title=None):
    """一轮 rollout 时间占比饼图。fracs={'推理':0.7,'验证':0.07,...}(0-1)。"""
    if not _OK: return None
    d = _prep(out_dir); path = os.path.join(d, "time_breakdown_pie.png")
    items = [(k, v) for k, v in fracs.items() if v and v > 0]
    if not items: return None
    labels = [k for k, _ in items]; vals = [v for _, v in items]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    wedges, _t, _a = ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.1f}%",
                            colors=colors[:len(vals)], startangle=90,
                            wedgeprops=dict(width=0.55, edgecolor="white"),
                            pctdistance=0.75, textprops=dict(fontsize=11))
    ax.set_title(title or _lbl("一轮 rollout 时间占比(墙钟)", "Rollout Wall-Clock Breakdown"),
                 fontsize=13, weight="bold")
    ax.axis("equal")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return "charts/time_breakdown_pie.png"

def lines_timeseries(out_dir, name, series: dict, xlabel, ylabel, title):
    """多线时序。series={label:[(t_min,val),...]}。"""
    if not _OK: return None
    d = _prep(out_dir); path = os.path.join(d, name + ".png")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for lab, pts in series.items():
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.3, label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(alpha=.3); ax.legend(fontsize=9, ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return "charts/" + name + ".png"

def hist(out_dir, name, vals, xlabel, title, bins=40, vline=None, vline_label=""):
    if not _OK: return None
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs: return None
    d = _prep(out_dir); path = os.path.join(d, name + ".png")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(xs, bins=bins, color="#4C72B0", edgecolor="white", alpha=.85)
    if vline is not None:
        ax.axvline(vline, color="#C44E52", ls="--", lw=1.5, label=vline_label or f"limit {vline}")
        ax.legend(fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(_lbl("请求数", "count")); ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return "charts/" + name + ".png"

def barh_ops(out_dir, name, pairs, xlabel, title):
    """水平条形(per-op 通过率等)。pairs=[(label,val),...] 已排序。"""
    if not _OK: return None
    if not pairs: return None
    d = _prep(out_dir); path = os.path.join(d, name + ".png")
    labels = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, max(3, len(pairs) * 0.28)))
    colors = ["#C44E52" if v < 50 else "#DD8452" if v < 80 else "#55A868" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return "charts/" + name + ".png"

def bars_simple(out_dir, name, pairs, ylabel, title):
    """竖直条形(error_type / 引擎吞吐)。"""
    if not _OK: return None
    if not pairs: return None
    d = _prep(out_dir); path = os.path.join(d, name + ".png")
    labels = [str(p[0]) for p in pairs]; vals = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(vals)), vals, color="#4C72B0", edgecolor="white")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(alpha=.3, axis="y")
    for i, v in enumerate(vals): ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return "charts/" + name + ".png"
