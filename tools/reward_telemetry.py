#!/usr/bin/env python3
"""Reward 分布 + GRPO 死组遥测(纯读盘,零训练侵入)。

读一个 run 的全部 ses_*.json,给出:
  (a) reward 各档 session 数 + 占比(按 ladder 语义分档);
  (b) 每题组(g######)的组内 reward 方差 → 死组(std≈0)占比,分 raw / 有效
      (排除 INFRA/ERROR 后)两口径;
  (c) 截断惩罚生效面(truncation_penalty>0 的 session 数、扣分量分布);
  (d) fail 段地板决策依据:blank/AST 没过/submission_missing 各有多少、
      它们落在哪些组、降地板能拆开多少死组。

用法:
  python tools/reward_telemetry.py <run_dir> [--json out.json]
其中 <run_dir> = output/ascend_operator/runs/polar_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

_GROUP_RE = re.compile(r"-(g\d{6})-")


def _evaluation(ses: dict[str, Any]) -> dict[str, Any]:
    return (ses.get("trajectory", {}).get("metadata", {}) or {}).get("evaluation") or {}


def _group_key(ses: dict[str, Any]) -> str | None:
    tid = str(ses.get("task_id") or "")
    m = _GROUP_RE.search(tid)
    return m.group(1) if m else None


def _reward_band(ev: dict[str, Any], status: str) -> str:
    """把一条 session 归到人类可读的档(与 operator_reward ladder 对齐)。"""
    if status in ("ERROR", "TIMEOUT"):
        return "infra/error(不计分)"
    m = ev.get("metrics") or {}
    success = bool(m.get("success", ev.get("success", False)))
    if success:
        sp = (ev.get("speedup_vs_torch")
              or (m.get("perf_data") or {}).get("speedup_vs_torch"))
        try:
            sp = float(sp)
        except (TypeError, ValueError):
            sp = 1.0
        if sp >= 2.0:
            return "success >=2x"
        if sp >= 1.0:
            return "success 1-2x"
        return "success <1x"
    et = str(m.get("error_type") or ev.get("error_type") or "")
    ast_ok = bool(m.get("ast_check_ok", False))
    corr_ok = bool(m.get("correctness_ok", False))
    if corr_ok:
        return "fail: correctness_ok(0.4)"
    if not ast_ok:
        if et == "submission_missing":
            return "fail: submission_missing(0.2)"
        return "fail: AST没过(0.2)"
    if et == "ascendc_compile_failed":
        return "fail: 编译没过(0.25)"
    if et in ("op_not_registered", "ascendc_run_crashed"):
        return "fail: 崩溃/未注册(0.3)"
    if et in ("correctness_failed", "output_precheck_failed"):
        return "fail: 精度/输出错(0.35)"
    return "fail: 其他(0.3)"


def load_run(run_dir: str) -> list[dict[str, Any]]:
    rows = []
    for f in glob.glob(os.path.join(run_dir, "**", "ses_*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = _evaluation(d)
        rows.append({
            "session_id": d.get("session_id"),
            "status": d.get("status"),
            "group": _group_key(d),
            "reward": ev.get("reward"),
            "band": _reward_band(ev, str(d.get("status") or "")),
            "trunc_penalty": ev.get("truncation_penalty") or 0.0,
            "trunc_events": ev.get("truncation_events") or 0,
            "error_type": ev.get("error_type"),
        })
    return rows


def _std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    bands = Counter(r["band"] for r in rows)

    # 组内方差 / 死组(两口径:raw = 全部;有效 = 排除 infra/error)
    groups_raw: dict[str, list[float]] = defaultdict(list)
    groups_eff: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["group"] is None or r["reward"] is None:
            continue
        groups_raw[r["group"]].append(float(r["reward"]))
        if r["status"] not in ("ERROR", "TIMEOUT"):
            groups_eff[r["group"]].append(float(r["reward"]))

    def dead_stats(groups: dict[str, list[float]]) -> dict[str, Any]:
        multi = {g: v for g, v in groups.items() if len(v) > 1}
        dead = {g: v for g, v in multi.items() if _std(v) < 1e-6}
        return {
            "n_groups_multi": len(multi),
            "n_dead": len(dead),
            "dead_frac": round(len(dead) / len(multi), 3) if multi else 0.0,
            "dead_groups": {g: round(v[0], 3) for g, v in sorted(dead.items())},
        }

    # 反事实:把 blank/AST没过/submission_missing 的 0.2 降到 0.0,能新拆开几个死组
    floor_bands = {"fail: submission_missing(0.2)", "fail: AST没过(0.2)"}
    revived = 0
    for g, v in groups_eff.items():
        if len(v) <= 1 or _std(v) >= 1e-6:
            continue
        members = [r for r in rows if r["group"] == g and r["status"] not in ("ERROR", "TIMEOUT")]
        cf = [0.0 if r["band"] in floor_bands else float(r["reward"]) for r in members]
        if _std(cf) >= 1e-6:
            revived += 1

    trunc_hit = [r for r in rows if r["trunc_penalty"] > 0]
    return {
        "n_sessions": n,
        "bands": dict(bands.most_common()),
        "dead_raw": dead_stats(groups_raw),
        "dead_effective": dead_stats(groups_eff),
        "floor_counterfactual": {
            "note": "把 0.2 floor 档降到 0.0 后,原死组里能新拆开的数量",
            "revived_dead_groups": revived,
        },
        "truncation_penalty": {
            "n_sessions_penalized": len(trunc_hit),
            "penalty_frac": round(len(trunc_hit) / n, 3) if n else 0.0,
            "events_hist": dict(Counter(r["trunc_events"] for r in trunc_hit).most_common()),
        },
    }


def _fmt(report: dict[str, Any]) -> str:
    L = [f"总 session: {report['n_sessions']}", "", "== reward 分档 =="]
    for band, c in report["bands"].items():
        pct = 100 * c / max(report["n_sessions"], 1)
        L.append(f"  {band:32} {c:4}  ({pct:4.1f}%)")
    for key, title in [("dead_raw", "死组(raw,含 infra)"), ("dead_effective", "死组(有效,排除 infra/error)")]:
        d = report[key]
        L += ["", f"== {title} ==",
              f"  多样本组: {d['n_groups_multi']}  死组: {d['n_dead']}  占比: {d['dead_frac']}"]
        if d["dead_groups"]:
            L.append(f"  死组明细(组:同分值): {d['dead_groups']}")
    cf = report["floor_counterfactual"]
    L += ["", "== fail 地板反事实(0.2→0.0) ==",
          f"  {cf['note']}: {cf['revived_dead_groups']} 个"]
    tp = report["truncation_penalty"]
    L += ["", "== 截断惩罚生效面 ==",
          f"  被扣 session: {tp['n_sessions_penalized']} ({100*tp['penalty_frac']:.1f}%)  截断次数分布: {tp['events_hist']}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None, help="额外落一份机器可读 json")
    args = ap.parse_args()
    rows = load_run(args.run_dir)
    if not rows:
        print(f"no ses_*.json under {args.run_dir}")
        raise SystemExit(1)
    report = analyze(rows)
    print(_fmt(report))
    if args.json:
        json.dump(report, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
