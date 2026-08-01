#!/usr/bin/env python3
"""T5–T8 常驻聚合 daemon:自动扫最新 run_dir,周期重建四张派生表落文件(不再手工跑离线脚本)。

每 --interval 秒:
  - T5 rollout_span:每 session 的时间拆解(复用 build_spans.process_session)→ rollout_span.jsonl + span.jsonl
  - T6 verify_job:每次算子验证(npu_lease_status.*.json 的 lease_wait/exec + metrics.json 结果)→ verify_job.jsonl
  - T7 rollout:每 session 汇总(ses_*.json:timing/reward/tokens + span 派生 infer/verify/tool/wait_frac)→ rollout.jsonl
  - T8 rollout_step:每 step 聚合(goodput/长度分位/engine 均衡[从 T4 engine-*.jsonl])→ rollout_step.jsonl
全部覆盖写到 <run_dir>/telemetry_derived/。run_dir 自动取 --runs-root 下最新含 rollout_results 的目录。

用法:
  python3 telemetry_aggregator.py --runs-root /mnt/share/.../runs --interval 30
  python3 telemetry_aggregator.py --run-dir <run_dir> --once
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_spans  # noqa: E402  复用 process_session / _lease_refine


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return time.monotonic()


def _dist(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "min": xs[0], "p50": q(.50), "p90": q(.90), "p99": q(.99),
            "max": xs[-1], "mean": round(st.mean(xs), 1)}


def _load_jsonl(pattern):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        for line in open(f, errors="replace"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _pick_run_dir(runs_root):
    cands = []
    for d in glob.glob(os.path.join(runs_root, "*")):
        if os.path.isdir(os.path.join(d, "rollout_results")) or glob.glob(os.path.join(d, "**", "ses_*.json"), recursive=True):
            try:
                cands.append((os.path.getmtime(d), d))
            except Exception:
                pass
    return max(cands)[1] if cands else None


def _session_spans(run_dir):
    """{session_id: span_summary} + 全 span 列表。复用 build_spans。"""
    summaries, all_spans = {}, []
    sdirs = [d for d in glob.glob(os.path.join(run_dir, "polar_sessions", "**", "session-*"), recursive=True)
             if os.path.isdir(d)]
    for sd in sdirs:
        sid = os.path.basename(sd).replace("session-", "")
        ts = glob.glob(os.path.join(sd, ".claude", "**", "*.jsonl"), recursive=True)
        if not ts:
            continue
        transcript = max(ts, key=lambda p: os.path.getsize(p))
        summary, spans = build_spans.process_session(transcript, sid, sd)
        if summary:
            summaries[sid] = summary
            all_spans.extend(spans)
    return summaries, sdirs, all_spans


def _verify_jobs(run_dir, sdirs):
    """T6:每个验证结果 metrics.json 一行(验证池负载 + 结果);同 session 有 npu_lease_status 则附 lease_wait/exec。"""
    jobs = []
    for sd in sdirs:
        sid = os.path.basename(sd).replace("session-", "")
        # lease 计时(带补丁的 operator_runtime 才有)
        leases = []
        for f in glob.glob(os.path.join(sd, "**", "npu_lease_status.*.json"), recursive=True):
            try:
                d = json.load(open(f))
                leases.append(d)
            except Exception:
                pass
        lw = next((d.get("wait_seconds") for d in leases if d.get("wait_seconds") is not None), None)
        ex = next((d.get("exec_seconds") for d in leases if d.get("exec_seconds") is not None), None)
        card = next((d.get("device_id") for d in leases if d.get("device_id") is not None), None)
        for f in glob.glob(os.path.join(sd, "**", "metrics.json"), recursive=True):
            try:
                mj = json.load(open(f))
            except Exception:
                continue
            perf = mj.get("perf_data") or {}
            jobs.append({
                "session_id": sid, "op_name": mj.get("op_name"),
                "verify_card_id": card, "lease_wait_s": lw, "exec_s": ex,
                "ast_check_ok": mj.get("ast_check_ok"), "correctness_ok": mj.get("correctness_ok"),
                "success": mj.get("success"),
                "speedup": perf.get("speedup") if isinstance(perf, dict) else None,
                "error_type": mj.get("error_type"),
            })
    return jobs


def _rollouts(run_dir, spans_by_sid):
    """T7:每 ses_*.json 一行 + 关联 span 派生占比。"""
    ses = _load_jsonl(os.path.join(run_dir, "rollout_results", "**", "ses_*.json"))
    rows = []
    for s in ses:
        md = (s.get("trajectory", {}) or {}).get("metadata", {}) or {}
        timing = s.get("timing", {}) or {}
        sid = s.get("session_id") or md.get("session_id")
        sp = spans_by_sid.get(str(sid), {})          # 短 slug vs 长 UUID → 多数对不上,best-effort
        ev = md.get("evaluation", {}) or {}          # reward 在 trajectory.metadata.evaluation
        rows.append({
            "session_id": sid, "task_id": s.get("task_id"), "rollout_step": md.get("rollout_step"),
            "group_id": md.get("group_id"), "node_id": s.get("node_id"),
            "status": s.get("status"), "error": s.get("error"),
            "policy_version": md.get("policy_version"), "op_name": (md.get("task_metadata") or {}).get("op_name"),
            "register_ms": timing.get("register_to_init_queue_ms"), "init_ms": timing.get("init_ms"),
            "run_ms": timing.get("run_ms"), "postrun_ms": timing.get("postrun_ms"),
            "num_turns": sp.get("n_turns") or md.get("trace_count"),   # 优先 span,回落 ses trace_count
            "reward": ev.get("reward"), "outcome_reward": ev.get("outcome_reward"),
            "inference_ms": sp.get("inference_ms"), "verify_ms": sp.get("verify_ms"),
            "tool_ms": sp.get("tool_ms"), "wait_cpu_ms": sp.get("wait_cpu_ms"),
            "infer_frac": sp.get("infer_frac"),
            "noninfer_frac": (round(1 - sp["infer_frac"], 4) if isinstance(sp.get("infer_frac"), (int, float)) else None),
        })
    return rows


def _engine_balance(run_dir, engine_dir):
    """T8 engine 均衡:优先用 T4 逐请求 engine-*.jsonl 的请求数/生成 token 份额 + CV。"""
    eng = _load_jsonl(os.path.join(engine_dir, "*.jsonl")) if engine_dir else []
    if not eng:
        return None
    cnt = Counter(r.get("engine_id") for r in eng if r.get("engine_id"))
    tok = defaultdict(int)
    for r in eng:
        tok[r.get("engine_id")] += int(r.get("num_generation_tokens") or 0)
    counts = [v for k, v in cnt.items() if k]
    cv = round(st.pstdev(counts) / st.mean(counts), 3) if len(counts) > 1 and st.mean(counts) else 0.0
    return {"requests_per_engine": dict(cnt), "gen_tokens_per_engine": dict(tok), "request_cv": cv}


def _rollout_steps(rollouts, run_dir, engine_dir, summaries):
    """T8:按 rollout_step 聚合。block④ 占比因 session 命名不通用无法 per-step join → 给全局分布。"""
    by_step = defaultdict(list)
    for r in rollouts:
        by_step[r.get("rollout_step")].append(r)
    bal = _engine_balance(run_dir, engine_dir)  # 全局(T4 逐请求无 rollout_step,暂全局)
    sp = list(summaries.values())
    block4 = {  # 全局(所有 polar_session span 的时间拆解占比)
        "infer_frac": _dist([s.get("infer_frac") for s in sp]),
        "verify_frac": _dist([s.get("verify_frac") for s in sp]),
        "tool_frac": _dist([s.get("tool_frac") for s in sp]),
        "wait_cpu_frac": _dist([s.get("wait_cpu_frac") for s in sp]),
    } if sp else None
    out = []
    for step, rs in by_step.items():
        stat = Counter(r.get("status") for r in rs)
        n = len(rs)
        comp = stat.get("COMPLETED", 0)
        ab = stat.get("ERROR", 0) + stat.get("TIMEOUT", 0) + stat.get("ABORTED", 0)
        out.append({
            "rollout_step": step, "num_sessions": n, "status": dict(stat),
            "num_completed": comp, "num_aborted": ab,
            "goodput": round(comp / n, 3) if n else None,
            "reward_dist": _dist([r.get("reward") for r in rs]),
            "run_ms_dist": _dist([r.get("run_ms") for r in rs]),
            "num_turns_dist": _dist([r.get("num_turns") for r in rs]),
            "engine_balance": bal,
            "block4_frac_global": block4,
        })
    return out


def _write(out_dir, name, rows):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def build_once(run_dir, engine_dir):
    out_dir = os.path.join(run_dir, "telemetry_derived")
    os.makedirs(out_dir, exist_ok=True)
    summaries, sdirs, all_spans = _session_spans(run_dir)
    n5 = _write(out_dir, "rollout_span.jsonl", list(summaries.values()))
    _write(out_dir, "span.jsonl", all_spans)
    n6 = _write(out_dir, "verify_job.jsonl", _verify_jobs(run_dir, sdirs))
    rollouts = _rollouts(run_dir, summaries)
    n7 = _write(out_dir, "rollout.jsonl", rollouts)
    n8 = _write(out_dir, "rollout_step.jsonl", _rollout_steps(rollouts, run_dir, engine_dir, summaries))
    return out_dir, n5, n6, n7, n8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.environ.get("POLAR_RUNS_ROOT"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--engine-dir", default=os.environ.get("POLAR_ENGINE_METRICS_DIR",
                    "/mnt/share/polar_engine_metrics"))
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if not args.run_dir and not args.runs_root:
        raise SystemExit("需 --run-dir <dir> 或 --runs-root <runs 父目录>(或设 POLAR_RUNS_ROOT)")

    while True:
        run_dir = args.run_dir or _pick_run_dir(args.runs_root)
        if not run_dir:
            print(f"[aggregator] 暂无 run(扫 {args.runs_root}),{args.interval}s 后重试", flush=True)
        else:
            try:
                out_dir, n5, n6, n7, n8 = build_once(run_dir, args.engine_dir)
                print(f"[aggregator] {time.strftime('%H:%M:%S')} {os.path.basename(run_dir)} → "
                      f"span={n5} verify={n6} rollout={n7} step={n8} @ {out_dir}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[aggregator] err: {e}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
