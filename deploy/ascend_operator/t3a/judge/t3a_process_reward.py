#!/usr/bin/env python3
"""[R5] t3a attempt stream → 过程分(process reward)合成器,与 operator_reward 同尺度(±0.10)。

输入: t3a_attempt_stream.jsonl(evaluate 判决 + promote 事件)
输出: process_reward.json —— per-attempt credit 序列 + 汇总,供 operator_reward 通道消费。

映射(±0.10 带宽内,与 t2a 过程奖励同尺度):
  promote(前进)            : +0.06
  PASS(全 case)            : +0.04
  A 类(编译/崩溃)           : -0.03
  D 类(精度不匹配,比对拍没跑完强): -0.01
  同签名连续第 2 次起        : 每次额外 -0.02(连撞递减,信号直指 79% 同错连撞)
  无源码变化重跑(dedup 式)   : -0.04
单项 credit 截断到 ±0.10;汇总(total)截断到 ±0.10(与 t2a 的 process reward 带宽一致)。

一致性核对(V3 平移):stream 里 PASS 的快照若被 judge 判挂,该次 credit 作废并记 anomaly。
"""

from __future__ import annotations

import json
import sys

CLAMP = 0.10

# [F5] 只有四个评测脚本的判决产生 credit;build/validate 等官方流程的正常辅助步骤
# 记中性事件(credit 0)——照流程走路不该被罚(原映射会把一次正常编译记成 -0.07)。
_EVAL_SCRIPTS = frozenset({
    "evaluate_ascendc.sh", "verification_ascendc.py",
    "evaluate_tilelang.sh", "verification_tilelang.py",
})


def _clamp(x: float) -> float:
    return max(-CLAMP, min(CLAMP, x))


def synthesize(stream_path: str, judge_metrics: dict | None = None) -> dict:
    attempts: list[dict] = []
    events: list[dict] = []
    with open(stream_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    credits: list[dict] = []
    streak_sig = None
    streak_n = 0
    last_was_eval = False
    for ev in events:
        if ev.get("event") == "promote":
            # promote 只在该进步由评测 attempt 触发时计分
            c = 0.06 if last_was_eval else 0.0
            credits.append({"event": "promote", "credit": c, "ts": ev.get("ts")})
            continue
        script = ev.get("script") or ""
        if script not in _EVAL_SCRIPTS:
            credits.append({"event": "helper", "script": script, "credit": 0.0, "ts": ev.get("ts")})
            last_was_eval = False
            continue
        last_was_eval = True
        cls = ev.get("classification")
        if cls == "PASS":
            c = 0.04
        elif cls == "D":
            c = -0.01
        else:  # A 类及其他
            c = -0.03
        # 同签名连撞递减
        sig = (cls, bool(ev.get("case_total")))
        if sig == streak_sig:
            streak_n += 1
            if streak_n >= 2:
                c -= 0.02
        else:
            streak_sig, streak_n = sig, 1
        if ev.get("exit_code") == 0 and cls is None:
            c -= 0.04  # dedup 式重跑(无判决但零退出)
        credits.append({
            "event": "attempt",
            "classification": cls,
            "case_pass": ev.get("case_pass"),
            "case_total": ev.get("case_total"),
            "credit": round(c, 4),
            "ts": ev.get("ts"),
        })

    anomalies: list[str] = []
    if judge_metrics is not None:
        judge_ok = bool(judge_metrics.get("correctness_ok") or judge_metrics.get("success"))
        if not judge_ok:
            for c in credits:
                if c.get("classification") == "PASS" and c.get("credit", 0) > 0:
                    anomalies.append(
                        f"PASS attempt at ts={c.get('ts')} but judge rejected the artifact; credit voided")
                    c["credit"] = 0.0
                    c["anomaly"] = "judge_rejected"

    total = _clamp(sum(c.get("credit", 0.0) for c in credits))
    return {
        "schema_version": 1,
        "source": stream_path,
        "per_attempt": credits,
        "total": round(total, 4),
        "band": CLAMP,
        "anomalies": anomalies,
        "counts": {
            "attempts": sum(1 for c in credits if c.get("event") == "attempt"),
            "promotes": sum(1 for c in credits if c.get("event") == "promote"),
        },
    }


if __name__ == "__main__":
    stream = sys.argv[1]
    metrics = None
    if len(sys.argv) > 2:
        metrics = json.load(open(sys.argv[2], encoding="utf-8"))
    out = synthesize(stream, metrics)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
