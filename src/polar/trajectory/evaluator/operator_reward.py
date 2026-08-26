"""Operator-gen reward kernel (harness-agnostic): judge metrics.json -> rllm-facing outcome.

Reused from the user's openhands_sdk (triton_eval_pipeline.sh `metrics.json` schema_version 2 +
openhands_agent._reward_from_metrics ladder), with the key TRAINING-CORRECTNESS addition:

  INFRA failures (judge container / task-setup broke) are NOT scored 0 — they map to a RETRY
  signal (status=ERROR), so infra flakes do not poison training with false negatives. This is the
  same discipline as 'no zero-fill' on logprobs: never fabricate a bad signal from an infra failure.
  OPERATOR failures (the agent's kernel is bad/missing) get the real 0..1.0 reward ladder.

Pure + dependency-free; unit-tested standalone (`python operator_reward.py`) or via pytest.
The Polar `operator_judge` evaluator imports `judge_outcome` to turn judge metrics into
(session status, reward) — ERROR => rllm retries; COMPLETED => reward used.
"""

from __future__ import annotations

import math
import os

# error_type values that mean the eval COULDN'T RUN for infra/setup reasons (retry, do not score).
# Everything else (ast_check_failed / correctness_failed / *compile* / *lowering* / ub_overflow /
# benchmark_failed / implementation_missing / submission_missing) is the AGENT's fault -> real signal.
INFRA_ERROR_TYPES = frozenset({
    "task_missing",              # the task spec file isn't there -> setup issue
    "input_load_failed",         # get_input / *.json not found -> setup issue
    "judge_container_failed",    # the judge container itself crashed
    "judge_metrics_unreadable",  # judge ran but metrics.json corrupt
    "judge_no_metrics",          # judge produced no metrics.json
    "npu_runtime_unavailable",   # Ascend runtime/device visibility failed before operator code ran
    "submission_fetch_failed",   # the artifact existed but transport failed (source container gone,
                                 # connection/timeout) -> can't distinguish from a good run being
                                 # cut off; retry instead of scoring the agent 0.2 for our plumbing.
                                 # NOTE: plain "submission_missing" (file genuinely absent) stays
                                 # the AGENT's fault and keeps its real 0.2 signal.
})


def classify_infra_error_text(text: str | None) -> str | None:
    """Classify known infra failures from the full judge error log.

    The Triton pipeline wraps verify failures with "correctness" wording, so an Ascend init
    failure can otherwise be mislabeled as ``correctness_failed`` even though no submitted operator
    code ran. Keep this intentionally narrow: only device/runtime-init signatures map to infra.
    """
    if not text:
        return None
    lowered = str(text).lower().replace(" ", "")
    spaced = str(text).lower()
    if (
        "aclinit" in lowered
        and (
            "invaliddeviceid" in lowered
            or "deviceiderror" in lowered
            or "getdevicecntfailed" in lowered
            or "resource_busy" in lowered
            or "rtgetdevmsgexecutionfailed" in lowered
        )
    ):
        return "npu_runtime_unavailable"
    if "ptacallaclapifailed" in lowered and "invaliddeviceid" in lowered:
        return "npu_runtime_unavailable"
    if "inputerrordeviceid" in lowered or (
        "invalid device id" in spaced and "torch_npu/csrc/core/npu/sys_ctrl" in spaced
    ):
        return "npu_runtime_unavailable"
    return None


# --------------------- 用例通过率旋钮(correctness 失败档)---------------------
#
# correctness_failed / output_precheck_failed(对拍跑完但未全过)不再是固定分:
#   reward = 0.3 + α × (cases_passed / cases_total)
# 分子分母来自 judge fresh 容器对数据集原版用例的实测(inject_baseline 覆盖工程副本,
# agent 无法操纵);只对「对拍跑完」的两档生效 —— 崩溃/编译挂时对拍未完成,case 统计
# 不可信,不给通过率分。档内渐近 0.4 不触碰:全过即离开本档(correctness_ok=true),
# 0.4 恒留给「全对但 benchmark 挂」。统计缺失(旧格式/脚本被杀)回退 0.35 = 旧固定档。

POLAR_CASE_PASS_WEIGHT_ENV = "POLAR_CASE_PASS_WEIGHT"   # α,默认 0.10;<=0 回退固定 0.35
CASE_PASS_FALLBACK = 0.35


def case_pass_weight(env: dict | None = None) -> float:
    source = os.environ if env is None else env
    try:
        return float(source.get(POLAR_CASE_PASS_WEIGHT_ENV, "0.10") or "0.10")
    except (TypeError, ValueError):
        return 0.10


def reward_from_metrics(metrics: dict, env: dict | None = None) -> float:
    """Authoritative ladder(2026-08 版:失败侧 [0, 0.4],correctness 失败档接入通过率).

    not success:  correctness_ok -> 0.4
                  ast_check_ok  -> 按 error_type 细分「编译→能跑→跑完」的进度:
                      ascendc_compile_failed                 -> 0.1  (AST 过、编译没过)
                      op_not_registered / ascendc_run_crashed -> 0.2  (编译过但没能有效跑完)
                      correctness_failed / output_precheck_failed
                        -> 0.3 + α×用例通过率  (对拍跑完、结果不对;统计缺失回退 0.35)
                      其他(阶段挂但类型未知)                 -> 0.25 (兜底中间档)
                  else          -> 0.0  (AST 没过 / 没调真 op:没产出真算子,记 0)
    success:      0.75 + 0.25*tanh(ln speedup)   # 0.5(<-0x) .. 0.75(1x hold) .. ->1.0(soft, no cap)

    档间距拉大的动机:失败侧从 [0.2,0.4] 扩到 [0,0.4],且 0.35 固定档改 10 刻度连续档
    (10-case 集),消灭「挂 1 个 case 与全挂同分」的组内零方差 dead group。(0.4, 0.5)
    空挡不动:任何「未全过」严格劣于「全过但 benchmark 挂」,正确性门控语义不变。
    """
    if not bool(metrics.get("success", False)):
        if bool(metrics.get("correctness_ok", False)):
            return 0.4
        if not bool(metrics.get("ast_check_ok", False)):
            return 0.0
        et = str(metrics.get("error_type") or "")
        if et == "ascendc_compile_failed":
            return 0.1
        if et in ("op_not_registered", "ascendc_run_crashed"):
            return 0.2
        # correctness_failed(数值差异)与 output_precheck_failed(形状/dtype/NaN 前置
        # 检查不通过)同档同公式:两者都是「对拍跑完了、结果不对」,通过率天然区分
        # 「差点全对」与「全错」;给 agent 的修改方向区分仍由 fail_hint 的 label 承担。
        if et in ("correctness_failed", "output_precheck_failed"):
            w = case_pass_weight(env)
            if w <= 0.0:
                return CASE_PASS_FALLBACK
            passed, total = metrics.get("cases_passed"), metrics.get("cases_total")
            if (isinstance(passed, int) and isinstance(total, int)
                    and not isinstance(passed, bool) and 0 <= passed <= total and total > 0):
                # min(...,0.999) 只防脏数据(ratio==1 却判 correctness_failed 的不一致),
                # 正常档内 ratio ≤ (N-1)/N 由分档互斥保证,不触碰 0.4 档。
                return 0.3 + w * min(passed / total, 0.999)
            return CASE_PASS_FALLBACK
        return 0.25
    try:
        speedup = float((metrics.get("perf_data") or {}).get("speedup_vs_torch", 1.0))
    except (TypeError, ValueError):
        speedup = float("nan")  # non-numeric -> judge_outcome's finiteness gate treats it as infra
    # Soft-saturating speedup: 0.75 + 0.25*tanh(ln speedup), written as the algebraic
    # identity (s^2-1)/(s^2+1) (no math import, s=0 -> 0.5, no overflow on huge s).
    # Continuous (no hard 2x cap) so equal-speedup success clones don't collapse to a
    # single 1.0 -> zero-std GRPO dead group; bounded (->1.0) so tail speedups (e.g. a
    # 700x measurement artifact) can't dominate the correctness ladder (0.3->0.75=+0.45).
    return 0.75 + 0.25 * (speedup * speedup - 1.0) / (speedup * speedup + 1.0)


def is_infra_failure(metrics: dict | None) -> bool:
    """True iff the eval couldn't run for infra reasons (=> retry, never score)."""
    if metrics is None:
        return True  # no metrics at all == judge didn't run == infra
    return str(metrics.get("error_type") or "") in INFRA_ERROR_TYPES


def judge_outcome(metrics: dict | None) -> dict:
    """judge metrics -> {status, retry, reward, error_type, reason} for the Polar evaluator.

    infra failure  -> status='ERROR',     retry=True,  reward=None   (rllm retries; NOT a 0-reward episode)
    operator result-> status='COMPLETED', retry=False, reward=0..1.0
    """
    if is_infra_failure(metrics):
        et = (metrics or {}).get("error_type") or "no_metrics"
        return {"status": "ERROR", "retry": True, "reward": None,
                "error_type": et, "reason": f"infra failure ({et}) -> retry, not scored"}
    # Malformed perf on a 'success' (NaN/inf/negative speedup) == garbage metrics, NOT a real reward.
    # Treat as infra (retry) so a single NaN never poisons the whole GRPO group's normalized advantage.
    if bool(metrics.get("success", False)):
        try:
            sp = float((metrics.get("perf_data") or {}).get("speedup_vs_torch", 1.0))
        except (TypeError, ValueError):
            sp = float("nan")
        if not math.isfinite(sp) or sp < 0:
            return {"status": "ERROR", "retry": True, "reward": None,
                    "error_type": "judge_metrics_unreadable",
                    "reason": f"non-finite/negative speedup ({sp!r}) -> malformed metrics, retry"}
    reward = reward_from_metrics(metrics)
    return {"status": "COMPLETED", "retry": False, "reward": reward,
            "error_type": metrics.get("error_type"), "reason": "scored"}


# --------------------- 截断事件惩罚(训练信号)---------------------
#
# 空截断(思考打满 max_output、零产出被 CLI 丢弃)在 outcome 阶梯里原本零成本:salvage 把
# session 救回后与干净 session 同分(run 205655 实测:截断>=3 的 session 均 0.313,零截断
# 0.323)——组内无差异 -> 没有负方向 -> 截断永不收敛。这里按截断「事件数」从标量 reward
# 轻扣:adjusted = max(reward - min(λ·T, cap), floor)。GRPO 只看组内相对排名,小额扣分即
# 产生学习压力;截断段 token 本体仍不过梯度(loss_mask 闸不动),12K 负梯度力度失控与
# 内容误伤的风险都不引入。
#
# 口径按「次」不按 token 长度:目标是「顶到帽之前必须产出」,不是「想得短」——长而有效
# 的思考不受罚。λ 来自 205655 场离线回放:0.01 -> 17% 组内对子翻转 / 9% 跨档(温和起效);
# 0.03 -> 42% 跨档(过激)。POLAR_TRUNCATION_PENALTY=0 整体回退。

POLAR_TRUNCATION_PENALTY_ENV = "POLAR_TRUNCATION_PENALTY"
POLAR_TRUNCATION_PENALTY_CAP_ENV = "POLAR_TRUNCATION_PENALTY_CAP"
POLAR_TRUNCATION_PENALTY_FLOOR_ENV = "POLAR_TRUNCATION_PENALTY_FLOOR"


def truncation_penalty_knobs(env: dict | None = None) -> tuple[float, float, float]:
    """(每次截断扣分 λ, 总扣分封顶 cap, 总分下限 floor)。λ<=0 即整体关闭。"""
    source = os.environ if env is None else env

    def _f(key: str, default: str) -> float:
        try:
            return float(source.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    return (
        _f(POLAR_TRUNCATION_PENALTY_ENV, "0.01"),
        _f(POLAR_TRUNCATION_PENALTY_CAP_ENV, "0.05"),
        # 阶梯 floor 降到 0(AST 档=0)后,floor 默认必须同步为 0:
        # max(0 - 扣分, 0.15) 会把重截断的 0 档轨迹倒挂着抬到 0.15。
        _f(POLAR_TRUNCATION_PENALTY_FLOOR_ENV, "0.0"),
    )


def apply_truncation_penalty(reward: float, truncation_events: int) -> tuple[float, float]:
    """reward -> (调整后 reward, 扣掉的分)。扣 0.0 = 未启用或无截断事件。"""
    lam, cap, floor = truncation_penalty_knobs()
    if lam <= 0.0 or truncation_events <= 0:
        return reward, 0.0
    deducted = min(lam * truncation_events, cap)
    return max(reward - deducted, floor), deducted


# --------------------- 过程奖励(process reward)---------------------
#
# 设计:dev_docs/dev_04_process_reward_design.md;实现:dev_docs/dev_05_process_reward_implementation.md。
# 数据源是 agent 侧固定评测入口自动落盘的 process_info.json(tools/process_track.py,
# 经 $ARTIFACTS_DIR bind mount 直通宿主机)。原则与上方各机制同一套纪律:
#   C1 outcome 主导:R_process 总摆幅 ±Δ(默认 0.10),永远不能把 0.2 的失败抬过 0.75 的成功;
#   C2 不可作弊:只信 events 原始序列(计分全部从 events 重算,不信自报 summary),
#      且终局必须与 judge 实测 metrics 一致(validate_process_info V3);
#   C3 不制造 dead group:主项随「首次通过步数」连续变化;
#   C4 infra 不染指:process 在 judge_outcome 判 COMPLETED 之后才合并,retry 分支走不到。

POLAR_PROCESS_REWARD_ENV = "POLAR_PROCESS_REWARD"          # 默认 1;<=0 整体回退纯 outcome
POLAR_PROCESS_REWARD_CAP_ENV = "POLAR_PROCESS_REWARD_CAP"  # 总摆幅 Δ,默认 0.10

# 分量权重按 Δ 等比:0.06/0.04/0.04(Δ=0.10 时)。灰度期只暴露总开关 + 总摆幅两个旋钮。
_W_FIRST_PASS_RATIO = 0.6
_W_OPT_GAIN_RATIO = 0.4
_W_REPEAT_RATIO = 0.4
_REPEAT_UNIT = 0.01        # 同一 (stage, error_type) 连击每次扣分(与截断惩罚 λ 同款量级)
_OPT_GAIN_FULL = 0.2       # speedup 提升多少拿满 opt_gain 的兑现半份

PROCESS_INFO_SCHEMA_VERSION = 1


def process_reward_knobs(env: dict | None = None) -> tuple[float, bool]:
    """(总摆幅 Δ, 是否启用)。POLAR_PROCESS_REWARD<=0 即整体关闭。"""
    source = os.environ if env is None else env

    def _f(key: str, default: str) -> float:
        try:
            return float(source.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    cap = _f(POLAR_PROCESS_REWARD_CAP_ENV, "0.10")
    enabled = _f(POLAR_PROCESS_REWARD_ENV, "1") > 0.0
    return cap, enabled


def terminal_stage_status(metrics: dict) -> tuple[str, str]:
    """judge 实测 metrics -> 期望的终局 (stage, status)。

    与 tools/process_track.py derive_stage_substeps 同一张推导表(两侧无依赖,
    靠单测锁定一致性):ast→compile→verify→benchmark→done 单调递进。
    """
    ast_ok = bool(metrics.get("ast_check_ok", False))
    corr_ok = bool(metrics.get("correctness_ok", False))
    success = bool(metrics.get("success", False))
    if not ast_ok:
        stage = "ast"
    elif not corr_ok:
        stage = "compile" if str(metrics.get("error_type") or "") == "ascendc_compile_failed" else "verify"
    elif not success:
        stage = "benchmark"
    else:
        stage = "done"
    return stage, "pass" if success else "fail"


def validate_process_info(data: object, final_metrics: dict) -> tuple[list[dict] | None, str]:
    """process_info.json -> (洗净的 events, 校验结论)。events=None 即 process 分量记 0。

    V1 格式:schema_version / events 数组 / global_step 严格递增;
    V3 终局一致(防伪造,关键校验):judge 在干净 runtime 重跑出的 metrics 是改不了的,
        agent 的过程记录必须与它对得上 ——
        judge 实测 success  => events 里必须真有过 stage=done 的 pass;
        judge 实测不 success => events 里不允许存在任何 pass(报过成功却复现不了 = 伪造或腐烂)。
        注意比「末条事件」而不是「每条」:judge 判的是 .best.tar.gz(R1 反退化),agent
        末次评测可能是 best 之后的回退尝试,严格比末条会误杀合法轨迹;
    V4 speedup 合法性:非法值降级为 None(不整篇作废)。
    """
    if not isinstance(data, dict) or data.get("schema_version") != PROCESS_INFO_SCHEMA_VERSION:
        return None, "bad_schema"
    raw_events = data.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        return None, "no_events"
    events: list[dict] = []
    steps: list[int] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            return None, "bad_event"
        ev = dict(raw)
        step = ev.get("global_step")
        if not isinstance(step, int):
            return None, "bad_step"
        steps.append(step)
        sp = ev.get("speedup_vs_torch")
        if sp is not None:
            try:
                sp = float(sp)
            except (TypeError, ValueError):
                sp = None
            if sp is None or not math.isfinite(sp) or sp < 0:
                ev["speedup_vs_torch"] = None
        events.append(ev)
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        return None, "non_monotonic_steps"
    _, expected_status = terminal_stage_status(final_metrics)
    has_done_pass = any(e.get("status") == "pass" and e.get("stage") == "done" for e in events)
    has_any_pass = any(e.get("status") == "pass" for e in events)
    if expected_status == "pass" and not has_done_pass:
        return None, "terminal_mismatch"      # judge 复现了成功,过程记录却没有
    if expected_status != "pass" and has_any_pass:
        return None, "terminal_mismatch"      # 过程记录报了成功,judge 复现不了
    return events, "ok"


def _consecutive_repeat_max(events: list[dict]) -> int:
    """相邻「失败」事件 (stage, error_type) 相同的最长连击(stage 在推进不算打转)。

    只数 fail:连续 pass(如 optimization 阶段反复刷 speedup)是有效尝试,
    「没提升」已由 B_opt_gain 的兑现半份表达,再用连击扣分会与进阶段奖励自相矛盾。
    """
    best, run, prev = 0, 0, None
    for ev in events:
        if ev.get("status") == "pass":
            run, prev = 0, None
            continue
        key = (ev.get("stage"), ev.get("error_type"))
        run = run + 1 if key == prev else 1
        prev = key
        best = max(best, run)
    return best


def _first_pass_step_of_deepest(events: list[dict]) -> int | None:
    """已解锁最深子步骤(benchmark>verify>compile>ast)首次 pass 的 global_step。"""
    first: dict[str, int] = {}
    for ev in events:
        step = ev.get("global_step")
        for name, state in (ev.get("substeps") or {}).items():
            if state == "pass" and name not in first and isinstance(step, int):
                first[name] = step
    for name in ("benchmark", "verify", "compile", "ast"):
        if name in first:
            return first[name]
    return None


def process_reward(
    events: list[dict],
    metrics: dict,
    env: dict | None = None,
) -> tuple[float, dict]:
    """已校验 events -> (R_process ∈ [-Δ, +Δ], 分量明细)。

    R_process = clamp(B_first_pass + B_opt_gain − P_repeat, −Δ, +Δ)
      B_first_pass:达到同一 outcome 用的评测次数越少越高(主项,连续);
      B_opt_gain:仅 success 轨迹 —— 进 optimization 阶段半份,speedup 兑现再半份;
      P_repeat:同一 (stage, error_type) 连击,超 1 次起扣 0.01/次,封顶 w_rep。
    """
    cap, enabled = process_reward_knobs(env)
    if not enabled:
        return 0.0, {"disabled": "env"}
    w_fp = _W_FIRST_PASS_RATIO * cap
    w_opt = _W_OPT_GAIN_RATIO * cap
    w_rep = _W_REPEAT_RATIO * cap

    # B_first_pass
    k = _first_pass_step_of_deepest(events)
    budget = 0
    for ev in events:  # generation 预算;缺失回退到实际步数(兜底不分母为 0)
        if ev.get("phase") == "generation" and isinstance(ev.get("phase_limit"), int):
            budget = ev["phase_limit"]
            break
    if budget <= 0:
        budget = max((e.get("global_step") or 1 for e in events), default=1)
    b_fp = w_fp * max(0, budget + 1 - k) / budget if k else 0.0

    # B_opt_gain(只对成功轨迹有意义;失败轨迹此项 0,不影响失败组内排序)
    b_opt = 0.0
    opt_entered = False
    opt_gain = 0.0
    if bool(metrics.get("success", False)):
        opt_entered = any(e.get("phase") == "optimization" for e in events)
        speedups = [
            (e.get("global_step") or 0, float(e["speedup_vs_torch"]))
            for e in events
            if e.get("speedup_vs_torch") is not None and e.get("status") == "pass"
        ]
        if speedups:
            first_success_sp = next(
                sp for _, sp in sorted(speedups)  # 首个 pass 事件的 speedup
            )
            best_sp = max(sp for _, sp in speedups)
            opt_gain = max(0.0, best_sp - first_success_sp)
        b_opt = w_opt * (0.5 * float(opt_entered) + 0.5 * min(1.0, opt_gain / _OPT_GAIN_FULL))

    # P_repeat
    repeat_max = _consecutive_repeat_max(events)
    p_rep = min(_REPEAT_UNIT * max(0, repeat_max - 1), w_rep)

    total = max(-cap, min(cap, b_fp + b_opt - p_rep))
    return total, {
        "b_first_pass": round(b_fp, 6),
        "b_opt_gain": round(b_opt, 6),
        "p_repeat": round(p_rep, 6),
        "first_pass_step": k,
        "budget": budget,
        "opt_entered": opt_entered,
        "opt_gain_speedup": round(opt_gain, 6),
        "repeat_max": repeat_max,
        "cap": cap,
    }


# --------------------------------- tests ---------------------------------

def _mk_events(specs: list[tuple[str, str, str | None, float | None]], budget: int = 6) -> list[dict]:
    """构造 events:(stage, status, error_type, speedup) 列表 -> 合法 events 数组。"""
    sub = {
        "ast": {"ast": "fail", "compile": "skip", "verify": "skip", "benchmark": "skip"},
        "compile": {"ast": "pass", "compile": "fail", "verify": "skip", "benchmark": "skip"},
        "verify": {"ast": "pass", "compile": "pass", "verify": "fail", "benchmark": "skip"},
        "benchmark": {"ast": "pass", "compile": "pass", "verify": "pass", "benchmark": "fail"},
        "done": {"ast": "pass", "compile": "pass", "verify": "pass", "benchmark": "pass"},
    }
    events = []
    seen_pass = False
    for i, (stage, status, et, sp) in enumerate(specs, start=1):
        phase = "optimization" if seen_pass else "generation"
        events.append({
            "global_step": i, "kind": "eval", "phase": phase,
            "phase_step": i, "phase_limit": budget,
            "stage": stage, "status": status, "error_type": et,
            "substeps": sub[stage], "speedup_vs_torch": sp,
        })
        seen_pass = seen_pass or status == "pass"
    return events


def _mk_info(events: list[dict]) -> dict:
    return {"schema_version": 1, "events": events, "milestones": []}


def _mk_metrics(success: bool, ast: bool = True, corr: bool = True,
                error_type: str | None = None, speedup: float | None = 1.0) -> dict:
    return {"success": success, "ast_check_ok": ast, "correctness_ok": corr,
            "error_type": error_type,
            "perf_data": {"speedup_vs_torch": speedup} if speedup is not None else None}


def test_validate_rejects_forgery():
    m_ok = _mk_metrics(True)
    events = _mk_events([("done", "pass", None, 1.1)])
    # non_monotonic
    bad = _mk_info([dict(events[0], global_step=2), dict(events[0], global_step=1)])
    assert validate_process_info(bad, m_ok)[0] is None
    assert validate_process_info(bad, m_ok)[1] == "non_monotonic_steps"
    # bad_schema / no_events
    assert validate_process_info({"schema_version": 99, "events": events}, m_ok)[1] == "bad_schema"
    assert validate_process_info(_mk_info([]), m_ok)[1] == "no_events"
    assert validate_process_info("junk", m_ok)[1] == "bad_schema"
    # terminal_mismatch 两个方向
    assert validate_process_info(_mk_info(_mk_events([("compile", "fail", "ascendc_compile_failed", None)])), m_ok)[1] == "terminal_mismatch"
    m_fail = _mk_metrics(False, corr=False, error_type="correctness_failed")
    assert validate_process_info(_mk_info(events), m_fail)[1] == "terminal_mismatch"
    # 合法轨迹(judge 判 best、agent 末次是回退尝试)不应被误杀
    legit = _mk_events([("done", "pass", None, 1.1), ("compile", "fail", "ascendc_compile_failed", None)])
    assert validate_process_info(_mk_info(legit), m_ok)[1] == "ok"


def test_validate_sanitizes_speedup():
    events = _mk_events([("done", "pass", None, 1.1)])
    events[0]["speedup_vs_torch"] = float("nan")
    cleaned, why = validate_process_info(_mk_info(events), _mk_metrics(True))
    assert why == "ok" and cleaned[0]["speedup_vs_torch"] is None
    events[0]["speedup_vs_torch"] = -3.0
    cleaned, why = validate_process_info(_mk_info(events), _mk_metrics(True))
    assert why == "ok" and cleaned[0]["speedup_vs_torch"] is None


def test_process_reward_first_pass_gradient():
    env = {POLAR_PROCESS_REWARD_ENV: "1", POLAR_PROCESS_REWARD_CAP_ENV: "0.10"}
    m = _mk_metrics(True, speedup=1.0)
    rewards = []
    for k in (1, 3, 6):
        specs = [("compile", "fail", "ascendc_compile_failed", None)] * (k - 1) + [("done", "pass", None, 1.0)]
        events, why = validate_process_info(_mk_info(_mk_events(specs)), m)
        assert why == "ok"
        r, comp = process_reward(events, m, env)
        rewards.append(comp["b_first_pass"])
    assert rewards[0] > rewards[1] > rewards[2] >= 0.0
    assert abs(rewards[0] - 0.06) < 1e-9  # 一次通过拿满主项


def test_process_reward_opt_gain_only_on_success():
    env = {POLAR_PROCESS_REWARD_ENV: "1", POLAR_PROCESS_REWARD_CAP_ENV: "0.10"}
    events = _mk_events([("done", "pass", None, 1.0), ("done", "pass", None, 1.2)])
    m_ok = _mk_metrics(True, speedup=1.2)
    _, comp = process_reward(events, m_ok, env)
    assert comp["opt_entered"] is True and comp["b_opt_gain"] > 0.0
    # 提升 0.2 拿满兑现半份:0.5*w + 0.5*w = w = 0.04
    assert abs(comp["b_opt_gain"] - 0.04) < 1e-9
    # 失败轨迹 opt 分量恒 0
    fail_events = _mk_events([("verify", "fail", "correctness_failed", None)])
    m_fail = _mk_metrics(False, corr=False, error_type="correctness_failed", speedup=None)
    r, comp = process_reward(fail_events, m_fail, env)
    assert comp["b_opt_gain"] == 0.0


def test_process_reward_repeat_penalty_cap():
    env = {POLAR_PROCESS_REWARD_ENV: "1", POLAR_PROCESS_REWARD_CAP_ENV: "0.10"}
    specs = [("compile", "fail", "ascendc_compile_failed", None)] * 10
    events = _mk_events(specs)
    _, comp = process_reward(events, _mk_metrics(False, corr=False, error_type="ascendc_compile_failed", speedup=None), env)
    assert comp["repeat_max"] == 10
    assert abs(comp["p_repeat"] - 0.04) < 1e-9  # 封顶 w_rep
    # 连击 1 不扣;两类错误交替(阶段在推进)不算打转
    alt = _mk_events([("compile", "fail", "ascendc_compile_failed", None),
                      ("verify", "fail", "correctness_failed", None)])
    _, comp = process_reward(alt, _mk_metrics(False, corr=False, error_type="correctness_failed", speedup=None), env)
    assert comp["repeat_max"] == 1 and comp["p_repeat"] == 0.0
    # 连续 pass(optimization 刷 speedup)不算连击,也不打断后的 fail 重新从 1 计
    passes = _mk_events([("done", "pass", None, 1.0), ("done", "pass", None, 1.1),
                         ("verify", "fail", "correctness_failed", None)])
    _, comp = process_reward(passes, _mk_metrics(False, corr=False, error_type="correctness_failed", speedup=None), env)
    assert comp["repeat_max"] == 1 and comp["p_repeat"] == 0.0


def test_process_reward_disabled_by_env():
    events = _mk_events([("done", "pass", None, 1.0)])
    r, comp = process_reward(events, _mk_metrics(True), {POLAR_PROCESS_REWARD_ENV: "0"})
    assert r == 0.0 and comp == {"disabled": "env"}


def test_process_reward_bounded():
    env = {POLAR_PROCESS_REWARD_ENV: "1", POLAR_PROCESS_REWARD_CAP_ENV: "0.10"}
    best = _mk_events([("done", "pass", None, 1.0)] + [("done", "pass", None, 9.9)] * 5)
    r, _ = process_reward(best, _mk_metrics(True, speedup=9.9), env)
    assert -0.10 <= r <= 0.10
    worst = _mk_events([("compile", "fail", "ascendc_compile_failed", None)] * 50)
    r, _ = process_reward(worst, _mk_metrics(False, corr=False, error_type="ascendc_compile_failed", speedup=None), env)
    assert -0.10 <= r <= 0.10


def test_terminal_stage_status_table():
    assert terminal_stage_status(_mk_metrics(False, ast=False, corr=False)) == ("ast", "fail")
    assert terminal_stage_status(_mk_metrics(False, corr=False, error_type="ascendc_compile_failed")) == ("compile", "fail")
    assert terminal_stage_status(_mk_metrics(False, corr=False, error_type="correctness_failed")) == ("verify", "fail")
    assert terminal_stage_status(_mk_metrics(False, corr=True, error_type="benchmark_failed")) == ("benchmark", "fail")
    assert terminal_stage_status(_mk_metrics(True)) == ("done", "pass")


def test_ladder_not_success():
    assert reward_from_metrics({"success": False, "correctness_ok": True, "ast_check_ok": True}) == 0.4
    assert reward_from_metrics({"success": False, "correctness_ok": False, "ast_check_ok": False}) == 0.0
    # ast 过后的 error_type 细分(新阶梯:0.1/0.2/0.25 + 通过率档)
    assert reward_from_metrics({"success": False, "ast_check_ok": True, "error_type": "ascendc_compile_failed"}) == 0.1
    assert reward_from_metrics({"success": False, "ast_check_ok": True, "error_type": "op_not_registered"}) == 0.2
    assert reward_from_metrics({"success": False, "ast_check_ok": True, "error_type": "ascendc_run_crashed"}) == 0.2
    assert reward_from_metrics({"success": False, "ast_check_ok": True, "error_type": None}) == 0.25  # 未知类型兜底中间档
    # 对拍跑完但未全过:0.3 + 0.10×通过率;统计缺失/越界 → 回退 0.35(旧固定档)
    base = {"success": False, "ast_check_ok": True, "error_type": "correctness_failed"}
    assert reward_from_metrics(base) == 0.35
    assert reward_from_metrics({**base, "cases_passed": 0, "cases_total": 10}) == 0.3
    assert abs(reward_from_metrics({**base, "cases_passed": 5, "cases_total": 10}) - 0.35) < 1e-9
    assert abs(reward_from_metrics({**base, "cases_passed": 9, "cases_total": 10}) - 0.39) < 1e-9
    assert reward_from_metrics({**base, "cases_passed": None, "cases_total": 10}) == 0.35
    assert reward_from_metrics({**base, "cases_passed": 11, "cases_total": 10}) == 0.35  # 越界脏数据
    assert reward_from_metrics({**base, "cases_passed": 0, "cases_total": 0}) == 0.35    # 空 case 集
    # 形状/dtype/NaN 前置检查不通过与数值差异同档同公式(通过率天然区分全错与差点全对)
    pre = {"success": False, "ast_check_ok": True, "error_type": "output_precheck_failed"}
    assert pre and abs(reward_from_metrics({**pre, "cases_passed": 8, "cases_total": 9}) - (0.3 + 0.1 * 8 / 9)) < 1e-9
    # ratio==1 的不一致脏数据(clamp 到 0.999):严格低于 0.4 档,不破「未全过 < 全过」的序
    assert reward_from_metrics({**base, "cases_passed": 10, "cases_total": 10}) < 0.4
    # 旋钮关闭(灰度回滚)→ 固定 0.35
    assert reward_from_metrics({**base, "cases_passed": 9, "cases_total": 10},
                               env={POLAR_CASE_PASS_WEIGHT_ENV: "0"}) == 0.35
    assert case_pass_weight({POLAR_CASE_PASS_WEIGHT_ENV: "garbage"}) == 0.10  # 垃圾值回落默认


def test_ladder_success_speedup():
    assert reward_from_metrics({"success": True, "perf_data": {"speedup_vs_torch": 0.0}}) == 0.5
    assert reward_from_metrics({"success": True, "perf_data": {"speedup_vs_torch": 1.0}}) == 0.75
    assert round(reward_from_metrics({"success": True, "perf_data": {"speedup_vs_torch": 2.0}}), 4) == 0.9  # 2x no longer hard-capped
    assert round(reward_from_metrics({"success": True, "perf_data": {"speedup_vs_torch": 9.0}}), 4) == 0.9939  # soft-saturating toward 1.0
    assert reward_from_metrics({"success": True, "perf_data": None}) == 0.75  # default speedup 1.0


def test_is_infra_failure():
    assert is_infra_failure(None) is True
    assert is_infra_failure({"error_type": "judge_container_failed"}) is True
    assert is_infra_failure({"error_type": "task_missing"}) is True
    assert is_infra_failure({"error_type": "npu_runtime_unavailable"}) is True
    assert is_infra_failure({"error_type": "correctness_failed"}) is False    # operator
    assert is_infra_failure({"error_type": "submission_missing"}) is False    # operator
    assert is_infra_failure({"error_type": "output_precheck_failed"}) is False  # operator(输出结构不对) (agent didn't deliver)
    assert is_infra_failure({"success": True, "error_type": None}) is False


def test_judge_outcome_infra_retries_not_scored():
    o = judge_outcome({"error_type": "judge_no_metrics", "success": False})
    assert o["status"] == "ERROR" and o["retry"] is True and o["reward"] is None
    o2 = judge_outcome(None)
    assert o2["status"] == "ERROR" and o2["retry"] is True
    o3 = judge_outcome({"error_type": "npu_runtime_unavailable", "success": False})
    assert o3["status"] == "ERROR" and o3["retry"] is True and o3["reward"] is None


def test_classify_infra_error_text_detects_npu_init_not_shape_mismatch():
    text = "RuntimeError: aclInit, error code is 107001\n[Error]: Invalid device ID."
    assert classify_infra_error_text(text) == "npu_runtime_unavailable"
    wrapped = "数值验证失败: PTA call acl api failed\ninput error deviceId:0"
    assert classify_infra_error_text(wrapped) == "npu_runtime_unavailable"
    assert classify_infra_error_text("验证失败: mismatch max diff 0.1") is None


def test_judge_outcome_operator_failure_scored():
    o = judge_outcome({"success": False, "ast_check_ok": False, "error_type": "correctness_failed"})
    assert o["status"] == "COMPLETED" and o["retry"] is False and o["reward"] == 0.0
    o2 = judge_outcome({"success": False, "correctness_ok": True, "error_type": "benchmark_failed"})
    assert o2["reward"] == 0.4 and o2["status"] == "COMPLETED"


def test_judge_outcome_success():
    o = judge_outcome({"success": True, "perf_data": {"speedup_vs_torch": 2.0}, "error_type": None})
    assert o["status"] == "COMPLETED" and round(o["reward"], 4) == 0.9 and o["retry"] is False


def test_malformed_speedup_is_infra_not_nan_reward():
    # NaN/inf/negative/garbage speedup on a 'success' must -> infra retry (reward None), never a
    # non-finite/garbage reward that would poison the GRPO group's normalized advantage.
    for bad in (float("nan"), float("inf"), -1.0, "oops"):
        o = judge_outcome({"success": True, "perf_data": {"speedup_vs_torch": bad}, "error_type": None})
        assert o["status"] == "ERROR" and o["retry"] is True and o["reward"] is None, bad


def test_truncation_penalty_deduct_cap_floor():
    saved = {k: os.environ.pop(k, None) for k in (
        POLAR_TRUNCATION_PENALTY_ENV, POLAR_TRUNCATION_PENALTY_CAP_ENV, POLAR_TRUNCATION_PENALTY_FLOOR_ENV)}
    try:
        # 默认 λ=0.01:2 次截断扣 0.02
        adj, ded = apply_truncation_penalty(0.35, 2)
        assert abs(adj - 0.33) < 1e-9 and abs(ded - 0.02) < 1e-9
        # cap=0.05:10 次截断只扣 0.05
        adj, ded = apply_truncation_penalty(0.35, 10)
        assert abs(adj - 0.30) < 1e-9 and abs(ded - 0.05) < 1e-9
        # floor=0.0(默认,配合 AST 档=0):cap 生效后 0.2 档重截断压到 0.15;0 档轨迹
        # 扣分被 floor 兜底在 0,不会倒挂抬分
        adj, ded = apply_truncation_penalty(0.2, 10)
        assert abs(adj - 0.15) < 1e-9 and abs(ded - 0.05) < 1e-9
        adj, ded = apply_truncation_penalty(0.0, 3)
        assert abs(adj - 0.0) < 1e-9
        # 无截断不动
        assert apply_truncation_penalty(0.35, 0) == (0.35, 0.0)
        # env=0 整体回退
        os.environ[POLAR_TRUNCATION_PENALTY_ENV] = "0"
        assert apply_truncation_penalty(0.35, 3) == (0.35, 0.0)
        # env 自定义 λ/cap/floor
        os.environ[POLAR_TRUNCATION_PENALTY_ENV] = "0.02"
        os.environ[POLAR_TRUNCATION_PENALTY_CAP_ENV] = "0.03"
        os.environ[POLAR_TRUNCATION_PENALTY_FLOOR_ENV] = "0.1"
        adj, ded = apply_truncation_penalty(0.35, 5)
        assert abs(adj - 0.32) < 1e-9 and abs(ded - 0.03) < 1e-9  # 0.02*5=0.10 被 cap 到 0.03
        # 显式 env dict(测试/调用方隔离)
        assert truncation_penalty_knobs({POLAR_TRUNCATION_PENALTY_ENV: "x"})[0] == 0.01  # 垃圾值回落默认
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    import sys
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
