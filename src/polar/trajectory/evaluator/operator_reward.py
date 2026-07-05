"""Operator-gen reward kernel (harness-agnostic): judge metrics.json -> rllm-facing outcome.

Reused from the user's openhands_sdk (triton_eval_pipeline.sh `metrics.json` schema_version 2 +
openhands_agent._reward_from_metrics ladder), with the key TRAINING-CORRECTNESS addition:

  INFRA failures (judge container / task-setup broke) are NOT scored 0 — they map to a RETRY
  signal (status=ERROR), so infra flakes do not poison training with false negatives. This is the
  same discipline as 'no zero-fill' on logprobs: never fabricate a bad signal from an infra failure.
  OPERATOR failures (the agent's kernel is bad/missing) get the real 0.2..1.0 reward ladder.

Pure + dependency-free; unit-tested standalone (`python operator_reward.py`) or via pytest.
The Polar `operator_judge` evaluator imports `judge_outcome` to turn judge metrics into
(session status, reward) — ERROR => rllm retries; COMPLETED => reward used.
"""

from __future__ import annotations

import math

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


def reward_from_metrics(metrics: dict) -> float:
    """Authoritative ladder (mirrors openhands_agent._reward_from_metrics).

    not success:  correctness_ok -> 0.4 | ast_check_ok -> 0.3 | else -> 0.2
    success:      0.75 + 0.25*tanh(ln speedup)   # 0.5(<-0x) .. 0.75(1x hold) .. ->1.0(soft, no cap)
    """
    if not bool(metrics.get("success", False)):
        if bool(metrics.get("correctness_ok", False)):
            return 0.4
        if bool(metrics.get("ast_check_ok", False)):
            return 0.3
        return 0.2
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
    operator result-> status='COMPLETED', retry=False, reward=0.2..1.0
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


# --------------------------------- tests ---------------------------------

def test_ladder_not_success():
    assert reward_from_metrics({"success": False, "correctness_ok": True, "ast_check_ok": True}) == 0.4
    assert reward_from_metrics({"success": False, "correctness_ok": False, "ast_check_ok": True}) == 0.3
    assert reward_from_metrics({"success": False, "correctness_ok": False, "ast_check_ok": False}) == 0.2


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
    assert is_infra_failure({"error_type": "submission_missing"}) is False    # operator (agent didn't deliver)
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
    assert o["status"] == "COMPLETED" and o["retry"] is False and o["reward"] == 0.2
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
