#!/usr/bin/env python3
"""
Verify a Polar SessionResult is a TRAINING-READY trajectory (design §0.5.8 #2 + RL-sample audit).

Run right after a Polar smoke / operator rollout to assert — WITHOUT waiting for e2e training — that
captured Traces are safe to feed GRPO. Two layers, each split into hard PROBLEMS (training-unsafe ->
exit 1) and WARNINGS (inspect, but not fatal):

  TRACE level — the token contract rllm's bridge consumes:
    * response_ids / prompt_ids non-empty
    * loss_mask (if present): len == len(response_ids), values in {0,1}; --expect-per-request -> all 1
    * response_logprobs (if present): len == len(response_ids)
    * VALUE realness (audit): no fabricated/zero-fill logprob (==0.0) at a TRAINABLE position
      (a sampled token with logprob 0.0 == prob 1.0 is implausible -> record_utils 0.0-filled a
      missing logprob); constant logprob vector -> warn
    * logprob_integrity (audit): record_utils flags token_id<->logprob misattribution / missing
      logprobs into trace.metadata -> any count > 0 is a hard problem
    * finish_reason == 'length' -> warn (truncated; trained at full weight unless masked downstream)

  SESSION level — the reward<->status contract (the infra-retry / no-false-negative-poison invariant):
    * status != COMPLETED (ERROR/TIMEOUT) -> NO trace may carry a reward (infra must retry, not score)
    * any attached reward must be finite and within the ladder band [0.2, 1.0]
    * COMPLETED with traces but no reward -> warn (evaluator may not have attached one)

  # verify a saved SessionResult, OR a /rollout/task/<id> status with results[]:
  python examples/ascend/verify_trajectory.py operator_session.json
  # require per_request semantics (our locked builder for Claude Code):
  python examples/ascend/verify_trajectory.py operator_session.json --expect-per-request
  # self-test (no input, verifies this checker's own logic):
  python examples/ascend/verify_trajectory.py --self-test

Exit 0 = all traces+sessions OK (warnings allowed), 1 = any hard problem (or no traces found).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

# operator reward ladder band (operator_reward.reward_from_metrics): 0.2..0.4 operator-fail, 0.5..1.0 success.
REWARD_LO, REWARD_HI = 0.2, 1.0


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sessions(doc: Any) -> list[dict[str, Any]]:
    """Normalize input into a list of SessionResult-like dicts."""
    if isinstance(doc, list):
        return [s for s in doc if isinstance(s, dict)]
    if isinstance(doc, dict):
        if "results" in doc and isinstance(doc["results"], list):  # task-status shape
            return [s for s in doc["results"] if isinstance(s, dict)]
        if "trajectory" in doc:  # single SessionResult
            return [doc]
    return []


def check_trace(tr: dict[str, Any], expect_per_request: bool) -> tuple[list[str], list[str]]:
    """Returns (problems, warnings). problems => training-unsafe; warnings => inspect."""
    problems: list[str] = []
    warnings: list[str] = []
    resp = tr.get("response_ids") or []
    prompt = tr.get("prompt_ids") or []
    if not resp:
        problems.append("empty response_ids")
    if not prompt:
        problems.append("empty prompt_ids")

    mask = tr.get("loss_mask")
    if mask is not None:
        if len(mask) != len(resp):
            problems.append(f"loss_mask len {len(mask)} != response_ids len {len(resp)}")
        if any(m not in (0, 1) for m in mask):
            problems.append("loss_mask has values outside {0,1}")
        if expect_per_request and resp and any(m != 1 for m in mask):
            problems.append("expected per_request (loss_mask all 1) but found 0s")

    lp = tr.get("response_logprobs")
    if lp is not None:
        if len(lp) != len(resp):
            problems.append(f"response_logprobs len {len(lp)} != response_ids len {len(resp)}")
        else:
            # zero-fill signature: a SAMPLED (trainable) token with logprob exactly 0.0 == prob 1.0 is
            # implausible -> almost certainly record_utils 0.0-filling a missing logprob.
            m = mask if (mask is not None and len(mask) == len(lp)) else [1] * len(lp)
            zero_trainable = [i for i, (mm, v) in enumerate(zip(m, lp)) if mm == 1 and _is_num(v) and v == 0.0]
            if zero_trainable:
                problems.append(f"fabricated/zero-fill logprob (==0.0) at {len(zero_trainable)} trainable position(s)")
            non_null = [v for v in lp if _is_num(v)]
            if len(non_null) >= 2 and len(set(non_null)) == 1:
                warnings.append("response_logprobs is a constant vector (degenerate — likely not real)")

    # token_id<->logprob misattribution / missing-logprob, flagged by Polar record_utils._logprob_integrity
    integ = tr.get("metadata")
    integ = integ.get("logprob_integrity") if isinstance(integ, dict) else None
    if isinstance(integ, dict) and (integ.get("misattributed") or integ.get("missing")):
        problems.append(
            f"logprob_integrity violation (misattributed={integ.get('misattributed')}, missing={integ.get('missing')})"
        )

    if str(tr.get("finish_reason")) == "length":
        warnings.append("finish_reason==length (truncated; ensure mask_max_response_length_exceeded downstream)")

    return problems, warnings


def check_session(s: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Reward<->status contract: infra/error must NOT be scored; rewards finite + in ladder band."""
    problems: list[str] = []
    warnings: list[str] = []
    traj = s.get("trajectory") or {}
    status = str(s.get("status") or traj.get("status") or "").upper()
    traces = traj.get("traces") or []
    rewards = [tr.get("reward") for tr in traces]
    present = [r for r in rewards if r is not None]

    if status == "COMPLETED":
        if traces and not present:
            warnings.append("COMPLETED but no trace carries a reward (evaluator didn't attach one?)")
    else:  # ERROR / TIMEOUT / ...
        if present:
            problems.append(
                f"status={status} but {len(present)} trace(s) carry a reward — infra/error must NOT be "
                f"scored (false-negative poison; adapter drops these, but capture should not attach them)"
            )

    for r in present:
        if not _is_num(r) or not math.isfinite(r):
            problems.append(f"non-finite reward {r!r}")
        elif not (REWARD_LO <= r <= REWARD_HI):
            problems.append(f"reward {r} outside ladder band [{REWARD_LO}, {REWARD_HI}]")

    return problems, warnings


def verify(doc: Any, expect_per_request: bool) -> int:
    sessions = _sessions(doc)
    if not sessions:
        print("[verify] no SessionResult/trajectory found in input")
        return 1
    total_traces = bad = warned = 0
    sess_bad = 0
    for si, s in enumerate(sessions):
        traj = s.get("trajectory") or {}
        status = s.get("status") or traj.get("status")
        traces = traj.get("traces") or []
        sp, sw = check_session(s)
        if sp:
            sess_bad += 1
        line = f"  session[{si}] status={status} traces={len(traces)}"
        if sp:
            line += "  >> " + "; ".join(sp)
        if sw:
            line += "  (warn: " + "; ".join(sw) + ")"
        print(line)
        for ti, tr in enumerate(traces):
            total_traces += 1
            problems, warnings = check_trace(tr, expect_per_request)
            n_resp = len(tr.get("response_ids") or [])
            n_mask = len(tr.get("loss_mask") or [])
            n_lp = len(tr.get("response_logprobs") or []) if tr.get("response_logprobs") is not None else None
            tag = "BAD" if problems else ("warn" if warnings else "OK ")
            if problems:
                bad += 1
            elif warnings:
                warned += 1
            extra = ""
            if problems:
                extra = "  >> " + "; ".join(problems)
            elif warnings:
                extra = "  ~ " + "; ".join(warnings)
            print(f"    [{tag}] trace[{ti}] resp={n_resp} mask={n_mask} logprobs={n_lp}{extra}")
    ok = total_traces > 0 and bad == 0 and sess_bad == 0
    print(
        f"\n[verify] {total_traces - bad}/{total_traces} traces OK"
        f" ({warned} warn), {len(sessions) - sess_bad}/{len(sessions)} sessions OK"
        + ("; per_request enforced" if expect_per_request else "")
    )
    print(f"[verify] {'PASS — training-ready' if ok else 'FAIL — training-unsafe (see >> above)'}")
    return 0 if ok else 1


def _self_test() -> int:
    cases: list[tuple[str, bool]] = []

    def probs(tr, pr=False):
        return check_trace(tr, pr)[0]

    good = {"response_ids": [1, 2, 3], "prompt_ids": [9, 9], "loss_mask": [1, 1, 1],
            "response_logprobs": [-0.1, -0.2, -0.3]}
    cases.append(("good trace -> no problems", probs(good) == []))
    cases.append(("good trace per_request -> no problems", probs(good, True) == []))

    # existing length/mask discipline
    cases.append(("mask len mismatch -> problem",
                  bool(probs({"response_ids": [1, 2, 3], "prompt_ids": [9], "loss_mask": [1, 1]}))))
    cases.append(("logprobs len mismatch -> problem",
                  bool(probs({"response_ids": [1, 2], "prompt_ids": [9], "response_logprobs": [-0.1]}))))
    cases.append(("empty response_ids -> problem", bool(probs({"response_ids": [], "prompt_ids": [9]}))))
    cases.append(("mask not 0/1 -> problem",
                  bool(probs({"response_ids": [1], "prompt_ids": [9], "loss_mask": [2]}))))
    cases.append(("per_request with a 0 -> problem",
                  bool(probs({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 0]}, True))))
    cases.append(("same 0-mask trace, non-per_request -> OK",
                  probs({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 0]}) == []))

    # NEW: value realness
    cases.append(("zero-fill logprob at trainable pos -> problem",
                  bool(probs({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 1],
                              "response_logprobs": [-0.1, 0.0]}))))
    cases.append(("0.0 logprob at a MASKED pos -> OK (not trained)",
                  probs({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 0],
                         "response_logprobs": [-0.1, 0.0]}) == []))
    cases.append(("logprob_integrity flag -> problem",
                  bool(probs({"response_ids": [1], "prompt_ids": [9], "loss_mask": [1],
                              "response_logprobs": [-0.1],
                              "metadata": {"logprob_integrity": {"misattributed": 1, "missing": 0}}}))))
    cases.append(("constant logprob vector -> warning (not problem)",
                  check_trace({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 1],
                               "response_logprobs": [-0.5, -0.5]}, False) == ([], [
                      "response_logprobs is a constant vector (degenerate — likely not real)"])))
    cases.append(("finish_reason==length -> warning (not problem)",
                  check_trace({"response_ids": [1], "prompt_ids": [9], "loss_mask": [1],
                               "response_logprobs": [-0.1], "finish_reason": "length"}, False)[0] == []))

    # NEW: session reward<->status contract
    sp_err = check_session({"status": "ERROR", "trajectory": {"traces": [{"reward": 0.5}]}})[0]
    cases.append(("ERROR session with a reward -> problem", bool(sp_err)))
    sp_ok = check_session({"status": "ERROR", "trajectory": {"traces": [{"reward": None}]}})[0]
    cases.append(("ERROR session, reward None -> OK", sp_ok == []))
    sp_oob = check_session({"status": "COMPLETED", "trajectory": {"traces": [{"reward": 1.5}]}})[0]
    cases.append(("reward out of band -> problem", bool(sp_oob)))
    sp_nan = check_session({"status": "COMPLETED", "trajectory": {"traces": [{"reward": float("nan")}]}})[0]
    cases.append(("NaN reward -> problem", bool(sp_nan)))
    sp_good = check_session({"status": "COMPLETED", "trajectory": {"traces": [{"reward": 0.75}]}})[0]
    cases.append(("COMPLETED reward 0.75 -> OK", sp_good == []))

    # input normalization
    cases.append(("normalize results[] shape", len(_sessions({"results": [{"trajectory": {}}, {"trajectory": {}}]})) == 2))
    cases.append(("normalize single SessionResult", len(_sessions({"trajectory": {"traces": []}})) == 1))
    cases.append(("normalize list shape", len(_sessions([{"trajectory": {}}])) == 1))
    cases.append(("normalize junk -> empty", _sessions({"foo": 1}) == []))

    all_ok = True
    for label, passed in cases:
        print(f"  [{'OK' if passed else 'XX'}] {label}")
        all_ok = all_ok and passed
    print(f"\n[self-test] {'PASS — verifier logic correct' if all_ok else 'FAIL — verifier has a bug'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Polar trajectory training-contract verifier")
    ap.add_argument("path", nargs="?", help="SessionResult / task-status JSON file")
    ap.add_argument("--expect-per-request", action="store_true", help="require loss_mask all 1")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.path:
        ap.error("provide a JSON path (or use --self-test)")
    with open(args.path) as f:
        doc = json.load(f)
    return verify(doc, args.expect_per_request)


if __name__ == "__main__":
    sys.exit(main())
