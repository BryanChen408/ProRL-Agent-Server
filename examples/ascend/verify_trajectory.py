#!/usr/bin/env python3
"""
Verify a Polar SessionResult is a TRAINING-READY trajectory (design §0.5.8 #2).

Run right after a Polar smoke (e.g. the calculator example) to assert — WITHOUT
waiting for e2e training — that every captured Trace satisfies the contract rllm's
bridge will consume: aligned prompt_ids / response_ids / loss_mask / response_logprobs.
This is the same length discipline Polar enforces in Trace._validate_response_lengths
and rllm enforces in Step.model_post_init — checked here on real captured data.

  # verify a saved SessionResult, OR a /rollout/task/<id> status with results[]:
  python scripts/polar_smoke/verify_trajectory.py rollout_results/.../session.json
  # require per_request semantics (our locked builder for Claude Code):
  python scripts/polar_smoke/verify_trajectory.py result.json --expect-per-request
  # self-test (no input, verifies this checker's own logic):
  python scripts/polar_smoke/verify_trajectory.py --self-test

Per-Trace assertions:
  * response_ids non-empty
  * prompt_ids non-empty
  * loss_mask (if present): len == len(response_ids) and values in {0,1}
  * response_logprobs (if present): len == len(response_ids)
  * --expect-per-request: loss_mask is all 1 (per_request => whole response trainable)

Exit 0 = all traces OK, 1 = any violation (or no traces found).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


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


def check_trace(tr: dict[str, Any], expect_per_request: bool) -> list[str]:
    problems: list[str] = []
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
    if lp is not None and len(lp) != len(resp):
        problems.append(f"response_logprobs len {len(lp)} != response_ids len {len(resp)}")
    return problems


def verify(doc: Any, expect_per_request: bool) -> int:
    sessions = _sessions(doc)
    if not sessions:
        print("[verify] no SessionResult/trajectory found in input")
        return 1
    total_traces = 0
    bad = 0
    for si, s in enumerate(sessions):
        traj = s.get("trajectory") or {}
        status = s.get("status") or traj.get("status")
        traces = traj.get("traces") or []
        print(f"  session[{si}] status={status} traces={len(traces)}")
        for ti, tr in enumerate(traces):
            total_traces += 1
            problems = check_trace(tr, expect_per_request)
            n_resp = len(tr.get("response_ids") or [])
            n_mask = len(tr.get("loss_mask") or [])
            n_lp = len(tr.get("response_logprobs") or []) if tr.get("response_logprobs") is not None else None
            tag = "OK " if not problems else "BAD"
            print(f"    [{tag}] trace[{ti}] resp={n_resp} mask={n_mask} logprobs={n_lp}"
                  + ("" if not problems else "  >> " + "; ".join(problems)))
            if problems:
                bad += 1
    print(f"\n[verify] {total_traces - bad}/{total_traces} traces OK across {len(sessions)} session(s)"
          + (f"; per_request enforced" if expect_per_request else ""))
    return 0 if (total_traces > 0 and bad == 0) else 1


def _self_test() -> int:
    cases: list[tuple[str, bool]] = []

    good = {"response_ids": [1, 2, 3], "prompt_ids": [9, 9], "loss_mask": [1, 1, 1],
            "response_logprobs": [-0.1, -0.2, -0.3]}
    cases.append(("good trace -> no problems", check_trace(good, False) == []))
    cases.append(("good trace per_request -> no problems", check_trace(good, True) == []))

    cases.append(("mask len mismatch -> problem",
                  bool(check_trace({"response_ids": [1, 2, 3], "prompt_ids": [9], "loss_mask": [1, 1]}, False))))
    cases.append(("logprobs len mismatch -> problem",
                  bool(check_trace({"response_ids": [1, 2], "prompt_ids": [9], "response_logprobs": [-0.1]}, False))))
    cases.append(("empty response_ids -> problem",
                  bool(check_trace({"response_ids": [], "prompt_ids": [9]}, False))))
    cases.append(("mask not 0/1 -> problem",
                  bool(check_trace({"response_ids": [1], "prompt_ids": [9], "loss_mask": [2]}, False))))
    cases.append(("per_request with a 0 -> problem",
                  bool(check_trace({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 0]}, True))))
    cases.append(("same 0-mask trace, non-per_request -> OK",
                  check_trace({"response_ids": [1, 2], "prompt_ids": [9], "loss_mask": [1, 0]}, False) == []))

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
