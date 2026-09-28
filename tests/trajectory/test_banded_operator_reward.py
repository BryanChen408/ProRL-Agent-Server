"""Stage ordering, valid outcomes, and the n<=8 GRPO sign boundary (CPU only)."""

import itertools

import pytest

from polar.trajectory.evaluator.operator_reward import banded_reward_from_metrics


def test_banded_reward_stages_and_speed():
    base = {"ast_check_ok": True, "correctness_ok": False, "success": False}
    for error, expected in [("ascendc_compile_failed", 0), ("op_not_registered", .02),
                            ("ascendc_run_crashed", .02), ("ascendc_load_failed", .02),
                            ("ascendc_launch_failed", .02), ("ascendc_run_timeout", .02),
                            ("stateful_impl_detected", .02), ("unknown", 0)]:
        assert banded_reward_from_metrics({**base, "error_type": error}) == expected
    for passed, expected in [(0, .03), (1, .044), (4, .086)]:
        for error in ("correctness_failed", "output_precheck_failed"):
            assert banded_reward_from_metrics({**base, "error_type": error,
                                               "cases_passed": passed, "cases_total": 5}) == pytest.approx(expected)
    assert banded_reward_from_metrics({**base, "ast_check_ok": False}) == 0
    for speedup, expected in [(None, .9), (.1, .900990099), (1, .95), (2, .98), (1e300, 1)]:
        assert banded_reward_from_metrics({"correctness_ok": True,
                                           "perf_data": {"speedup_vs_torch": speedup}}) == pytest.approx(expected)
    # Missing/invalid case counts cannot fabricate partial progress.
    for passed, total in [(None, None), (True, 5), (6, 5), (1, 0)]:
        assert banded_reward_from_metrics({**base, "error_type": "correctness_failed",
                                           "cases_passed": passed, "cases_total": total}) == .03
    with pytest.raises(ValueError, match="terminal correctness"):
        banded_reward_from_metrics({"success": False})
    with pytest.raises(ValueError, match="terminal correctness"):
        banded_reward_from_metrics({"success": True})
    for metrics in ({"success": True, "correctness_ok": False},
                    {"correctness_ok": True, "ast_check_ok": False}):
        with pytest.raises(ValueError, match="inconsistent"):
            banded_reward_from_metrics(metrics)


def test_all_mixed_group_corner_signs_up_to_eight():
    count = 0
    for n in range(2, 9):
        for rewards in itertools.product((0., .1, .9, 1.), repeat=n):
            if not any(r >= .9 for r in rewards) or not any(r <= .1 for r in rewards):
                continue
            mean = sum(rewards) / n
            assert all((r - mean if r >= .9 else mean - r) >= .025 - 1e-12 for r in rewards)
            count += 1
    assert count == 86360
