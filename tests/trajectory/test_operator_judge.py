"""Unit tests for the operator_judge evaluator (fake runtimes; no Docker/NPU).

Exercises the evaluate() orchestration + the infra/operator split end-to-end against in-memory
runtimes. Standalone: `python tests/trajectory/test_operator_judge.py` | or via pytest.
(The pure reward ladder is tested separately in operator_reward.py.)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from polar.runtime.base import BaseRuntime
    from polar.runtime.models import ExecResult
    from polar.trajectory.evaluator.operator_judge import OperatorJudgeEvaluator
    from polar.trajectory.models import Trace, Trajectory
    _DEPS = True
except Exception as _exc:  # noqa: BLE001 — polar/pydantic not importable on a bare host
    _DEPS = False
    _IMPORT_ERR = _exc

OP = "add"
SUB = f"output/submission/{OP}_impl.py"
METRICS = "judge_out/metrics.json"


if _DEPS:

    class FakeRuntime(BaseRuntime):
        """In-memory runtime: `files` maps remote_path -> content (None/absent => download raises)."""

        def __init__(self, files: dict | None = None, exec_rc: int = 0, exec_rcs: list[int] | None = None) -> None:
            self.files = files or {}
            self.exec_rc = exec_rc
            self.exec_rcs = list(exec_rcs or [])
            self.uploaded: list[tuple[str, str]] = []
            self.execs: list[str] = []
            self.exec_envs: list[dict] = []

        @property
        def runtime_id(self) -> str:
            return "fake"

        async def start(self) -> None: ...
        async def stop(self) -> None: ...

        async def exec(self, command, *, cwd=None, env=None, timeout_sec=None):
            self.execs.append(command)
            self.exec_envs.append(dict(env or {}))
            rc = self.exec_rcs.pop(0) if self.exec_rcs else self.exec_rc
            return ExecResult(stdout="judge ran\n", stderr="", return_code=rc)

        async def upload_file(self, local_path: str, remote_path: str) -> None:
            self.uploaded.append((local_path, remote_path))

        async def upload_dir(self, local_path: str, remote_path: str) -> None: ...

        async def download_file(self, remote_path: str, local_path: str) -> None:
            if remote_path not in self.files or self.files[remote_path] is None:
                raise FileNotFoundError(remote_path)
            Path(local_path).write_text(self.files[remote_path])

        async def download_dir(self, remote_path: str, local_path: str) -> None: ...

    def _run(metrics, *, impl=True, exec_rc=0, refresh=True, with_fresh=True, agent_files=None, traces=None,
             process_info=None, budget_status=None):
        ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
        if agent_files is None:
            agent_files = {SUB: "# kernel"} if impl else {}
        agent = FakeRuntime(files=agent_files)
        judge = FakeRuntime(files=({METRICS: json.dumps(metrics)} if metrics is not None else {}),
                            exec_rc=exec_rc)
        with tempfile.TemporaryDirectory() as d:
            if process_info is not None:
                Path(d, "process_info.json").write_text(json.dumps(process_info))
            if budget_status is not None:
                Path(d, "pipeline_budget_status.json").write_text(json.dumps(budget_status))
            return asyncio.run(ev.evaluate(
                Trajectory(status="COMPLETED", traces=list(traces or [])),
                runtime=agent,
                fresh_eval_runtime=(judge if with_fresh else None),
                refresh_runtime=refresh,
                artifacts_dir=d, env={}, timeout_seconds=None,
                session_id="s", task_id="t",
            )), agent, judge


def test_success_speedup_reward():
    res, _agent, judge = _run({"success": True, "perf_data": {"speedup_vs_torch": 2.0}})
    assert res.outcome_reward == 0.9   # 0.75+0.25*(s^2-1)/(s^2+1), s=2 -> 0.9
    assert res.metadata["success"] is True and res.metadata["error_type"] is None
    assert len(judge.uploaded) == 1 and len(judge.execs) == 1  # impl crossed + judge ran
    # 无 process_info.json:分量 0,与改造前逐分一致(优雅回退)
    assert res.metadata["process_reward"] == 0.0
    assert res.metadata["process_validation"] == "missing"


# ------------------------- process reward(dev_04/dev_05)-------------------------

_PROCESS_ENVS = ("POLAR_PROCESS_REWARD", "POLAR_PROCESS_REWARD_CAP")


def _mk_process_info(specs, budget=6):
    """(stage, status, error_type, speedup) 列表 -> 合法 process_info dict。"""
    sub = {
        "compile": {"ast": "pass", "compile": "fail", "verify": "skip", "benchmark": "skip"},
        "verify": {"ast": "pass", "compile": "pass", "verify": "fail", "benchmark": "skip"},
        "done": {"ast": "pass", "compile": "pass", "verify": "pass", "benchmark": "pass"},
    }
    events, seen_pass = [], False
    for i, (stage, status, et, sp) in enumerate(specs, start=1):
        events.append({
            "global_step": i, "kind": "eval",
            "phase": "optimization" if seen_pass else "generation",
            "phase_step": i, "phase_limit": budget,
            "stage": stage, "status": status, "error_type": et,
            "substeps": sub[stage], "speedup_vs_torch": sp,
        })
        seen_pass = seen_pass or status == "pass"
    return {"schema_version": 1, "events": events, "milestones": []}


def _process_env_saved():
    return {k: os.environ.pop(k, None) for k in _PROCESS_ENVS}


def _process_env_restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_process_reward_merges_into_score():
    # 一次通过(k=1)+ 进 optimization 且 speedup 1.0->1.2 拿满:process = 0.06+0.04 = 0.10(cap)
    saved = _process_env_saved()
    try:
        info = _mk_process_info([("done", "pass", None, 1.0), ("done", "pass", None, 1.2)])
        res, *_ = _run({"success": True, "perf_data": {"speedup_vs_torch": 1.2}},
                       process_info=info)
        raw = 0.75 + 0.25 * (1.2**2 - 1) / (1.2**2 + 1)
        assert abs(res.metadata["reward_outcome_raw"] - raw) < 1e-9
        assert abs(res.metadata["process_reward"] - 0.10) < 1e-9
        assert res.metadata["process_validation"] == "ok"
        assert abs(res.outcome_reward - min(raw + 0.10, 1.0)) < 1e-9
        comp = res.metadata["process_components"]
        assert abs(comp["b_first_pass"] - 0.06) < 1e-9
        assert abs(comp["b_opt_gain"] - 0.04) < 1e-9
        assert comp["p_repeat"] == 0.0
    finally:
        _process_env_restore(saved)


def test_process_reward_same_outcome_different_path():
    # 同样最终 1.0x:一次通过 vs 压哨连挂 5 次 compile,两条轨迹必须拉开
    saved = _process_env_saved()
    try:
        metrics = {"success": True, "perf_data": {"speedup_vs_torch": 1.0}}
        clean, *_ = _run(metrics, process_info=_mk_process_info([("done", "pass", None, 1.0)]))
        messy, *_ = _run(metrics, process_info=_mk_process_info(
            [("compile", "fail", "ascendc_compile_failed", None)] * 5
            + [("done", "pass", None, 1.0)]))
        assert clean.outcome_reward > messy.outcome_reward
        # clean:b_fp 满 0.06;raw=0.75 -> 0.81
        assert abs(clean.outcome_reward - 0.81) < 1e-9, clean.outcome_reward
        # messy:k=6,b_fp=0.06*(6+1-6)/6=0.01;p_repeat=0.04 -> raw+0.01-0.04=0.72
        assert abs(messy.outcome_reward - 0.72) < 1e-9, messy.outcome_reward
    finally:
        _process_env_restore(saved)


def test_process_info_forged_zeroes_component_not_outcome():
    # 过程记录虚报成功,但 judge 实测 correctness_failed:terminal_mismatch ->
    # process 分量 0,outcome 不受影响(0.35 原样),不 retry、不额外罚
    saved = _process_env_saved()
    try:
        forged = _mk_process_info([("done", "pass", None, 1.5)])
        res, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                        "error_type": "correctness_failed"},
                       process_info=forged)
        assert res.outcome_reward == 0.35
        assert res.metadata["process_reward"] == 0.0
        assert res.metadata["process_validation"] == "terminal_mismatch"
        assert res.metadata["process_components"] == {"disabled": "terminal_mismatch"}
    finally:
        _process_env_restore(saved)


def test_process_v2_budget_count_mismatch():
    # 预算计数器(gen=5)比事件数(1)多 = 删过事件 -> 分量 0
    saved = _process_env_saved()
    try:
        info = _mk_process_info([("done", "pass", None, 1.0)])
        res, *_ = _run({"success": True, "perf_data": {"speedup_vs_torch": 1.0}},
                       process_info=info,
                       budget_status={"gen_count": 5, "opt_count": 0})
        assert res.metadata["process_validation"] == "budget_count_mismatch"
        assert res.metadata["process_reward"] == 0.0
        assert res.outcome_reward == 0.75
    finally:
        _process_env_restore(saved)


def test_process_reward_env_off():
    # POLAR_PROCESS_REWARD=0 整体回退纯 outcome(校验仍跑,便于遥测观察)
    saved = _process_env_saved()
    os.environ["POLAR_PROCESS_REWARD"] = "0"
    try:
        info = _mk_process_info([("done", "pass", None, 1.0)])
        res, *_ = _run({"success": True, "perf_data": {"speedup_vs_torch": 1.0}},
                       process_info=info)
        assert res.outcome_reward == 0.75
        assert res.metadata["process_reward"] == 0.0
        assert res.metadata["process_validation"] == "ok"
        assert res.metadata["process_components"] == {"disabled": "env"}
    finally:
        _process_env_restore(saved)


def test_process_reward_then_truncation_penalty_order():
    # 合并顺序:先 process 再截断。fail 0.35 + process(连挂 3 次 verify:
    # b_fp=0.06(k=1,compile 首过) − p_rep=0.02 = +0.04) − 截断 2 次 0.02 = 0.37
    saved = _process_env_saved()
    saved.update({k: os.environ.pop(k, None) for k in _PENALTY_ENVS})
    try:
        info = _mk_process_info([("verify", "fail", "correctness_failed", None)] * 3)
        res, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                        "error_type": "correctness_failed"},
                       process_info=info,
                       traces=[Trace(finish_reason="length"), Trace(finish_reason="length")])
        assert abs(res.metadata["process_reward"] - 0.04) < 1e-9
        assert abs(res.metadata["truncation_penalty"] - 0.02) < 1e-9
        assert abs(res.outcome_reward - 0.37) < 1e-9, res.outcome_reward
    finally:
        _process_env_restore(saved)


_PENALTY_ENVS = ("POLAR_TRUNCATION_PENALTY", "POLAR_TRUNCATION_PENALTY_CAP", "POLAR_TRUNCATION_PENALTY_FLOOR")


def _penalty_env_saved():
    return {k: os.environ.pop(k, None) for k in _PENALTY_ENVS}


def _penalty_env_restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_truncation_events_penalize_reward():
    # finish_reason=length 的 trace 按次轻扣(默认 λ=0.01):0.35 档 2 次截断 -> 0.33
    saved = _penalty_env_saved()
    try:
        res, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                        "error_type": "correctness_failed"},
                       traces=[Trace(finish_reason="length"), Trace(finish_reason="length"),
                               Trace(finish_reason="tool_calls")])
        assert abs(res.outcome_reward - 0.33) < 1e-9, res.outcome_reward
        assert res.metadata["truncation_events"] == 2
        assert abs(res.metadata["truncation_penalty"] - 0.02) < 1e-9
        # 无截断 trace:不扣,审计字段在
        res2, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                         "error_type": "correctness_failed"},
                        traces=[Trace(finish_reason="stop")])
        assert res2.outcome_reward == 0.35
        assert res2.metadata["truncation_events"] == 0
        assert res2.metadata["truncation_penalty"] == 0.0
    finally:
        _penalty_env_restore(saved)


def test_truncation_penalty_cap_and_floor():
    saved = _penalty_env_saved()
    try:
        # 总扣分封顶 0.05:0.35 档 10 次截断 -> 0.30(不无限叠加)
        many = [Trace(finish_reason="length") for _ in range(10)]
        res, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                        "error_type": "correctness_failed"}, traces=many)
        assert abs(res.outcome_reward - 0.30) < 1e-9, res.outcome_reward
        # 下限 0.15:submission_missing 的 0.2 档重压不穿到 0
        res2, *_ = _run(None, impl=False, traces=many)
        assert abs(res2.outcome_reward - 0.15) < 1e-9, res2.outcome_reward
        assert res2.metadata["error_type"] == "submission_missing"
    finally:
        _penalty_env_restore(saved)


def test_truncation_penalty_env_off():
    # POLAR_TRUNCATION_PENALTY=0 整体回退:截断 trace 在场也不扣
    saved = _penalty_env_saved()
    try:
        os.environ["POLAR_TRUNCATION_PENALTY"] = "0"
        res, *_ = _run({"success": False, "ast_check_ok": True, "correctness_ok": False,
                        "error_type": "correctness_failed"},
                       traces=[Trace(finish_reason="length") for _ in range(3)])
        assert res.outcome_reward == 0.35
        assert res.metadata["truncation_events"] == 3
        assert res.metadata["truncation_penalty"] == 0.0
    finally:
        _penalty_env_restore(saved)


def test_operator_failure_scored_not_raised():
    res, *_ = _run({"success": False, "ast_check_ok": False, "correctness_ok": False,
                    "error_type": "correctness_failed"})
    assert res.outcome_reward == 0.2 and res.metadata["error_type"] == "correctness_failed"
    res2, *_ = _run({"success": False, "correctness_ok": True, "error_type": "benchmark_failed"})
    assert res2.outcome_reward == 0.4


def test_downloads_metrics_error_log_artifact():
    ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
    agent = FakeRuntime(files={SUB: "# kernel"})
    judge = FakeRuntime(files={
        METRICS: json.dumps({"success": False, "ast_check_ok": True, "correctness_ok": False,
                             "error_type": "correctness_failed"}),
        "judge_out/metrics_error.log": "shape mismatch detail",
    })
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
        ))
        metrics_error_path = Path(res.metadata["metrics_error_path"])
        assert metrics_error_path.is_file()
        assert metrics_error_path.read_text() == "shape mismatch detail"


def test_npu_init_failure_reclassified_as_infra_retry():
    ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
    agent = FakeRuntime(files={SUB: "# kernel"})
    judge = FakeRuntime(files={
        METRICS: json.dumps({"success": False, "ast_check_ok": True, "correctness_ok": False,
                             "error_type": "correctness_failed"}),
        "judge_out/metrics_error.log": (
            "数值验证失败: RuntimeError: aclInit, error code is 107001\n"
            "[ERROR] PTA call acl api failed\n"
            "[Error]: Invalid device ID.\n"
            "input error deviceId:0 is err:0x7010003\n"
        ),
    })
    with tempfile.TemporaryDirectory() as d:
        try:
            asyncio.run(ev.evaluate(
                Trajectory(status="COMPLETED", traces=[]),
                runtime=agent,
                fresh_eval_runtime=judge,
                refresh_runtime=True,
                artifacts_dir=d,
                env={},
                timeout_seconds=None,
                session_id="s",
                task_id="t",
            ))
        except RuntimeError as e:
            assert "npu_runtime_unavailable" in str(e)
        else:
            raise AssertionError("expected RuntimeError on NPU init infra failure")
        metrics = json.loads((Path(d) / "metrics.json").read_text())
        assert metrics["error_type"] == "npu_runtime_unavailable"
        assert metrics["original_error_type"] == "correctness_failed"


def test_infra_no_metrics_raises():
    try:
        _run(None)  # judge produced no metrics.json
    except RuntimeError as e:
        assert "metrics.json" in str(e)
    else:
        raise AssertionError("expected RuntimeError on missing metrics")


def test_infra_timeout_raises():
    try:
        _run({"success": True}, exec_rc=-1)  # judge pipeline timed out
    except TimeoutError:
        return
    raise AssertionError("expected TimeoutError on judge timeout")


def test_submission_missing_is_operator_floor():
    res, _agent, judge = _run({"success": True}, impl=False)  # agent wrote no kernel
    assert res.outcome_reward == 0.2 and res.metadata["error_type"] == "submission_missing"
    assert len(judge.execs) == 0  # judge never ran (nothing to score)


def test_workdir_resolves_relative_submission_to_absolute():
    # Regression (V4 false-negative): the agent writes the kernel under its workdir, but docker cp
    # resolves a bare relative path against the container ROOT -> the judge found NOTHING and floored
    # every rollout to 0.2 (submission_missing) even on a correct, fast kernel. With workdir set, the
    # relative submission_path must be resolved to the ABSOLUTE path for both download and upload.
    wd = "/opt/workspace/agent_workdir"
    abs_best = f"{wd}/{SUB[:-3]}.best.py"
    ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh",
                                metrics_path=METRICS, workdir=wd)
    agent = FakeRuntime(files={abs_best: "# best kernel"})   # ONLY the absolute path exists
    # judge writes metrics under cwd=workdir too -> operator_judge must download it via the absolute path.
    judge = FakeRuntime(files={f"{wd}/{METRICS}": json.dumps({"success": True,
                                                              "perf_data": {"speedup_vs_torch": 2.0}})})
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent, fresh_eval_runtime=judge, refresh_runtime=True,
            artifacts_dir=d, env={}, timeout_seconds=None, session_id="s", task_id="t",
        ))
    assert res.metadata["error_type"] is None and res.outcome_reward == 0.9   # found + scored, NOT missing
    assert res.metadata["submission_used"] == SUB[:-3] + ".best.py"           # logical (relative) label kept
    assert judge.uploaded and judge.uploaded[0][1] == f"{wd}/{SUB}"           # uploaded to the ABSOLUTE dest


def test_prefers_best_impl_then_falls_back():
    best = SUB[:-3] + ".best.py"
    res, *_ = _run({"success": True, "perf_data": {"speedup_vs_torch": 2.0}},
                   agent_files={best: "# best", SUB: "# final"})
    assert res.metadata["submission_used"] == best          # best-so-far wins (R1)
    res2, *_ = _run({"success": True}, agent_files={SUB: "# final"})
    assert res2.metadata["submission_used"] == SUB           # no best -> final


def test_refresh_without_fresh_runtime_raises():
    try:
        _run({"success": True}, with_fresh=False, refresh=True)
    except RuntimeError as e:
        assert "fresh_eval_runtime" in str(e)
    else:
        raise AssertionError("expected RuntimeError when refresh_runtime=true but no fresh runtime")


def test_host_submission_artifact_skips_agent_download():
    ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
    agent = FakeRuntime(files={})
    judge = FakeRuntime(files={METRICS: json.dumps({"success": True, "perf_data": {"speedup_vs_torch": 2.0}})})
    with tempfile.TemporaryDirectory() as d:
        host_impl = Path(d) / "submission_impl.py"
        host_impl.write_text("# host artifact")
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=None,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
            submission_host_path=str(host_impl),
            submission_used=SUB,
        ))
    assert res.outcome_reward == 0.9   # 0.75+0.25*(s^2-1)/(s^2+1), s=2 -> 0.9
    assert res.metadata["submission_used"] == SUB
    assert judge.uploaded and judge.uploaded[0][1] == SUB
    assert agent.files == {}


def test_missing_submission_artifact_scores_without_judge_runtime():
    ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=None,
            fresh_eval_runtime=None,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
            submission_missing=True,
        ))
    assert res.outcome_reward == 0.2
    assert res.metadata["error_type"] == "submission_missing"


def test_cannbot_judge_runs_native_verify_benchmark_without_budget_env():
    wd = "/opt/workspace/agent_workdir"
    verify_dir = f"{wd}/judge_out/cannbot_verify"
    verify_result = f"{verify_dir}/verify_result.json"
    perf_result = f"{verify_dir}/perf_result.json"
    ev = OperatorJudgeEvaluator(
        op_name=OP,
        judge_mode="cannbot",
        task_path=f"input/{OP}.py",
        workdir=wd,
    )
    agent = FakeRuntime(files={f"{wd}/output/optimized_code.py": "# impl"})
    judge = FakeRuntime(files={
        verify_result: json.dumps({"op_name": OP, "total_cases": 1, "passed_cases": 1, "failed_cases": 0}),
        perf_result: json.dumps({
            "op_name": OP,
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "speedup_vs_torch": 2.0,
        }),
    })
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={
                "POLAR_GEN_PIPELINE_MAX": "3",
                "POLAR_OPT_PIPELINE_MAX": "1",
                "POLAR_PIPELINE_PHASE": "generation",
                "POLAR_NPU_LEASE_POOL": "0-1",
            },
            timeout_seconds=None,
            session_id="s",
            task_id="t",
        ))

    assert res.outcome_reward == 0.9   # 0.75+0.25*(s^2-1)/(s^2+1), s=2 -> 0.9
    assert res.metadata["submission_used"] == "output/optimized_code.py"
    assert len(judge.execs) == 3
    assert "stage_verifier_inputs.py" in judge.execs[0]
    assert "verify.py" in judge.execs[1]
    assert "benchmark.py" in judge.execs[2]
    for env in judge.exec_envs:
        assert "POLAR_GEN_PIPELINE_MAX" not in env
        assert "POLAR_OPT_PIPELINE_MAX" not in env
        assert "POLAR_PIPELINE_PHASE" not in env
        assert env["POLAR_NPU_LEASE_POOL"] == "0-1"
    metrics = res.metadata["metrics"]
    assert metrics["success"] is True
    assert metrics["perf_data"]["speedup_vs_torch"] == 2.0


def test_cannbot_judge_prefers_phase5_final_artifact():
    wd = "/opt/workspace/agent_workdir"
    verify_dir = f"{wd}/judge_out/cannbot_verify"
    verify_result = f"{verify_dir}/verify_result.json"
    perf_result = f"{verify_dir}/perf_result.json"
    ev = OperatorJudgeEvaluator(op_name=OP, judge_mode="cannbot", workdir=wd)
    agent = FakeRuntime(files={
        f"{wd}/{OP}_generated.py": "# final",
        f"{wd}/output/optimized_code.py": "# stale optimized",
    })
    judge = FakeRuntime(files={
        verify_result: json.dumps({"op_name": OP, "total_cases": 1, "passed_cases": 1, "failed_cases": 0}),
        perf_result: json.dumps({
            "op_name": OP,
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "speedup_vs_torch": 2.0,
        }),
    })
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
        ))

    assert res.metadata["submission_used"] == f"{OP}_generated.py"
    assert res.outcome_reward == 0.9   # 0.75+0.25*(s^2-1)/(s^2+1), s=2 -> 0.9


def test_cannbot_judge_verify_failure_does_not_run_benchmark():
    wd = "/opt/workspace/agent_workdir"
    verify_dir = f"{wd}/judge_out/cannbot_verify"
    verify_result = f"{verify_dir}/verify_result.json"
    ev = OperatorJudgeEvaluator(op_name=OP, judge_mode="cannbot", workdir=wd)
    agent = FakeRuntime(files={f"{wd}/output/generated_code.py": "# impl"})
    judge = FakeRuntime(files={
        verify_result: json.dumps({
            "op_name": OP,
            "total_cases": 2,
            "passed_cases": 1,
            "failed_cases": 1,
            "failures": [{"case_idx": 2, "error_type": "AssertionError"}],
        }),
    })
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
        ))

    assert res.outcome_reward == 0.35   # 六档阶梯:跑完但精度错(correctness_failed)→ 0.35
    assert res.metadata["submission_used"] == "output/generated_code.py"
    assert len(judge.execs) == 2
    assert all("benchmark.py" not in command for command in judge.execs)
    assert res.metadata["error_type"] == "correctness_failed"


def test_cannbot_judge_verify_crash_without_json_scores_operator_failure():
    wd = "/opt/workspace/agent_workdir"
    ev = OperatorJudgeEvaluator(op_name=OP, judge_mode="cannbot", workdir=wd)
    agent = FakeRuntime(files={f"{wd}/output/generated_code.py": "# impl"})
    judge = FakeRuntime(files={}, exec_rcs=[0, 1])
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(ev.evaluate(
            Trajectory(status="COMPLETED", traces=[]),
            runtime=agent,
            fresh_eval_runtime=judge,
            refresh_runtime=True,
            artifacts_dir=d,
            env={},
            timeout_seconds=None,
            session_id="s",
            task_id="t",
        ))

    assert res.outcome_reward == 0.2
    assert res.metadata["error_type"] == "correctness_failed"
    assert len(judge.execs) == 2


if __name__ == "__main__":
    if not _DEPS:
        print(f"[skip] polar/pydantic not importable: {_IMPORT_ERR}")
        sys.exit(0)
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
