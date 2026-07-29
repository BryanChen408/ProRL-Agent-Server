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
    from polar.trajectory.models import Trajectory
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

    def _run(metrics, *, impl=True, exec_rc=0, refresh=True, with_fresh=True, agent_files=None):
        ev = OperatorJudgeEvaluator(op_name=OP, judge_command="bash pipeline.sh", metrics_path=METRICS)
        if agent_files is None:
            agent_files = {SUB: "# kernel"} if impl else {}
        agent = FakeRuntime(files=agent_files)
        judge = FakeRuntime(files=({METRICS: json.dumps(metrics)} if metrics is not None else {}),
                            exec_rc=exec_rc)
        with tempfile.TemporaryDirectory() as d:
            return asyncio.run(ev.evaluate(
                Trajectory(status="COMPLETED", traces=[]),
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

    assert res.outcome_reward == 0.3
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
