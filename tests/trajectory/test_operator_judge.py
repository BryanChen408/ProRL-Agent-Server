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

        def __init__(self, files: dict | None = None, exec_rc: int = 0) -> None:
            self.files = files or {}
            self.exec_rc = exec_rc
            self.uploaded: list[tuple[str, str]] = []
            self.execs: list[str] = []

        @property
        def runtime_id(self) -> str:
            return "fake"

        async def start(self) -> None: ...
        async def stop(self) -> None: ...

        async def exec(self, command, *, cwd=None, env=None, timeout_sec=None):
            self.execs.append(command)
            return ExecResult(stdout="judge ran\n", stderr="", return_code=self.exec_rc)

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
    assert res.outcome_reward == 1.0
    assert res.metadata["success"] is True and res.metadata["error_type"] is None
    assert len(judge.uploaded) == 1 and len(judge.execs) == 1  # impl crossed + judge ran


def test_operator_failure_scored_not_raised():
    res, *_ = _run({"success": False, "ast_check_ok": False, "correctness_ok": False,
                    "error_type": "correctness_failed"})
    assert res.outcome_reward == 0.2 and res.metadata["error_type"] == "correctness_failed"
    res2, *_ = _run({"success": False, "correctness_ok": True, "error_type": "benchmark_failed"})
    assert res2.outcome_reward == 0.4


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
