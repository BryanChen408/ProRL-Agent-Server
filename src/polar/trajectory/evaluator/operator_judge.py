"""``operator_judge`` evaluator — authoritative reward for Triton operator-gen on Ascend.

Flow (mirrors the user's openhands_sdk ``_judge_and_record``, but Polar-native):

  1. pull the agent's submitted kernel out of the AGENT runtime;
  2. drop ONLY that file into a CLEAN judge runtime (``refresh_runtime``: the agent can't have
     tampered with the canonical eval — that comes from the judge runtime's ``eval_prepare``);
  3. run the canonical eval pipeline THERE -> ``metrics.json``;
  4. map metrics -> reward via the shared, harness-agnostic ladder (:mod:`operator_reward`).

INFRA failures (judge couldn't run: container/setup/timeout/no-metrics) **raise** — the gateway turns
that into ``trajectory.status="ERROR"`` (node.py), rllm sees ``finished=False`` and retries, instead
of poisoning training with a false-negative 0 reward. OPERATOR results (bad/missing kernel) get the
real 0.2..1.0 ladder.

Everything image-specific is CONFIG (never hardcoded) so the agent image's python/conda/entrypoint can
change freely. Config (``EvaluatorSpec.config``):

  op_name         (str, required)
  judge_command   (str, required)  — shell run INSIDE the judge runtime; MUST write metrics.json
  submission_path (str, default ``output/submission/{op_name}_impl.py``) — path in the AGENT runtime
  submission_dest (str, default = submission_path)                       — path in the JUDGE runtime
  metrics_path    (str, default ``judge_out/metrics.json``)              — where judge_command writes it
  workdir         (str, optional) — cwd for judge_command
  judge_timeout   (float, default 1800)

Set ``evaluator.refresh_runtime: true`` in the request so a fresh judge runtime is provided (anti-cheat).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.evaluator.operator_reward import judge_outcome
from polar.trajectory.models import EvalResult, Trajectory

logger = logging.getLogger(__name__)


class OperatorJudgeEvaluator(BaseTrajectoryEvaluator):
    """Authoritative operator-gen reward: re-run the submitted kernel under the canonical eval."""

    MODE = "operator_judge"

    def __init__(
        self,
        *,
        op_name: str,
        judge_command: str,
        submission_path: str | None = None,
        submission_dest: str | None = None,
        metrics_path: str = "judge_out/metrics.json",
        workdir: str | None = None,
        judge_timeout: float = 1800.0,
        **_: Any,
    ) -> None:
        if not str(op_name).strip():
            raise ValueError("operator_judge requires 'op_name'")
        if not str(judge_command).strip():
            raise ValueError("operator_judge requires 'judge_command'")
        self.op_name = op_name
        self.judge_command = judge_command
        self.submission_path = submission_path or f"output/submission/{op_name}_impl.py"
        self.submission_dest = submission_dest or self.submission_path
        self.metrics_path = metrics_path
        self.workdir = workdir
        self.judge_timeout = float(judge_timeout)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        source = runtime.get("runtime")
        if not isinstance(source, BaseRuntime):
            raise RuntimeError("operator_judge requires a live agent runtime")
        fresh = runtime.get("fresh_eval_runtime")
        if bool(runtime.get("refresh_runtime")) and not isinstance(fresh, BaseRuntime):
            raise RuntimeError("operator_judge: refresh_runtime=true but no fresh_eval_runtime provided")
        judge_rt: BaseRuntime = fresh if isinstance(fresh, BaseRuntime) else source
        if judge_rt is source:
            logger.warning(
                "operator_judge running in the AGENT runtime (no fresh judge): anti-cheat weakened; "
                "set evaluator.refresh_runtime=true for the authoritative reward"
            )

        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        env = runtime.get("env") if isinstance(runtime.get("env"), dict) else {}
        timeout_cap = runtime.get("timeout_seconds")
        timeout = self.judge_timeout if timeout_cap is None else min(self.judge_timeout, float(timeout_cap))

        # 1) pull the submitted kernel out of the AGENT runtime. Absent == agent delivered nothing
        #    -> OPERATOR failure (floor reward), NOT infra (mirrors _judge_and_record).
        local_impl = artifacts_dir / "submission_impl.py"
        try:
            await source.download_file(self.submission_path, str(local_impl))
        except Exception as exc:  # noqa: BLE001
            return self._scored(
                {"success": False, "ast_check_ok": False, "correctness_ok": False,
                 "error_type": "submission_missing",
                 "error": f"no submission at {self.submission_path}: {exc!r}"},
                artifacts_dir,
            )

        # 2) place ONLY the impl into the judge runtime (canonical pipeline comes from eval_prepare).
        if judge_rt is not source:
            await judge_rt.upload_file(str(local_impl), self.submission_dest)

        # 3) run the canonical eval pipeline inside the judge runtime -> metrics.json.
        result = await judge_rt.exec(self.judge_command, cwd=self.workdir, env=env, timeout_sec=timeout)
        (artifacts_dir / "judge.stdout.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code == -1:
            raise TimeoutError(f"operator_judge: judge pipeline timed out after {timeout}s")  # infra -> retry

        # 4) read metrics.json. Missing/garbled after the judge ran == INFRA -> raise (retry, never score 0).
        local_metrics = artifacts_dir / "metrics.json"
        try:
            await judge_rt.download_file(self.metrics_path, str(local_metrics))
            metrics = json.loads(local_metrics.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"operator_judge: no readable metrics.json at {self.metrics_path} "
                f"(judge exit={result.return_code}; see {artifacts_dir / 'judge.stdout.log'}): {exc!r}"
            ) from exc

        return self._scored(metrics, artifacts_dir)

    def _scored(self, metrics: dict, artifacts_dir: Path) -> EvalResult:
        """metrics -> EvalResult; infra failures raise (=> session ERROR => retry)."""
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        outcome = judge_outcome(metrics)
        if outcome["retry"]:  # INFRA: do NOT fabricate a 0 reward — raise so the trainer retries.
            raise RuntimeError(
                f"operator_judge infra failure ({outcome['error_type']}) -> retry; "
                f"metrics at {artifacts_dir / 'metrics.json'}"
            )
        return EvalResult(
            outcome_reward=outcome["reward"],
            metadata={
                "mode": self.MODE,
                "op_name": self.op_name,
                "reward": outcome["reward"],
                "success": bool(metrics.get("success", False)),
                "error_type": outcome["error_type"],
                "speedup_vs_torch": (metrics.get("perf_data") or {}).get("speedup_vs_torch"),
                "metrics_path": str(artifacts_dir / "metrics.json"),
                "judge_stdout_path": str(artifacts_dir / "judge.stdout.log"),
                "metrics": metrics,
            },
        )
