"""``operator_judge`` evaluator — authoritative reward for Triton operator-gen on Ascend.

Flow (mirrors the user's openhands_sdk ``_judge_and_record``, but Polar-native):

  1. pull the agent's submitted kernel out of the AGENT runtime, unless the gateway
     already supplied a host ``submission_host_path`` for lazy fresh-runtime judging;
  2. drop that file into a fresh judge runtime when ``refresh_runtime`` is enabled;
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
  workdir         (str, optional) — cwd for judge_command; ALSO the base a *relative* submission_path /
                  submission_dest is resolved against before the docker cp / bind-mount transfer (the
                  agent writes the kernel under its workdir, but ``docker cp`` resolves a bare relative
                  path against the container ROOT — without this they'd never meet -> a deterministic
                  false-negative ``submission_missing``). Absolute paths pass through unchanged.
  judge_timeout   (float, default 1800)

Set ``evaluator.refresh_runtime: true`` in the request so final scoring uses a fresh judge runtime.
"""

from __future__ import annotations

import json
import logging
import posixpath
import shlex
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.evaluator.operator_reward import (
    apply_truncation_penalty,
    classify_infra_error_text,
    judge_outcome,
    process_reward,
    validate_process_info,
)
from polar.trajectory.models import EvalResult, Trajectory

logger = logging.getLogger(__name__)


class OperatorJudgeEvaluator(BaseTrajectoryEvaluator):
    """Authoritative operator-gen reward: re-run the submitted kernel under the canonical eval."""

    MODE = "operator_judge"
    LEGACY_MODE = "legacy"
    CANNBOT_MODE = "cannbot"
    CANNBOT_BUDGET_ENV_KEYS = frozenset(
        {
            "POLAR_GEN_PIPELINE_MAX",
            "POLAR_OPT_PIPELINE_MAX",
            "POLAR_PIPELINE_PHASE",
            "POLAR_PIPELINE_STATUS_FILE",
        }
    )

    def __init__(
        self,
        *,
        op_name: str,
        judge_command: str = "",
        judge_mode: str = LEGACY_MODE,
        submission_path: str | None = None,
        submission_candidates: list[str] | None = None,
        submission_dest: str | None = None,
        metrics_path: str = "judge_out/metrics.json",
        metrics_error_path: str = "judge_out/metrics_error.log",
        workdir: str | None = None,
        judge_timeout: float = 1800.0,
        cannbot_runtime_root: str = "/opt/canonical/cannbot",
        task_path: str | None = None,
        verify_dir: str = "judge_out/cannbot_verify",
        triton_impl_name: str = "triton_ascend_impl",
        **_: Any,
    ) -> None:
        if not str(op_name).strip():
            raise ValueError("operator_judge requires 'op_name'")
        self.judge_mode = str(judge_mode or self.LEGACY_MODE).strip().lower()
        if self.judge_mode not in {self.LEGACY_MODE, self.CANNBOT_MODE}:
            raise ValueError(f"operator_judge unsupported judge_mode: {judge_mode!r}")
        if self.judge_mode == self.LEGACY_MODE and not str(judge_command).strip():
            raise ValueError("operator_judge requires 'judge_command'")
        self.op_name = op_name
        self.judge_command = judge_command
        self.submission_path = submission_path or self._default_submission_path()
        self.submission_candidates = list(submission_candidates or self._default_submission_candidates())
        self.submission_dest = submission_dest or self._default_submission_dest()
        self.metrics_path = metrics_path
        self.metrics_error_path = metrics_error_path
        self.workdir = workdir
        self.judge_timeout = float(judge_timeout)
        self.cannbot_runtime_root = cannbot_runtime_root.rstrip("/")
        self.task_path = task_path or f"input/{op_name}.py"
        self.verify_dir = verify_dir
        self.triton_impl_name = triton_impl_name

    def _default_submission_path(self) -> str:
        if self.judge_mode == self.CANNBOT_MODE:
            return f"{self.op_name}_generated.py"
        return f"output/submission/{self.op_name}_impl.py"

    def _default_submission_candidates(self) -> list[str]:
        if self.judge_mode == self.CANNBOT_MODE:
            return [
                f"{self.op_name}_generated.py",
                "output/optimized_code.py",
                "output/generated_code.py",
            ]
        candidates: list[str] = []
        if self.submission_path.endswith(".py"):
            candidates.append(self.submission_path[:-3] + ".best.py")
        candidates.append(self.submission_path)
        return candidates

    def _default_submission_dest(self) -> str:
        if self.judge_mode == self.CANNBOT_MODE:
            return "judge_out/cannbot_submission.py"
        return self.submission_path

    def _abs(self, path: str) -> str:
        """Resolve a runtime path against ``workdir`` so the transfer actually finds it.

        The agent writes the submission under its workdir (e.g. ``/opt/workspace/agent_workdir/
        output/submission/...``), but ``DockerRuntime.download_file`` -> ``docker cp`` resolves a bare
        relative path against the container ROOT (``/output/submission/...``) and the bind-mount fast
        path only covers ``/polar/session`` — so a relative path is found by NEITHER and the judge
        reports ``submission_missing`` even on a perfect kernel (deterministic false-negative that
        floors every rollout to 0.2 and zeroes the GRPO group's advantage). Joining ``workdir`` makes
        ``docker cp`` hit the real file. Absolute paths and a missing workdir pass through unchanged.
        Assumes the judge runtime mirrors the agent's workdir layout (it does: same WORKDIR + eval_prepare).
        """
        if self.workdir and not posixpath.isabs(path):
            return posixpath.join(self.workdir, path)
        return path

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        source = runtime.get("runtime")
        fresh = runtime.get("fresh_eval_runtime")
        refresh_runtime = bool(runtime.get("refresh_runtime"))
        submission_host_path = runtime.get("submission_host_path")
        submission_missing = bool(runtime.get("submission_missing"))
        if (
            not submission_missing
            and submission_host_path is None
            and not isinstance(source, BaseRuntime)
        ):
            raise RuntimeError("operator_judge requires a live agent runtime")
        if refresh_runtime and not submission_missing and not isinstance(fresh, BaseRuntime):
            raise RuntimeError("operator_judge: refresh_runtime=true but no fresh_eval_runtime provided")
        judge_rt = fresh if isinstance(fresh, BaseRuntime) else source
        if not isinstance(judge_rt, BaseRuntime):
            if submission_missing:
                judge_rt = None
            else:
                raise RuntimeError("operator_judge requires a live judge runtime")
        if isinstance(judge_rt, BaseRuntime) and judge_rt is source:
            logger.warning(
                "operator_judge running in the agent runtime; "
                "set evaluator.refresh_runtime=true for fresh-runtime final scoring"
            )

        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        env = runtime.get("env") if isinstance(runtime.get("env"), dict) else {}
        timeout_cap = runtime.get("timeout_seconds")
        timeout = self.judge_timeout if timeout_cap is None else min(self.judge_timeout, float(timeout_cap))

        # 截断事件计数(训练信号,见 operator_reward.apply_truncation_penalty)。优先读
        # builder 的 completion 级统计(空截断轮修复后不再单独成 trace);旧落盘没有
        # 该字段时回退到按 finish_reason=length 的 trace 数近似。
        truncation_events = (trajectory.metadata or {}).get("truncation_events")
        if truncation_events is None:
            truncation_events = sum(
                1 for t in (trajectory.traces or []) if t.finish_reason == "length"
            )

        if submission_missing:
            return self._scored(
                {"success": False, "ast_check_ok": False, "correctness_ok": False,
                 "error_type": "submission_missing",
                 "error": f"no submission in candidates: {self.submission_candidates}"},
                artifacts_dir, submission_used=None,
                truncation_events=truncation_events,
            )

        # 1) pull the submitted kernel out of the AGENT runtime. R1 anti-regression (mirrors
        #    _judge_and_record): prefer the best-so-far successful impl the fixed entry saves on each
        #    success ({op}_impl.best.py) so a later optimization that breaks the kernel can't drag the
        #    reward below what was already achieved; fall back to the final impl. Absent == agent
        #    delivered nothing -> OPERATOR failure (floor reward), NOT infra.
        local_impl = artifacts_dir / "submission_impl.py"
        picked: str | None
        if submission_host_path is not None:
            local_impl = Path(str(submission_host_path))
            if not local_impl.is_file():
                return self._scored(
                    {"success": False, "ast_check_ok": False, "correctness_ok": False,
                     "error_type": "submission_missing",
                     "error": f"host submission artifact is missing: {local_impl}"},
                    artifacts_dir, submission_used=None,
                    truncation_events=truncation_events,
                )
            picked_value = runtime.get("submission_used")
            picked = str(picked_value) if picked_value else str(local_impl)
        else:
            assert isinstance(source, BaseRuntime)
            picked = None
            # Keep每个候选的真实失败原因。原实现是 `except Exception: continue`,把
            # "文件不存在"(agent 没交 → 记 0.2 合理)和"源容器已销毁 / 传输失败"
            # (infra 故障 → 应 retry 不计分)塌缩成同一条 submission_missing,
            # 事后无从区分 —— 实测有 session 打包成功 10 次仍报 missing 而查不下去。
            attempts: list[str] = []
            for cand in self.submission_candidates:
                try:
                    await source.download_file(self._abs(cand), str(local_impl))
                    picked = cand  # report the logical (relative) path; _abs is a transfer detail
                    break
                except Exception as exc:  # noqa: BLE001 — try the next candidate (best -> final)
                    attempts.append(f"{cand}: {type(exc).__name__}: {exc}")
                    continue
            if picked is None:
                detail = "; ".join(attempts) if attempts else "no candidates configured"
                # 传输层故障(容器没了/连不上/超时)与"文件确实不存在"要分开:前者是 infra,
                # 后者才是 agent 的锅。判据用异常类型名,不看文案(文案随 runtime 实现变)。
                transport = any(
                    kind in a
                    for a in attempts
                    for kind in (
                        "ConnectionError", "TimeoutError", "asyncio.TimeoutError",
                        "ContainerNotFound", "RuntimeNotAvailable", "OSError",
                    )
                )
                return self._scored(
                    {"success": False, "ast_check_ok": False, "correctness_ok": False,
                     "error_type": "submission_fetch_failed" if transport else "submission_missing",
                     "error": f"no submission in candidates: {self.submission_candidates} | {detail}"},
                    artifacts_dir, submission_used=None,
                    truncation_events=truncation_events,
                )

        # 2) place ONLY the impl into the judge runtime (canonical pipeline comes from eval_prepare).
        assert isinstance(judge_rt, BaseRuntime)
        if judge_rt is not source or submission_host_path is not None:
            await judge_rt.upload_file(str(local_impl), self._abs(self.submission_dest))

        if self.judge_mode == self.CANNBOT_MODE:
            return await self._evaluate_cannbot(
                judge_rt,
                artifacts_dir,
                env,
                timeout,
                submission_used=picked,
                truncation_events=truncation_events,
            )

        # 3) run the canonical eval pipeline inside the judge runtime -> metrics.json.
        result = await judge_rt.exec(self.judge_command, cwd=self.workdir, env=env, timeout_sec=timeout)
        (artifacts_dir / "judge.stdout.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code == -1:
            raise TimeoutError(f"operator_judge: judge pipeline timed out after {timeout}s")  # infra -> retry

        # 4) read metrics.json. Missing/garbled after the judge ran == INFRA -> raise (retry, never score 0).
        #    _abs: judge_command writes it relative to cwd=workdir, so download via the absolute path too
        #    (same docker-cp-resolves-against-container-root trap as the submission).
        local_metrics = artifacts_dir / "metrics.json"
        try:
            await judge_rt.download_file(self._abs(self.metrics_path), str(local_metrics))
            metrics = json.loads(local_metrics.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"operator_judge: no readable metrics.json at {self.metrics_path} "
                f"(judge exit={result.return_code}; see {artifacts_dir / 'judge.stdout.log'}): {exc!r}"
            ) from exc

        local_metrics_error = artifacts_dir / "metrics_error.log"
        try:
            await judge_rt.download_file(self._abs(self.metrics_error_path), str(local_metrics_error))
        except Exception:
            local_metrics_error = None
        if local_metrics_error is not None:
            metrics = self._with_error_log_infra_classification(metrics, local_metrics_error)
            local_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

        return self._scored(
            metrics,
            artifacts_dir,
            submission_used=picked,
            metrics_error_path=str(local_metrics_error) if local_metrics_error is not None else None,
            truncation_events=truncation_events,
        )

    async def _evaluate_cannbot(
        self,
        judge_rt: BaseRuntime,
        artifacts_dir: Path,
        env: dict,
        timeout: float,
        *,
        submission_used: str | None,
        truncation_events: int = 0,
    ) -> EvalResult:
        verify_dir = self._abs(self.verify_dir)
        verify_result_path = posixpath.join(verify_dir, "verify_result.json")
        perf_result_path = posixpath.join(verify_dir, "perf_result.json")
        cannbot_env = self._cannbot_env(env)
        logs: list[str] = []

        commands = [
            self._cannbot_stage_command(verify_dir),
            self._cannbot_verify_command(verify_dir, verify_result_path),
        ]
        verify_rc = 0
        for command in commands:
            result = await judge_rt.exec(command, cwd=self.workdir, env=cannbot_env, timeout_sec=timeout)
            logs.append(self._format_command_log(command, result.return_code, result.stdout, result.stderr))
            if result.return_code == -1:
                (artifacts_dir / "judge.stdout.log").write_text("".join(logs))
                raise TimeoutError(f"operator_judge cannbot command timed out after {timeout}s: {command}")
            if result.return_code != 0:
                verify_rc = result.return_code
                if command != commands[-1]:
                    (artifacts_dir / "judge.stdout.log").write_text("".join(logs))
                    raise RuntimeError(
                        f"operator_judge cannbot staging failed "
                        f"(exit={result.return_code}; see {artifacts_dir / 'judge.stdout.log'})"
                    )
                break

        verify_data = await self._download_json(
            judge_rt,
            verify_result_path,
            artifacts_dir / "verify_result.json",
            label="verify_result.json",
            required=False,
        )
        if verify_data is None:
            combined_log = "".join(logs)
            (artifacts_dir / "judge.stdout.log").write_text(combined_log)
            infra_type = classify_infra_error_text(combined_log)
            if infra_type:
                raise RuntimeError(
                    f"operator_judge cannbot infra failure ({infra_type}); "
                    f"see {artifacts_dir / 'judge.stdout.log'}"
                )
            return self._scored(
                {
                    "success": False,
                    "ast_check_ok": False,
                    "correctness_ok": False,
                    "error_type": "correctness_failed",
                    "error": "verify.py failed before writing verify_result.json",
                    "verify_return_code": verify_rc,
                },
                artifacts_dir,
                submission_used=submission_used,
                truncation_events=truncation_events,
            )
        if not self._cannbot_verify_ok(verify_data):
            (artifacts_dir / "judge.stdout.log").write_text("".join(logs))
            return self._scored(
                self._cannbot_metrics(verify_data=verify_data, perf_data=None, verify_rc=verify_rc),
                artifacts_dir,
                submission_used=submission_used,
                truncation_events=truncation_events,
            )

        benchmark_command = self._cannbot_benchmark_command(verify_dir, perf_result_path)
        result = await judge_rt.exec(benchmark_command, cwd=self.workdir, env=cannbot_env, timeout_sec=timeout)
        logs.append(self._format_command_log(benchmark_command, result.return_code, result.stdout, result.stderr))
        (artifacts_dir / "judge.stdout.log").write_text("".join(logs))
        if result.return_code == -1:
            raise TimeoutError(f"operator_judge cannbot benchmark timed out after {timeout}s")

        perf_data = await self._download_json(
            judge_rt,
            perf_result_path,
            artifacts_dir / "perf_result.json",
            label="perf_result.json",
        )
        return self._scored(
            self._cannbot_metrics(
                verify_data=verify_data,
                perf_data=perf_data,
                verify_rc=verify_rc,
                benchmark_rc=result.return_code,
            ),
            artifacts_dir,
            submission_used=submission_used,
            truncation_events=truncation_events,
        )

    def _cannbot_stage_command(self, verify_dir: str) -> str:
        return self._shell_command(
            [
                "python3",
                f"{self.cannbot_runtime_root}/runtime/stage_verifier_inputs.py",
                "--op-name",
                self.op_name,
                "--task",
                self._abs(self.task_path),
                "--impl",
                self._abs(self.submission_dest),
                "--verify-dir",
                verify_dir,
                "--triton-impl-name",
                self.triton_impl_name,
            ]
        )

    def _cannbot_verify_command(self, verify_dir: str, output_path: str) -> str:
        return self._shell_command(
            [
                "python3",
                f"{self.cannbot_runtime_root}/skills/triton-op-verifier/scripts/verify.py",
                "--op_name",
                self.op_name,
                "--verify_dir",
                verify_dir,
                "--triton_impl_name",
                self.triton_impl_name,
                "--output",
                output_path,
            ]
        )

    def _cannbot_benchmark_command(self, verify_dir: str, output_path: str) -> str:
        return self._shell_command(
            [
                "python3",
                f"{self.cannbot_runtime_root}/skills/triton-op-verifier/scripts/benchmark.py",
                "--op_name",
                self.op_name,
                "--verify_dir",
                verify_dir,
                "--triton_impl_name",
                self.triton_impl_name,
                "--output",
                output_path,
            ]
        )

    @staticmethod
    def _shell_command(args: list[str]) -> str:
        return " ".join(shlex.quote(str(arg)) for arg in args)

    @classmethod
    def _cannbot_env(cls, env: dict) -> dict:
        return {str(k): v for k, v in env.items() if str(k) not in cls.CANNBOT_BUDGET_ENV_KEYS}

    @staticmethod
    def _format_command_log(command: str, return_code: int, stdout: str | None, stderr: str | None) -> str:
        return (
            f"\n$ {command}\n"
            f"[exit={return_code}]\n"
            f"{stdout or ''}"
            f"{stderr or ''}"
        )

    async def _download_json(
        self,
        judge_rt: BaseRuntime,
        remote_path: str,
        local_path: Path,
        *,
        label: str,
        required: bool = True,
    ) -> dict | None:
        try:
            await judge_rt.download_file(remote_path, str(local_path))
        except Exception as exc:  # noqa: BLE001
            if not required:
                return None
            raise RuntimeError(f"operator_judge cannbot: no readable {label} at {remote_path}: {exc!r}") from exc
        try:
            data = json.loads(local_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"operator_judge cannbot: malformed {label} at {remote_path}: {exc!r}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"operator_judge cannbot: {label} is not a JSON object")
        return data

    @staticmethod
    def _cannbot_verify_ok(verify_data: dict) -> bool:
        try:
            total = int(verify_data.get("total_cases") or 0)
            passed = int(verify_data.get("passed_cases") or 0)
        except Exception:
            return False
        return total > 0 and passed == total

    @staticmethod
    def _cannbot_benchmark_ok(perf_data: dict | None) -> bool:
        if not perf_data:
            return False
        try:
            total = int(perf_data.get("total_cases") or 0)
            passed = int(perf_data.get("passed_cases") or 0)
        except Exception:
            return False
        return total > 0 and passed == total and perf_data.get("speedup_vs_torch") is not None

    def _cannbot_metrics(
        self,
        *,
        verify_data: dict,
        perf_data: dict | None,
        verify_rc: int = 0,
        benchmark_rc: int | None = None,
    ) -> dict:
        if not self._cannbot_verify_ok(verify_data):
            return {
                "success": False,
                "ast_check_ok": True,
                "correctness_ok": False,
                "error_type": "correctness_failed",
                "verify_result": verify_data,
                "verify_return_code": verify_rc,
            }
        if not self._cannbot_benchmark_ok(perf_data):
            return {
                "success": False,
                "ast_check_ok": True,
                "correctness_ok": True,
                "error_type": "benchmark_failed",
                "verify_result": verify_data,
                "perf_data": perf_data,
                "benchmark_return_code": benchmark_rc,
            }
        return {
            "success": True,
            "ast_check_ok": True,
            "correctness_ok": True,
            "error_type": None,
            "verify_result": verify_data,
            "perf_data": perf_data,
            "benchmark_return_code": benchmark_rc,
        }

    @staticmethod
    def _with_error_log_infra_classification(metrics: dict, error_log_path: Path) -> dict:
        """Correct stale/over-broad pipeline labels using the full error log.

        Older pipeline copies wrapped every verify failure as correctness_failed. When the full log
        shows an Ascend init/device-visibility failure, no operator code ran, so the judge must retry
        instead of scoring a false correctness failure.
        """
        try:
            infra_type = classify_infra_error_text(error_log_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return metrics
        if not infra_type or metrics.get("error_type") == infra_type:
            return metrics
        return {**metrics, "original_error_type": metrics.get("error_type"), "error_type": infra_type}

    def _load_process_events(
        self, artifacts_dir: Path, metrics: dict
    ) -> tuple[list[dict] | None, str]:
        """读 process_info.json(agent 侧固定入口经 $ARTIFACTS_DIR bind mount 直落宿主机,
        与 metrics.json 同目录),跑 V1-V5 校验;任一不过 -> (None, 原因),process 分量记 0,
        不影响 outcome、不产生 infra retry。详见 operator_reward.validate_process_info。
        """
        raw_path = artifacts_dir / "process_info.json"
        if not raw_path.is_file():
            return None, "missing"  # V5:未接入/被删/老 session —— 与改造前行为一致
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None, "unreadable"  # V1
        events, why = validate_process_info(data, metrics)
        if events is None:
            return None, why
        # V2:与 pipeline_budget_status.json 交叉比对(同一 artifacts_dir)。只查「计数器
        # 比事件多 = 删过事件」一个方向:计数器在脚本尾才 +1,最后一次评测中途崩溃会少 1,
        # 不能用 != 冤枉正常轨迹。文件缺席(老 run/cannbot)跳过此校验。
        budget_path = artifacts_dir / "pipeline_budget_status.json"
        if budget_path.is_file():
            try:
                budget = json.loads(budget_path.read_text(encoding="utf-8"))
                counted = int(budget.get("gen_count") or 0) + int(budget.get("opt_count") or 0)
                if counted > len(events):
                    return None, "budget_count_mismatch"
            except Exception:  # noqa: BLE001 —— 预算文件损坏不株连 process 分
                pass
        return events, "ok"

    def _scored(
        self,
        metrics: dict,
        artifacts_dir: Path,
        *,
        submission_used: str | None = None,
        metrics_error_path: str | None = None,
        truncation_events: int = 0,
    ) -> EvalResult:
        """metrics -> EvalResult; infra failures raise (=> session ERROR => retry)."""
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        outcome = judge_outcome(metrics)
        if outcome["retry"]:  # INFRA: do NOT fabricate a 0 reward — raise so the trainer retries.
            raise RuntimeError(
                f"operator_judge infra failure ({outcome['error_type']}) -> retry; "
                f"metrics at {artifacts_dir / 'metrics.json'}"
            )
        # 过程奖励(dev_04/dev_05):infra retry 分支在上面已经 raise,走不到这里(C4)。
        # 合并顺序:先 process 塑形(±Δ,floor 0.15 / ceil 1.0),再截断惩罚(独立作用于
        # 最终分,其内部 floor 依然兜底)。校验不过/文件缺失 -> 分量 0,与改造前逐分一致。
        events, process_why = self._load_process_events(artifacts_dir, metrics)
        if events is not None:
            r_proc, process_components = process_reward(events, metrics)
        else:
            r_proc, process_components = 0.0, {"disabled": process_why}
        base_reward = min(max(outcome["reward"] + r_proc, 0.15), 1.0)
        # 截断事件轻扣(空截断在阶梯里原本零成本,salvage 救回后与干净 session 同分 -> 无负
        # 方向 -> 永不收敛;见 operator_reward.apply_truncation_penalty)。截断段 token 本体
        # 仍不过梯度。扣量写进 metadata,原 reward 可还原(reward + truncation_penalty)。
        reward, truncated_deduction = apply_truncation_penalty(base_reward, truncation_events)
        return EvalResult(
            outcome_reward=reward,
            metadata={
                "mode": self.MODE,
                "op_name": self.op_name,
                "reward": reward,
                "reward_outcome_raw": outcome["reward"],  # 原 outcome 分;R = raw + process_reward - truncation_penalty
                "process_reward": r_proc,
                "process_components": process_components,
                "process_validation": process_why,
                "success": bool(metrics.get("success", False)),
                "error_type": outcome["error_type"],
                "speedup_vs_torch": (metrics.get("perf_data") or {}).get("speedup_vs_torch"),
                "truncation_events": truncation_events,
                "truncation_penalty": truncated_deduction,
                "submission_used": submission_used,  # which impl scored (best-so-far vs final)
                "metrics_path": str(artifacts_dir / "metrics.json"),
                "metrics_error_path": metrics_error_path,
                "judge_stdout_path": str(artifacts_dir / "judge.stdout.log"),
                "metrics": metrics,
            },
        )
