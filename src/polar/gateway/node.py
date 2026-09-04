"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import shutil
import time
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import httpx

from polar.gateway.dispatcher import (
    DispatcherSnapshot,
    ManagedSession,
    SessionDispatcher,
    SessionStage,
)
from polar.gateway.inflight import InflightGenerationTracker
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.agent.base import BaseHarness
from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult
from polar.run_namespace import run_dir_name, run_id_from_metadata
from polar.rollout.agent_actions import extract_completed_agent_actions
from polar.rollout.artifacts import persist_profiling_artifacts
from polar.rollout.models import (
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    NodeStageMetrics,
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
)
from polar.rollout.observability import SessionObservability
from polar.rollout.timer import StageTimer
from polar.rollout.trace_exporter import build_chrome_trace_document
from polar.runtime.base import BaseRuntime
from polar.runtime.factory import create_runtime
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.models import EvalResult, EvaluatorSpec, StrategySpec, Trajectory
from polar.trajectory.registry import StrategyRegistry

logger = logging.getLogger(__name__)


def _is_retryable_judge_infra(exc: BaseException) -> bool:
    """operator_judge 的 infra 失败(判 judge 级重试) vs 真错误(不重试)。

    infra 失败 = golden 缺失/NPU 不可用/容器传输失败/超时等环境故障,重起 runtime 重判
    可能恢复;其他异常(bug、config 错、真算子失败)重判结果一样,不重。
    operator_judge 把 infra 失败 raise 为带 "infra failure" 的 RuntimeError 或 TimeoutError。
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    return "infra failure" in str(exc).lower()


class GatewayExecutionTimeout(TimeoutError):
    """Raised when a session exhausts its shared gateway execution budget."""


class GatewayNodeManager:
    """Run the INIT/READY/RUN/POST_RUN lifecycle on one gateway node."""

    def __init__(
        self,
        *,
        node_id: str,
        gateway_url: str,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
        storage: SessionStore,
        session_registry: SessionRegistry,
        builders: StrategyRegistry,
        evaluators: StrategyRegistry,
        default_runtime: RuntimeSpec | None = None,
        session_base_dir: str | None = None,
        rollout_server_url: str | None = None,
        heartbeat_interval_seconds: int = 30,
        inflight: InflightGenerationTracker | None = None,
        session_affinity_release_url: str | None = None,
        # ── Per-session trace artifacts ──
        enable_session_trace_wandb: bool = False,
        enable_session_trace_json: bool = True,
        session_trace_wandb_project: str = "polar-session-traces",
        persist_traces_dir: str | None = None,
        persist_session_artifacts: bool = True,
        session_artifacts_max_bytes: int = 2 * 1024 * 1024 * 1024,
        session_artifacts_max_files: int = 1000,
        prometheus_enabled: bool = True,
        rl_insight_url: str | None = None,
        otlp_endpoint: str | None = None,
        otlp_headers: dict[str, str] | None = None,
        otlp_include_action_content: bool = False,
        observability_export_timeout_seconds: float = 3.0,
        observability_registration_refresh_seconds: float = 30.0,
        observability_service_name: str = "polar-gateway",
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.storage = storage
        self.inflight = inflight
        self.session_registry = session_registry
        self.builders = builders
        self.evaluators = evaluators
        self.default_runtime = default_runtime
        self._session_base_dir = session_base_dir
        # Per-session tracing
        self._enable_session_trace_wandb = enable_session_trace_wandb
        self._enable_session_trace_json = enable_session_trace_json
        self._session_trace_wandb_project = session_trace_wandb_project
        self._persist_traces_dir = Path(persist_traces_dir) if persist_traces_dir else None
        self._persist_session_artifacts = persist_session_artifacts
        self._session_artifacts_max_bytes = session_artifacts_max_bytes
        self._session_artifacts_max_files = session_artifacts_max_files
        self.observability = SessionObservability(
            node_id=node_id,
            gateway_url=self.gateway_url,
            prometheus_enabled=prometheus_enabled,
            rl_insight_url=rl_insight_url,
            otlp_endpoint=otlp_endpoint,
            otlp_headers=otlp_headers,
            otlp_include_action_content=otlp_include_action_content,
            export_timeout_seconds=observability_export_timeout_seconds,
            registration_refresh_seconds=observability_registration_refresh_seconds,
            service_name=observability_service_name,
        )
        self._observability_registration_task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(timeout=30.0)
        self._dispatcher = SessionDispatcher(
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
        )
        self._dispatcher.on_init = self._handle_init
        self._dispatcher.on_run = self._handle_run
        self._dispatcher.on_postrun = self._handle_postrun
        self._dispatcher.on_stage_change = self._handle_dispatcher_stage_change

        self._rollout_server_url = rollout_server_url.rstrip("/") if rollout_server_url else None
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._control_client: httpx.AsyncClient | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._policy_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._session_affinity_release_url = (
            session_affinity_release_url.rstrip("/")
            if session_affinity_release_url
            else None
        )

    async def start(self) -> None:
        await self._dispatcher.start()
        await self.observability.register_prometheus_target(self._client)
        if self.observability.registration_enabled:
            self._observability_registration_task = asyncio.create_task(
                self.observability.refresh_prometheus_registration(self._client)
            )
        if self._rollout_server_url is not None:
            self._control_client = httpx.AsyncClient(
                base_url=self._rollout_server_url, timeout=15.0
            )
            await self._register_with_rollout_server()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        if self._observability_registration_task is not None:
            self._observability_registration_task.cancel()
            await asyncio.gather(
                self._observability_registration_task,
                return_exceptions=True,
            )
            self._observability_registration_task = None
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self._control_client is not None:
            await self._control_client.aclose()
            self._control_client = None
        await self._dispatcher.stop()
        cleanup_tasks = list(self._policy_cleanup_tasks)
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._policy_cleanup_tasks.clear()
        await self._client.aclose()

    async def _register_with_rollout_server(self) -> None:
        if self._control_client is None:
            return
        try:
            response = await self._control_client.post(
                "/nodes/register",
                json=NodeRegistrationRequest(
                    node_id=self.node_id,
                    gateway_url=self.gateway_url,
                    max_init_workers=self.max_init_workers,
                    max_run_workers=self.max_run_workers,
                    max_postrun_workers=self.max_postrun_workers,
                    heartbeat_interval_seconds=self._heartbeat_interval_seconds,
                ).model_dump(mode="json"),
            )
            response.raise_for_status()
        except Exception:
            logger.warning("Node registration failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        assert self._control_client is not None
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                metrics = await self.stage_metrics()
                response = await self._control_client.post(
                    f"/nodes/{self.node_id}/heartbeat",
                    json=NodeHeartbeatRequest(metrics=metrics).model_dump(mode="json"),
                )
                if response.status_code == 404:
                    await self._register_with_rollout_server()
                    continue
                response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Node heartbeat failed", exc_info=True)

    async def dispatch(self, request: SessionDispatchRequest) -> None:
        session_id = request.session_id
        if self.session_registry.get(session_id) is not None:
            raise ValueError(
                f"session {session_id} already exists; rollout session IDs are single-use"
            )

        session_dir: Path | None = None
        try:
            info = self.session_registry.register(
                session_id,
                task_id=request.task_id,
                registered=True,
                status=SessionStatus.REGISTERED,
                metadata=dict(request.metadata),
            )
            self.storage.ensure_session(
                info.session_id,
                model_requested=None,
                model_used=None,
                api_type=None,
                task_id=info.task_id,
                created_at=info.created_at.isoformat(),
                metadata=dict(request.metadata),
            )

            timer = StageTimer()
            timer.mark("dispatch", "started")
            session_parent = Path(self._session_base_dir) if self._session_base_dir else None
            run_dir = run_dir_name(run_id_from_metadata(request.task_id, request.metadata))
            if session_parent is not None and run_dir:
                session_parent = session_parent / run_dir
                session_parent.mkdir(parents=True, exist_ok=True)
            session_dir = Path(
                mkdtemp(
                    prefix=f"session-{session_id[:8]}-",
                    dir=str(session_parent) if session_parent is not None else None,
                )
            )
            artifacts_dir = session_dir / "artifacts"
            artifacts_dir.mkdir()
            (session_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
            await self._dispatcher.enqueue(
                ManagedSession(
                    request=request,
                    timer=timer,
                    session_dir=session_dir,
                    artifacts_dir=artifacts_dir,
                )
            )
        except Exception:
            self.storage.delete_session(session_id)
            self.session_registry.remove(session_id)
            if session_dir is not None:
                await self._remove_session_dir_best_effort(session_dir, session_id)
            raise

    async def cancel(self, session_id: str, *, reason: str | None = None) -> bool:
        managed = await self._dispatcher.cancel(session_id, reason=reason)
        cancelled = managed is not None
        if cancelled and reason != "pipeline_budget_exceeded":
            await self._close_inflight_generations(session_id, reason=reason or "cancel")
        if managed is not None and reason == "policy_cutoff":
            cleanup_task = asyncio.create_task(
                self._cleanup_policy_cutoff_session(managed),
                name=f"polar-policy-cutoff-{session_id}",
            )
            cleanup_tasks = getattr(self, "_policy_cleanup_tasks", None)
            if cleanup_tasks is None:
                cleanup_tasks = set()
                self._policy_cleanup_tasks = cleanup_tasks
            cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(cleanup_tasks.discard)
        return cancelled

    async def _cleanup_policy_cutoff_session(self, managed: ManagedSession) -> None:
        """Stop agent/evaluator containers after a logical policy cutoff.

        The DELETE request returns after this task is scheduled, so training waits for
        the inference fence but not Docker teardown.  Runtime ``stop`` is idempotent;
        normal POSTRUN cleanup may race this task safely.
        """
        prewarm = managed.eval_prewarm_task
        if prewarm is not None and not prewarm.done():
            prewarm.cancel()

        runtimes: list[BaseRuntime] = []
        for runtime in (managed.runtime, managed.eval_runtime):
            if runtime is not None and all(runtime is not existing for existing in runtimes):
                runtimes.append(runtime)
        if runtimes:
            results = await asyncio.gather(
                *(runtime.cancel() for runtime in runtimes),
                return_exceptions=True,
            )
            for runtime, result in zip(runtimes, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning(
                        "Policy-cutoff cleanup failed for runtime %s session %s: %s",
                        runtime.runtime_id,
                        managed.session_id,
                        result,
                    )
        if prewarm is not None:
            await asyncio.gather(prewarm, return_exceptions=True)

    async def active_sessions(self) -> int:
        return await self._dispatcher.active_count()

    async def stage_metrics(self) -> NodeStageMetrics:
        snapshot = await self._dispatcher.snapshot()
        return self._snapshot_to_metrics(snapshot)

    def record_llm_call(
        self, session_id: str, *,
        acquire_wait_ms: float = 0.0,
        prepare_ms: float = 0.0,
        sglang_wait_ms: float = 0.0,
        normalize_ms: float = 0.0,
        roundtrip_ms: float = 0.0,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        trace_timing: dict[str, int] | None = None,
        trace_id: str | None = None,
        engine_name: str | None = None,
        engine_url: str | None = None,
        engine_metrics: dict[str, float | int] | None = None,
    ) -> None:
        """Record LLM inference timing on the active session's StageTimer."""
        managed = self._dispatcher._sessions.get(session_id)
        if managed is not None:
            managed.timer.record_llm_call(
                acquire_wait_ms=acquire_wait_ms,
                prepare_ms=prepare_ms,
                sglang_wait_ms=sglang_wait_ms,
                normalize_ms=normalize_ms,
                roundtrip_ms=roundtrip_ms,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                trace_timing=trace_timing,
                trace_id=trace_id,
                engine_name=engine_name,
                engine_url=engine_url,
                engine_metrics=engine_metrics,
            )
        self.observability.record_inference(
            engine_name=engine_name,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            roundtrip_ms=roundtrip_ms,
            engine_metrics=engine_metrics,
        )
        # Also log for direct API sessions — visible in gateway logs
        logger.info(
            "llm_call session=%s engine=%s acquire=%.0fms prepare=%.0fms inference=%.0fms normalize=%.0fms p_t=%d r_t=%d",
            session_id, engine_name or "unknown", acquire_wait_ms, prepare_ms, sglang_wait_ms,
            normalize_ms, prompt_tokens, response_tokens,
        )

    def patch_last_llm_post_ms(self, session_id: str, post_ms: float) -> None:
        """Patch the last recorded LLM call's post_ms (I: storage+format timing)."""
        managed = self._dispatcher._sessions.get(session_id)
        if managed is not None and managed.timer._llm_calls:
            managed.timer._llm_calls[-1]["post_ms"] = round(post_ms, 2)

    def patch_last_llm_gateway_total_ms(self, session_id: str, gateway_total_ms: float) -> None:
        """Patch the last recorded LLM call's gateway_total_ms (12→17: arrival→departure)."""
        managed = self._dispatcher._sessions.get(session_id)
        if managed is not None and managed.timer._llm_calls:
            managed.timer._llm_calls[-1]["gateway_total_ms"] = round(gateway_total_ms, 2)

    # ── Agent-side gap (Stage 18+19: tool execution + client overhead) ──

    def compute_agent_side_gap_ms(self, session_id: str, arrival_time: float) -> float:
        """Compute the agent-side gap between the previous LLM response departure
        and the current request arrival.  Returns 0.0 for the first LLM call
        in a session.
        """
        managed = self._dispatcher._sessions.get(session_id)
        if managed is None or managed.last_llm_departure_at is None:
            return 0.0
        return max(0.0, (arrival_time - managed.last_llm_departure_at) * 1000.0)

    def mark_llm_departure(
        self,
        session_id: str,
        departure_time: float,
        departure_time_ns: int | None = None,
    ) -> None:
        """Record the timestamp when the Gateway sent the LLM response back to the agent."""
        managed = self._dispatcher._sessions.get(session_id)
        if managed is not None:
            managed.last_llm_departure_at = departure_time
            managed.last_llm_departure_at_ns = departure_time_ns

    def patch_last_llm_agent_side_gap_ms(
        self,
        session_id: str,
        gap_ms: float,
        original_request: dict | None = None,
        gap_finished_at_ns: int | None = None,
    ) -> None:
        """Patch the agent gap and classify tool results from the request."""
        managed = self._dispatcher._sessions.get(session_id)
        if managed is None:
            return
        actions = extract_completed_agent_actions(original_request or {})
        managed.timer.patch_last_llm_agent_side_gap(
            gap_ms=gap_ms,
            actions=actions,
            gap_started_at_ns=managed.last_llm_departure_at_ns,
            gap_finished_at_ns=gap_finished_at_ns,
        )

    # ── Per-session trace artifacts (W&B + Chrome Trace JSON) ──

    async def _dump_session_trace_artifacts(
        self,
        managed: ManagedSession,
        result: SessionResult,
    ) -> None:
        """Export per-session observability before the session directory is removed.

        Outputs are independently gated by configuration:

        * **Chrome Trace JSON** — ``{session_dir}/trace.json``, also persisted to
          ``{persist_traces_dir}/{session_id}.json`` when configured.
        * **W&B offline run** — ``{session_dir}/wandb/``, also persisted to
          ``{persist_traces_dir}/{session_id}.wandb/`` when configured.
        * **OTLP/HTTP** — correlated parent/child spans for RL-Insight Tempo.

        **Without ``persist_traces_dir`` both outputs are deleted when the session
        directory is cleaned up.**  Set ``persist_traces_dir`` to a durable path
        (e.g. ``/data/polar-traces/``) to keep them for later analysis.
        """
        timing = managed.timer.to_session_timing()
        request = managed.request
        session_id = request.session_id

        await self.observability.record_session(
            self._client,
            timing=timing,
            session_id=session_id,
            task_id=request.task_id,
            status=result.status,
            metadata={
                **request.metadata,
                "profiling_artifact_count": (
                    (result.metadata.get("profiling_artifacts") or {}).get("artifact_count", 0)
                ),
            },
        )

        # ── Option C: Chrome Trace Event JSON ──
        if self._enable_session_trace_json:
            try:
                trace_document = build_chrome_trace_document(
                    timing,
                    session_id=session_id,
                    node_id=self.node_id,
                    task_id=request.task_id,
                )
                trace_document["metadata"]["profilingArtifacts"] = result.metadata.get(
                    "profiling_artifacts", {}
                )
                trace_json = json.dumps(trace_document, ensure_ascii=False)
                # Always write inside the session directory (for debugging).
                managed.session_dir.mkdir(parents=True, exist_ok=True)
                trace_path = managed.session_dir / "trace.json"
                trace_path.write_text(trace_json, encoding="utf-8")
                logger.debug("Chrome Trace JSON written: %s (%d events)",
                             trace_path, len(trace_document["traceEvents"]))
                # Persist to durable directory.
                if self._persist_traces_dir is not None:
                    self._persist_traces_dir.mkdir(parents=True, exist_ok=True)
                    persist_path = self._persist_traces_dir / f"{session_id}.json"
                    persist_path.write_text(trace_json, encoding="utf-8")
                    logger.info("Chrome Trace persisted: %s", persist_path)
                else:
                    logger.warning(
                        "Chrome Trace for session %s will be deleted with "
                        "session dir. Set persist_traces_dir to keep it.",
                        session_id,
                    )
            except Exception:
                logger.exception("Failed to export Chrome Trace for session %s",
                                 session_id)

        # ── Option A: Per-session W&B offline run ──
        if self._enable_session_trace_wandb:
            try:
                import wandb
                wandb_dir = managed.session_dir / "wandb"
                wandb_dir.mkdir(exist_ok=True)
                os_module = __import__("os")
                os_module.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=self._session_trace_wandb_project,
                    name=session_id,
                    dir=str(wandb_dir),
                    settings=wandb.Settings(mode="offline", console="off"),
                    config={
                        "session_id": session_id,
                        "task_id": request.task_id,
                        "node_id": self.node_id,
                    },
                )
                # ── Stage timing summary ──
                _log_session_summary_wandb(timing)
                # ── Per-call agent_side_gap line chart ──
                for idx, call in enumerate(timing.llm_calls):
                    wandb.log({
                        "polar/llm/round": call.get("round", idx + 1),
                        "polar/llm/agent_side_gap_ms": call.get("agent_side_gap_ms", 0.0),
                        "polar/llm/sglang_wait_ms": call.get("sglang_wait_ms", 0.0),
                        "polar/llm/roundtrip_ms": call.get("roundtrip_ms", 0.0),
                        "polar/llm/gateway_total_ms": call.get("gateway_total_ms", 0.0),
                    })
                # ── Per-tool-exec data points ──
                for te in timing.tool_execs:
                    wandb.log({
                        "polar/tool/idx": te.get("idx", 0),
                        "polar/tool/duration_ms": te.get("duration_ms", 0.0),
                        "polar/tool/exit_code": te.get("exit_code", 0),
                    })
                wandb.finish(exit_code=0, quiet=True)
                logger.info("Per-session W&B run written: %s", wandb_dir)
                # Persist to durable directory.
                if self._persist_traces_dir is not None:
                    persist_wandb = self._persist_traces_dir / f"{session_id}.wandb"
                    # Remove stale copy if it exists, then copy fresh.
                    if persist_wandb.exists():
                        shutil.rmtree(persist_wandb, ignore_errors=True)
                    shutil.copytree(str(wandb_dir), str(persist_wandb))
                    logger.info("Per-session W&B persisted: %s", persist_wandb)
                else:
                    logger.warning(
                        "W&B run for session %s will be deleted with "
                        "session dir. Set persist_traces_dir to keep it.",
                        session_id,
                    )
            except Exception:
                logger.exception("Failed to write per-session W&B for session %s",
                                 session_id)

    def _handle_dispatcher_stage_change(self, managed: ManagedSession) -> None:
        status = {
            SessionStage.INIT: SessionStatus.INITIALIZING,
            SessionStage.READY: SessionStatus.READY,
            SessionStage.RUNNING: SessionStatus.RUNNING,
            SessionStage.POSTRUN: SessionStatus.POST_RUN,
        }.get(managed.stage)
        if status is not None:
            self.session_registry.set_status(managed.request.session_id, status)

    # ------------------------------------------------------------------
    # INIT stage
    # ------------------------------------------------------------------

    async def _handle_init(self, managed: ManagedSession) -> None:
        request = managed.request
        self._start_execution_deadline(managed)
        managed.timer.mark("init", "started")
        try:
            if managed.cancel_requested:
                return
            runtime_spec = self._resolve_runtime_spec(request)
            managed.timer.mark("runtime_create", "started")
            runtime = create_runtime(runtime_spec, request.session_id, managed.session_dir)
            managed.runtime = runtime
            if managed.cancel_requested:
                await runtime.cancel()
                return
            await self._await_with_budget(runtime.start(), managed)
            managed.timer.mark("runtime_create", "finished")
            # ── Record docker sub-operation timing (read from runtime) ──
            if hasattr(runtime, "docker_create_ms"):
                self._mark_docker_op(managed, "docker_create", getattr(runtime, "docker_create_ms", 0.0))
                self._mark_docker_op(managed, "docker_start", getattr(runtime, "docker_start_ms", 0.0))
            if managed.cancel_requested:
                await runtime.cancel()
                return
            # Run ordered prepare actions
            managed.timer.mark("prepare", "started")
            await self._run_runtime_prepare(runtime, runtime_spec, request, managed)
            managed.timer.mark("prepare", "finished")
        except GatewayExecutionTimeout as exc:
            managed.final_result = self._timeout_result(request, managed.timer, str(exc))
        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Initialization cancelled for session %s", request.session_id)
            else:
                logger.exception("Initialization failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"runtime initialization failed: {exc}",
                )
        finally:
            managed.timer.mark("init", "finished")

    @staticmethod
    def _mark_docker_op(managed: ManagedSession, op: str, duration_ms: float) -> None:
        """Record a docker sub-operation duration as a timer mark pair.

        The timer computes durations from ``op_started`` → ``op_finished``
        mark pairs.  This helper injects a synthetic pair with a known
        duration so the timing flows into SessionTiming without changing
        the public mark interface.
        """
        if duration_ms <= 0:
            return
        now = managed.timer._marks.get("dispatch_started", 0.0) or 0.0
        managed.timer._marks[f"{op}_started"] = now
        managed.timer._marks[f"{op}_finished"] = now + duration_ms / 1000.0

    def _resolve_runtime_spec(self, request: SessionDispatchRequest) -> RuntimeSpec:
        spec = request.runtime or self.default_runtime
        if spec is None:
            raise RuntimeError(
                "no runtime configured: request has no runtime and gateway "
                "node has no default_runtime"
            )
        return spec

    def _resolve_eval_runtime_spec(self, request: SessionDispatchRequest) -> RuntimeSpec:
        if request.evaluator is not None and request.evaluator.runtime is not None:
            return request.evaluator.runtime
        return self._resolve_runtime_spec(request)

    async def _run_runtime_prepare(
        self,
        runtime: BaseRuntime,
        spec: RuntimeSpec,
        request: SessionDispatchRequest,
        managed: ManagedSession,
        *,
        actions: list | None = None,
        log_prefix: str = "prepare",
        honor_cancel: bool = True,
    ) -> None:
        """Execute an ordered prepare action list (``spec.prepare`` by default).

        ``honor_cancel``:agent prepare 在取消时应尽早停(True)。但 **eval/judge prepare 传 False** ——
        取消(如 pipeline_budget_exceeded)只该停 agent,不该连"给 agent 已产出物打分的判分准备"也跳过;
        跳过会导致 judge 在没铺 input/ 的空容器上判分 → 假 input_load_failed → 丢弃产出 + 无谓 retry。
        """
        steps = actions if actions is not None else spec.prepare
        base_env = self._runtime_env(request, managed, runtime_override=runtime)
        for i, action in enumerate(steps):
            if honor_cancel and managed.cancel_requested:
                return
            if action.type == "upload_file":
                await runtime.upload_file(action.source, action.target)
            elif action.type == "upload_dir":
                await runtime.upload_dir(action.source, action.target)
            elif action.type == "exec":
                merged_env = {**base_env, **(action.env or {})}
                effective_cwd = action.cwd or runtime.runtime_session_dir
                result = await runtime.exec(
                    action.command,
                    cwd=effective_cwd,
                    env=merged_env,
                    timeout_sec=self._remaining_budget(managed),
                )
                log_dir = managed.session_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._write_exec_log(
                    log_dir, f"{log_prefix}.{i:02d}", result.stdout, result.stderr
                )
                if result.return_code == -1:
                    raise RuntimeError(f"{log_prefix} action {i} timed out")
                if result.return_code != 0:
                    raise RuntimeError(
                        f"{log_prefix} action {i} failed with exit code {result.return_code}"
                    )

    # ------------------------------------------------------------------
    # RUN stage
    # ------------------------------------------------------------------

    async def _handle_run(self, managed: ManagedSession) -> None:
        request = managed.request
        if managed.final_result is not None or managed.cancel_requested:
            return
        managed.timer.mark("run", "started")

        harness: BaseHarness | None = None
        try:
            runtime = managed.runtime
            if runtime is None:
                raise RuntimeError("runtime is required for execution")

            if not self._use_lazy_eval_runtime(request):
                self._start_eval_prewarm(managed)
            harness = self._resolve_agent_harness(request)

            # Setup
            managed.timer.mark("harness_setup", "started")
            await self._await_with_budget(harness.setup(runtime), managed)
            managed.timer.mark("harness_setup", "finished")

            # Run
            managed.timer.mark("agent_exec", "started")
            steps = harness.run_steps(request.instruction)
            env = self._runtime_env(request, managed, include_agent_env=True)
            agent_result = await self._run_exec_inputs(runtime, steps, env, managed)
            managed.timer.mark("agent_exec", "finished")

            # Postprocess always runs so harnesses can collect artifacts from
            # failed or timed-out agent runs before post-run evaluation.
            managed.timer.mark("harness_postprocess", "started")
            await self._await_with_budget(harness.postprocess(runtime, agent_result), managed)
            managed.timer.mark("harness_postprocess", "finished")
            managed.agent_result = agent_result

        except GatewayExecutionTimeout as exc:
            # Don't set final_result — let _handle_postrun build a partial
            # trajectory from the completions captured so far.
            managed.agent_result = AgentRunResult(
                status="timeout", return_code=-1, error=str(exc),
            )
        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Agent execution cancelled for session %s", request.session_id)
            else:
                logger.exception("Agent execution failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"agent execution failed: {exc}",
                )
        finally:
            if harness is not None:
                managed.postrun_steps = harness.postrun_steps()
            managed.timer.mark("run", "finished")

    def _resolve_agent_harness(self, request: SessionDispatchRequest) -> BaseHarness:
        return create_harness(request.agent)

    async def _run_exec_inputs(
        self,
        runtime: BaseRuntime,
        steps: list[ExecInput],
        env: dict[str, str],
        managed: ManagedSession,
    ) -> AgentRunResult:
        """Execute a list of ExecInput steps and return an AgentRunResult."""
        log_dir = managed.session_dir / "logs" / "agent"
        log_dir.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(steps):
            if managed.cancel_requested:
                return AgentRunResult(
                    status="failed", return_code=-1, error="cancelled"
                )
            merged_env = {**env, **(step.env or {})}
            t_step_start = asyncio.get_event_loop().time()
            t_step_start_ns = time.time_ns()
            result = await runtime.exec(
                step.command,
                cwd=step.cwd,
                env=merged_env,
                timeout_sec=self._remaining_budget(managed),
            )
            step_finished_ns = time.time_ns()
            step_duration_ms = (asyncio.get_event_loop().time() - t_step_start) * 1000.0
            managed.timer.record_tool_exec(
                command=step.command,
                duration_ms=step_duration_ms,
                exit_code=result.return_code,
                started_at_ns=t_step_start_ns,
                finished_at_ns=step_finished_ns,
            )
            self._write_exec_log(
                log_dir, f"step.{i:02d}", result.stdout, result.stderr
            )
            if result.return_code == -1:
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error=f"step {i} timed out",
                    metadata=self._step_metadata(log_dir, i, managed),
                )
            if result.return_code != 0:
                return AgentRunResult(
                    status="failed",
                    return_code=result.return_code,
                    error=f"step {i} exited with code {result.return_code}",
                    metadata=self._step_metadata(log_dir, i, managed),
                )

        return AgentRunResult(
            status="completed",
            return_code=0,
            metadata=self._step_metadata(log_dir, len(steps) - 1, managed),
        )

    # ------------------------------------------------------------------
    # Evaluator runtime prewarm
    # ------------------------------------------------------------------

    def _start_eval_prewarm(self, managed: ManagedSession) -> None:
        """Spawn a background task to prewarm a fresh evaluator runtime."""
        request = managed.request
        if request.evaluator is None or not request.evaluator.refresh_runtime:
            return
        if self._use_lazy_eval_runtime(request):
            return
        if managed.eval_prewarm_task is not None:
            return
        managed.eval_prewarm_task = asyncio.create_task(
            self._prepare_eval_runtime(managed)
        )

    @staticmethod
    def _use_lazy_eval_runtime(request: SessionDispatchRequest) -> bool:
        evaluator = request.evaluator
        if evaluator is None or not evaluator.refresh_runtime:
            return False
        if evaluator.strategy != "operator_judge":
            return False
        return bool(evaluator.config.get("lazy_refresh_runtime"))

    @staticmethod
    def _judge_infra_retries(evaluator_spec) -> int:
        """judge 级 infra 重试次数(默认 2,即最多 3 次判分)。evaluator.config.judge_infra_retries 可调。"""
        config = getattr(evaluator_spec, "config", None)
        if isinstance(config, dict):
            try:
                return max(0, int(config.get("judge_infra_retries", 2)))
            except (TypeError, ValueError):
                pass
        return 2

    async def _verify_eval_golden_input(
        self,
        runtime: BaseRuntime,
        request: SessionDispatchRequest,
        actions: list | None,
    ) -> None:
        """eval_prepare 后核对 golden input/<op>.py 真的传上了;没传上就按动作补传并记日志。

        背景:input_load_failed(golden missing)在并发 judge 时间歇出现,根因待这条日志
        确认。这里对每个指向 input/<op>.py 的 upload_file 动作,核对容器内目标文件存在,
        不在就补传一次。只核对、不重跑整个 prepare,代价极小。
        """
        op_name = ""
        evaluator = getattr(request, "evaluator", None)
        config = getattr(evaluator, "config", None)
        if isinstance(config, dict):
            op_name = str(config.get("op_name") or "").strip()
        if not op_name:
            return
        for action in actions or []:
            if getattr(action, "type", None) != "upload_file":
                continue
            target = str(getattr(action, "target", "") or "")
            if not target.endswith(f"/input/{op_name}.py"):
                continue
            try:
                check = await runtime.exec(
                    f'test -f "{target}" && echo P || echo M', timeout_sec=30
                )
                present = "P" in (check.stdout or "")
            except Exception as exc:
                logger.warning(
                    "eval golden check failed session=%s op=%s target=%s: %s",
                    request.session_id, op_name, target, exc,
                )
                continue
            if present:
                logger.info(
                    "eval golden present session=%s op=%s target=%s",
                    request.session_id, op_name, target,
                )
                continue
            source = str(getattr(action, "source", "") or "")
            logger.warning(
                "eval golden MISSING after eval_prepare session=%s op=%s target=%s; "
                "re-upload from %s",
                request.session_id, op_name, target, source,
            )
            try:
                await runtime.upload_file(source, target)
                recheck = await runtime.exec(
                    f'test -f "{target}" && echo P || echo M', timeout_sec=30
                )
                ok = "P" in (recheck.stdout or "")
                logger.info(
                    "eval golden re-upload %s session=%s op=%s target=%s",
                    "OK" if ok else "STILL-MISSING",
                    request.session_id, op_name, target,
                )
            except Exception as exc:
                logger.warning(
                    "eval golden re-upload FAILED session=%s op=%s target=%s: %s",
                    request.session_id, op_name, target, exc,
                )

    async def _prepare_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Create and prepare a fresh runtime for the evaluator. Returns None on failure."""
        if managed.cancel_reason == "policy_cutoff":
            return None
        request = managed.request
        runtime_spec = self._resolve_eval_runtime_spec(request)
        eval_session_dir = managed.session_dir / "eval_runtime"
        eval_artifacts_dir = eval_session_dir / "artifacts"
        eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

        eval_runtime = create_runtime(
            runtime_spec, f"{request.session_id}-eval", eval_session_dir
        )
        managed.eval_runtime = eval_runtime
        try:
            if managed.cancel_reason == "policy_cutoff":
                await eval_runtime.cancel()
                if managed.eval_runtime is eval_runtime:
                    managed.eval_runtime = None
                return None
            await self._await_with_budget(eval_runtime.start(), managed)
            eval_actions = (
                runtime_spec.eval_prepare
                if runtime_spec.eval_prepare is not None
                else runtime_spec.prepare
            )
            await self._run_runtime_prepare(
                eval_runtime,
                runtime_spec,
                request,
                managed,
                actions=eval_actions,
                log_prefix="eval_prepare",
                # 取消(如 budget 超限)只停 agent;判分准备必须跑全,否则 judge 在空容器上判分
                # → 假 input_load_failed、丢弃 best-so-far 产出。见 deploy/ascend_operator/FIX-input_load_failed.md
                honor_cancel=False,
            )
            # 判分前自检: golden input/<op>.py 必须真的传上了。最终 judge 用 fresh
            # runtime(隔离),并发 judge 时 eval_prepare 的 golden 上传偶发没落上,
            # pipeline 一跑就 input_load_failed、整个 session 被判重试。在这里先核对,
            # 没传上就按 prepare 动作补传,把 golden missing 挡在 pipeline 开跑之前。
            await self._verify_eval_golden_input(eval_runtime, request, eval_actions)
            if managed.cancel_reason == "policy_cutoff":
                await eval_runtime.cancel()
                if managed.eval_runtime is eval_runtime:
                    managed.eval_runtime = None
                return None
            return eval_runtime
        except asyncio.CancelledError:
            with suppress(Exception):
                await eval_runtime.stop()
            if managed.eval_runtime is eval_runtime:
                managed.eval_runtime = None
            raise
        except Exception as exc:
            logger.warning(
                "Eval runtime prewarm failed for session %s: %s",
                request.session_id,
                exc,
            )
            with suppress(Exception):
                await eval_runtime.stop()
            if managed.eval_runtime is eval_runtime:
                managed.eval_runtime = None
            return None

    async def _acquire_prepared_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Await the prewarm task and return its runtime, if any."""
        task = managed.eval_prewarm_task
        if task is None:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self._remaining_budget(managed)
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout(
                "timed out waiting for a fresh evaluator runtime"
            ) from exc

    async def _drain_eval_prewarm_task(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Resolve the prewarm task during teardown. Cancel if still running."""
        task = managed.eval_prewarm_task
        if task is None:
            return None
        if not task.done():
            task.cancel()
        try:
            return await task
        except (asyncio.CancelledError, Exception):
            return None

    # ------------------------------------------------------------------
    # POSTRUN stage
    # ------------------------------------------------------------------

    async def _handle_postrun(self, managed: ManagedSession) -> None:
        request = managed.request
        result: SessionResult | None = managed.final_result
        managed.timer.mark("postrun", "started")
        try:
            if result is None:
                if managed.cancel_requested:
                    result = await self._build_cancelled_session_result(managed)
                else:
                    result = await self._build_session_result(managed)
        except GatewayExecutionTimeout as exc:
            result = self._timeout_result(request, managed.timer, str(exc))
        except Exception as exc:
            logger.exception("Post-run handling failed for session %s", request.session_id)
            result = self._error_result(request, managed.timer, f"post-run failed: {exc}")
        finally:
            managed.timer.mark("postrun", "finished")
            managed.timer.mark("teardown", "started")
            await self._run_postrun_steps(managed)
            stop_tasks = []
            eval_runtime = await self._drain_eval_prewarm_task(managed)
            if eval_runtime is not None:
                stop_tasks.append(
                    self._stop_runtime_best_effort(
                        eval_runtime, request.session_id, "eval runtime"
                    )
                )
                if managed.eval_runtime is eval_runtime:
                    managed.eval_runtime = None
            if managed.runtime is not None:
                stop_tasks.append(
                    self._stop_runtime_best_effort(
                        managed.runtime, request.session_id, "runtime"
                    )
                )
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            # ── Record docker kill / rm timing ──
            if managed.runtime is not None and hasattr(managed.runtime, "docker_kill_ms"):
                self._mark_docker_op(managed, "docker_kill", getattr(managed.runtime, "docker_kill_ms", 0.0))
                self._mark_docker_op(managed, "docker_rm", getattr(managed.runtime, "docker_rm_ms", 0.0))
            managed.timer.mark("teardown", "finished")
            managed.timer.mark("return", "finished")

        if result is None:
            result = self._error_result(
                request,
                managed.timer,
                "post-run finished without producing a session result",
            )
        profiling_artifacts = await asyncio.to_thread(
            self._persist_profiling_artifacts_manifest,
            managed,
        )
        if profiling_artifacts:
            result = result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "profiling_artifacts": profiling_artifacts,
                    }
                }
            )
        try:
            normalized = result.model_copy(
                update={
                    "timing": managed.timer.to_session_timing(),
                    "node_id": self.node_id,
                    "error": result.error or result.trajectory.error,
                }
            )
            self.session_registry.set_result(request.session_id, normalized)
            await self._close_inflight_generations(request.session_id, reason="postrun_result")
            self.storage.mark_session_closed(request.session_id, reason="postrun_result")
            self.storage.delete_session(request.session_id)
            await self.release_session_affinity_best_effort(request.session_id)
            managed.timer.mark("push_result", "started")
            if await self._push_result(request.callback_url, normalized):
                # Rollout server has acked; free the heavy payload but keep
                # status/task_id visible for debugging via the polling endpoint.
                self.session_registry.clear_result_payload(request.session_id)
            managed.timer.mark("push_result", "finished")
            # Export only after all measured stages are complete, but before cleanup.
            await self._dump_session_trace_artifacts(managed, normalized)
        finally:
            await self._remove_session_dir_best_effort(
                managed.session_dir, request.session_id
            )

    def _persist_profiling_artifacts_manifest(self, managed: ManagedSession) -> dict[str, Any]:
        if not self._persist_session_artifacts or self._persist_traces_dir is None:
            return {}
        try:
            return persist_profiling_artifacts(
                managed.artifacts_dir,
                self._persist_traces_dir,
                session_id=managed.session_id,
                max_total_bytes=self._session_artifacts_max_bytes,
                max_files=self._session_artifacts_max_files,
            )
        except Exception:
            logger.exception("Failed to persist profiling artifacts for %s", managed.session_id)
            return {}

    async def _close_inflight_generations(
        self,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> None:
        if self.inflight is None:
            return
        try:
            await self.inflight.close_session(session_id, reason=reason)
        except Exception:
            logger.warning(
                "Failed to close inflight generations for session %s",
                session_id,
                exc_info=True,
            )

    async def release_session_affinity_best_effort(self, session_id: str) -> None:
        """Tell an optional external inference router that a session is terminal.

        This is performance-only cleanup.  A missing or unhealthy endpoint must
        never change the session result, callback, reward, or teardown path; the
        router's TTL and policy-boundary clear remain the fallback.
        """
        url = self._session_affinity_release_url
        if url is None:
            return
        try:
            response = await self._client.post(
                url,
                json={"session_id": session_id},
                timeout=1.0,
            )
            response.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to release inference affinity for terminal session %s via %s",
                session_id,
                url,
                exc_info=True,
            )

    async def _build_session_result(self, managed: ManagedSession) -> SessionResult:
        request = managed.request
        agent_result = managed.agent_result
        if agent_result is None:
            return self._error_result(
                request,
                managed.timer,
                "session did not produce an agent result",
            )

        self.session_registry.set_status(request.session_id, SessionStatus.BUILDING)
        managed.timer.mark("build", "started")
        try:
            trajectory = await self._await_with_budget(
                asyncio.to_thread(self._build_trajectory, request),
                managed,
            )
        finally:
            managed.timer.mark("build", "finished")

        error = trajectory.error
        if agent_result.status == "timeout":
            trajectory = trajectory.model_copy(
                update={"status": "TIMEOUT", "error": agent_result.error or error}
            )
        elif agent_result.status == "failed":
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": agent_result.error or error}
            )

        managed.timer.mark("eval", "started")
        try:
            if request.evaluator is not None:
                self.session_registry.set_status(request.session_id, SessionStatus.EVALUATING)
                trajectory = await self._run_eval(
                    request,
                    trajectory,
                    agent_result=agent_result,
                    managed=managed,
                )
        except GatewayExecutionTimeout as exc:
            # Preserve the built trajectory even when eval times out.
            logger.warning("Eval timed out for session %s: %s", request.session_id, exc)
            if trajectory.status not in ("TIMEOUT", "ERROR"):
                trajectory = trajectory.model_copy(
                    update={"status": "TIMEOUT", "error": f"eval timed out: {exc}"}
                )
        except Exception as exc:
            logger.exception("Eval failed for session %s", request.session_id)
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )
        finally:
            managed.timer.mark("eval", "finished")

        error = trajectory.error or error
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status=trajectory.status,
            trajectory=trajectory,
            timing=managed.timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    async def _build_cancelled_session_result(self, managed: ManagedSession) -> SessionResult:
        request = managed.request
        if managed.cancel_reason != "pipeline_budget_exceeded":
            return self._cancelled_result(request, managed.timer)

        if managed.agent_result is None:
            managed.agent_result = AgentRunResult(
                status="failed",
                return_code=-1,
                error="pipeline budget exceeded",
            )

        result = await self._build_session_result(managed)
        if not result.trajectory.traces:
            return self._error_result(
                request,
                managed.timer,
                "pipeline budget exceeded before any trainable traces were captured",
            )

        trajectory_metadata = {
            **result.trajectory.metadata,
            "termination_reason": "pipeline_budget_exceeded",
            "cancelled_partial": True,
        }
        trajectory = result.trajectory.model_copy(
            update={
                "status": "COMPLETED",
                "error": None,
                "metadata": trajectory_metadata,
            }
        )
        metadata = {
            **result.metadata,
            "termination_reason": "pipeline_budget_exceeded",
            "cancelled_partial": True,
        }
        return result.model_copy(
            update={
                "status": SessionStatus.COMPLETED,
                "trajectory": trajectory,
                "error": None,
                "metadata": metadata,
            }
        )

    def _build_trajectory(self, request: SessionDispatchRequest) -> Trajectory:
        completion_session = self.storage.load_completion_session(request.session_id)
        builder = self.builders.create(request.builder)
        result = builder.build(completion_session)
        if asyncio.iscoroutine(result):
            trajectory = asyncio.run(result)
        else:
            trajectory = result
        return Trajectory.model_validate(trajectory)

    async def _run_eval(
        self,
        request: SessionDispatchRequest,
        trajectory: Trajectory,
        *,
        agent_result: AgentRunResult,
        managed: ManagedSession,
    ) -> Trajectory:
        evaluator_spec = request.evaluator
        if evaluator_spec is None:
            return trajectory
        if managed.cancel_reason == "policy_cutoff":
            return trajectory.model_copy(
                update={"status": "ERROR", "error": "cancelled at policy cutoff"}
            )

        live_runtime = managed.runtime
        if live_runtime is None:
            raise RuntimeError("runtime is required for evaluation")

        fresh_eval_runtime: BaseRuntime | None = None
        eval_runtime_spec = self._resolve_runtime_spec(request)
        if evaluator_spec.refresh_runtime:
            eval_runtime_spec = self._resolve_eval_runtime_spec(request)
            if self._use_lazy_eval_runtime(request):
                return await self._run_lazy_eval(
                    request,
                    trajectory,
                    agent_result=agent_result,
                    managed=managed,
                    eval_runtime_spec=eval_runtime_spec,
                )
            fresh_eval_runtime = await self._acquire_prepared_eval_runtime(managed)
            if fresh_eval_runtime is None:
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": "refresh_runtime=true requires a fresh runtime: eval runtime prewarm did not produce a usable runtime",
                    }
                )

        # Convert EvaluatorSpec to StrategySpec for registry
        strategy_spec = StrategySpec(
            strategy=evaluator_spec.strategy,
            config=evaluator_spec.config,
        )

        max_attempts = 1 + self._judge_infra_retries(evaluator_spec)
        judge_rt = fresh_eval_runtime
        eval_result = None
        last_exc: Exception | None = None
        evaluator = self.evaluators.create(strategy_spec)
        for attempt in range(1, max_attempts + 1):
            if managed.cancel_reason == "policy_cutoff":
                last_exc = RuntimeError("cancelled at policy cutoff")
                break
            try:
                eval_result = await self._await_with_budget(
                    evaluator.evaluate(
                        trajectory,
                        session_id=request.session_id,
                        task_id=request.task_id,
                        session_dir=managed.session_dir,
                        artifacts_dir=managed.artifacts_dir,
                        agent_result=agent_result,
                        env=self._evaluator_env(evaluator_spec, eval_runtime_spec),
                        timeout_seconds=self._remaining_budget(managed),
                        runtime=live_runtime,
                        fresh_eval_runtime=judge_rt,
                        runtime_spec=eval_runtime_spec,
                        refresh_runtime=evaluator_spec.refresh_runtime,
                    ),
                    managed,
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if managed.cancel_reason == "policy_cutoff":
                    break
                if attempt >= max_attempts or not _is_retryable_judge_infra(exc):
                    break
                # judge 级重试: infra 失败(golden 缺失/NPU 不可用/容器传输/超时)只重起
                # judge runtime 重判,不牵连整个 session(agent 不用重跑)。.best.tar.gz
                # 还在宿主机,重判代价远小于 session 级重试。
                logger.warning(
                    "judge infra failure (attempt %d/%d) for session %s: %s; "
                    "retrying judge only",
                    attempt, max_attempts, request.session_id, exc,
                )
                if evaluator_spec.refresh_runtime:
                    if isinstance(judge_rt, BaseRuntime):
                        with suppress(Exception):
                            await judge_rt.stop()
                        if managed.eval_runtime is judge_rt:
                            managed.eval_runtime = None
                    judge_rt = await self._prepare_eval_runtime(managed)
                    if judge_rt is None:
                        logger.warning(
                            "judge retry: could not prepare a fresh runtime for session %s",
                            request.session_id,
                        )
                        break
                # refresh_runtime=false 时在 source(agent)runtime 原地重试,对瞬时 NPU/超时有效。

        if judge_rt is not None and judge_rt is not fresh_eval_runtime:
            # 重试期间新建的 eval runtime 不在 managed.eval_prewarm_task 里,teardown 只清
            # prewarm 那个;在这里显式停掉,避免容器泄漏。
            with suppress(Exception):
                await judge_rt.stop()
            if managed.eval_runtime is judge_rt:
                managed.eval_runtime = None

        if last_exc is not None:
            logger.exception(
                "Evaluator %s failed for session %s",
                evaluator_spec.strategy,
                request.session_id,
            )
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {last_exc}"}
            )

        return self._merge_eval_result(trajectory, eval_result, evaluator_spec)

    async def _run_lazy_eval(
        self,
        request: SessionDispatchRequest,
        trajectory: Trajectory,
        *,
        agent_result: AgentRunResult,
        managed: ManagedSession,
        eval_runtime_spec: RuntimeSpec,
    ) -> Trajectory:
        evaluator_spec = request.evaluator
        if evaluator_spec is None:
            return trajectory
        if managed.cancel_reason == "policy_cutoff":
            return trajectory.model_copy(
                update={"status": "ERROR", "error": "cancelled at policy cutoff"}
            )

        live_runtime = managed.runtime
        if live_runtime is None:
            raise RuntimeError("runtime is required for lazy evaluation")

        submission_context = await self._extract_operator_judge_submission(
            managed, evaluator_spec
        )

        # Harness postrun steps still belong to the agent runtime. Run them
        # before stopping the runtime, then clear the list so teardown does not
        # repeat them.
        await self._run_postrun_steps(managed)
        managed.postrun_steps = []

        try:
            await live_runtime.stop()
        except Exception as exc:  # noqa: BLE001
            return trajectory.model_copy(
                update={
                    "status": "ERROR",
                    "error": (
                        "lazy refresh_runtime could not stop agent runtime before "
                        f"starting evaluator runtime: {exc}"
                    ),
                }
            )
        managed.runtime = None

        strategy_spec = StrategySpec(
            strategy=evaluator_spec.strategy,
            config=evaluator_spec.config,
        )

        fresh_eval_runtime: BaseRuntime | None = None
        judge_rt: BaseRuntime | None = None
        last_exc: Exception | None = None
        eval_result = None
        try:
            if not submission_context.get("submission_missing"):
                fresh_eval_runtime = await self._prepare_eval_runtime(managed)
                judge_rt = fresh_eval_runtime
                if fresh_eval_runtime is None:
                    return trajectory.model_copy(
                        update={
                            "status": "ERROR",
                            "error": (
                                "refresh_runtime=true requires a fresh runtime: "
                                "lazy eval runtime did not produce a usable runtime"
                            ),
                        }
                    )

            max_attempts = 1 + self._judge_infra_retries(evaluator_spec)
            evaluator = self.evaluators.create(strategy_spec)
            for attempt in range(1, max_attempts + 1):
                if managed.cancel_reason == "policy_cutoff":
                    last_exc = RuntimeError("cancelled at policy cutoff")
                    break
                try:
                    eval_result = await self._await_with_budget(
                        evaluator.evaluate(
                            trajectory,
                            session_id=request.session_id,
                            task_id=request.task_id,
                            session_dir=managed.session_dir,
                            artifacts_dir=managed.artifacts_dir,
                            agent_result=agent_result,
                            env=self._evaluator_env(evaluator_spec, eval_runtime_spec),
                            timeout_seconds=self._remaining_budget(managed),
                            runtime=None,
                            fresh_eval_runtime=judge_rt,
                            runtime_spec=eval_runtime_spec,
                            refresh_runtime=evaluator_spec.refresh_runtime,
                            **submission_context,
                        ),
                        managed,
                    )
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if managed.cancel_reason == "policy_cutoff":
                        break
                    if attempt >= max_attempts or not _is_retryable_judge_infra(exc):
                        break
                    # judge 级重试: infra 失败只重起 judge runtime 重判,不牵连 session。
                    logger.warning(
                        "judge infra failure (attempt %d/%d) for session %s: %s; "
                        "retrying judge only",
                        attempt, max_attempts, request.session_id, exc,
                    )
                    if submission_context.get("submission_missing"):
                        break
                    if isinstance(judge_rt, BaseRuntime):
                        with suppress(Exception):
                            await judge_rt.stop()
                        if managed.eval_runtime is judge_rt:
                            managed.eval_runtime = None
                    judge_rt = await self._prepare_eval_runtime(managed)
                    if judge_rt is None:
                        logger.warning(
                            "judge retry: could not prepare a fresh runtime for session %s",
                            request.session_id,
                        )
                        break
        except Exception as exc:
            logger.exception(
                "Evaluator %s failed for session %s",
                evaluator_spec.strategy,
                request.session_id,
            )
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )
        finally:
            if judge_rt is not None:
                await self._stop_runtime_best_effort(
                    judge_rt, request.session_id, "eval runtime"
                )
                if managed.eval_runtime is judge_rt:
                    managed.eval_runtime = None

        if last_exc is not None:
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {last_exc}"}
            )

        return self._merge_eval_result(trajectory, eval_result, evaluator_spec)

    async def _extract_operator_judge_submission(
        self,
        managed: ManagedSession,
        evaluator_spec: EvaluatorSpec,
    ) -> dict[str, Any]:
        runtime = managed.runtime
        if runtime is None:
            raise RuntimeError("runtime is required to extract operator submission")

        candidates = self._operator_judge_submission_candidates(evaluator_spec)
        artifact_dir = managed.artifacts_dir / "operator_judge"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        local_impl = artifact_dir / "submission_impl.py"
        if local_impl.exists():
            local_impl.unlink()

        # 先看 bind-mount 的 session 目录 —— pack_submission.sh 每次更新 best 都往那儿镜像一份。
        # 该目录直通宿主机,所以这条路径:① 不经容器传输(容器已死也能取);
        # ② 不受 agent 事后 `rm -rf output/submission/*.tar.gz` 影响 —— 取件发生在 agent
        # 跑完之后,删了就真没了(实测有 session 这么干过,通配符把 .best 一起带走)。
        mirror_dir = managed.session_dir / "submission"
        for mirrored in sorted(mirror_dir.glob("*_impl.best.tar.gz")):
            if mirrored.is_file() and mirrored.stat().st_size > 0:
                logger.info(
                    "operator_judge: using session-dir mirror for session %s: %s",
                    managed.request.session_id,
                    mirrored.name,
                )
                return {
                    "submission_host_path": str(mirrored),
                    "submission_used": f"session_mirror/{mirrored.name}",
                }

        for logical_path, runtime_path in candidates:
            try:
                await runtime.download_file(runtime_path, str(local_impl))
                return {
                    "submission_host_path": str(local_impl),
                    "submission_used": logical_path,
                }
            except Exception:
                logger.debug(
                    "operator_judge lazy submission candidate missing for session %s: %s",
                    managed.request.session_id,
                    runtime_path,
                    exc_info=True,
                )

        return {"submission_missing": True, "submission_used": None}

    def _operator_judge_submission_candidates(
        self,
        evaluator_spec: EvaluatorSpec,
    ) -> list[tuple[str, str]]:
        config = evaluator_spec.config
        op_name = str(config.get("op_name") or "").strip()
        workdir_value = config.get("workdir")
        workdir = str(workdir_value) if workdir_value else None

        configured_candidates = config.get("submission_candidates")
        if isinstance(configured_candidates, list) and configured_candidates:
            logical_candidates = [str(path) for path in configured_candidates if str(path).strip()]
        elif str(config.get("judge_mode") or "").strip().lower() == "cannbot":
            logical_candidates = [
                f"{op_name}_generated.py",
                "output/optimized_code.py",
                "output/generated_code.py",
            ]
        else:
            submission_path = str(
                config.get("submission_path") or f"output/submission/{op_name}_impl.py"
            )
            logical_candidates = []
            if submission_path.endswith(".py"):
                logical_candidates.append(submission_path[:-3] + ".best.py")
            logical_candidates.append(submission_path)
        return [
            (path, self._operator_judge_abs_path(path, workdir))
            for path in logical_candidates
        ]

    @staticmethod
    def _operator_judge_abs_path(path: str, workdir: str | None) -> str:
        if workdir and not posixpath.isabs(path):
            return posixpath.join(workdir, path)
        return path

    @staticmethod
    def _merge_eval_result(
        trajectory: Trajectory,
        eval_result: EvalResult,
        evaluator_spec: EvaluatorSpec,
    ) -> Trajectory:
        """Apply rewards from EvalResult to trajectory traces."""
        traces = list(trajectory.traces)

        if eval_result.trace_rewards is not None:
            if len(eval_result.trace_rewards) != len(traces):
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": (
                            f"evaluator returned {len(eval_result.trace_rewards)} "
                            f"trace_rewards but trajectory has {len(traces)} traces"
                        ),
                    }
                )
            traces = [
                trace.model_copy(update={"reward": reward})
                for trace, reward in zip(traces, eval_result.trace_rewards)
            ]
        elif eval_result.outcome_reward is not None and traces:
            # Broadcast trajectory-level reward 
            traces = [
                trace.model_copy(update={"reward": eval_result.outcome_reward})
                for trace in traces
            ]

        eval_metadata = {
            "strategy": evaluator_spec.strategy,
            "outcome_reward": eval_result.outcome_reward,
            "trace_rewards": eval_result.trace_rewards,
            **eval_result.metadata,
        }
        metadata = {**trajectory.metadata, "evaluation": eval_metadata}
        return trajectory.model_copy(update={"traces": traces, "metadata": metadata})

    # ------------------------------------------------------------------
    # Environment and helpers
    # ------------------------------------------------------------------

    def _runtime_env(
        self,
        request: SessionDispatchRequest,
        managed: ManagedSession,
        *,
        include_agent_env: bool = False,
        runtime_override: BaseRuntime | None = None,
    ) -> dict[str, str]:
        runtime = runtime_override or managed.runtime
        if runtime is None:
            session_dir = str(managed.session_dir)
            artifacts_dir = str(managed.artifacts_dir)
            logs_dir = str(managed.session_dir / "logs")
            agent_log_dir = str(managed.session_dir / "logs" / "agent")
            runtime_env: dict[str, str] = {}
        else:
            session_dir = runtime.runtime_session_dir
            artifacts_dir = runtime.runtime_artifacts_dir
            logs_dir = runtime.runtime_logs_dir
            agent_log_dir = runtime.runtime_agent_log_dir
            runtime_env = dict(runtime.spec.env)
        agent_env = dict(request.agent.env) if include_agent_env else {}
        return {
            "ANTHROPIC_BASE_URL": self.gateway_url,
            "ANTHROPIC_API_KEY": request.session_id,
            "OPENAI_BASE_URL": f"{self.gateway_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": request.session_id,
            "GOOGLE_API_URL": self.gateway_url,
            "GOOGLE_API_KEY": request.session_id,
            "SESSION_ID": request.session_id,
            "TASK_ID": request.task_id,
            "SESSION_DIR": session_dir,
            "ARTIFACTS_DIR": artifacts_dir,
            "LOGS_DIR": logs_dir,
            "AGENT_LOG_DIR": agent_log_dir,
            **{key: str(value) for key, value in runtime_env.items()},
            **{key: str(value) for key, value in agent_env.items()},
        }

    @staticmethod
    def _evaluator_env(
        evaluator_spec: EvaluatorSpec,
        runtime_spec: RuntimeSpec,
    ) -> dict[str, str]:
        return {
            **{key: str(value) for key, value in runtime_spec.env.items()},
            **{key: str(value) for key, value in evaluator_spec.env.items()},
        }

    @staticmethod
    def _write_exec_log(
        log_dir: Path, prefix: str, stdout: str | None, stderr: str | None
    ) -> None:
        if stdout:
            (log_dir / f"{prefix}.stdout.log").write_text(stdout)
        if stderr:
            (log_dir / f"{prefix}.stderr.log").write_text(stderr)

    @staticmethod
    def _step_metadata(log_dir: Path, step_index: int, managed: ManagedSession) -> dict:
        return {
            "log_dir": str(log_dir),
            "last_step": step_index,
            "cwd": str(managed.session_dir),
        }

    def _error_result(
        self,
        request: SessionDispatchRequest,
        timer: StageTimer,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="ERROR",
            trajectory=Trajectory(
                status="ERROR",
                metadata={
                    "builder": request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    def _timeout_result(
        self,
        request: SessionDispatchRequest,
        timer: StageTimer,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="TIMEOUT",
            trajectory=Trajectory(
                status="TIMEOUT",
                metadata={
                    "builder": request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    def _cancelled_result(self, request: SessionDispatchRequest, timer: StageTimer) -> SessionResult:
        return self._error_result(request, timer, "session cancelled")

    async def _push_result(self, callback_url: str | None, result: SessionResult) -> bool:
        """POST the terminal result to the rollout server. Return True on success."""
        if not callback_url:
            return False
        try:
            response = await self._client.post(callback_url, json=result.model_dump(mode="json"))
            response.raise_for_status()
            return True
        except Exception:
            logger.warning(
                "Failed to deliver callback for session %s to %s",
                result.session_id,
                callback_url,
                exc_info=True,
            )
            return False

    @staticmethod
    def _snapshot_to_metrics(snapshot: DispatcherSnapshot) -> NodeStageMetrics:
        return NodeStageMetrics(
            init_queue_depth=snapshot.init_queue_depth,
            init_inflight=snapshot.init_inflight,
            ready_depth=snapshot.ready_depth,
            run_inflight=snapshot.run_inflight,
            postrun_queue_depth=snapshot.postrun_queue_depth,
            postrun_inflight=snapshot.postrun_inflight,
        )

    def _remaining_budget(self, managed: ManagedSession) -> float:
        deadline = managed.execution_deadline
        if deadline is None:
            raise RuntimeError("session execution deadline was not initialized")
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise GatewayExecutionTimeout("session execution timeout")
        return remaining

    async def _await_with_budget(
        self,
        awaitable,
        managed: ManagedSession,
    ):
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=self._remaining_budget(managed),
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout("session execution timeout") from exc

    @staticmethod
    def _start_execution_deadline(managed: ManagedSession) -> None:
        if managed.execution_deadline is not None:
            return
        managed.execution_deadline = (
            asyncio.get_running_loop().time()
            + managed.request.remaining_timeout_seconds
        )

    async def _run_postrun_steps(self, managed: ManagedSession) -> None:
        if managed.cancel_reason == "policy_cutoff":
            return
        if not managed.postrun_steps or managed.runtime is None:
            return
        log_dir = managed.session_dir / "logs" / "teardown"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = self._runtime_env(managed.request, managed, include_agent_env=True)
        for i, step in enumerate(managed.postrun_steps):
            try:
                merged_env = {**env, **(step.env or {})}
                result = await managed.runtime.exec(
                    step.command,
                    cwd=step.cwd,
                    env=merged_env,
                    timeout_sec=self._remaining_budget(managed),
                )
                self._write_exec_log(
                    log_dir,
                    f"step.{i:02d}",
                    result.stdout,
                    result.stderr,
                )
            except Exception:
                logger.debug(
                    "Teardown step failed for session %s",
                    managed.request.session_id,
                    exc_info=True,
                )

    async def _stop_runtime_best_effort(
        self,
        runtime: BaseRuntime,
        session_id: str,
        label: str,
    ) -> None:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "Failed to stop %s for session %s",
                label,
                session_id,
                exc_info=True,
            )

    async def _remove_session_dir_best_effort(
        self,
        session_dir: Path,
        session_id: str,
    ) -> None:
        # 诊断开关:置 POLAR_KEEP_SESSION_DIR=1 时保留 session 目录(claude 转录/submission/metrics),
        # 供跨容器从共享盘 /home/docker/polar_sessions 读取定位 reward 根因。默认仍删(原行为)。
        if os.environ.get("POLAR_KEEP_SESSION_DIR"):
            logger.info("POLAR_KEEP_SESSION_DIR set; KEEP session dir for %s: %s", session_id, session_dir)
            return
        try:
            await asyncio.to_thread(shutil.rmtree, session_dir)
        except FileNotFoundError:
            return
        except Exception:
            logger.warning(
                "Failed to remove session directory for session %s",
                session_id,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _log_session_summary_wandb(timing) -> None:
    """Log session-level stage timing to the current (per-session) W&B run."""
    import wandb
    summary: dict[str, float] = {}

    def _set(key: str, value: float) -> None:
        if value > 0:
            summary[f"polar/session_ms/{key}"] = value

    # coarse
    _set("register_to_init_queue", timing.register_to_init_queue_ms)
    _set("init", timing.init_ms)
    _set("run", timing.run_ms)
    _set("postrun", timing.postrun_ms)
    _set("total", timing.total_ms)
    # init breakdown
    _set("init_runtime_create", timing.init_runtime_create_ms)
    _set("init_prepare", timing.init_prepare_ms)
    _set("ready_wait", timing.ready_wait_ms)
    # run breakdown
    _set("run_harness_setup", timing.run_harness_setup_ms)
    _set("run_agent_exec", timing.run_agent_exec_ms)
    _set("run_harness_postprocess", timing.run_harness_postprocess_ms)
    # postrun breakdown
    _set("postrun_build", timing.postrun_build_ms)
    _set("postrun_eval", timing.postrun_eval_ms)
    _set("postrun_teardown", timing.postrun_teardown_ms)
    _set("postrun_push_result", timing.postrun_push_result_ms)
    # LLM aggregates
    summary["polar/llm_call_count"] = float(timing.llm_call_count)
    _set("llm_sglang_wait", timing.llm_total_ms)
    _set("llm_request_roundtrip", timing.llm_request_total_ms)
    _set("llm_agent_side_total", timing.llm_agent_side_total_ms)

    wandb.log(summary)
