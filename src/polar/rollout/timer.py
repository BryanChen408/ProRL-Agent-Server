"""Per-session stage timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from polar.rollout.models import SessionTiming


# Finer marks (``build``, ``eval``, ``teardown``) still write into the mark
# dictionary for debug logs but are rolled into ``postrun_ms`` in the public
# schema.
_POSTRUN_MARKS: tuple[str, ...] = ("postrun", "build", "eval", "teardown")


@dataclass(slots=True)
class StageTimer:
    """Record stable durations plus wall-clock anchors for trace correlation."""

    _marks: dict[str, float] = field(default_factory=dict)
    _clock_monotonic_origin: float = field(default_factory=time.monotonic)
    _clock_epoch_origin_ns: int = field(default_factory=time.time_ns)

    # Accumulators for per-LLM-call timing (populated by gateway proxy).
    _llm_call_count: int = 0
    _llm_total_ms: float = 0.0
    _llm_request_total_ms: float = 0.0
    _llm_agent_side_total_ms: float = 0.0   # total tool + client overhead between LLM calls
    _llm_calls: list = field(default_factory=list)  # per-call detail
    _tool_execs: list = field(default_factory=list)  # per-tool-execution detail

    def mark(self, stage: str, event: str) -> None:
        """Mark a stage start or finish."""
        self._marks[f"{stage}_{event}"] = time.monotonic()

    def monotonic_to_epoch_ns(self, value: float) -> int:
        """Map a monotonic timestamp onto the timer's wall-clock epoch anchor."""
        delta_ns = int((value - self._clock_monotonic_origin) * 1_000_000_000)
        return self._clock_epoch_origin_ns + delta_ns

    def record_llm_call(
        self, *,
        acquire_wait_ms: float = 0.0,
        prepare_ms: float = 0.0,
        sglang_wait_ms: float = 0.0,
        normalize_ms: float = 0.0,
        roundtrip_ms: float = 0.0,
        post_ms: float = 0.0,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        trace_timing: dict[str, int] | None = None,
        trace_id: str | None = None,
        engine_url: str | None = None,
        engine_metrics: dict[str, float | int] | None = None,
    ) -> None:
        """Record one LLM inference call's timing (called by gateway proxy).

        Segments (see proxy.py completion() for detailed definitions):
          B. acquire_wait_ms  — semaphore wait
          C. prepare_ms       — deepcopy + engine.prepare_request
          D-G. sglang_wait_ms — httpx POST → SGLang → response
          H. normalize_ms     — engine.normalize_response
          I. post_ms          — storage.save_message + transform_response
        """
        self._llm_call_count += 1
        self._llm_total_ms += sglang_wait_ms
        self._llm_request_total_ms += roundtrip_ms
        call = {
            "round": self._llm_call_count,
            "acquire_wait_ms": round(acquire_wait_ms, 2),
            "prepare_ms": round(prepare_ms, 2),
            "sglang_wait_ms": round(sglang_wait_ms, 2),
            "normalize_ms": round(normalize_ms, 2),
            "post_ms": round(post_ms, 2),
            "roundtrip_ms": round(roundtrip_ms, 2),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
        }
        if trace_timing:
            call["trace_timing"] = {
                str(key): int(value)
                for key, value in trace_timing.items()
                if isinstance(value, int) and value >= 0
            }
        if trace_id:
            call["trace_id"] = trace_id
        if engine_url:
            call["engine_url"] = engine_url
        if engine_metrics:
            call["engine_metrics"] = {
                str(key): value
                for key, value in engine_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            }
        self._llm_calls.append(call)

    def record_tool_exec(
        self, *,
        command: str = "",
        duration_ms: float = 0.0,
        exit_code: int = 0,
        started_at_ns: int | None = None,
        finished_at_ns: int | None = None,
    ) -> None:
        """Record one agent tool execution (bash/sed/python etc.).

        Called by ``_run_exec_inputs`` inside the Gateway for every ExecInput
        step the agent harness emits.  Each call is a separate per-step data
        point that can be plotted or embedded in Chrome Trace spans.
        """
        tool_exec = {
            "idx": len(self._tool_execs),
            "command": command[:256],   # truncate long commands
            "duration_ms": round(duration_ms, 2),
            "exit_code": exit_code,
        }
        if started_at_ns is not None:
            tool_exec["started_at_ns"] = started_at_ns
        if finished_at_ns is not None:
            tool_exec["finished_at_ns"] = finished_at_ns
        self._tool_execs.append(tool_exec)

    def patch_last_llm_agent_side_gap(
        self,
        *,
        gap_ms: float,
        actions: list[dict] | None = None,
        gap_started_at_ns: int | None = None,
        gap_finished_at_ns: int | None = None,
    ) -> None:
        """Attach the pre-call agent gap and its completed actions.

        A request reports only the total interval between the previous response
        and this request.  When several tool results arrive together (parallel
        tool use), the total is split evenly so the synthetic Perfetto timeline
        remains additive.  The trace metadata explicitly marks this allocation
        as estimated.
        """
        if not self._llm_calls:
            return

        safe_gap_ms = max(0.0, gap_ms)
        call = self._llm_calls[-1]
        call["agent_side_gap_ms"] = round(safe_gap_ms, 2)
        if gap_started_at_ns is not None:
            call["agent_side_gap_started_at_ns"] = gap_started_at_ns
        if gap_finished_at_ns is not None:
            call["agent_side_gap_finished_at_ns"] = gap_finished_at_ns
        self._llm_agent_side_total_ms += safe_gap_ms

        normalized = [dict(action) for action in (actions or [])]
        if safe_gap_ms > 0 and not normalized:
            normalized = [{
                "kind": "think",
                "tool_name": "",
                "tool_use_id": "",
                "summary": "agent think",
                "is_error": False,
            }]
        if not normalized:
            return

        share_ms = safe_gap_ms / len(normalized)
        assigned_ms = 0.0
        parallel = len(normalized) > 1
        for index, action in enumerate(normalized):
            duration_ms = (
                safe_gap_ms - assigned_ms
                if index == len(normalized) - 1
                else round(share_ms, 2)
            )
            assigned_ms += duration_ms
            action["duration_ms"] = round(max(0.0, duration_ms), 2)
            action["duration_estimated"] = True
            action["gap_allocation"] = "equal_share" if parallel else "whole_gap"
            action["parallel"] = parallel
        call["agent_actions"] = normalized

    def to_session_timing(self) -> SessionTiming:
        """Return durations for every phase of the session lifecycle.

        Coarse aggregates (init_ms / run_ms / postrun_ms) are computed as
        sums of their finer children when those children are populated,
        preserving backward compatibility with existing dashboards.
        """
        # ── postrun children ──
        build_ms = self._duration_ms("build")
        eval_ms = self._duration_ms("eval")
        teardown_ms = self._duration_ms("teardown")
        docker_kill_ms = self._duration_ms("docker_kill")
        docker_rm_ms = self._duration_ms("docker_rm")
        postrun_ms = self._postrun_span_ms()

        # ── init children ──
        runtime_create_ms = self._duration_ms("runtime_create")
        docker_create_ms = self._duration_ms("docker_create")
        docker_start_ms = self._duration_ms("docker_start")
        prepare_ms = self._duration_ms("prepare")
        init_ms = self._duration_ms("init")

        # ── run children ──
        harness_setup_ms = self._duration_ms("harness_setup")
        agent_exec_ms = self._duration_ms("agent_exec")
        postprocess_ms = self._duration_ms("harness_postprocess")
        run_ms = self._duration_ms("run")

        # ── other ──
        ready_wait_ms = self._span_ms("init_finished", "run_started")
        push_result_ms = self._duration_ms("push_result")

        # ── total ──
        total_ms = self._span_ms("dispatch_started", "return_finished")

        stage_spans = self._stage_spans()
        trace_start_ns = min(
            (span["started_at_ns"] for span in stage_spans),
            default=None,
        )
        trace_end_ns = max(
            (span["finished_at_ns"] for span in stage_spans),
            default=None,
        )

        return SessionTiming(
            schema_version=2,
            trace_start_time_ns=trace_start_ns,
            trace_end_time_ns=trace_end_ns,
            stage_spans=stage_spans,
            # coarse
            register_to_init_queue_ms=self._span_ms("dispatch_started", "init_started"),
            init_ms=init_ms if init_ms > 0 else runtime_create_ms + prepare_ms,
            run_ms=run_ms if run_ms > 0 else harness_setup_ms + agent_exec_ms + postprocess_ms,
            postrun_ms=postrun_ms if postrun_ms > 0 else build_ms + eval_ms + teardown_ms + push_result_ms,
            # init breakdown
            init_runtime_create_ms=runtime_create_ms,
            init_docker_create_ms=docker_create_ms,
            init_docker_start_ms=docker_start_ms,
            init_prepare_ms=prepare_ms,
            # ready
            ready_wait_ms=ready_wait_ms,
            # run breakdown
            run_harness_setup_ms=harness_setup_ms,
            run_agent_exec_ms=agent_exec_ms,
            run_harness_postprocess_ms=postprocess_ms,
            # LLM
            llm_call_count=self._llm_call_count,
            llm_total_ms=self._llm_total_ms,
            llm_request_total_ms=self._llm_request_total_ms,
            llm_agent_side_total_ms=self._llm_agent_side_total_ms,
            llm_calls=list(self._llm_calls),
            tool_execs=list(self._tool_execs),
            # postrun breakdown
            postrun_build_ms=build_ms,
            postrun_eval_ms=eval_ms,
            postrun_teardown_ms=teardown_ms,
            postrun_docker_kill_ms=docker_kill_ms,
            postrun_docker_rm_ms=docker_rm_ms,
            postrun_push_result_ms=push_result_ms,
            # total
            total_ms=total_ms,
        )

    def _stage_spans(self) -> list[dict]:
        """Return completed stage intervals with epoch-nanosecond boundaries."""
        spans: list[dict] = []
        for key, started in self._marks.items():
            if not key.endswith("_started"):
                continue
            stage = key.removesuffix("_started")
            finished = self._marks.get(f"{stage}_finished")
            if finished is None:
                continue
            started_at_ns = self.monotonic_to_epoch_ns(started)
            finished_at_ns = self.monotonic_to_epoch_ns(max(started, finished))
            spans.append({
                "name": stage,
                "started_at_ns": started_at_ns,
                "finished_at_ns": finished_at_ns,
                "duration_ms": round((finished_at_ns - started_at_ns) / 1_000_000, 2),
            })
        for name, start_mark, end_mark in (
            ("session", "dispatch_started", "return_finished"),
            ("register_to_init_queue", "dispatch_started", "init_started"),
            ("ready_wait", "init_finished", "run_started"),
        ):
            started = self._marks.get(start_mark)
            finished = self._marks.get(end_mark)
            if started is None or finished is None:
                continue
            started_at_ns = self.monotonic_to_epoch_ns(started)
            finished_at_ns = self.monotonic_to_epoch_ns(max(started, finished))
            spans.append({
                "name": name,
                "started_at_ns": started_at_ns,
                "finished_at_ns": finished_at_ns,
                "duration_ms": round((finished_at_ns - started_at_ns) / 1_000_000, 2),
            })
        spans.sort(key=lambda span: (span["started_at_ns"], span["finished_at_ns"]))
        return spans

    def _duration_ms(self, stage: str) -> float:
        started = self._marks.get(f"{stage}_started")
        finished = self._marks.get(f"{stage}_finished") or started
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _span_ms(self, start_mark: str, end_mark: str) -> float:
        started = self._marks.get(start_mark)
        finished = self._marks.get(end_mark)
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _postrun_span_ms(self) -> float:
        """Earliest postrun-family started to latest postrun-family finished."""
        starts = [
            self._marks[f"{stage}_started"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_started" in self._marks
        ]
        finishes = [
            self._marks[f"{stage}_finished"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_finished" in self._marks
        ]
        if not starts or not finishes:
            return 0.0
        return max(0.0, (max(finishes) - min(starts)) * 1000.0)
