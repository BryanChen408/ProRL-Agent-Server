"""Low-cardinality Prometheus metrics and OTLP/HTTP session trace export."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

import httpx

from polar.rollout.models import NodeStageMetrics, SessionTiming
from polar.run_namespace import run_id_from_metadata

logger = logging.getLogger(__name__)

_DURATION_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 900.0, 3600.0)
_INFERENCE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
_RATIO_BUCKETS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
_ENGINE_LATENCIES = {
    "request": "roundtrip_ms",
    "queue": "queue_ms",
    "ttft": "ttft_ms",
    "prefill": "prefill_ms",
    "decode": "decode_ms",
}


class SessionObservability:
    """Own gateway metrics and best-effort export into an RL-Insight stack."""

    def __init__(
        self,
        *,
        node_id: str,
        gateway_url: str,
        prometheus_enabled: bool = True,
        rl_insight_url: str | None = None,
        otlp_endpoint: str | None = None,
        otlp_headers: dict[str, str] | None = None,
        export_timeout_seconds: float = 3.0,
        service_name: str = "polar-gateway",
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url
        self.prometheus_enabled = prometheus_enabled
        self.rl_insight_url = rl_insight_url.rstrip("/") if rl_insight_url else None
        self.otlp_endpoint = otlp_endpoint
        self.otlp_headers = dict(otlp_headers or {})
        self.export_timeout_seconds = export_timeout_seconds
        self.service_name = service_name
        self._sessions: dict[str, int] = defaultdict(int)
        self._llm_calls = 0
        self._tokens: dict[str, int] = defaultdict(int)
        self._inference_requests: dict[str, int] = defaultdict(int)
        self._inference_tokens: dict[tuple[str, str], int] = defaultdict(int)
        self._inference_histograms: dict[tuple[str, str], dict[str, Any]] = {}
        self._cached_prompt_tokens: dict[str, int] = defaultdict(int)
        self._duration_count = 0
        self._duration_sum = 0.0
        self._duration_buckets: dict[float, int] = defaultdict(int)
        self._export_failures = 0

    def record_inference(
        self,
        *,
        engine_name: str | None,
        prompt_tokens: int,
        response_tokens: int,
        roundtrip_ms: float,
        engine_metrics: dict[str, float | int] | None = None,
    ) -> None:
        """Update scrape-visible counters as soon as one inference call completes."""
        engine = _safe_label((engine_name or "unknown").lower())
        prompt = max(0, int(prompt_tokens))
        response = max(0, int(response_tokens))
        self._llm_calls += 1
        self._tokens["prompt"] += prompt
        self._tokens["response"] += response
        self._inference_requests[engine] += 1
        self._inference_tokens[(engine, "prompt")] += prompt
        self._inference_tokens[(engine, "response")] += response

        metrics = dict(engine_metrics or {})
        metrics["roundtrip_ms"] = roundtrip_ms
        for metric, field in _ENGINE_LATENCIES.items():
            value = metrics.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _observe_histogram(
                    self._inference_histograms,
                    (engine, metric),
                    max(0.0, float(value) / 1000.0),
                    _INFERENCE_BUCKETS,
                )
        cached = metrics.get("num_cached_tokens")
        if isinstance(cached, (int, float)) and not isinstance(cached, bool):
            self._cached_prompt_tokens[engine] += max(0, int(cached))
        cache_hit = metrics.get("prefix_cache_hit_pct")
        if isinstance(cache_hit, (int, float)) and not isinstance(cache_hit, bool):
            _observe_histogram(
                self._inference_histograms,
                (engine, "prefix_cache_hit_ratio"),
                min(1.0, max(0.0, float(cache_hit) / 100.0)),
                _RATIO_BUCKETS,
            )

    async def register_prometheus_target(self, client: httpx.AsyncClient) -> None:
        """Register this gateway's `/metrics` target with RL-Insight when configured."""
        if not self.prometheus_enabled or not self.rl_insight_url:
            return
        parsed = urlparse(self.gateway_url)
        target = parsed.netloc or parsed.path
        try:
            response = await client.post(
                f"{self.rl_insight_url}/api/v1/prometheus/targets",
                json={
                    "job_name": "polar-gateway",
                    "targets": [{"target": target, "labels": {"node_id": self.node_id}}],
                },
                timeout=self.export_timeout_seconds,
            )
            response.raise_for_status()
            logger.info("Registered Polar metrics target %s with RL-Insight", target)
        except Exception:
            logger.warning("Failed to register metrics target with RL-Insight", exc_info=True)

    async def record_session(
        self,
        client: httpx.AsyncClient,
        *,
        timing: SessionTiming,
        session_id: str,
        task_id: str,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        """Update local metrics and export one correlated OTLP trace, best effort."""
        status_label = _safe_label(status.lower())
        self._sessions[status_label] += 1
        duration_seconds = max(0.0, timing.total_ms / 1000.0)
        self._duration_count += 1
        self._duration_sum += duration_seconds
        for bucket in _DURATION_BUCKETS:
            if duration_seconds <= bucket:
                self._duration_buckets[bucket] += 1

        if not self.otlp_endpoint:
            return
        payload = _build_otlp_payload(
            timing,
            service_name=self.service_name,
            node_id=self.node_id,
            session_id=session_id,
            task_id=task_id,
            status=status,
            metadata=metadata,
        )
        if not payload:
            return
        try:
            headers = {"Content-Type": "application/json", **self.otlp_headers}
            response = await client.post(
                self.otlp_endpoint,
                content=_json_bytes(payload),
                headers=headers,
                timeout=self.export_timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            self._export_failures += 1
            logger.warning("Failed to export session %s trace over OTLP", session_id, exc_info=True)

    def render_prometheus(self, stages: NodeStageMetrics) -> str:
        """Render Prometheus text format without adding a runtime dependency."""
        if not self.prometheus_enabled:
            return ""
        label = f'node_id="{_escape_label(self.node_id)}"'
        lines = [
            "# HELP polar_sessions_total Completed rollout sessions by terminal status.",
            "# TYPE polar_sessions_total counter",
        ]
        for status, value in sorted(self._sessions.items()):
            lines.append(f'polar_sessions_total{{{label},status="{status}"}} {value}')
        lines.extend([
            "# HELP polar_llm_calls_total LLM calls completed by rollout sessions.",
            "# TYPE polar_llm_calls_total counter",
            f"polar_llm_calls_total{{{label}}} {self._llm_calls}",
            "# HELP polar_llm_tokens_total LLM tokens by direction.",
            "# TYPE polar_llm_tokens_total counter",
        ])
        for direction in ("prompt", "response"):
            lines.append(
                f'polar_llm_tokens_total{{{label},direction="{direction}"}} '
                f'{self._tokens[direction]}'
            )
        lines.extend([
            "# HELP polar_inference_requests_total Completed inference requests by engine.",
            "# TYPE polar_inference_requests_total counter",
        ])
        for engine, value in sorted(self._inference_requests.items()):
            lines.append(
                f'polar_inference_requests_total{{{label},engine="{engine}"}} {value}'
            )
        lines.extend([
            "# HELP polar_inference_tokens_total Inference tokens by engine and direction.",
            "# TYPE polar_inference_tokens_total counter",
        ])
        for (engine, direction), value in sorted(self._inference_tokens.items()):
            lines.append(
                f'polar_inference_tokens_total{{{label},engine="{engine}",direction="{direction}"}} {value}'
            )
        lines.extend([
            "# HELP polar_inference_cached_prompt_tokens_total Cached prompt tokens reported by inference engines.",
            "# TYPE polar_inference_cached_prompt_tokens_total counter",
        ])
        for engine, value in sorted(self._cached_prompt_tokens.items()):
            lines.append(
                f'polar_inference_cached_prompt_tokens_total{{{label},engine="{engine}"}} {value}'
            )
        for metric, help_text in (
            ("request", "End-to-end inference request duration."),
            ("queue", "Inference engine queue duration."),
            ("ttft", "Inference time to first token."),
            ("prefill", "Inference prefill duration."),
            ("decode", "Inference decode duration."),
            ("prefix_cache_hit_ratio", "Inference prefix-cache hit ratio."),
        ):
            metric_name = f"polar_inference_{metric}_seconds"
            buckets = _INFERENCE_BUCKETS
            if metric == "request":
                metric_name = "polar_inference_request_duration_seconds"
            elif metric == "prefix_cache_hit_ratio":
                metric_name = "polar_inference_prefix_cache_hit_ratio"
                buckets = _RATIO_BUCKETS
            lines.extend([f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} histogram"])
            for engine in sorted(self._inference_requests):
                state = self._inference_histograms.get((engine, metric))
                if state is not None:
                    _render_histogram(lines, metric_name, label, engine, state, buckets)
        lines.extend([
            "# HELP polar_session_duration_seconds End-to-end rollout session duration.",
            "# TYPE polar_session_duration_seconds histogram",
        ])
        for bucket in _DURATION_BUCKETS:
            lines.append(
                f'polar_session_duration_seconds_bucket{{{label},le="{bucket:g}"}} '
                f"{self._duration_buckets[bucket]}"
            )
        lines.extend([
            f'polar_session_duration_seconds_bucket{{{label},le="+Inf"}} {self._duration_count}',
            f"polar_session_duration_seconds_sum{{{label}}} {self._duration_sum:g}",
            f"polar_session_duration_seconds_count{{{label}}} {self._duration_count}",
            "# HELP polar_observability_export_failures_total Failed OTLP exports.",
            "# TYPE polar_observability_export_failures_total counter",
            f"polar_observability_export_failures_total{{{label}}} {self._export_failures}",
            "# HELP polar_gateway_sessions Current sessions by scheduler stage.",
            "# TYPE polar_gateway_sessions gauge",
        ])
        for stage, value in stages.model_dump().items():
            lines.append(f'polar_gateway_sessions{{{label},stage="{stage}"}} {value}')
        return "\n".join(lines) + "\n"


def _observe_histogram(
    states: dict[tuple[str, str], dict[str, Any]],
    key: tuple[str, str],
    value: float,
    buckets: tuple[float, ...],
) -> None:
    state = states.setdefault(key, {"count": 0, "sum": 0.0, "buckets": defaultdict(int)})
    state["count"] += 1
    state["sum"] += value
    for bucket in buckets:
        if value <= bucket:
            state["buckets"][bucket] += 1


def _render_histogram(
    lines: list[str],
    metric_name: str,
    node_label: str,
    engine: str,
    state: dict[str, Any],
    buckets: tuple[float, ...],
) -> None:
    labels = f'{node_label},engine="{engine}"'
    for bucket in buckets:
        lines.append(
            f'{metric_name}_bucket{{{labels},le="{bucket:g}"}} {state["buckets"][bucket]}'
        )
    lines.extend([
        f'{metric_name}_bucket{{{labels},le="+Inf"}} {state["count"]}',
        f'{metric_name}_sum{{{labels}}} {state["sum"]:g}',
        f'{metric_name}_count{{{labels}}} {state["count"]}',
    ])


def _build_otlp_payload(
    timing: SessionTiming,
    *,
    service_name: str,
    node_id: str,
    session_id: str,
    task_id: str,
    status: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    boundaries = [
        (int(span["started_at_ns"]), int(span["finished_at_ns"]))
        for span in timing.stage_spans
        if span.get("started_at_ns") is not None and span.get("finished_at_ns") is not None
    ]
    for call in timing.llm_calls:
        trace = call.get("trace_timing") or {}
        if trace.get("request_started_at_ns") and trace.get("response_finished_at_ns"):
            boundaries.append(
                (int(trace["request_started_at_ns"]), int(trace["response_finished_at_ns"]))
            )
    if not boundaries:
        return None

    trace_id = hashlib.sha256(session_id.encode()).hexdigest()[:32]
    root_span_id = _span_id(session_id, "root")
    start_ns = min(start for start, _ in boundaries)
    end_ns = max(end for _, end in boundaries)
    identity = _trace_identity(task_id, session_id, metadata)
    spans = [
        _otlp_span(
            trace_id,
            root_span_id,
            "agent_session",
            start_ns,
            end_ns,
            attributes={
                **identity,
                "node_id": node_id,
                "status": status,
                "llm_call_count": timing.llm_call_count,
                "monitor.trace_source": "polar_session",
            },
        )
    ]
    for index, stage in enumerate(timing.stage_spans):
        spans.append(
            _otlp_span(
                trace_id,
                _span_id(session_id, f"stage:{index}:{stage.get('name')}"),
                f"polar.{stage.get('name', 'stage')}",
                int(stage["started_at_ns"]),
                int(stage["finished_at_ns"]),
                parent_span_id=root_span_id,
                attributes={**identity, "stage": str(stage.get("name", ""))},
            )
        )
    for index, call in enumerate(timing.llm_calls):
        trace = call.get("trace_timing") or {}
        call_start = trace.get("request_started_at_ns")
        call_end = trace.get("response_finished_at_ns")
        if not call_start or not call_end:
            continue
        call_span_id = _span_id(session_id, f"llm:{index}")
        spans.append(
            _otlp_span(
                trace_id,
                call_span_id,
                "gateway_generation",
                int(call_start),
                int(call_end),
                parent_span_id=root_span_id,
                attributes={
                    **identity,
                    "turn": int(call.get("round", index + 1)),
                    "prompt_tokens": int(call.get("prompt_tokens", 0)),
                    "response_tokens": int(call.get("response_tokens", 0)),
                    "polar.trace_id": call.get("trace_id", ""),
                    "engine": call.get("engine_name", "unknown"),
                    "engine_url": call.get("engine_url", ""),
                    **(call.get("engine_metrics") or {}),
                },
            )
        )
        for segment in ("acquire", "prepare", "sglang", "normalize", "post"):
            segment_start = trace.get(f"{segment}_started_at_ns")
            segment_end = trace.get(f"{segment}_finished_at_ns")
            if segment_start is None or segment_end is None:
                continue
            spans.append(
                _otlp_span(
                    trace_id,
                    _span_id(session_id, f"llm:{index}:{segment}"),
                    f"gateway_generation.{'inference' if segment == 'sglang' else segment}",
                    int(segment_start),
                    int(segment_end),
                    parent_span_id=call_span_id,
                    attributes={
                        **identity,
                        "turn": int(call.get("round", index + 1)),
                        **(
                            {
                                "engine": str(call.get("engine_name") or "unknown"),
                                "engine_url": str(call.get("engine_url") or ""),
                            }
                            if segment == "sglang"
                            else {}
                        ),
                    },
                )
            )
        for detail, detail_start, detail_end in _engine_metric_intervals(call):
            spans.append(
                _otlp_span(
                    trace_id,
                    _span_id(session_id, f"llm:{index}:engine:{detail}"),
                    f"gateway_generation.engine.{detail}",
                    detail_start,
                    detail_end,
                    parent_span_id=call_span_id,
                    attributes={
                        **identity,
                        "turn": int(call.get("round", index + 1)),
                        "position_derived": True,
                        "duration_measured": True,
                    },
                )
            )
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attribute("service.name", service_name)]},
            "scopeSpans": [{
                "scope": {"name": "polar.rollout", "version": "2"},
                "spans": spans,
            }],
        }]
    }


def _engine_metric_intervals(call: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Anchor engine durations within the measured upstream request interval."""
    metrics = call.get("engine_metrics") or {}
    trace = call.get("trace_timing") or {}
    start = trace.get("sglang_started_at_ns")
    end = trace.get("sglang_finished_at_ns")
    if not isinstance(start, int) or not isinstance(end, int):
        return []
    intervals: list[tuple[str, int, int]] = []
    cursor = start
    prefill_seen = False
    for key, name in (("queue_ms", "queue"), ("prefill_ms", "prefill")):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        interval_end = min(end, cursor + int(max(0.0, float(value)) * 1_000_000))
        intervals.append((name, cursor, interval_end))
        cursor = interval_end
        prefill_seen = prefill_seen or key == "prefill_ms"
    if not prefill_seen and isinstance(metrics.get("ttft_ms"), (int, float)):
        ttft_end = min(end, start + int(max(0.0, float(metrics["ttft_ms"])) * 1_000_000))
        intervals.append(("time_to_first_token", start, ttft_end))
        cursor = max(cursor, ttft_end)
    decode = metrics.get("decode_ms")
    if isinstance(decode, (int, float)) and not isinstance(decode, bool):
        decode_end = min(end, cursor + int(max(0.0, float(decode)) * 1_000_000))
        intervals.append(("decode", cursor, decode_end))
    return intervals


def _trace_identity(task_id: str, session_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": "polar",
        "experiment_name": run_id_from_metadata(task_id, metadata) or "polar",
        "sample": metadata.get("group_id", task_id),
        "session": metadata.get("sample_pos", session_id),
        "traj": metadata.get("trace_index", 0),
        "uid": metadata.get("op_name", ""),
        "global_steps": metadata.get("rollout_step", ""),
        "session_id": session_id,
        "task_id": task_id,
        "policy_version": metadata.get("policy_version", ""),
        "profiling_artifact_count": metadata.get("profiling_artifact_count", 0),
    }


def _otlp_span(
    trace_id: str,
    span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    *,
    parent_span_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(max(start_ns, end_ns)),
        "attributes": [_attribute(key, value) for key, value in (attributes or {}).items()],
        "status": {"code": 1},
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float) and math.isfinite(value):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _span_id(session_id: str, name: str) -> str:
    return hashlib.sha256(f"{session_id}:{name}".encode()).hexdigest()[:16]


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _json_bytes(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


__all__ = ["SessionObservability"]
