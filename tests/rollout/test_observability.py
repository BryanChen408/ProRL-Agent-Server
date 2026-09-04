from __future__ import annotations

import asyncio
import json

import httpx

import polar.gateway.server as gateway_server
from polar.rollout.models import NodeStageMetrics, SessionStatus, SessionTiming
from polar.rollout.observability import SessionObservability


def _timing() -> SessionTiming:
    start = 1_700_000_000_000_000_000
    return SessionTiming(
        schema_version=2,
        trace_start_time_ns=start,
        trace_end_time_ns=start + 2_000_000_000,
        total_ms=2_000,
        llm_call_count=1,
        stage_spans=[{
            "name": "session",
            "started_at_ns": start,
            "finished_at_ns": start + 2_000_000_000,
            "duration_ms": 2_000,
        }],
        llm_calls=[{
            "round": 1,
            "prompt_tokens": 10,
            "response_tokens": 4,
            "trace_id": "session-1:1",
            "engine_name": "vllm",
            "engine_url": "http://engine:8000",
            "engine_metrics": {
                "queue_ms": 10.0,
                "prefill_ms": 90.0,
                "decode_ms": 500.0,
                "num_cached_tokens": 8,
            },
            "trace_timing": {
                "request_started_at_ns": start + 100_000_000,
                "sglang_started_at_ns": start + 200_000_000,
                "sglang_finished_at_ns": start + 900_000_000,
                "response_finished_at_ns": start + 1_000_000_000,
            },
        }],
    )


def test_prometheus_metrics_are_low_cardinality() -> None:
    async def _run() -> str:
        exporter = SessionObservability(node_id="node-a", gateway_url="http://node-a:8081")
        async with httpx.AsyncClient() as client:
            await exporter.record_session(
                client,
                timing=_timing(),
                session_id="secret-session-id",
                task_id="task-1",
                status=SessionStatus.COMPLETED,
                metadata={},
            )
        return exporter.render_prometheus(NodeStageMetrics(run_inflight=2))

    metrics = asyncio.run(_run())

    assert 'polar_sessions_total{node_id="node-a",status="completed"} 1' in metrics
    assert 'polar_llm_tokens_total{node_id="node-a",direction="prompt"} 10' in metrics
    assert 'polar_gateway_sessions{node_id="node-a",stage="run_inflight"} 2' in metrics
    assert "secret-session-id" not in metrics


def test_otlp_export_has_one_trace_and_parent_child_spans() -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200)

    async def _run() -> None:
        exporter = SessionObservability(
            node_id="node-a",
            gateway_url="http://node-a:8081",
            otlp_endpoint="http://tempo:4318/v1/traces",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await exporter.record_session(
                client,
                timing=_timing(),
                session_id="session-1",
                task_id="run-1--task-2",
                status=SessionStatus.COMPLETED,
                metadata={"group_id": 7, "sample_pos": 2, "rollout_step": 11},
            )

    asyncio.run(_run())

    spans = captured[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root = next(span for span in spans if span["name"] == "agent_session")
    generation = next(span for span in spans if span["name"] == "gateway_generation")
    engine = next(span for span in spans if span["name"] == "gateway_generation.inference")
    prefill = next(
        span for span in spans if span["name"] == "gateway_generation.engine.prefill"
    )
    assert {span["traceId"] for span in spans} == {root["traceId"]}
    assert generation["parentSpanId"] == root["spanId"]
    assert engine["parentSpanId"] == generation["spanId"]
    engine_attributes = {
        item["key"]: next(iter(item["value"].values()))
        for item in engine["attributes"]
    }
    assert engine_attributes["engine"] == "vllm"
    assert engine_attributes["engine_url"] == "http://engine:8000"
    assert prefill["parentSpanId"] == generation["spanId"]
    assert int(prefill["endTimeUnixNano"]) - int(prefill["startTimeUnixNano"]) == 90_000_000
    attributes = {
        item["key"]: next(iter(item["value"].values()))
        for item in root["attributes"]
    }
    assert attributes["project"] == "polar"
    assert attributes["sample"] == "7"
    assert attributes["global_steps"] == "11"


def test_rl_insight_target_registration_uses_gateway_authority() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200)

    async def _run() -> None:
        exporter = SessionObservability(
            node_id="node-a",
            gateway_url="http://10.0.0.8:8081",
            rl_insight_url="http://insight:18080",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await exporter.register_prometheus_target(client)

    asyncio.run(_run())

    assert requests == [{
        "job_name": "polar-gateway",
        "targets": [{"target": "10.0.0.8:8081", "labels": {"node_id": "node-a"}}],
    }]


def test_gateway_metrics_endpoint_uses_live_stage_snapshot(monkeypatch) -> None:
    exporter = SessionObservability(node_id="node-a", gateway_url="http://node-a:8081")

    class NodeManager:
        observability = exporter

        async def stage_metrics(self) -> NodeStageMetrics:
            return NodeStageMetrics(ready_depth=3)

    monkeypatch.setattr(
        gateway_server,
        "get_state",
        lambda: type("State", (), {"node_manager": NodeManager()})(),
    )

    response = asyncio.run(gateway_server.prometheus_metrics())

    assert response.media_type.startswith("text/plain")
    assert 'polar_gateway_sessions{node_id="node-a",stage="ready_depth"} 3' in response.body.decode()
