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
            "agent_side_gap_started_at_ns": start + 1_000_000_000,
            "agent_side_gap_finished_at_ns": start + 1_200_000_000,
            "agent_actions": [{
                "kind": "edit",
                "tool_name": "Edit",
                "tool_use_id": "tool-1",
                "summary": "edit kernel.cpp",
                "target": "kernel.cpp",
                "duration_ms": 200,
                "duration_estimated": True,
                "is_error": False,
            }],
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
        exporter.record_inference(
            engine_name="vllm",
            prompt_tokens=10,
            response_tokens=4,
            roundtrip_ms=700,
            engine_metrics={
                "queue_ms": 10,
                "ttft_ms": 100,
                "prefill_ms": 90,
                "decode_ms": 500,
                "num_cached_tokens": 8,
                "prefix_cache_hit_pct": 80,
            },
        )
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
    assert 'polar_llm_tokens_total{node_id="node-a",direction="response"} 4' in metrics
    assert 'polar_inference_requests_total{node_id="node-a",engine="vllm"} 1' in metrics
    assert (
        'polar_inference_decode_seconds_count{node_id="node-a",engine="vllm"} 1'
        in metrics
    )
    assert (
        'polar_inference_cached_prompt_tokens_total{node_id="node-a",engine="vllm"} 8'
        in metrics
    )
    assert (
        'polar_inference_prefix_cache_hit_ratio_sum{node_id="node-a",engine="vllm"} 0.8'
        in metrics
    )
    assert 'polar_gateway_sessions{node_id="node-a",stage="run_inflight"} 2' in metrics
    assert "secret-session-id" not in metrics


def test_inference_metrics_are_visible_before_session_completion() -> None:
    exporter = SessionObservability(node_id="node-a", gateway_url="http://node-a:8081")

    exporter.record_inference(
        engine_name="sglang",
        prompt_tokens=20,
        response_tokens=5,
        roundtrip_ms=250,
    )

    metrics = exporter.render_prometheus(NodeStageMetrics(run_inflight=1))
    assert 'polar_llm_calls_total{node_id="node-a"} 1' in metrics
    assert 'polar_inference_tokens_total{node_id="node-a",engine="sglang",direction="response"} 5' in metrics
    assert 'polar_sessions_total' in metrics


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
    action = next(span for span in spans if span["name"] == "agent.tool.edit")
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
    assert action["parentSpanId"] == root["spanId"]
    action_attributes = {
        item["key"]: next(iter(item["value"].values()))
        for item in action["attributes"]
    }
    assert action_attributes["tool.name"] == "Edit"
    assert "tool.target" not in action_attributes
    assert int(prefill["endTimeUnixNano"]) - int(prefill["startTimeUnixNano"]) == 90_000_000
    attributes = {
        item["key"]: next(iter(item["value"].values()))
        for item in root["attributes"]
    }
    assert attributes["project"] == "polar"
    assert attributes["sample"] == "7"
    assert attributes["global_steps"] == "11"


def test_otlp_action_content_is_opt_in_and_bounded() -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200)

    async def _run() -> None:
        exporter = SessionObservability(
            node_id="node-a",
            gateway_url="http://node-a:8081",
            otlp_endpoint="http://tempo:4318/v1/traces",
            otlp_include_action_content=True,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await exporter.record_session(
                client,
                timing=_timing(),
                session_id="session-1",
                task_id="task-1",
                status=SessionStatus.COMPLETED,
                metadata={},
            )

    asyncio.run(_run())
    spans = captured[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    action = next(span for span in spans if span["name"] == "agent.tool.edit")
    attributes = {
        item["key"]: next(iter(item["value"].values()))
        for item in action["attributes"]
    }
    assert attributes["action.summary"] == "edit kernel.cpp"
    assert attributes["tool.target"] == "kernel.cpp"


def test_rl_insight_target_registration_uses_gateway_authority() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200)

    async def _run() -> None:
        exporter = SessionObservability(
            node_id="node-a",
            gateway_url="http://10.0.0.8:8081",
            inference_url="http://10.0.0.9:8000/v1",
            inference_engine="vllm",
            rl_insight_url="http://insight:18080",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await exporter.register_prometheus_target(client)

    asyncio.run(_run())

    assert requests == [
        {
            "job_name": "polar-gateway",
            "targets": [{"target": "10.0.0.8:8081", "labels": {"node_id": "node-a"}}],
        },
        {
            "job_name": "polar-inference-engine",
            "targets": [{
                "target": "10.0.0.9:8000",
                "labels": {"node_id": "node-a", "engine_type": "vllm"},
            }],
        },
    ]


def test_prometheus_registration_refresh_recovers_after_rl_insight_starts() -> None:
    attempts = 0
    recovered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        recovered.set()
        return httpx.Response(200)

    async def _run() -> None:
        exporter = SessionObservability(
            node_id="node-a",
            gateway_url="http://10.0.0.8:8081",
            rl_insight_url="http://insight:18080",
            registration_refresh_seconds=0.001,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await exporter.register_prometheus_target(client) is False
            task = asyncio.create_task(exporter.refresh_prometheus_registration(client))
            await asyncio.wait_for(recovered.wait(), timeout=1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert exporter._registration_healthy is True

    asyncio.run(_run())
    assert attempts == 3


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
