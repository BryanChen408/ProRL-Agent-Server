from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import polar.gateway.server as gateway_server
from polar.gateway.completion_metrics import extract_engine_metrics
from polar.gateway.completion_writer import CompletionWriter
from polar.gateway.storage import SessionStore


def _response(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "cmpl-test",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


def test_engine_metrics_are_normalized_and_bounded() -> None:
    metrics = extract_engine_metrics({
        "metrics": {
            "ttft_ms": "12.5",
            "decode_ms": 80,
            "num_cached_tokens": "16",
            "ignored": 1,
        },
        "timings": {
            "ttft_ms": 99,
            "queue_ms": -1,
            "prefix_cache_hit_pct": float("inf"),
        },
    })

    assert metrics == {
        "ttft_ms": 12.5,
        "decode_ms": 80.0,
        "num_cached_tokens": 16,
    }


def test_session_store_exposes_completion_metrics() -> None:
    store = SessionStore()
    store.save_message(
        "sess1",
        {"model": "served", "messages": []},
        _response(10, 4),
        original_request={"model": "requested"},
        model_requested="requested",
        model_used="served",
        api_type="anthropic",
        task_id="task1",
        metadata={"task_id": "task1"},
        latency_ms=200.0,
    )
    store.save_message(
        "sess1",
        {"model": "served", "messages": []},
        _response(20, 6),
        original_request={"model": "requested"},
        model_requested="requested",
        model_used="served",
        api_type="anthropic",
        task_id="task1",
        metadata={"task_id": "task1"},
        latency_ms=300.0,
        streaming=True,
    )

    payload = store.completion_metrics_summary(task_id="task1")

    assert payload["summary"]["request_count"] == 2
    assert payload["summary"]["prompt_tokens"] == 30
    assert payload["summary"]["completion_tokens"] == 10
    assert payload["summary"]["latency_ms_total"] == 500.0
    assert payload["summary"]["completion_tokens_per_second"] == 20.0
    assert len(payload["sessions"]) == 1
    session_metrics = payload["sessions"][0]["completion_metrics"]
    assert session_metrics["request_count"] == 2
    assert session_metrics["latest"]["streaming"] is True
    assert session_metrics["latest"]["sequence"] == 2


def test_session_store_drops_late_completion_after_close() -> None:
    store = SessionStore()
    first_id = store.save_message(
        "sess1",
        {"model": "served", "messages": []},
        _response(10, 4),
        task_id="task1",
    )
    assert first_id is not None

    store.mark_session_closed("sess1", reason="postrun_result")
    assert store.delete_session("sess1") == 1

    late_id = store.save_message(
        "sess1",
        {"model": "served", "messages": []},
        _response(20, 6),
        task_id="task1",
    )

    assert late_id is None
    assert store.load_completion_session("sess1").completions == []
    summary = store.late_completion_summary()
    assert summary["late_message_drop_count"] == 1
    assert summary["recent_closed_sessions"][-1]["session_id"] == "sess1"
    assert summary["recent_closed_sessions"][-1]["reason"] == "postrun_result"


def test_gateway_completion_metrics_endpoint_returns_summary(monkeypatch) -> None:
    store = SessionStore()
    store.save_message(
        "sess1",
        {"model": "served", "messages": []},
        _response(10, 4),
        original_request={"model": "requested"},
        model_requested="requested",
        model_used="served",
        api_type="anthropic",
        task_id="task1",
        metadata={"task_id": "task1"},
        latency_ms=200.0,
    )
    monkeypatch.setattr(
        gateway_server,
        "get_state",
        lambda: SimpleNamespace(storage=store, node=SimpleNamespace(id="node-a")),
    )

    payload = asyncio.run(
        gateway_server.list_completion_metrics(task_id="task1", session_id=None)
    )

    assert payload["node_id"] == "node-a"
    assert payload["summary"]["request_count"] == 1
    assert payload["sessions"][0]["session_id"] == "sess1"


def test_completion_writer_persists_metric_jsonl(tmp_path: Path) -> None:
    async def _run() -> None:
        writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
        await writer.start()
        store = SessionStore(completion_writer=writer)
        store.save_message(
            "sess1",
            {"model": "served", "messages": []},
            _response(10, 4),
            original_request={"model": "requested"},
            model_requested="requested",
            model_used="served",
            api_type="anthropic",
            task_id="task1",
            metadata={"task_id": "task1"},
            latency_ms=250.0,
        )
        await asyncio.sleep(0.2)
        await writer.close()

    asyncio.run(_run())

    metrics_path = tmp_path / "task_task1" / "sessions" / "sess1" / "completion_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess1"
    assert rows[0]["task_id"] == "task1"
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["completion_tokens"] == 4
    assert rows[0]["latency_ms"] == 250.0
