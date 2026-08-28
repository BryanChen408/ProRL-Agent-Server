from __future__ import annotations

import asyncio
from types import SimpleNamespace

from polar.gateway.inflight import InflightGenerationTracker, request_fingerprint
from polar.gateway.node import GatewayNodeManager
from polar.gateway.proxy import UpstreamError
from polar.gateway.storage import SessionStore
from polar.gateway.transform.openai_chat import OpenAIChatTransformer
import polar.gateway.server as gateway_server


def test_request_fingerprint_ignores_stream_transport_fields() -> None:
    base = {
        "model": "served",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "stream": False,
    }
    retry = {
        **base,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    assert request_fingerprint(base) == request_fingerprint(retry)


def test_inflight_tracker_coalesces_duplicate_generation() -> None:
    async def _run() -> None:
        tracker = InflightGenerationTracker()
        calls = 0
        release = asyncio.Event()

        async def factory() -> dict:
            nonlocal calls
            calls += 1
            await release.wait()
            return {"choices": [{"message": {"content": "ok"}}]}

        request = {"model": "served", "messages": [{"role": "user", "content": "hello"}]}
        first = asyncio.create_task(tracker.run("sess1", request, factory))
        await asyncio.sleep(0)
        second = asyncio.create_task(tracker.run("sess1", request, factory))
        await asyncio.sleep(0)
        release.set()

        first_result, second_result = await asyncio.gather(first, second)

        assert calls == 1
        assert first_result.response == second_result.response
        assert {first_result.should_save, second_result.should_save} == {True, False}
        assert {first_result.coalesced, second_result.coalesced} == {True, False}
        status = tracker.status()
        assert status["coalesced_request_count"] == 1
        assert status["active"] == 0

    asyncio.run(_run())


def test_inflight_tracker_session_close_cancels_active_generation() -> None:
    async def _run() -> None:
        tracker = InflightGenerationTracker()

        async def factory() -> dict:
            await asyncio.sleep(10)
            return {"choices": []}

        request = {"model": "served", "messages": [{"role": "user", "content": "hello"}]}
        task = asyncio.create_task(tracker.run("sess1", request, factory))
        await asyncio.sleep(0)

        closed = await tracker.close_session("sess1", reason="postrun_result")

        assert closed == 1
        try:
            await task
        except UpstreamError as exc:
            assert "session closed" in str(exc)
        else:
            raise AssertionError("expected active generation to be cancelled")
        status = tracker.status()
        assert status["closed_generation_count"] == 1

    asyncio.run(_run())


def test_inflight_tracker_session_close_fences_future_generation() -> None:
    """DELETE-before-register cannot leave an old request parked until resume."""

    async def _run() -> None:
        tracker = InflightGenerationTracker()
        calls = 0

        async def factory() -> dict:
            nonlocal calls
            calls += 1
            return {"choices": []}

        closed = await tracker.close_session("sess1", reason="policy_cutoff")
        assert closed == 0

        try:
            await tracker.run(
                "sess1",
                {"model": "served", "messages": []},
                factory,
            )
        except UpstreamError as exc:
            assert "session closed" in str(exc)
            assert "policy_cutoff" in str(exc)
        else:
            raise AssertionError("expected future generation to be fenced")

        assert calls == 0
        assert tracker.status()["closed_sessions"] == 1

    asyncio.run(_run())


def test_non_streaming_handler_coalesces_duplicate_request_and_saves_once(monkeypatch) -> None:
    async def _run() -> None:
        tracker = InflightGenerationTracker()
        storage = SessionStore()
        release = asyncio.Event()
        calls = 0

        class FakeInference:
            base_url = "http://engine"

            async def completion(
                self,
                request: dict,
                *,
                trace_headers: dict[str, str] | None = None,
            ) -> dict:
                nonlocal calls
                del trace_headers
                calls += 1
                await release.wait()
                return {
                    "id": "cmpl-test",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                }

        monkeypatch.setattr(
            gateway_server,
            "get_state",
            lambda: SimpleNamespace(
                inference=FakeInference(),
                inflight=tracker,
                storage=storage,
            ),
        )

        openai_request = {
            "model": "served",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        original_request = {
            "model": "requested",
            "messages": [{"role": "user", "content": "hello"}],
        }
        first = asyncio.create_task(
            gateway_server._handle_non_streaming(
                gateway_server.APIType.OPENAI_CHAT,
                OpenAIChatTransformer(),
                openai_request,
                original_request,
                "sess1",
                original_model="requested",
                session_info=None,
            )
        )
        await asyncio.sleep(0)
        second = asyncio.create_task(
            gateway_server._handle_non_streaming(
                gateway_server.APIType.OPENAI_CHAT,
                OpenAIChatTransformer(),
                dict(openai_request),
                dict(original_request),
                "sess1",
                original_model="requested",
                session_info=None,
            )
        )
        await asyncio.sleep(0)
        release.set()

        responses = await asyncio.gather(first, second)

        assert calls == 1
        assert [response.status_code for response in responses] == [200, 200]
        session = storage.load_completion_session("sess1")
        assert len(session.completions) == 1
        assert tracker.status()["coalesced_request_count"] == 1

    asyncio.run(_run())


def test_non_streaming_handler_rejects_closed_session_without_upstream(monkeypatch) -> None:
    async def _run() -> None:
        tracker = InflightGenerationTracker()
        storage = SessionStore()
        storage.mark_session_closed("sess1", reason="postrun_result")
        calls = 0

        class FakeInference:
            async def completion(self, request: dict) -> dict:
                nonlocal calls
                calls += 1
                return {"choices": []}

        monkeypatch.setattr(
            gateway_server,
            "get_state",
            lambda: SimpleNamespace(
                inference=FakeInference(),
                inflight=tracker,
                storage=storage,
            ),
        )

        response = await gateway_server._handle_non_streaming(
            gateway_server.APIType.OPENAI_CHAT,
            OpenAIChatTransformer(),
            {"model": "served", "messages": []},
            {"model": "requested", "messages": []},
            "sess1",
            original_model="requested",
            session_info=None,
        )

        assert response.status_code == 502
        assert calls == 0

    asyncio.run(_run())


def test_inflight_status_endpoint_returns_tracker_status(monkeypatch) -> None:
    tracker = InflightGenerationTracker()
    monkeypatch.setattr(
        gateway_server,
        "get_state",
        lambda: SimpleNamespace(inflight=tracker),
    )

    payload = asyncio.run(gateway_server.inference_inflight_status())

    assert payload["active"] == 0
    assert payload["coalesced_request_count"] == 0


def test_node_manager_close_inflight_generations_is_best_effort() -> None:
    async def _run() -> None:
        class FakeInflight:
            def __init__(self):
                self.calls = []

            async def close_session(self, session_id: str, *, reason: str | None = None) -> int:
                self.calls.append((session_id, reason))
                return 1

        inflight = FakeInflight()
        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        manager.inflight = inflight

        await manager._close_inflight_generations("sess1", reason="postrun_result")

        assert inflight.calls == [("sess1", "postrun_result")]

    asyncio.run(_run())
