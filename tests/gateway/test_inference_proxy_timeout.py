from __future__ import annotations

import asyncio

import pytest

from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient


def test_inference_client_uses_default_liveness_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS", raising=False)

    client = InferenceClient("http://127.0.0.1:30000", SGLangEngine())

    assert client.generation_status()["request_timeout_seconds"] == 900.0


def test_inference_client_reads_liveness_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS", "7200")

    client = InferenceClient("http://127.0.0.1:30000", SGLangEngine())

    assert client.generation_status()["request_timeout_seconds"] == 7200.0


def test_inference_client_allows_explicit_liveness_timeout() -> None:
    client = InferenceClient(
        "http://127.0.0.1:30000",
        SGLangEngine(),
        liveness_timeout_seconds=123.0,
    )

    assert client.generation_status()["request_timeout_seconds"] == 123.0


def test_inference_client_rejects_invalid_liveness_timeout_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="positive number"):
        InferenceClient("http://127.0.0.1:30000", SGLangEngine())


def test_pause_timeout_keeps_admission_paused_and_reports_not_drained() -> None:
    async def run() -> None:
        client = InferenceClient("http://127.0.0.1:30000", SGLangEngine())
        client._inflight_generations = 1

        status = await client.pause_generation(timeout_seconds=0.001)

        assert status["paused"] is True
        assert status["drained"] is False
        assert status["timed_out"] is True
        assert status["inflight"] == 1

        blocked = asyncio.create_task(client._acquire_generation_slot())
        await asyncio.sleep(0)
        assert not blocked.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked

    asyncio.run(run())


def test_pause_reports_drained_when_no_generation_is_active() -> None:
    async def run() -> None:
        client = InferenceClient("http://127.0.0.1:30000", SGLangEngine())

        status = await client.pause_generation(timeout_seconds=0.001)

        assert status["paused"] is True
        assert status["drained"] is True
        assert status["timed_out"] is False
        assert status["inflight"] == 0

    asyncio.run(run())


def test_pause_can_close_admission_without_waiting_for_drain() -> None:
    async def run() -> None:
        client = InferenceClient("http://127.0.0.1:30000", SGLangEngine())
        client._inflight_generations = 1

        status = await client.pause_generation(
            timeout_seconds=300.0,
            wait_for_drain=False,
        )

        assert status["paused"] is True
        assert status["drained"] is False
        assert status["timed_out"] is False
        assert status["wait_for_drain"] is False
        assert status["inflight"] == 1

    asyncio.run(run())
