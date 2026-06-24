from __future__ import annotations

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
