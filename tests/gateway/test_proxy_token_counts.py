from __future__ import annotations

import asyncio

from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient


class _Response:
    is_success = True

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _HTTPClient:
    is_closed = False

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def post(self, *args, **kwargs) -> _Response:  # noqa: ANN002, ANN003
        return _Response(self._payload)


def _complete(payload: dict) -> InferenceClient:
    async def run() -> InferenceClient:
        client = InferenceClient("http://engine", SGLangEngine())
        client._client = _HTTPClient(payload)
        await client.completion({"model": "served", "messages": []})
        return client

    return asyncio.run(run())


def test_completion_counts_exact_returned_token_ids() -> None:
    client = _complete({
        "usage": {"prompt_tokens": 11, "completion_tokens": 99},
        "choices": [{"token_ids": [101, 102, 103]}],
    })

    assert client.last_prompt_tokens == 11
    assert client.last_response_tokens == 3


def test_completion_falls_back_to_standard_usage_without_token_ids() -> None:
    client = _complete({
        "usage": {"prompt_tokens": 17, "completion_tokens": 5},
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
    })

    assert client.last_prompt_tokens == 17
    assert client.last_response_tokens == 5


def test_completion_tolerates_missing_or_invalid_usage() -> None:
    client = _complete({"usage": None, "choices": [{"message": {"content": "answer"}}]})

    assert client.last_prompt_tokens == 0
    assert client.last_response_tokens == 0
