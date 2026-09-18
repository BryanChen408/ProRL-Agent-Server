"""Check the effective wire budget survives capture and masks only the delimiter."""

import asyncio
import json

import httpx
import pytest

from polar.gateway.engine import VLLMEngine
from polar.gateway.proxy import InferenceClient
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord


@pytest.mark.parametrize("forced_indices", [None, [], [1]])
def test_budget_capture_and_training_mask(monkeypatch, forced_indices):
    monkeypatch.setenv("POLAR_THINKING_TOKEN_BUDGET", "2")
    monkeypatch.setenv("POLAR_MASK_REASONING", "0")
    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "token_ids": [3, 4, 5],
                "finish_reason": "tool_calls",
                "message": {"reasoning": "thought", "tool_calls": [{"id": "call_1"}]},
                "logprobs": {
                    "content": [
                        {"token": "thought", "logprob": -0.1},
                        {"token": "</think>", "logprob": -8.0},
                        {"token": "<tool_call>", "logprob": -0.3},
                    ]
                },
            }
        ],
    }
    if forced_indices is not None:
        response["choices"][0]["thinking_budget_forced_token_indices"] = forced_indices

    def handler(request):
        wire = json.loads(request.content)
        assert wire["thinking_token_budget"] == 2
        assert wire["max_tokens"] == 8
        return httpx.Response(200, json=response)

    async def run():
        client = InferenceClient("http://test", VLLMEngine())
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as http:
            client._client = http
            return await client.completion({"messages": [], "max_tokens": 8})

    captured = asyncio.run(run())
    record = CompletionRecord(
        completion_id="budget",
        timestamp="2026-09-18T00:00:00Z",
        request={"messages": []},
        response=captured,
    )
    trace = build_trace_from_completion(record)
    assert trace.loss_mask == ([1, 1, 1] if forced_indices == [] else [1, 0, 1])
    assert trace.response_ids == [3, 4, 5]
    assert trace.response_logprobs == [-0.1, -8.0, -0.3]
    expected = {
        "budget": 2,
        "end_token_indices": [1] if forced_indices is None else forced_indices,
    }
    if forced_indices is not None:
        expected["source"] = "engine"
    assert trace.metadata["thinking_budget_loss_mask"] == expected
    assert trace.finish_reason == "tool_calls"

    record.response.pop("_polar_thinking_token_budget")
    record.response["choices"][0].pop("thinking_budget_forced_token_indices", None)
    assert build_trace_from_completion(record).loss_mask == [1, 1, 1]
    record.request["thinking_token_budget"] = 2
    record.response["choices"][0]["finish_reason"] = "length"
    monkeypatch.setenv("POLAR_MASK_TRUNCATED", "1")
    assert build_trace_from_completion(record).loss_mask == [0, 0, 0]


@pytest.mark.parametrize("indices", [[True], [-1], [3], [0], [1, 1], "1", {}])
def test_bad_forcing_attribution_rejects_training_trace(indices):
    record = CompletionRecord(
        completion_id="bad",
        timestamp="2026-09-18T00:00:00Z",
        request={},
        response={
            "_polar_thinking_token_budget": 2,
            "prompt_token_ids": [1],
            "choices": [
                {
                    "token_ids": [3, 4, 5],
                    "message": {"content": "body"},
                    "thinking_budget_forced_token_indices": indices,
                    "logprobs": {
                        "content": [
                            {"token": "thought", "logprob": -0.1},
                            {"token": "</think>", "logprob": -0.2},
                            {"token": "body", "logprob": -0.3},
                        ]
                    },
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="forced token attribution"):
        build_trace_from_completion(record)
