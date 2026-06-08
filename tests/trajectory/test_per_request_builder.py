from __future__ import annotations

import asyncio

from polar.trajectory.builder.per_request import PerRequestBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def test_per_request_builder_returns_error_for_empty_session() -> None:
    session = CompletionSession(
        session_id="session-1",
        metadata={"group_id": "g1", "policy_version": 7},
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no completions"
    assert trajectory.metadata["builder"] == "per_request"
    assert trajectory.metadata["group_id"] == "g1"
    assert trajectory.metadata["policy_version"] == 7
    assert trajectory.traces == []


def test_per_request_builder_emits_one_trace_per_completion() -> None:
    session = CompletionSession(
        session_id="session-1",
        task_id="task-1",
        model_requested="requested",
        model_used="served",
        api_type="openai_chat",
        metadata={"rollout_step": 3},
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                },
                response={
                    "choices": [
                        {
                            "input_token_ids": [1, 2],
                            "token_ids": [3, 4],
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token_id": 3, "logprob": -0.1},
                                    {"token_id": 4, "logprob": -0.2},
                                ]
                            },
                        }
                    ]
                },
                metadata={"completion_metadata": True},
            )
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert trajectory.metadata["record_count"] == 1
    assert trajectory.metadata["trace_count"] == 1
    assert trajectory.metadata["rollout_step"] == 3
    trace = trajectory.traces[0]
    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [3, 4]
    assert trace.loss_mask == [1, 1]
    assert trace.prompt_messages == [{"role": "user", "content": "Say hi"}]
    assert trace.response_messages == [{"role": "assistant", "content": "Hi"}]
    assert trace.tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert trace.response_logprobs == [-0.1, -0.2]
    # healthy completion -> no integrity flag added (zero behavior change)
    assert "logprob_integrity" not in (trace.metadata or {})


def test_logprob_integrity_flagged_for_misattribution_and_missing() -> None:
    # content[1].token_id (99) != response token_ids[1] (4) -> misattributed; content[0] has no
    # logprob -> would be silently 0.0-filled (fake prob 1.0) -> missing. record_utils must flag both
    # in metadata so the rllm adapter can reject the trace before it trains GRPO.
    session = CompletionSession(
        session_id="s", task_id="t",
        completions=[CompletionRecord(
            completion_id="c1",
            request={"messages": [{"role": "user", "content": "hi"}]},
            response={"choices": [{
                "token_ids": [3, 4],
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token_id": 3}, {"token_id": 99, "logprob": -0.2}]},
            }]},
            metadata={},
        )],
    )
    trace = asyncio.run(PerRequestBuilder().build(session)).traces[0]
    assert trace.metadata["logprob_integrity"] == {"misattributed": 1, "missing": 1}
