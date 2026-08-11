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
                timestamp="2026-01-01T00:00:00+00:00",
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


def test_per_request_builder_filters_non_trainable_completions() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _normal_record("keep-1", [1], [10]),
            _side_read_record("side-1"),
            _truncated_record("truncated-1"),
            _empty_record("empty-1"),
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert len(trajectory.traces) == 1
    assert "source_completion_ids" not in trajectory.traces[0].metadata
    assert trajectory.metadata["record_count"] == 4
    assert trajectory.metadata["trace_count"] == 1
    completion_filter = trajectory.metadata["completion_filter"]
    assert completion_filter["input_completions"] == 4
    assert completion_filter["kept_completions"] == 1
    assert completion_filter["excluded_completions"] == 3
    assert completion_filter["excluded_reasons"] == {
        "empty_completion": 1,
        "non_agent_side_completion": 1,
        "persisted_truncated_completion": 1,
    }
    assert set(completion_filter["excluded_completion_ids"]) == {
        "side-1",
        "truncated-1",
        "empty-1",
    }
    assert {
        (item["completion_id"], item["reason"])
        for item in completion_filter["excluded"]
    } == {
        ("side-1", "non_agent_side_completion"),
        ("truncated-1", "persisted_truncated_completion"),
        ("empty-1", "empty_completion"),
    }


def test_per_request_builder_returns_error_when_filter_removes_everything() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _side_read_record("side-1"),
            _empty_record("empty-1"),
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no trainable completions after completion filter"
    assert trajectory.traces == []
    assert set(trajectory.metadata["completion_filter"]["excluded_completion_ids"]) == {
        "side-1",
        "empty-1",
    }


def test_per_request_builder_keeps_normal_single_user_request_with_tools() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            CompletionRecord(
                completion_id="keep",
                original_request={
                    "messages": [{"role": "user", "content": "# Triton Ascend 基础知识参考手册"}],
                    "tools": [{"type": "function", "function": {"name": "Read"}}],
                },
                request={
                    "messages": [{"role": "user", "content": "# Triton Ascend 基础知识参考手册"}],
                    "tools": [{"type": "function", "function": {"name": "Read"}}],
                },
                response={
                    "choices": [
                        {
                            "input_token_ids": [1],
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": {"content": [{"token_id": 10, "logprob": -0.1}]},
                        }
                    ]
                },
            )
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert len(trajectory.traces) == 1
    assert trajectory.metadata["completion_filter"]["excluded_completions"] == 0


def test_qwen_reasoning_tokens_are_loss_masked_without_changing_token_alignment() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={"system": "harness", "messages": [{"role": "user", "content": "Write code"}]},
                response={
                    "choices": [
                        {
                            "input_token_ids": [1, 2],
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Need a plan.",
                                "content": "Use the tool.",
                            },
                            "finish_reason": "tool_calls",
                            "logprobs": {
                                "content": [
                                    {"token": "Need", "token_id": 10, "logprob": -0.1},
                                    {"token": "</think>", "token_id": 11, "logprob": -0.2},
                                    {"token": "\n\n", "token_id": 12, "logprob": -0.3},
                                    {"token": "Use", "token_id": 13, "logprob": -0.4},
                                ]
                            },
                        }
                    ]
                },
                metadata={},
            )
        ],
    )

    trace = asyncio.run(PerRequestBuilder().build(session)).traces[0]

    assert trace.response_ids == [10, 11, 12, 13]
    assert trace.response_logprobs == [-0.1, -0.2, -0.3, -0.4]
    assert trace.loss_mask == [0, 0, 1, 1]
    assert trace.metadata["reasoning_loss_mask"] == {
        "masked_tokens": 2,
        "end_token_index": 1,
    }


def test_logprob_integrity_flagged_for_misattribution_and_missing() -> None:
    # content[1].token_id (99) != response token_ids[1] (4) -> misattributed; content[0] has no
    # logprob -> would be silently 0.0-filled (fake prob 1.0) -> missing. record_utils must flag both
    # in metadata so the rllm adapter can reject the trace before it trains GRPO.
    session = CompletionSession(
        session_id="s", task_id="t",
        completions=[CompletionRecord(
            completion_id="c1",
            request={"system": "harness", "messages": [{"role": "user", "content": "hi"}]},
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


def _normal_record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=f"2026-01-01T00:00:{len(completion_id):02d}+00:00",
        # `system` marks agent-side traffic so record_filters keeps the record.
        request={"system": "harness", "messages": [{"role": "user", "content": completion_id}]},
        response={
            "choices": [
                {
                    "input_token_ids": prompt_ids,
                    "message": {"role": "assistant", "content": completion_id},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": -0.1}
                            for token_id in response_ids
                        ]
                    },
                }
            ]
        },
    )


def _side_read_record(completion_id: str) -> CompletionRecord:
    # A bare request the harness emitted outside the agent loop, carrying the body of
    # a file the agent just Read. Dropped on shape (lone user message, no system /
    # tools / SDK fields), so any payload is covered — not just one known document.
    request = {
        "messages": [
            {
                "role": "user",
                "content": "# Copyright (c) 2025 Huawei\nfunction(ascendc_compile_kernel)\nendfunction()\n",
            }
        ],
    }
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=f"2026-01-01T00:00:{len(completion_id):02d}+00:00",
        original_request=request,
        request=request,
        response={
            "choices": [
                {
                    "input_token_ids": [1],
                    "message": {"role": "assistant", "content": "doc review"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"token_id": 10, "logprob": -0.1}]},
                }
            ]
        },
    )


def _truncated_record(completion_id: str) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=f"2026-01-01T00:00:{len(completion_id):02d}+00:00",
        request={"messages": [{"role": "user", "content": completion_id}]},
        response={"id": "r1", "__truncated": True},
    )


def _empty_record(completion_id: str) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=f"2026-01-01T00:00:{len(completion_id):02d}+00:00",
        request={"messages": [{"role": "user", "content": completion_id}]},
        response={"id": "r1", "choices": []},
    )


def _length_record(completion_id: str, prompt_ids: list[int], response_ids: list[int]) -> CompletionRecord:
    """finish_reason=length 的截断 completion(思考被 max_tokens 掐断,content 为空)。"""
    return CompletionRecord(
        completion_id=completion_id,
        timestamp="2026-01-01T00:00:00+00:00",
        request={"system": "harness", "messages": [{"role": "user", "content": completion_id}]},
        response={
            "choices": [
                {
                    "input_token_ids": prompt_ids,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "没写完的 deliberation……",
                    },
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": -0.1}
                            for token_id in response_ids
                        ]
                    },
                }
            ]
        },
    )


def test_length_truncated_completion_is_masked_out(monkeypatch) -> None:
    # 截断段(finish_reason=length)默认整段 loss_mask=0:不进梯度,防止训练奖励
    # 「写不完的思考」(带正奖励的截断段毒性最大)。
    monkeypatch.delenv("POLAR_MASK_TRUNCATED", raising=False)
    monkeypatch.setenv("POLAR_MASK_REASONING", "0")  # 隔离 reasoning mask 的干扰
    session = CompletionSession(
        session_id="session-1",
        completions=[_length_record("len-1", [1, 2], [10, 11, 12])],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    trace = trajectory.traces[0]
    assert trace.finish_reason == "length"
    assert trace.loss_mask == [0, 0, 0]
    note = (trace.metadata or {}).get("reasoning_loss_mask") or {}
    assert note.get("reason") == "finish_reason_length"
    assert note.get("truncated_tokens") == 3


def test_length_truncated_mask_gate_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_MASK_TRUNCATED", "0")
    monkeypatch.setenv("POLAR_MASK_REASONING", "0")
    session = CompletionSession(
        session_id="session-1",
        completions=[_length_record("len-1", [1, 2], [10, 11, 12])],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.traces[0].loss_mask == [1, 1, 1]
