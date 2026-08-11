from __future__ import annotations

import asyncio

from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession

EOT = 99


def _record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    finish_reason: str = "stop",
    prompt_messages: list[dict] | None = None,
    metadata: dict | None = None,
) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        # `system` marks this as agent-side traffic; without it the record has the
        # bare shape record_filters drops as a non-agent-side completion.
        request={
            "system": "harness",
            "messages": prompt_messages or [{"role": "user", "content": completion_id}],
        },
        response={
            "choices": [
                {
                    "input_token_ids": list(prompt_ids),
                    "message": {"role": "assistant", "content": completion_id},
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "content": [
                            {
                                "token": f"t{token_id}",
                                "token_id": token_id,
                                "logprob": -0.01 * (idx + 1),
                                "bytes": [],
                            }
                            for idx, token_id in enumerate(response_ids)
                        ]
                    },
                }
            ]
        },
        metadata=metadata or {},
    )


def _build(records: list[CompletionRecord], *, eot: int | None = EOT):
    return asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=eot).build(
            CompletionSession(session_id="session-1", completions=records)
        )
    )


def test_linear_main_agent_tool_chain_merges_with_interstitial_masked() -> None:
    records = [
        _record("00-main1", [1, 2], [10, EOT]),
        _record(
            "01-main2",
            [1, 2, 10, EOT, 50, 51],
            [20, EOT],
            prompt_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "main1"},
                {"role": "tool", "content": "tool-result"},
            ],
        ),
    ]

    trajectory = _build(records)

    assert len(trajectory.traces) == 1
    trace = trajectory.traces[0]
    assert trace.response_ids == [10, EOT, 50, 51, 20, EOT]
    assert trace.loss_mask == [1, 1, 0, 0, 1, 1]
    assert trace.response_logprobs == [-0.01, -0.02, 0.0, 0.0, -0.01, -0.02]
    assert trace.metadata["source_completion_ids"] == ["00-main1", "01-main2"]
    assert trace.metadata["kept_completion_count"] == 2
    assert trajectory.metadata["reconstruction_stats"]["completions_preserved"] == 2
    assert trajectory.metadata["reconstruction_stats"]["completions_dropped"] == 0


def test_prefix_merge_filters_side_truncated_and_empty_completions_before_grouping() -> None:
    records = [
        _record("00-main1", [1, 2], [10, EOT]),
        _side_read_record("01-side"),
        _truncated_record("02-truncated"),
        _empty_record("03-empty"),
        _record(
            "04-main2",
            [1, 2, 10, EOT, 50],
            [20, EOT],
            prompt_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "main1"},
                {"role": "tool", "content": "tool-result"},
            ],
        ),
    ]

    trajectory = _build(records)

    assert trajectory.status == "COMPLETED"
    assert len(trajectory.traces) == 1
    assert trajectory.traces[0].metadata["source_completion_ids"] == ["00-main1", "04-main2"]
    stats = trajectory.metadata["reconstruction_stats"]
    assert stats["raw_completions_total"] == 5
    assert stats["completions_total"] == 2
    assert stats["completions_preserved"] == 2
    assert stats["completions_dropped"] == 0
    completion_filter = trajectory.metadata["completion_filter"]
    assert completion_filter["input_completions"] == 5
    assert completion_filter["kept_completions"] == 2
    assert completion_filter["excluded_completions"] == 3
    assert completion_filter["excluded_reasons"] == {
        "empty_completion": 1,
        "non_agent_side_completion": 1,
        "persisted_truncated_completion": 1,
    }
    assert set(completion_filter["excluded_completion_ids"]) == {
        "01-side",
        "02-truncated",
        "03-empty",
    }
    assert {
        (item["completion_id"], item["reason"])
        for item in completion_filter["excluded"]
    } == {
        ("01-side", "non_agent_side_completion"),
        ("02-truncated", "persisted_truncated_completion"),
        ("03-empty", "empty_completion"),
    }


def test_prefix_merge_returns_error_when_filter_removes_everything() -> None:
    trajectory = _build([_side_read_record("00-side"), _empty_record("01-empty")])

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no trainable completions after completion filter"
    assert trajectory.traces == []
    assert set(trajectory.metadata["completion_filter"]["excluded_completion_ids"]) == {
        "00-side",
        "01-empty",
    }


def test_prefix_merge_preserves_reasoning_loss_mask() -> None:
    records = [
        CompletionRecord(
            completion_id="00-main1",
            request={"system": "harness", "messages": [{"role": "user", "content": "q"}]},
            response={
                "choices": [
                    {
                        "input_token_ids": [1, 2],
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Need a plan.",
                            "content": "main1",
                        },
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "Need", "token_id": 10, "logprob": -0.01},
                                {"token": "</think>", "token_id": 11, "logprob": -0.02},
                                {"token": "main1", "token_id": 12, "logprob": -0.03},
                                {"token": f"t{EOT}", "token_id": EOT, "logprob": -0.04},
                            ]
                        },
                    }
                ]
            },
        ),
        _record(
            "01-main2",
            [1, 2, 10, 11, 12, EOT, 50],
            [20, EOT],
            prompt_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "main1"},
                {"role": "tool", "content": "tool-result"},
            ],
        ),
    ]

    trajectory = _build(records)
    trace = trajectory.traces[0]

    assert trace.response_ids == [10, 11, 12, EOT, 50, 20, EOT]
    assert trace.loss_mask == [0, 0, 1, 1, 0, 1, 1]
    assert trace.response_logprobs == [-0.01, -0.02, -0.03, -0.04, 0.0, -0.01, -0.02]


def test_interleaved_main_and_subagent_prefixes_form_separate_chains() -> None:
    records = [
        _record("00-main1", [1], [10, EOT], metadata={"agent": "main"}),
        _record("01-sub1", [7], [70, EOT], metadata={"agent": "sub"}),
        _record("02-main2", [1, 10, EOT, 50], [20, EOT], metadata={"agent": "main"}),
        _record("03-sub2", [7, 70, EOT, 80], [90, EOT], metadata={"agent": "sub"}),
    ]

    trajectory = _build(records)

    assert len(trajectory.traces) == 2
    source_ids = [trace.metadata["source_completion_ids"] for trace in trajectory.traces]
    assert ["00-main1", "02-main2"] in source_ids
    assert ["01-sub1", "03-sub2"] in source_ids
    assert trajectory.metadata["reconstruction_stats"]["completions_preserved"] == 4
    assert trajectory.metadata["reconstruction_stats"]["completions_dropped"] == 0


def test_compact_or_prefix_break_is_preserved_as_new_trace() -> None:
    records = [
        _record("00-main1", [1, 2], [10, EOT]),
        _record("01-compact-main2", [1, 2, 777, 778], [20, EOT]),
    ]

    trajectory = _build(records)

    assert len(trajectory.traces) == 2
    assert trajectory.traces[0].metadata["source_completion_ids"] == ["00-main1"]
    assert trajectory.traces[0].metadata["break_reason"] == "interstitial_split_failed"
    assert trajectory.traces[1].metadata["source_completion_ids"] == ["01-compact-main2"]
    stats = trajectory.metadata["reconstruction_stats"]
    assert stats["break_reasons"] == {"interstitial_split_failed": 1}
    assert stats["completions_preserved"] == 2
    assert stats["completions_dropped"] == 0


def test_eot_unavailable_preserves_every_completion_as_safe_segments() -> None:
    records = [
        _record("00-main1", [1], [10], finish_reason="length"),
        _record("01-main2", [1, 10, 50], [20], finish_reason="length"),
    ]

    trajectory = _build(records, eot=None)

    assert len(trajectory.traces) == 2
    assert trajectory.traces[0].metadata["source_completion_ids"] == ["00-main1"]
    assert trajectory.traces[0].metadata["break_reason"] == "eot_unavailable"
    assert trajectory.traces[1].metadata["source_completion_ids"] == ["01-main2"]
    stats = trajectory.metadata["reconstruction_stats"]
    assert stats["break_reasons"] == {"eot_unavailable": 1}
    assert stats["completions_preserved"] == 2
    assert stats["completions_dropped"] == 0


def test_longest_matching_prefix_wins_for_grouping_collision() -> None:
    records = [
        _record("00-b1", [1, 2], [20, EOT]),
        _record("01-a1", [1], [10, EOT]),
        _record("02-b2", [1, 2, 20, EOT, 50], [21, EOT]),
    ]

    trajectory = _build(records)

    source_ids = [trace.metadata["source_completion_ids"] for trace in trajectory.traces]
    assert ["00-b1", "02-b2"] in source_ids
    assert ["01-a1"] in source_ids
    merged = next(
        trace for trace in trajectory.traces
        if trace.metadata["source_completion_ids"] == ["00-b1", "02-b2"]
    )
    assert merged.loss_mask == [1, 1, 0, 1, 1]


def _side_read_record(completion_id: str) -> CompletionRecord:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "# Triton Ascend 基础知识参考手册\n\n本文档汇集 Triton Ascend 编程的基础知识。",
            }
        ],
    }
    return CompletionRecord(
        completion_id=completion_id,
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
        request={"messages": [{"role": "user", "content": completion_id}]},
        response={"id": "r1", "__truncated": True},
    )


def _empty_record(completion_id: str) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        request={"messages": [{"role": "user", "content": completion_id}]},
        response={"id": "r1", "choices": []},
    )


def test_prefix_merge_masks_length_segment_but_keeps_normal_segments(monkeypatch) -> None:
    # 链内混排:正常段继续训练,截断段(finish_reason=length)整段置 0。
    monkeypatch.delenv("POLAR_MASK_TRUNCATED", raising=False)
    monkeypatch.setenv("POLAR_MASK_REASONING", "0")
    records = [
        _record("00-main1", [1, 2], [10, EOT], finish_reason="tool_calls"),
        _record(
            "01-main2-length",
            [1, 2, 10, EOT, 50, 51],
            [20, 21],  # 截断段:无 EOT,finish_reason=length
            finish_reason="length",
            prompt_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "main1"},
                {"role": "tool", "content": "tool-result"},
            ],
        ),
    ]

    trajectory = _build(records)

    assert len(trajectory.traces) == 1
    trace = trajectory.traces[0]
    assert trace.response_ids == [10, EOT, 50, 51, 20, 21]
    assert trace.loss_mask == [1, 1, 0, 0, 0, 0]
    assert trace.metadata["kept_completion_count"] == 2


def test_prefix_merge_length_mask_gate_off_restores_old_behavior(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_MASK_TRUNCATED", "0")
    monkeypatch.setenv("POLAR_MASK_REASONING", "0")
    records = [
        _record("00-main1", [1, 2], [10, EOT], finish_reason="tool_calls"),
        _record(
            "01-main2-length",
            [1, 2, 10, EOT, 50, 51],
            [20, 21],
            finish_reason="length",
            prompt_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "main1"},
                {"role": "tool", "content": "tool-result"},
            ],
        ),
    ]

    trajectory = _build(records)

    trace = trajectory.traces[0]
    assert trace.loss_mask == [1, 1, 0, 0, 1, 1]  # 截断段照常训练(旧行为)
