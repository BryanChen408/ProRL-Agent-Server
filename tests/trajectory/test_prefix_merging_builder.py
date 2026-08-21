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


def test_prefix_merge_preserves_reasoning_loss_mask(monkeypatch) -> None:
    # 本用例专测「掩零」行为,显式钉回旧默认;新默认(训 CoT)见下一条用例。
    monkeypatch.setenv("POLAR_MASK_REASONING", "1")
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


# ---------------------------------------------------------------------------
# upstream 传输抖动:可恢复,不再整条作废(混权重仍然作废)
# ---------------------------------------------------------------------------


def _one_turn_session(metadata: dict) -> CompletionSession:
    return CompletionSession(
        session_id="s-blip",
        metadata=metadata,
        completions=[
            CompletionRecord(
                completion_id="c0",
                request={"system": "harness", "messages": [{"role": "user", "content": "go"}]},
                response={"choices": [{
                    "input_token_ids": [1, 2],
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [
                        {"token": "t10", "token_id": 10, "logprob": -0.1, "bytes": []},
                        {"token": "eot", "token_id": 99, "logprob": -0.1, "bytes": []},
                    ]},
                }]},
                metadata={},
            )
        ],
    )


def test_upstream_blip_stays_trainable():
    """传输抖动可恢复:轨迹带真实数据留下,只在 metadata 里留痕。

    实测 092443:97% 的抖动之后 session 还跑了中位 30 轮并自然收尾,judge 正常评分;
    而失败请求不落盘,所以已记录的每一轮都是完整的。
    """
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"transport": 1}})
        )
    )
    assert traj.status == "COMPLETED"
    assert traj.error is None
    assert traj.metadata["upstream_failures"] == {"transport": 1}   # 可观测性不丢
    assert traj.traces and any(traj.traces[0].loss_mask)


def test_terminal_blip_still_discarded():
    """末尾抖断:会话就此停止,judge 评的是半成品 -> 仍然整条作废(原设计要治的假阴性)。

    blip 时已存 1 条,最终也是 1 条 -> completion_count <= last_at -> 判为截断。
    """
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"transport": 1},
                               "upstream_failures_last_at": 1})
        )
    )
    assert traj.status == "ERROR"
    assert "truncated the session" in (traj.error or "")


def test_recovered_blip_is_trainable():
    """中途抖动后恢复:blip 时已存 0 条,最终 1 条 -> 有后续产出 -> 可训练。"""
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"transport": 1},
                               "upstream_failures_last_at": 0})
        )
    )
    assert traj.status == "COMPLETED"
    assert traj.error is None
    assert traj.metadata["upstream_failures"] == {"transport": 1}


def test_terminal_context_overflow_is_trainable():
    """末尾 4xx(上下文撑爆):引擎在生成前就拒了请求 -> 没有半句话被掐断 -> 该学。

    实测 165820:3 条被丢的 session 全是 prompt 撞 249856 上限的 http_400,每条都有真实
    judge 分(0.25/0.29/0.30)与 5w-8.6w 可训练 token,且每次都是组内最大的样本 ——
    撑爆天然挑中最长的 episode,扔掉它们会把训练分布推向短会话。
    """
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"http_400": 1},
                               "upstream_failures_last_at": 1,
                               "upstream_failures_last_kind": "http_400"})
        )
    )
    assert traj.status == "COMPLETED"
    assert traj.error is None
    assert traj.metadata["upstream_failures"] == {"http_400": 1}   # 可观测性不丢
    assert traj.traces and any(traj.traces[0].loss_mask)


def test_terminal_overflow_without_last_kind_is_trainable():
    """老数据无 last_kind:失效种类全是 4xx 时同样豁免(键集足以判定)。"""
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"http_400": 2},
                               "upstream_failures_last_at": 1})
        )
    )
    assert traj.status == "COMPLETED"


def test_mixed_blip_ending_on_transport_still_discarded():
    """混合失效以 transport 收尾:最后那次可能留下半句话 -> 仍作废。

    092443 里 150 条带 upstream_failures 的 session 有 16 条同时含两种 kind,
    只看键集分不清是哪种收尾,last_kind 就是为这种情形加的。
    """
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"http_400": 1, "transport": 1},
                               "upstream_failures_last_at": 1,
                               "upstream_failures_last_kind": "transport"})
        )
    )
    assert traj.status == "ERROR"
    assert "truncated the session" in (traj.error or "")


def test_mixed_blip_without_last_kind_stays_conservative():
    """老数据 + 混合种类:分不清收尾者 -> 保守判为截断(不放宽)。"""
    traj = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(
            _one_turn_session({"upstream_failures": {"http_400": 1, "transport": 1},
                               "upstream_failures_last_at": 1})
        )
    )
    assert traj.status == "ERROR"


def test_policy_version_span_still_discarded():
    """混权重:策略身份不明,重要性比分母不唯一 -> 仍然整条作废。"""
    sess = _one_turn_session({})
    sess.completions[0].metadata = {"policy_version": 0}
    extra = CompletionRecord(
        completion_id="c1",
        request={"system": "harness", "messages": [{"role": "user", "content": "go"}]},
        response={"choices": [{
            "input_token_ids": [1, 2, 10, 99],
            "message": {"role": "assistant", "content": "more"},
            "finish_reason": "stop",
            "logprobs": {"content": [
                {"token": "t20", "token_id": 20, "logprob": -0.1, "bytes": []},
                {"token": "eot", "token_id": 99, "logprob": -0.1, "bytes": []},
            ]},
        }]},
        metadata={"policy_version": 1},
    )
    sess.completions.append(extra)
    traj = asyncio.run(PrefixMergingBuilder(end_of_turn_token_id=99).build(sess))
    assert traj.status == "ERROR"
    assert "policy_version span" in (traj.error or "")


def _trunc_record(completion_id, prompt_ids, response_ids, *, content=""):
    """空截断轮 fixture:finish=length + content 空 + 无 tool_calls(CLI 会丢弃的那类)。"""
    r = _record(completion_id, prompt_ids, response_ids, finish_reason="length")
    r.response["choices"][0]["message"]["content"] = content
    return r


# 生成头结构:im_start=1(prompt 首 token),gen-prompt = [1, 50, 51]
def test_empty_truncation_rejoins_chain_and_response_excluded():
    """空截断 + 后续带 limit 消息:不拆链、截断响应不入流、interstitial 掩零。

    复现实测场景(152826 op-3):C2 思考顶穿被 CLI 丢弃,C3 的 prompt 在 C2 的
    生成头位置变成 user 消息。旧行为:C2 自成 1-completion 全零孤儿链。
    新行为:C2 并入主链、响应不入流、limit 消息作为 interstitial 恒掩零。
    """
    p1 = [1, 10, 11, 99, 1, 50, 51]
    p2 = [1, 10, 11, 99, 1, 50, 51, 20, 99, 1, 60, 99, 1, 50, 51]
    p3 = [1, 10, 11, 99, 1, 50, 51, 20, 99, 1, 60, 99, 1, 70, 71, 99, 1, 50, 51]
    traj = _build([
        _record("c1", p1, [20, 99]),
        _trunc_record("c2", p2, [30, 31, 32]),
        _record("c3", p3, [40, 99]),
    ])
    st = traj.metadata["reconstruction_stats"]
    assert st["chains_total"] == 1, f"截断后被拆链: {st}"
    tr = traj.traces[0]
    assert tr.prompt_ids == p1
    assert 30 not in tr.response_ids and 31 not in tr.response_ids and 32 not in tr.response_ids, \
        "空截断的响应不应入流"
    assert tr.response_ids == [20, 99, 1, 60, 99, 1, 50, 51, 1, 70, 71, 99, 1, 50, 51, 40, 99]
    assert tr.loss_mask == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert traj.metadata["truncation_events"] == 1


def test_partial_truncation_keeps_response_in_stream():
    """半截截断(content 非空)会被 CLI 保留:响应正常入流、由 MASK_TRUNCATED 掩零。"""
    p1 = [1, 10, 11, 99, 1, 50, 51]
    p2 = [1, 10, 11, 99, 1, 50, 51, 20, 99, 1, 60, 99, 1, 50, 51]
    # 半截截断:CLI 保留 c2 的半截 assistant 回合([1,50,51,30,31,99],带 eot 收尾)
    p3 = [1, 10, 11, 99, 1, 50, 51, 20, 99, 1, 60, 99, 1, 50, 51, 30, 31, 99, 1, 70, 71, 99, 1, 50, 51]
    traj = _build([
        _record("c1", p1, [20, 99]),
        _trunc_record("c2", p2, [30, 31], content="partial text"),
        _record("c3", p3, [40, 99]),
    ])
    st = traj.metadata["reconstruction_stats"]
    assert st["chains_total"] == 1
    tr = traj.traces[0]
    assert 30 in tr.response_ids and 31 in tr.response_ids, "半截截断的响应必须入流"
    # MASK_TRUNCATED: 截断段(c2 的 [30,31])入流但掩零;工具结果与 limit 消息为 interstitial 掩零
    assert tr.response_ids == [20, 99, 1, 60, 99, 1, 50, 51, 30, 31, 99, 1, 70, 71, 99, 1, 50, 51, 40, 99]
    assert tr.loss_mask == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_reasoning_trains_by_default() -> None:
    """新默认(POLAR_MASK_REASONING 缺省=0):CoT 进训练,loss_mask 全 1。"""
    import os
    os.environ.pop("POLAR_MASK_REASONING", None)
    records = [
        CompletionRecord(
            completion_id="00-main1",
            request={"system": "harness", "messages": [{"role": "user", "content": "q"}]},
            response={
                "choices": [{
                    "input_token_ids": [1, 2],
                    "message": {"role": "assistant", "reasoning_content": "plan", "content": "main1"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [
                        {"token": "Need", "token_id": 10, "logprob": -0.01},
                        {"token": "</think>", "token_id": 11, "logprob": -0.02},
                        {"token": "main1", "token_id": 12, "logprob": -0.03},
                        {"token": f"t{EOT}", "token_id": EOT, "logprob": -0.04},
                    ]},
                }]
            },
        ),
    ]
    trajectory = _build(records)
    trace = trajectory.traces[0]
    assert trace.loss_mask == [1, 1, 1, 1], "默认下 CoT 应进训练(全 1)"
