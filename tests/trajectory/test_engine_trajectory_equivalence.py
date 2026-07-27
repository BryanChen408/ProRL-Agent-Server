"""Both backends must produce the same trajectory from the same generation.

SGLang (patched) and vLLM (`return_token_ids` + `VLLMEngine.normalize_response`)
expose the training fields in different response shapes. These tests pin that
the two shapes collapse to byte-identical ``Trace`` objects through the real
builders, so downstream training sees one trajectory regardless of engine.
"""

from __future__ import annotations

import asyncio

from polar.gateway.engine import VLLMEngine
from polar.trajectory.builder.per_request import PerRequestBuilder
from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord, CompletionSession

_EOT = 99  # synthetic end-of-turn token id


def _sglang_record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float],
    *,
    content: str,
    reasoning: str | None,
    finish_reason: str,
    prompt_messages: list[dict],
    response_message: dict,
) -> CompletionRecord:
    """Canonical SGLang shape: input_token_ids on the choice, token_id in logprobs."""
    message = {"role": "assistant", "content": content, **response_message}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    response = {
        "choices": [
            {
                "input_token_ids": list(prompt_ids),
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": {
                    "content": [
                        {"token": f"t{tid}", "token_id": tid, "logprob": lp, "bytes": []}
                        for tid, lp in zip(response_ids, logprobs)
                    ]
                },
            }
        ]
    }
    return CompletionRecord(
        completion_id=completion_id,
        # `system` marks agent-side traffic so record_filters keeps the record.
        request={"system": "harness", "messages": prompt_messages},
        response=response,
    )


def _vllm_record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float],
    *,
    content: str,
    reasoning: str | None,
    finish_reason: str,
    prompt_messages: list[dict],
    response_message: dict,
) -> CompletionRecord:
    """Native vLLM shape, passed through the gateway's normalize_response."""
    message = {"role": "assistant", "content": content, **response_message}
    if reasoning is not None:
        message["reasoning"] = reasoning  # vLLM names it `reasoning`
    response = {
        "prompt_token_ids": list(prompt_ids),  # top-level in vLLM
        "choices": [
            {
                "token_ids": list(response_ids),  # on the choice in vLLM
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": {
                    "content": [
                        {"token": f"t{tid}", "logprob": lp, "bytes": []}
                        for tid, lp in zip(response_ids, logprobs)
                    ]
                },
            }
        ],
    }
    response = VLLMEngine().normalize_response(response)
    return CompletionRecord(
        completion_id=completion_id,
        request={"system": "harness", "messages": prompt_messages},
        response=response,
    )


def _assert_traces_equal(a, b) -> None:
    assert a.prompt_ids == b.prompt_ids
    assert a.response_ids == b.response_ids
    assert a.loss_mask == b.loss_mask
    assert a.finish_reason == b.finish_reason
    assert a.response_messages == b.response_messages
    assert a.response_logprobs == b.response_logprobs


def test_single_turn_trace_is_identical_across_engines() -> None:
    common = dict(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, 12, 13],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        content="4",
        reasoning="thinking",
        finish_reason="stop",
        prompt_messages=[{"role": "user", "content": "2+2?"}],
        response_message={},
    )
    sg = build_trace_from_completion(_sglang_record("c1", **common))
    vllm = build_trace_from_completion(_vllm_record("c1", **common))

    _assert_traces_equal(sg, vllm)
    # And the actual values are the correct ones.
    assert vllm.prompt_ids == [1, 2, 3]
    assert vllm.response_ids == [10, 11, 12, 13]
    assert vllm.loss_mask == [1, 1, 1, 1]
    assert vllm.response_messages[0]["reasoning_content"] == "thinking"
    assert vllm.response_logprobs == [-0.1, -0.2, -0.3, -0.4]


def test_per_request_builder_is_identical_across_engines() -> None:
    common = dict(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, 12, 13],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        content="4",
        reasoning=None,
        finish_reason="stop",
        prompt_messages=[{"role": "user", "content": "2+2?"}],
        response_message={},
    )
    sg_traj = asyncio.run(
        PerRequestBuilder().build(
            CompletionSession(session_id="s", completions=[_sglang_record("c1", **common)])
        )
    )
    vllm_traj = asyncio.run(
        PerRequestBuilder().build(
            CompletionSession(session_id="s", completions=[_vllm_record("c1", **common)])
        )
    )
    _assert_traces_equal(sg_traj.traces[0], vllm_traj.traces[0])


def test_adapter_rollout_log_probs_are_identical_across_engines() -> None:
    from slime_bridge.adapter import _extract_rollout_log_probs

    common = dict(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, 12, 13],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        content="4",
        reasoning="thinking",
        finish_reason="stop",
        prompt_messages=[{"role": "user", "content": "2+2?"}],
        response_message={},
    )
    sg = build_trace_from_completion(_sglang_record("c1", **common))
    vllm = build_trace_from_completion(_vllm_record("c1", **common))

    kwargs = dict(
        response_len=4,
        loss_mask=[1, 1, 1, 1],
        require_trainable_logprobs=True,
        session_id="s",
        trace_index=0,
    )
    sg_lp = _extract_rollout_log_probs(sg, **kwargs)
    vllm_lp = _extract_rollout_log_probs(vllm, **kwargs)
    assert sg_lp == vllm_lp == [-0.1, -0.2, -0.3, -0.4]


def _two_turn_chain(record_fn) -> list[CompletionRecord]:
    """A valid 2-completion agent chain (C2.prompt == C1.prompt + C1.resp + tool)."""
    q1 = {"role": "user", "content": "Q1"}
    a1 = {"role": "assistant", "content": "A1"}
    tool = {"role": "tool", "content": "result"}  # interstitial, dropped by grouping
    c1 = record_fn(
        "c1",
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, _EOT],
        logprobs=[-0.1, -0.2, -0.3],
        content="A1",
        reasoning=None,
        finish_reason="stop",
        prompt_messages=[q1],
        response_message={},
    )
    # canonical_tail = [10, 11, _EOT, 50, 51] -> interstitial after _EOT = [50, 51]
    c2 = record_fn(
        "c2",
        prompt_ids=[1, 2, 3, 10, 11, _EOT, 50, 51],
        response_ids=[20, 21, _EOT],
        logprobs=[-0.5, -0.6, -0.7],
        content="A2",
        reasoning=None,
        finish_reason="stop",
        prompt_messages=[q1, a1, tool],
        response_message={},
    )
    return [c1, c2]


def test_prefix_merging_chain_is_identical_across_engines() -> None:
    builder = PrefixMergingBuilder(end_of_turn_token_id=_EOT)
    sg = asyncio.run(
        builder.build(CompletionSession(session_id="s", completions=_two_turn_chain(_sglang_record)))
    )
    vllm = asyncio.run(
        builder.build(CompletionSession(session_id="s", completions=_two_turn_chain(_vllm_record)))
    )

    assert len(sg.traces) == len(vllm.traces) == 1
    _assert_traces_equal(sg.traces[0], vllm.traces[0])
    # The merged stream is the prompt + raw responses + canonical interstitial,
    # with interstitial tokens masked out.
    assert vllm.traces[0].response_ids == [10, 11, _EOT, 50, 51, 20, 21, _EOT]
    assert vllm.traces[0].loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
