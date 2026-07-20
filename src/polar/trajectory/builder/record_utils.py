"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace

_THINK_END_TOKEN = "</think>"


def _logprob_content(choice: dict[str, Any]) -> list[Any] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    return content if isinstance(content, list) else None


def _extract_response_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    token_ids = choice.get("token_ids", response.get("token_ids"))
    if isinstance(token_ids, list):
        return list(token_ids)

    content = _logprob_content(choice)
    if isinstance(content, list):
        extracted = [
            int(item["token_id"])
            for item in content
            if isinstance(item, dict) and item.get("token_id") is not None
        ]
        if extracted:
            return extracted
    return []


def _extract_response_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Sampled-token logprob per position, aligned 1:1 with response_ids.

    The token id is intentionally dropped -- it is already in ``response_ids``
    at the same index; only the float is needed for training.
    """
    content = _logprob_content(choice)
    if isinstance(content, list):
        return [
            float(item.get("logprob", 0.0)) if isinstance(item, dict) else 0.0
            for item in content
        ]
    return None


def _logprob_integrity(choice: dict[str, Any], response_ids: list[int]) -> dict[str, int]:
    """Detect the two silent corruptions the length-only contract misses (see RL-sample audit):

      * misattributed: logprobs.content[i].token_id != response_ids[i] — each token would carry a
        logprob computed for a DIFFERENT token (e.g. an else-0 backend fallback), biasing the GRPO
        importance ratio. Lengths stay equal, so no existing guard fires.
      * missing: a content entry lacks a `logprob` — _extract_response_logprobs 0.0-fills it, which
        reads as probability 1.0 for a sampled token (fabricated). Length stays equal.

    Non-fatal here (keeps capture robust); recorded into trace.metadata so the rllm adapter
    (normalize_trace, which has per-task error handling) can REJECT the trace before it trains.
    """
    content = _logprob_content(choice)
    if not isinstance(content, list):
        return {}
    misattributed = missing = 0
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            missing += 1
            continue
        lp = item.get("logprob")
        # vLLM sets logprob to -9999.0 both as the field default (missing →
        # chat_completion/protocol.py:69) and as the clamp floor (serving.py:1448/1502),
        # so `is None` never fires for vLLM: a missing/degenerate logprob reads as a valid
        # very-negative value and fabricates probability into the GRPO ratio. Treat the
        # -9999.0 sentinel as missing so the adapter rejects the trace.
        if lp is None or (isinstance(lp, (int, float)) and lp <= -9999.0):
            missing += 1
        tid = item.get("token_id")
        if tid is not None and i < len(response_ids) and int(tid) != int(response_ids[i]):
            misattributed += 1
    return {"misattributed": misattributed, "missing": missing} if (misattributed or missing) else {}


def _response_loss_mask(
    choice: dict[str, Any], response_message: Any, response_ids: list[int]
) -> tuple[list[int], dict[str, Any]]:
    """Mask parsed Qwen-style hidden reasoning without changing token/logprob alignment."""

    mask = [1] * len(response_ids)
    if not mask or not isinstance(response_message, dict):
        return mask, {}

    reasoning_content = response_message.get("reasoning_content")
    if not reasoning_content:
        return mask, {}

    content = _logprob_content(choice)
    if not isinstance(content, list):
        return mask, {"masked_tokens": 0, "reason": "missing_logprobs_content"}

    end_index = None
    for idx, item in enumerate(content[: len(mask)]):
        if isinstance(item, dict) and item.get("token") == _THINK_END_TOKEN:
            end_index = idx
            break

    if end_index is None:
        return mask, {"masked_tokens": 0, "reason": "think_end_token_not_found"}

    # [CoT-train gate] POLAR_MASK_REASONING=0 → 训 reasoning(不 mask <think>…</think>),但仍记录 span
    #   到 metadata 供下游(cot_tis 监控)用。默认 "1" = 现状。放开时 loss_mask 保持全 1,reasoning 进训练。
    if os.environ.get("POLAR_MASK_REASONING", "1") == "0":
        return mask, {"masked_tokens": 0, "reasoning_span": end_index + 1, "reason": "reasoning_masking_disabled"}

    masked_tokens = end_index + 1
    for idx in range(masked_tokens):
        mask[idx] = 0
    return mask, {"masked_tokens": masked_tokens, "end_token_index": end_index}


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def build_trace_from_completion(completion: CompletionRecord) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = first_choice.get("input_token_ids") or response.get("prompt_token_ids")
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids = _extract_response_ids(response, first_choice)

    # Record (don't raise) token_id<->logprob misattribution / missing-logprob into metadata so the
    # rllm adapter can reject the trace before training (keeps capture itself robust). Adds the key
    # ONLY when there's a problem -> zero behavior change on healthy completions.
    md = deepcopy(completion.metadata)
    integ = _logprob_integrity(first_choice, response_ids)
    if integ:
        md = {**(md if isinstance(md, dict) else {}), "logprob_integrity": integ}
    loss_mask, reasoning_mask = _response_loss_mask(first_choice, response_message, response_ids)
    if reasoning_mask:
        md = {**(md if isinstance(md, dict) else {}), "reasoning_loss_mask": reasoning_mask}

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=loss_mask,
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=_extract_response_logprobs(first_choice),
        metadata=md,
    )
