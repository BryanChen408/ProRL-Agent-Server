"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace


def _extract_response_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    token_ids = choice.get("token_ids", response.get("token_ids"))
    if isinstance(token_ids, list):
        return list(token_ids)

    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
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
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
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
    lg = choice.get("logprobs")
    content = lg.get("content") if isinstance(lg, dict) else None
    if not isinstance(content, list):
        return {}
    misattributed = missing = 0
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            missing += 1
            continue
        if item.get("logprob") is None:
            missing += 1
        tid = item.get("token_id")
        if tid is not None and i < len(response_ids) and int(tid) != int(response_ids[i]):
            misattributed += 1
    return {"misattributed": misattributed, "missing": missing} if (misattributed or missing) else {}


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

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=_extract_response_logprobs(first_choice),
        metadata=md,
    )
