"""Filters for completion records that must not become trainable traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from polar.trajectory.models import CompletionRecord


@dataclass(frozen=True, slots=True)
class CompletionFilterResult:
    kept: list[CompletionRecord]
    metadata: dict[str, Any]


def filter_trainable_completions(
    completions: list[CompletionRecord],
) -> CompletionFilterResult:
    kept: list[CompletionRecord] = []
    excluded: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for completion in completions:
        reason = exclude_completion_reason(completion)
        if reason is None:
            kept.append(completion)
            continue
        reasons[reason] += 1
        excluded.append(
            {
                "completion_id": completion.completion_id,
                "reason": reason,
            }
        )

    return CompletionFilterResult(
        kept=kept,
        metadata={
            "input_completions": len(completions),
            "kept_completions": len(kept),
            "excluded_completions": len(excluded),
            "excluded_reasons": dict(sorted(reasons.items())),
            "excluded_completion_ids": [item["completion_id"] for item in excluded],
            "excluded": excluded,
        },
    )


def exclude_completion_reason(completion: CompletionRecord) -> str | None:
    if _has_truncated_marker(completion.model_dump(mode="python")):
        return "persisted_truncated_completion"
    if _has_empty_choices(completion):
        return "empty_completion"
    if _is_non_agent_side_completion(completion):
        return "non_agent_side_completion"
    return None


def _has_truncated_marker(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("__truncated") is True:
            return True
        return any(_has_truncated_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_truncated_marker(item) for item in value)
    return False


def _has_empty_choices(completion: CompletionRecord) -> bool:
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    return not (
        isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
    )


def _is_non_agent_side_completion(completion: CompletionRecord) -> bool:
    """True when a request cannot have come from the agent loop.

    Judged purely on request *shape*, never on payload text. Every agent-side
    call carries the harness system prompt, the tool schema, and at least one
    SDK-only field (``thinking`` / ``stream`` / ...); a request with a lone user
    message and none of those is a bare completion the harness emitted outside
    the agent loop — typically the body of a large file the agent just Read.

    Content matching used to gate this check against one hardcoded Triton
    handbook, so the same shape carrying any other payload (cmake sources,
    judge output, skill frontmatter) slipped through and became a trainable
    trace. Shape alone is sufficient and task-agnostic.
    """
    request = _request_for_shape_check(completion)
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return False
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if request.get("system") is not None:
        return False
    if any(isinstance(item, dict) and item.get("role") == "system" for item in messages):
        return False
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        return False
    if any(key in request for key in ("context_management", "thinking", "output_config", "stream")):
        return False
    return True


def _request_for_shape_check(completion: CompletionRecord) -> dict[str, Any]:
    for request in (completion.original_request, completion.request):
        if isinstance(request, dict) and request:
            return request
    return {}
