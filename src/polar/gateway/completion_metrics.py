"""Lightweight observability metrics for gateway completion calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _cached_prompt_tokens(usage: dict[str, Any]) -> int | None:
    direct = _int_or_none(usage.get("cached_tokens"))
    if direct is not None:
        return direct
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return _int_or_none(details.get("cached_tokens"))
    return None


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    reason = first.get("finish_reason")
    return str(reason) if reason is not None else None


def _usage(response: dict[str, Any]) -> dict[str, int | None]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_prompt_tokens": None,
        }
    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    total_tokens = _int_or_none(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": _cached_prompt_tokens(usage),
    }


def _tokens_per_second(tokens: int | None, latency_ms: float | None) -> float | None:
    if tokens is None or latency_ms is None or latency_ms <= 0:
        return None
    return tokens / (latency_ms / 1000.0)


def build_completion_metric_event(
    *,
    session_id: str,
    task_id: str | None,
    completion_id: str,
    sequence: int,
    api_type: str | None,
    model_requested: str | None,
    model_used: str | None,
    response: dict[str, Any],
    latency_ms: float | None,
    streaming: bool,
) -> dict[str, Any]:
    """Build a small per-completion event safe to expose in logs/UI."""
    usage = _usage(response)
    latency = _float_or_none(latency_ms)
    if latency is not None:
        latency = round(max(0.0, latency), 3)
    completion_tokens = usage["completion_tokens"]
    return {
        "schema_version": 1,
        "recorded_at": _utcnow_iso(),
        "session_id": session_id,
        "task_id": task_id,
        "completion_id": completion_id,
        "sequence": sequence,
        "api_type": api_type,
        "model_requested": model_requested,
        "model_used": model_used,
        "streaming": bool(streaming),
        "latency_ms": latency,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "total_tokens": usage["total_tokens"],
        "cached_prompt_tokens": usage["cached_prompt_tokens"],
        "completion_tokens_per_second": _tokens_per_second(completion_tokens, latency),
        "finish_reason": _finish_reason(response),
    }


@dataclass(slots=True)
class CompletionMetricsAggregate:
    """Bounded in-memory aggregate for one gateway session."""

    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0
    first_recorded_at: str | None = None
    latest_recorded_at: str | None = None
    latest: dict[str, Any] = field(default_factory=dict)

    def update(self, event: dict[str, Any]) -> None:
        self.request_count += 1
        self.prompt_tokens += int(event.get("prompt_tokens") or 0)
        self.completion_tokens += int(event.get("completion_tokens") or 0)
        self.total_tokens += int(event.get("total_tokens") or 0)
        self.cached_prompt_tokens += int(event.get("cached_prompt_tokens") or 0)
        latency_ms = _float_or_none(event.get("latency_ms"))
        if latency_ms is not None:
            self.latency_ms_total += max(0.0, latency_ms)
            self.latency_ms_max = max(self.latency_ms_max, latency_ms)
        recorded_at = str(event.get("recorded_at") or "")
        self.first_recorded_at = self.first_recorded_at or recorded_at or None
        self.latest_recorded_at = recorded_at or self.latest_recorded_at
        self.latest = dict(event)

    def as_dict(self) -> dict[str, Any]:
        mean_latency_ms: float | None = None
        if self.request_count:
            mean_latency_ms = self.latency_ms_total / self.request_count
        return {
            "schema_version": 1,
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "latency_ms_total": round(self.latency_ms_total, 3),
            "latency_ms_mean": round(mean_latency_ms, 3) if mean_latency_ms is not None else None,
            "latency_ms_max": round(self.latency_ms_max, 3),
            "completion_tokens_per_second": _tokens_per_second(
                self.completion_tokens,
                self.latency_ms_total,
            ),
            "first_recorded_at": self.first_recorded_at,
            "latest_recorded_at": self.latest_recorded_at,
            "latest": dict(self.latest),
        }


def combine_completion_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-session metric aggregates into a node-level summary."""
    request_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_prompt_tokens = 0
    latency_ms_total = 0.0
    latency_ms_max = 0.0
    for row in rows:
        metrics = row.get("completion_metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, dict):
            continue
        request_count += int(metrics.get("request_count") or 0)
        prompt_tokens += int(metrics.get("prompt_tokens") or 0)
        completion_tokens += int(metrics.get("completion_tokens") or 0)
        total_tokens += int(metrics.get("total_tokens") or 0)
        cached_prompt_tokens += int(metrics.get("cached_prompt_tokens") or 0)
        latency_ms_total += float(metrics.get("latency_ms_total") or 0.0)
        latency_ms_max = max(latency_ms_max, float(metrics.get("latency_ms_max") or 0.0))
    mean_latency_ms = latency_ms_total / request_count if request_count else None
    return {
        "schema_version": 1,
        "session_count": len(rows),
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "latency_ms_total": round(latency_ms_total, 3),
        "latency_ms_mean": round(mean_latency_ms, 3) if mean_latency_ms is not None else None,
        "latency_ms_max": round(latency_ms_max, 3),
        "completion_tokens_per_second": _tokens_per_second(
            completion_tokens,
            latency_ms_total,
        ),
    }
