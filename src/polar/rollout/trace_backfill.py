"""Backfill persisted Chrome/Perfetto session traces into an OTLP endpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx


def iter_trace_files(root: Path, *, limit: int | None = None) -> Iterable[Path]:
    """Yield bounded JSON inputs in stable order."""
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    yield from files if limit is None else files[:limit]


def load_trace_payload(
    path: Path,
    *,
    service_name: str = "polar-gateway",
    include_content: bool = False,
) -> dict[str, Any] | None:
    """Convert one versioned or legacy Chrome trace document to OTLP/HTTP JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        document: dict[str, Any] = {"schemaVersion": 1, "traceEvents": data}
    elif isinstance(data, dict):
        document = data
    else:
        return None
    raw_events = document.get("traceEvents")
    if not isinstance(raw_events, list):
        return None
    events = [event for event in raw_events if _is_complete_event(event)]
    if not events:
        return None

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    event_metadata = _event_metadata(raw_events)
    session_id = str(
        metadata.get("sessionId")
        or event_metadata.get("session_id")
        or path.stem.removesuffix(".trace")
    )
    task_id = str(metadata.get("taskId") or event_metadata.get("task_id") or "")
    node_id = str(metadata.get("nodeId") or event_metadata.get("node_id") or "")
    schema_version = int(document.get("schemaVersion") or event_metadata.get("schema_version") or 1)
    inferred = schema_version < 2 or min(int(event["ts"]) for event in events) < 1_000_000_000_000
    anchor_ns = 0
    if inferred:
        max_end_us = max(int(event["ts"]) + int(event["dur"]) for event in events)
        anchor_ns = path.stat().st_mtime_ns - max_end_us * 1000

    trace_id = hashlib.sha256(session_id.encode()).hexdigest()[:32]
    root_span_id = _span_id(session_id, "backfill-root")
    converted: list[dict[str, Any]] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, event in enumerate(events):
        start_ns = anchor_ns + int(event["ts"]) * 1000
        end_ns = start_ns + int(event["dur"]) * 1000
        starts.append(start_ns)
        ends.append(end_ns)
        attributes: dict[str, Any] = {
            "session_id": session_id,
            "task_id": task_id,
            "node_id": node_id,
            "project": "polar",
            "trace.backfilled": True,
            "timing.inferred": inferred,
            "chrome.category": str(event.get("cat") or ""),
            "chrome.pid": int(event.get("pid") or 0),
        }
        if include_content and isinstance(event.get("args"), dict):
            for key, value in event["args"].items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"event.{key}"] = _bounded(value, 2048)
        converted.append(
            _span(
                trace_id,
                _span_id(session_id, f"backfill:{index}"),
                str(event.get("name") or "event"),
                start_ns,
                end_ns,
                parent_span_id=root_span_id,
                attributes=attributes,
            )
        )
    root = _span(
        trace_id,
        root_span_id,
        "agent_session.backfilled",
        min(starts),
        max(ends),
        attributes={
            "session_id": session_id,
            "task_id": task_id,
            "node_id": node_id,
            "project": "polar",
            "trace.backfilled": True,
            "timing.inferred": inferred,
            "trace.schema_version": schema_version,
            "trace.source_file": path.name,
        },
    )
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attribute("service.name", service_name)]},
            "scopeSpans": [{
                "scope": {"name": "polar.trace_backfill", "version": "1"},
                "spans": [root, *converted],
            }],
        }]
    }


def backfill_trace_files(
    root: Path,
    *,
    otlp_endpoint: str,
    service_name: str = "polar-gateway",
    include_content: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    timeout_seconds: float = 10.0,
) -> dict[str, int]:
    """Export trace files one at a time and return deterministic counters."""
    counts = {"discovered": 0, "exported": 0, "skipped": 0, "failed": 0}
    with httpx.Client(timeout=timeout_seconds) as client:
        for path in iter_trace_files(root, limit=limit):
            counts["discovered"] += 1
            try:
                payload = load_trace_payload(
                    path,
                    service_name=service_name,
                    include_content=include_content,
                )
                if payload is None:
                    counts["skipped"] += 1
                    continue
                if not dry_run:
                    response = client.post(
                        otlp_endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                counts["exported"] += 1
            except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError):
                counts["failed"] += 1
    return counts


def _is_complete_event(event: Any) -> bool:
    return (
        isinstance(event, dict)
        and event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
        and event["dur"] >= 0
    )


def _event_metadata(events: list[Any]) -> dict[str, Any]:
    for event in events:
        if isinstance(event, dict) and event.get("name") == "session_metadata":
            args = event.get("args")
            return args if isinstance(args, dict) else {}
    return {}


def _span(
    trace_id: str,
    span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    *,
    parent_span_id: str | None = None,
    attributes: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(max(start_ns, end_ns)),
        "attributes": [_attribute(key, value) for key, value in attributes.items()],
        "status": {"code": 1},
    }
    if parent_span_id:
        result["parentSpanId"] = parent_span_id
    return result


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _span_id(session_id: str, suffix: str) -> str:
    return hashlib.sha256(f"{session_id}:{suffix}".encode()).hexdigest()[:16]


def _bounded(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


__all__ = ["backfill_trace_files", "iter_trace_files", "load_trace_payload"]
