from __future__ import annotations

import json
from pathlib import Path

from polar.cli import build_parser
from polar.rollout.trace_backfill import backfill_trace_files, load_trace_payload


def _attributes(span: dict) -> dict:
    return {
        item["key"]: next(iter(item["value"].values()))
        for item in span["attributes"]
    }


def test_versioned_trace_preserves_epoch_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "session.trace.json"
    path.write_text(json.dumps({
        "schemaVersion": 2,
        "metadata": {"sessionId": "session-1", "taskId": "task-1", "nodeId": "node-a"},
        "traceEvents": [{
            "name": "llm_call_1/inference", "ph": "X", "cat": "engine,vllm",
            "pid": 2, "ts": 1_700_000_000_000_000, "dur": 50_000,
            "args": {"engine": "vllm", "secret": "hidden"},
        }],
    }), encoding="utf-8")

    payload = load_trace_payload(path)
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    event = spans[1]
    assert event["startTimeUnixNano"] == "1700000000000000000"
    assert event["endTimeUnixNano"] == "1700000000050000000"
    assert _attributes(event)["timing.inferred"] is False
    assert not any(item["key"].startswith("event.") for item in event["attributes"])


def test_legacy_trace_is_anchored_and_marks_inferred_timing(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([
        {"name": "run", "ph": "X", "cat": "gateway", "pid": 1, "ts": 0, "dur": 1000},
        {"name": "session_metadata", "ph": "M", "args": {"session_id": "legacy-1"}},
    ]), encoding="utf-8")

    payload = load_trace_payload(path, include_content=True)
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root = spans[0]
    assert _attributes(root)["session_id"] == "legacy-1"
    assert _attributes(root)["timing.inferred"] is True
    assert int(root["endTimeUnixNano"]) - int(root["startTimeUnixNano"]) == 1_000_000


def test_backfill_dry_run_counts_valid_and_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "valid.json").write_text(json.dumps([
        {"name": "run", "ph": "X", "ts": 0, "dur": 1, "pid": 1}
    ]), encoding="utf-8")
    (tmp_path / "invalid.json").write_text("not-json", encoding="utf-8")

    counts = backfill_trace_files(
        tmp_path,
        otlp_endpoint="http://tempo:4318/v1/traces",
        dry_run=True,
    )

    assert counts == {"discovered": 2, "exported": 1, "skipped": 0, "failed": 1}


def test_backfill_cli_parses_safe_defaults() -> None:
    args = build_parser().parse_args([
        "backfill_traces",
        "/traces",
        "--otlp-endpoint",
        "http://tempo:4318/v1/traces",
    ])

    assert args.command == "backfill_traces"
    assert args.include_content is False
    assert args.dry_run is False
    assert args.limit is None
