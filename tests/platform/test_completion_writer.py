"""Tests for the gateway CompletionWriter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from polar.gateway.completion_writer import CompletionWriter, _truncate_value


def test_truncate_value_string() -> None:
    long = "a" * 100
    truncated = _truncate_value(long, max_bytes=20)
    assert isinstance(truncated, str)
    assert len(truncated.encode("utf-8")) <= 24  # plus ellipsis


def test_truncate_value_under_budget() -> None:
    short = {"foo": "bar"}
    assert _truncate_value(short, max_bytes=1024) == short


def test_truncate_value_reports_omitted_keys_from_first_dropped_key() -> None:
    value = {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": ["x" * 200],
        "usage": {"output_tokens": 1},
        "metadata": {"run_id": "r"},
    }

    truncated = _truncate_value(value, max_bytes=100)

    assert truncated["__truncated"] is True
    assert truncated["_truncated_keys_omitted"] == ["choices", "usage", "metadata"]


def test_writer_persists_records(tmp_path: Path) -> None:
    async def _run() -> None:
        writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
        await writer.start()
        for i in range(3):
            writer.enqueue(
                task_id="t1",
                session_id="sess1",
                completion_id=f"id{i}",
                record={"completion_id": f"id{i}", "payload": {"i": i}},
            )
        # Give the drain loop a moment to flush.
        await asyncio.sleep(0.2)
        await writer.close()

    asyncio.run(_run())

    out_dir = tmp_path / "task_t1" / "sessions" / "sess1" / "completions"
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 3
    first = json.loads(files[0].read_text())
    assert first["payload"]["i"] == 0


def test_writer_scopes_records_by_training_run_id(tmp_path: Path) -> None:
    async def _run() -> None:
        writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
        await writer.start()
        writer.enqueue(
            task_id="train-a-polar-op-0-0",
            session_id="sess1",
            completion_id="id0",
            record={"completion_id": "id0", "metadata": {"run_id": "train-a"}},
        )
        await asyncio.sleep(0.2)
        await writer.close()

    asyncio.run(_run())

    out = tmp_path / "run_train-a" / "task_train-a-polar-op-0-0" / "sessions" / "sess1" / "completions"
    assert sorted(out.glob("*.json"))


def test_writer_disabled_when_no_save_dir() -> None:
    async def _run() -> bool:
        writer = CompletionWriter(save_dir=None, enabled=True)
        await writer.start()  # no-op
        ok = writer.enqueue(task_id="t", session_id="s", completion_id="c", record={})
        await writer.close()
        return ok

    assert asyncio.run(_run()) is False


def test_writer_requires_task_id(tmp_path: Path) -> None:
    async def _run() -> bool:
        writer = CompletionWriter(save_dir=tmp_path)
        await writer.start()
        ok = writer.enqueue(task_id=None, session_id="s", completion_id="c", record={})
        await writer.close()
        return ok

    assert asyncio.run(_run()) is False
