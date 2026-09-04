from __future__ import annotations

from pathlib import Path

from polar.rollout.artifacts import persist_profiling_artifacts


def test_artifact_manifest_copies_profile_data_and_excludes_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "msprof").mkdir(parents=True)
    (source / "msprof" / "timeline.json").write_text('{"events": []}')
    (source / "verify_result.json").write_text('{"success": true}')
    (source / "submission_impl.py").write_text("raise SystemExit")

    manifest = persist_profiling_artifacts(
        source,
        tmp_path / "persisted",
        session_id="session-1",
        max_total_bytes=1024,
    )

    assert manifest["artifact_count"] == 2
    assert {item["kind"] for item in manifest["artifacts"]} == {
        "profiler",
        "verification",
    }
    assert not (tmp_path / "persisted/session-1.artifacts/submission_impl.py").exists()
    assert len(manifest["artifacts"][0]["sha256"]) == 64


def test_artifact_manifest_enforces_total_byte_budget(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.log").write_text("a" * 8)
    (source / "b.log").write_text("b" * 8)

    manifest = persist_profiling_artifacts(
        source,
        tmp_path / "persisted",
        session_id="session-1",
        max_total_bytes=8,
    )

    assert manifest["artifact_count"] == 1
    assert manifest["total_bytes"] == 8
    assert manifest["skipped_bytes"] == 8
