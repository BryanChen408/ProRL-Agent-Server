"""End-to-end FastAPI tests of the platform service."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from polar.platform.config import PlatformConfig
from polar.platform.server import create_app
from polar.rollout.artifacts import persist_profiling_artifacts


@pytest.fixture()
def topology_with_results(tmp_path: Path) -> Path:
    save_dir = tmp_path / "rollout_results"
    save_dir.mkdir(parents=True)
    trace_dir = save_dir / "perfetto_traces"
    trace_dir.mkdir()
    (trace_dir / "abc.json").write_text(
        json.dumps({
            "schemaVersion": 2,
            "traceEvents": [{
                "name": "run",
                "ph": "X",
                "ts": 1_700_000_000_000_000,
                "dur": 200_000,
                "pid": 1,
                "tid": "abc",
            }],
            "metadata": {"sessionId": "abc"},
        })
    )
    artifact_source = tmp_path / "session_artifacts"
    artifact_source.mkdir()
    (artifact_source / "msprof_summary.json").write_text('{"kernel": "Add"}')
    persist_profiling_artifacts(
        artifact_source,
        trace_dir,
        session_id="abc",
        max_total_bytes=1024,
    )
    task_dir = save_dir / "task_demo-claude_code-001"
    task_dir.mkdir()
    (task_dir / "ses_abc.json").write_text(
        json.dumps(
            {
                "session_id": "abc",
                "task_id": "demo-claude_code-001",
                "status": "COMPLETED",
                "node_id": "node-a",
                "timing": {
                    "register_to_init_queue_ms": 1.0,
                    "init_ms": 100.0,
                    "run_ms": 200.0,
                    "postrun_ms": 50.0,
                },
                "trajectory": {
                    "status": "COMPLETED",
                    "metadata": {
                        "api_type": "anthropic",
                        "model_used": "Qwen/Qwen3.5-4B",
                        "evaluation": {
                            "strategy": "test_on_output",
                            "outcome_reward": 1.0,
                        },
                    },
                    "traces": [{"reward": 1.0}],
                },
            }
        )
    )
    topology = tmp_path / "topology.yaml"
    topology.write_text(
        f"""
rollout:
  host: 127.0.0.1
  port: 9080
  public_url: http://127.0.0.1:9080
  save_dir: {save_dir}
gateway:
  heartbeat_interval_seconds: 30
  persist_traces_dir: {trace_dir}
  nodes:
    - id: gw-test
      host: 127.0.0.1
      port: 9081
      public_url: http://127.0.0.1:9081
      max_init_workers: 4
      max_run_workers: 4
      max_postrun_workers: 4
      model_served: Qwen/Qwen3.5-4B
      inference:
        engine: sglang
        base_url: http://127.0.0.1:9000
""".strip()
    )
    return topology


def _client(topology_path: Path) -> TestClient:
    config = PlatformConfig.from_topology(topology_path, port=9090)
    app = create_app(config)
    return TestClient(app)


def test_health_route(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "rollout_url" in body


def test_tasks_route_lists_filesystem(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        r = client.get("/api/tasks")
        assert r.status_code == 200
        tasks = r.json()["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "demo-claude_code-001"
        assert tasks[0]["harness"] == "claude_code"
        assert tasks[0]["status"] == "completed"
        assert tasks[0]["session_time_ms"] == 351.0


def test_task_detail_includes_sessions(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        r = client.get("/api/tasks/demo-claude_code-001")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "demo-claude_code-001"
        assert len(body["sessions"]) == 1
        assert body["sessions"][0]["session_id"] == "abc"


def test_session_detail_and_trajectory(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        r = client.get("/api/sessions/abc")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "COMPLETED"
        assert body["task_id"] == "demo-claude_code-001"
        assert body["timing"]["run_ms"] == 200.0

        r = client.get("/api/sessions/abc/trajectory")
        assert r.status_code == 200
        traj = r.json()
        assert traj["status"] == "COMPLETED"
        assert traj["traces"][0]["reward"] == 1.0

        r = client.get("/api/sessions/abc/evaluation")
        assert r.status_code == 200
        ev = r.json()
        assert ev["outcome_reward"] == 1.0
        assert ev["strategy"] == "test_on_output"


def test_session_trace_view_and_download(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        response = client.get("/api/sessions/abc/trace")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == 2
        assert payload["event_count"] == 1
        assert payload["trace_events"][0]["name"] == "run"
        assert payload["download_url"].endswith("/abc/trace/download")

        download = client.get(payload["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/json"
        assert "abc.trace.json" in download.headers["content-disposition"]


def test_session_artifact_manifest_and_download(topology_with_results: Path) -> None:
    with _client(topology_with_results) as client:
        response = client.get("/api/sessions/abc/artifacts")
        assert response.status_code == 200
        payload = response.json()
        assert payload["artifact_count"] == 1
        artifact = payload["artifacts"][0]
        assert artifact["kind"] == "profiler"
        assert artifact["relative_path"] == "msprof_summary.json"

        download = client.get(artifact["download_url"])
        assert download.status_code == 200
        assert download.json() == {"kernel": "Add"}

        missing = client.get("/api/sessions/abc/artifacts/not-in-manifest")
        assert missing.status_code == 404


def test_dashboard_has_no_submit_or_templates(topology_with_results: Path) -> None:
    """The dashboard is read-only. /api/templates and /api/submit are not API routes.

    The SPA fallback catches unknown paths and returns the HTML index; what we
    really want to check is that those URLs are not JSON API endpoints.
    """
    with _client(topology_with_results) as client:
        for path in ("/api/templates", "/api/models"):
            resp = client.get(path)
            content_type = resp.headers.get("content-type", "")
            assert "application/json" not in content_type, (
                f"{path} unexpectedly serves JSON; submit/templates should be gone"
            )
        # POST without a JSON router returns 405 from the SPA fallback (GET-only).
        resp = client.post("/api/submit", json={})
        assert resp.status_code in (404, 405)
