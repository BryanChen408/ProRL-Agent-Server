from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from polar.rollout import server


def test_operator_samples_endpoint_expands_and_submits(monkeypatch, tmp_path: Path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
  default_operator_profile: operator_npu
  operator_profiles:
    operator_npu:
      agent: {harness: claude_code}
      metadata:
        rendered_op: "{op_name}"
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
""".strip()
    )
    server.configure_server(str(topology_path))
    state = server.get_state()
    captured = {}

    async def submit_task(request):  # noqa: ANN001, ANN202
        captured["request"] = request
        return request.task_id

    async def noop() -> None:
        return None

    monkeypatch.setattr(state.manager, "submit_task", submit_task)
    monkeypatch.setattr(state.pipeline, "start", noop)
    monkeypatch.setattr(state.pipeline, "close", noop)

    with TestClient(server.app) as client:
        response = client.post(
            "/rollout/operator_samples/submit",
            json={
                "task_id": "task-1",
                "instruction": "do it",
                "num_samples": 2,
                "sample": {"op_name": "op", "group_index": 3},
                "metadata": {"policy_version": 7},
            },
        )

    assert response.status_code == 200
    task_request = captured["request"]
    assert task_request.task_id == "task-1"
    assert task_request.num_samples == 2
    assert task_request.agent.harness == "claude_code"
    assert task_request.metadata["operator_profile"] == "operator_npu"
    assert task_request.metadata["op_name"] == "op"
    assert task_request.metadata["rendered_op"] == "op"
    assert task_request.metadata["policy_version"] == 7


def test_rollout_admin_inference_pause_fans_out_to_gateway(monkeypatch, tmp_path: Path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout:
  public_url: http://127.0.0.1:8080
gateway:
  nodes:
    - id: n1
      public_url: http://127.0.0.1:8100
    - id: n2
      public_url: http://127.0.0.1:8101
""".strip()
    )
    server.configure_server(str(topology_path))
    state = server.get_state()
    calls = []

    async def noop() -> None:
        return None

    gateway_responses = iter([
        {"paused": True, "drained": True, "timed_out": False, "inflight": 0},
        {"paused": True, "drained": False, "timed_out": True, "inflight": 3},
    ])

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return next(gateway_responses)

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url: str, params=None):  # noqa: ANN001, ANN202
            calls.append((url, params))
            return _Response()

    monkeypatch.setattr(state.pipeline, "start", noop)
    monkeypatch.setattr(state.pipeline, "close", noop)
    monkeypatch.setattr(server.httpx, "AsyncClient", _Client)

    with TestClient(server.app) as client:
        response = client.post(
            "/rollout/admin/inference/pause",
            params={"timeout_seconds": 12},
        )

    assert response.status_code == 200
    assert calls == [
        ("http://127.0.0.1:8100/admin/inference/pause", {"timeout_seconds": 12.0}),
        ("http://127.0.0.1:8101/admin/inference/pause", {"timeout_seconds": 12.0}),
    ]
    assert response.json() == {
        "all_paused": True,
        "all_drained": False,
        "inflight": 3,
        "nodes": [
            {
                "node_id": "n1",
                "status": "ok",
                "response": {
                    "paused": True,
                    "drained": True,
                    "timed_out": False,
                    "inflight": 0,
                },
            },
            {
                "node_id": "n2",
                "status": "ok",
                "response": {
                    "paused": True,
                    "drained": False,
                    "timed_out": True,
                    "inflight": 3,
                },
            },
        ],
    }
