import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "examples"
    / "ascend"
    / "polar_dockerruntime_e2e"
    / "tools"
    / "probe_gateway_runtime.py"
)


class FakeGateway(BaseHTTPRequestHandler):
    status_by_session: dict[str, str] = {}
    error_by_session: dict[str, str | None] = {}
    posts: list[dict] = []
    deletes: list[str] = []
    mode = "completed"

    def log_message(self, *_args):
        return

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path.startswith("/sessions/"):
            session_id = self.path.rsplit("/", 1)[-1]
            status = self.status_by_session.get(session_id, "REGISTERED")
            error = self.error_by_session.get(session_id)
            payload = {
                "session_id": session_id,
                "task_id": "probe",
                "created_at": "2026-06-23T00:00:00Z",
                "completion_count": 0,
                "status": status,
            }
            if status in {"ERROR", "TIMEOUT"}:
                payload["result"] = {
                    "session_id": session_id,
                    "task_id": "probe",
                    "status": status,
                    "trajectory": {"traces": []},
                    "error": error,
                }
            self._json(200, payload)
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/sessions":
            self._json(404, {"detail": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode())
        self.posts.append(payload)
        session_id = payload["session_id"]
        if self.mode == "completed":
            self.status_by_session[session_id] = "COMPLETED"
        else:
            self.status_by_session[session_id] = "ERROR"
            self.error_by_session[session_id] = (
                "runtime initialization failed: [Errno 2] No such file or directory"
            )
        self._json(
            200,
            {
                "session_id": session_id,
                "task_id": payload["task_id"],
                "status": "REGISTERED",
                "node_id": "fake-node",
            },
        )

    def do_DELETE(self):
        if self.path.startswith("/sessions/"):
            self.deletes.append(self.path.rsplit("/", 1)[-1])
            self._json(200, {"session_id": self.deletes[-1], "deleted": True, "messages_deleted": 0})
            return
        self._json(404, {"detail": "not found"})


class GatewayServer:
    def __init__(self, mode: str):
        self.mode = mode
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        FakeGateway.status_by_session = {}
        FakeGateway.error_by_session = {}
        FakeGateway.posts = []
        FakeGateway.deletes = []
        FakeGateway.mode = self.mode
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _run_probe(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gateway-url",
            url,
            "--image",
            "sandbox:v1",
            "--pool",
            "8,9,10,11",
            "--timeout",
            "3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_gateway_runtime_probe_passes_and_dispatches_minimal_docker_runtime():
    with GatewayServer("completed") as url:
        result = _run_probe(url)

    assert result.returncode == 0, result.stderr
    assert "runtime probe completed" in result.stdout
    payload = FakeGateway.posts[0]
    assert payload["runtime"]["backend"] == "docker"
    assert payload["runtime"]["image"] == "sandbox:v1"
    assert payload["runtime"]["kwargs"]["ascend"]["pool"] == "8,9,10,11"
    assert payload["runtime"]["kwargs"]["ascend"]["lease_at_start"] is False
    assert payload["agent"]["harness"] == "shell"
    assert payload["runtime"]["workdir"] == "/polar/session"
    assert payload["agent"]["custom_shell"]["command"] == "true"
    assert payload["agent"]["custom_shell"]["cwd"] == "/polar/session"
    assert FakeGateway.deletes


def test_gateway_runtime_probe_fails_on_runtime_initialization_error():
    with GatewayServer("error") as url:
        result = _run_probe(url)

    assert result.returncode == 1
    assert "runtime initialization failed" in result.stderr
    assert "No such file or directory" in result.stderr
    assert not FakeGateway.deletes


def test_gateway_runtime_probe_fails_when_gateway_unreachable():
    with GatewayServer("completed") as url:
        port = int(url.rsplit(":", 1)[-1])
    result = _run_probe(f"http://127.0.0.1:{port}")
    assert result.returncode == 1
    assert "GET" in result.stderr
