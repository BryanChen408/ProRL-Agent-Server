from __future__ import annotations

import asyncio
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest

from polar.gateway.node import GatewayNodeManager
from polar.gateway.operator_best import PACK, best_root, best_submission, retain_best, evaluation_records
from polar.gateway import server
from polar.gateway.dispatcher import SessionStage
from polar.trajectory.builder.attempt_spans import best_ordinal, build_spans, verdict_score
from polar.trajectory.models import EvaluatorSpec


def session(tmp_path):
    workdir = tmp_path / "work"
    mounted = tmp_path / "sessions" / "session-test"
    workdir.mkdir()
    mounted.mkdir(parents=True)

    async def download(remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)

    spec = EvaluatorSpec(strategy="operator_judge", config={
        "op_name": "op_test", "workdir": str(workdir),
        "judge_command": "bash tools/ascendc_eval_pipeline.sh --op_name op_test",
    })
    managed = SimpleNamespace(
        session_dir=mounted, artifacts_dir=mounted / "artifacts",
        stage=SessionStage.RUNNING, operator_best_lock=asyncio.Lock(),
        request=SimpleNamespace(evaluator=spec, session_id="test", task_id="task-test"),
        runtime=SimpleNamespace(spec=SimpleNamespace(workdir=str(workdir)),
                                runtime_session_dir=str(mounted),
                                download_file=download, upload_file=download),
    )
    return managed, workdir


def inputs(workdir, name, **values):
    candidate = workdir / "output/.selfcheck/candidates" / f"{name}.tar.gz"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    # Promotion treats archives as opaque bytes; the existing pack tests cover tar creation.
    candidate.write_bytes(name.encode())
    metrics = dict(op_name="op_test", success=False, correctness_ok=False,
                   ast_check_ok=True, error_type="ascendc_compile_failed",
                   evaluated_candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), **values)
    result = workdir / "judge_out" / name / "metrics.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(metrics))
    return str(candidate), str(result)


def test_host_best_survives_local_deletion_and_matches_attempt_peak(tmp_path, monkeypatch):
    monkeypatch.setenv("POLAR_CASE_PASS_WEIGHT", "0.10")
    managed, workdir = session(tmp_path)
    best = best_submission(managed.session_dir, "op_test")
    local = workdir / "output/submission"

    async def run():
        await retain_best(managed, *inputs(workdir, "initial"))
        # MoeGating: delete only metadata while leaving the failed best.
        (local / ".op_test_impl.best.meta.json").unlink()
        candidate, result = inputs(workdir, "correct")
        data = json.loads(Path(result).read_text())
        data.update(success=True, correctness_ok=True, error_type=None,
                    perf_data={"speedup_vs_torch": 0.41907808755455467})
        Path(result).write_text(json.dumps(data))
        reply = await retain_best(managed, candidate, result)
        assert "best 已更新" in reply["message"]
        correct_record = reply["message"].split("evaluation_record=")[1].splitlines()[0]
        retry = await retain_best(managed, candidate, result)
        assert f"evaluation_record={correct_record}" in retry["message"]
        assert len(evaluation_records(managed.session_dir, "op_test")["records"]) == 2
        assert best.read_bytes() == b"correct"
        # Sum/Cat: delete local best, metadata, candidates and the public mirror.
        shutil.rmtree(workdir / "output")
        shutil.rmtree(managed.session_dir / "submission")
        reply = await retain_best(managed, *inputs(workdir, "regression"))
        assert "best 保持不变" in reply["message"]
        assert best.read_bytes() == b"correct"
        assert (local / best.name).read_bytes() == b"correct"
        # A tied correct candidate keeps the earlier candidate and attempt.
        candidate, result = inputs(workdir, "tied")
        data["evaluated_candidate_sha256"] = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
        Path(result).write_text(json.dumps(data))
        await retain_best(managed, candidate, result)
        assert best.read_bytes() == b"correct"
        # Judge must prefer the private copy even if the agent rewrites the mirror.
        (managed.session_dir / "submission" / best.name).write_bytes(b"wrong")
        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        context = await manager._extract_operator_judge_submission(managed, managed.request.evaluator)
        assert Path(context["submission_host_path"]).read_bytes() == b"correct"
        assert context["submission_used"] == "gateway_best/op_test_impl.best.tar.gz"
        assert not best.is_relative_to(managed.session_dir)
        best.unlink()  # interrupted meta -> best replacement still has a frozen source
        recovered = await manager._extract_operator_judge_submission(managed, managed.request.evaluator)
        assert recovered == context
        # The eager/in-place evaluator path must receive the same protected source.
        managed.cancel_reason = None
        managed.request.evaluator.refresh_runtime = False
        manager._resolve_runtime_spec = lambda request: managed.runtime.spec
        manager._evaluator_env = lambda *args: {}
        manager._remaining_budget = lambda managed: 60
        manager._merge_eval_result = lambda trajectory, result, spec: result

        async def direct(awaitable, managed):
            return await awaitable

        async def evaluate(trajectory, **kwargs):
            assert kwargs["submission_host_path"] == context["submission_host_path"]
            return "evaluated protected source"

        manager._await_with_budget = direct
        manager.evaluators = SimpleNamespace(create=lambda spec: SimpleNamespace(evaluate=evaluate))
        result = await manager._run_eval(managed.request, None, agent_result=None, managed=managed)
        assert result == "evaluated protected source"
        # Native attempt scoring uses exactly the same ladder and first-tie rule.
        bad = "[ascendc-eval] verdict — operator_valid=False ast_check_ok=True correctness_ok=False error_type=ascendc_compile_failed speedup_vs_torch=None"
        good = "[ascendc-eval] verdict — operator_valid=True ast_check_ok=True correctness_ok=True error_type=None speedup_vs_torch=0.41907808755455467"
        verdicts = {"c0": bad, "c1": good, "c2": bad, "c3": good}
        assert best_ordinal({str(i): (i, f"c{i}") for i in range(4)}, verdicts) == 1
        spans = build_spans([(i * 10, i, f"c{i}") for i in range(4)], verdicts, -1, 40)
        meta = json.loads(best.with_name(".op_test_impl.best.meta.json").read_text())
        ledger = evaluation_records(managed.session_dir, "op_test")
        assert ledger["best_record_id"] == correct_record == meta["evaluation_record_id"]
        assert ledger["records"][correct_record]["score"] == meta["reward_score"]
        assert len(ledger["records"]) == 4  # retry was not another evaluation
        assert spans[1][3] == meta["reward_score"] == verdict_score(data)
        assert spans[0][3] == spans[2][3] == 0.1
        assert spans[3][3] == spans[1][3]
        assert len(spans) == 4  # retaining a candidate creates no extra attempt

    asyncio.run(run())


def test_endpoint_rejects_mismatch_and_paths_without_changing_best(tmp_path, monkeypatch):
    managed, workdir = session(tmp_path)
    manager = GatewayNodeManager.__new__(GatewayNodeManager)

    async def get_session(session_id):
        return managed if session_id == "test" else None

    manager._dispatcher = SimpleNamespace(get_session=get_session)
    monkeypatch.setattr(server, "_state", SimpleNamespace(node_manager=manager))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            candidate, metrics = inputs(workdir, "first")
            url = "/sessions/test/operator_best"
            reply = await client.post(url, json=dict(candidate=candidate, metrics=metrics))
            assert reply.status_code == 200, reply.text
            best = best_submission(managed.session_dir, "op_test")
            original = best.read_bytes()
            Path(candidate).write_bytes(b"changed-after-evaluation")
            reply = await client.post(url, json=dict(candidate=candidate, metrics=metrics))
            assert reply.status_code == 400
            assert best.read_bytes() == original
            for bad in ("/etc/passwd", str(workdir / "../outside.tar.gz")):
                reply = await client.post(url, json=dict(candidate=bad, metrics=metrics))
                assert reply.status_code == 400
            reply = await client.post(url, json={"candidate": [], "metrics": metrics})
            assert reply.status_code == 400
            managed.stage = SessionStage.POSTRUN
            reply = await client.post(url, json=dict(candidate=candidate, metrics=metrics))
            assert reply.status_code == 404
            assert best.read_bytes() == original

    asyncio.run(run())


def test_infra_is_not_promoted_and_cleanup_respects_keep(tmp_path, monkeypatch):
    managed, workdir = session(tmp_path)

    async def run():
        candidate, result = inputs(workdir, "infra")
        data = json.loads(Path(result).read_text())
        data["error_type"] = "npu_runtime_unavailable"
        Path(result).write_text(json.dumps(data))
        reply = await retain_best(managed, candidate, result)
        assert "best 未更新" in reply["message"]
        assert not best_submission(managed.session_dir, "op_test").exists()
        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        monkeypatch.setenv("POLAR_KEEP_SESSION_DIR", "1")
        await manager._remove_session_dir_best_effort(managed.session_dir, "test")
        assert best_root(managed.session_dir).exists()
        monkeypatch.delenv("POLAR_KEEP_SESSION_DIR")
        await manager._remove_session_dir_best_effort(managed.session_dir, "test")
        assert not best_root(managed.session_dir).exists()

    asyncio.run(run())


def test_pack_client_uses_gateway_and_reports_transport_failure(tmp_path):
    managed, workdir = session(tmp_path)
    candidate, metrics = inputs(workdir, "client")
    fail = False
    failures = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if fail:
                failures.append(body)
                self.send_response(503)
                self.end_headers()
                return
            result = asyncio.run(retain_best(managed, body["candidate"], body["metrics"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    env = {**os.environ, "WORKDIR": str(workdir),
           "POLAR_OPERATOR_BEST_URL": f"http://127.0.0.1:{http.server_port}/operator_best"}
    command = ["bash", str(PACK), "op_test", "--promote", "--candidate", candidate, "--metrics", metrics]
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "Gateway 留存" in result.stdout
        assert best_submission(managed.session_dir, "op_test").read_bytes() == b"client"
        fail = True
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode != 0
        assert "Gateway best 留存失败" in result.stderr
        assert len(failures) == 2
        assert best_submission(managed.session_dir, "op_test").read_bytes() == b"client"
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


def test_banded_host_best_and_attempts_share_recorded_scores(tmp_path):
    managed, workdir = session(tmp_path)
    managed.request.evaluator.config["reward_scheme"] = "correctness_banded"

    async def run():
        first = inputs(workdir, "missing-cases")
        data = json.loads(Path(first[1]).read_text())
        data["error_type"] = "correctness_failed"
        Path(first[1]).write_text(json.dumps(data))
        await retain_best(managed, *first)
        best = best_submission(managed.session_dir, "op_test")
        second = inputs(workdir, "two-cases", cases_passed=2, cases_total=10)
        data = json.loads(Path(second[1]).read_text())
        data["error_type"] = "correctness_failed"
        Path(second[1]).write_text(json.dumps(data))
        await retain_best(managed, *second)
        assert best.read_bytes() == b"two-cases"
        records = evaluation_records(managed.session_dir, "op_test")
        assert sorted(r["score"] for r in records["records"].values()) == pytest.approx([.03, .044])
        assert records["records"][records["best_record_id"]]["score"] == pytest.approx(.044)
        correct = inputs(workdir, "correct")
        data = json.loads(Path(correct[1]).read_text())
        data.update(correctness_ok=True, error_type="benchmark_failed")
        Path(correct[1]).write_text(json.dumps(data))
        await retain_best(managed, *correct)
        assert best.read_bytes() == b"correct"
        retry = await retain_best(managed, *second)
        assert "best 保持不变" in retry["message"]
        assert best.read_bytes() == b"correct"
        records = evaluation_records(managed.session_dir, "op_test")
        assert len(records["records"]) == 3
        assert records["records"][records["best_record_id"]]["score"] == pytest.approx(.9)

    asyncio.run(run())
