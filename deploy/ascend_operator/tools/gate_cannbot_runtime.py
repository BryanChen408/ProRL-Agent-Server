#!/usr/bin/env python3
"""Live gate for the CANNBot operator runtime path.

This intentionally exercises Polar through the public rollout endpoint instead
of importing service internals:

1. submit one fixture operator sample and wait for completion;
2. verify the run produced CANNBot validation artifacts;
3. submit a small parallel batch and repeat the same checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TERMINAL_TASK_STATUSES = {"completed", "failed"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON body: {body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned unexpected JSON: {parsed!r}")
    return parsed


def _load_rollout_url(repo: Path, explicit: str) -> str:
    if explicit:
        return explicit.rstrip("/")
    profile = repo / "deploy/ascend_operator/profile.yaml"
    text = profile.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("rollout_url:"):
            return stripped.split(":", 1)[1].strip().rstrip("/")
    return "http://127.0.0.1:8080"


def _load_source(repo: Path, op: str) -> str:
    path = repo / "deploy/ascend_operator/fixtures/operator_assets/op_tasks" / f"{op}.py"
    return path.read_text(encoding="utf-8")


def _submit_one(
    *,
    rollout_url: str,
    repo: Path,
    op: str,
    suffix: str,
    timeout_seconds: float,
    request_timeout: float,
) -> str:
    source = _load_source(repo, op)
    task_id = f"gate-{suffix}-{int(time.time())}"
    payload = {
        "task_id": task_id,
        "instruction": (
            f"Implement the Ascend Triton operator `{op}`. "
            f"The prepared reference task is at `input/{op}.py`. "
            "Follow `./CLAUDE.md`."
        ),
        "num_samples": 1,
        "timeout_seconds": timeout_seconds,
        "sample": {
            "op_name": op,
            "task_source": source,
            "task_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
        "metadata": {"gate": suffix},
    }
    response = _request_json(
        "POST",
        f"{rollout_url}/rollout/operator_samples/submit",
        payload=payload,
        timeout=request_timeout,
    )
    print(f"[submit] {task_id} -> {response}")
    return task_id


def _wait_task(
    *,
    rollout_url: str,
    task_id: str,
    timeout: float,
    poll_interval: float,
    request_timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        data = _request_json(
            "GET",
            f"{rollout_url}/rollout/task/{urllib.parse.quote(task_id, safe='')}",
            timeout=request_timeout,
        )
        compact = {
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "total": data.get("total_sessions"),
            "completed": data.get("completed_sessions"),
            "results": len(data.get("results") or []),
        }
        if compact != last:
            print(f"[poll] {json.dumps(compact, ensure_ascii=False)}")
            last = compact
        if str(data.get("status")) in TERMINAL_TASK_STATUSES:
            return data
        time.sleep(poll_interval)
    raise TimeoutError(f"task {task_id} did not finish within {timeout:.0f}s")


def _latest_run_dir(repo: Path) -> Path:
    runs_root = repo / "output/ascend_operator/runs"
    runs = [path for path in runs_root.glob("*") if path.is_dir()]
    if not runs:
        raise RuntimeError(f"no run directory found under {runs_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _artifact_counts(session_dir: Path) -> dict[str, int]:
    return {
        "budget": len(list(session_dir.rglob("pipeline_budget_status.json"))),
        "verify": len(list(session_dir.rglob("verify_result*.json"))),
        "perf": len(list(session_dir.rglob("perf_result*.json"))),
        "generated": len(list(session_dir.rglob("generated_code.py"))),
        "optimized": len(list(session_dir.rglob("optimized_code.py"))),
    }


def _inspect_run(run_dir: Path, *, min_new_sessions: int) -> list[dict[str, Any]]:
    sessions_root = run_dir / "polar_sessions"
    session_dirs = sorted(sessions_root.glob("session-*"), key=lambda path: path.stat().st_mtime)
    if len(session_dirs) < min_new_sessions:
        raise RuntimeError(
            f"expected at least {min_new_sessions} session dirs in {sessions_root}, "
            f"found {len(session_dirs)}"
        )
    print(f"[inspect] run_dir={run_dir}")
    print(f"[inspect] sessions={len(session_dirs)}")
    summaries: list[dict[str, Any]] = []
    for session_dir in session_dirs[-min_new_sessions:]:
        counts = _artifact_counts(session_dir)
        summary = {"session": session_dir.name, **counts}
        summaries.append(summary)
        print(f"[session] {json.dumps(summary, ensure_ascii=False, sort_keys=True)}")
    return summaries


def _assert_task_completed(task: dict[str, Any]) -> None:
    if task.get("status") != "completed":
        raise RuntimeError(f"task did not complete: {task}")
    total = int(task.get("total_sessions") or 0)
    completed = int(task.get("completed_sessions") or 0)
    if total < 1 or completed != total:
        raise RuntimeError(f"task completed count mismatch: total={total} completed={completed}")


def _assert_has_validation_artifact(summaries: list[dict[str, Any]]) -> None:
    missing = [
        item["session"]
        for item in summaries
        if int(item.get("verify") or 0) < 1
    ]
    if missing:
        raise RuntimeError(f"sessions missing verify_result artifacts: {missing}")


def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    rollout_url = _load_rollout_url(repo, args.rollout_url)
    print(f"[gate] repo={repo}")
    print(f"[gate] rollout_url={rollout_url}")
    _request_json("GET", f"{rollout_url}/health", timeout=args.request_timeout)

    print("== Gate 2: single session ==")
    single_id = _submit_one(
        rollout_url=rollout_url,
        repo=repo,
        op=args.op,
        suffix="single",
        timeout_seconds=args.session_timeout,
        request_timeout=args.request_timeout,
    )
    single = _wait_task(
        rollout_url=rollout_url,
        task_id=single_id,
        timeout=args.wait_timeout,
        poll_interval=args.poll_interval,
        request_timeout=args.request_timeout,
    )
    _assert_task_completed(single)
    single_run = _latest_run_dir(repo)
    single_summaries = _inspect_run(single_run, min_new_sessions=1)
    _assert_has_validation_artifact(single_summaries)

    print("== Gate 3: parallel sessions ==")
    task_ids = [
        _submit_one(
            rollout_url=rollout_url,
            repo=repo,
            op=args.op,
            suffix=f"parallel-{idx}",
            timeout_seconds=args.session_timeout,
            request_timeout=args.request_timeout,
        )
        for idx in range(args.parallel)
    ]
    for task_id in task_ids:
        task = _wait_task(
            rollout_url=rollout_url,
            task_id=task_id,
            timeout=args.wait_timeout,
            poll_interval=args.poll_interval,
            request_timeout=args.request_timeout,
        )
        _assert_task_completed(task)
    parallel_run = _latest_run_dir(repo)
    parallel_summaries = _inspect_run(parallel_run, min_new_sessions=args.parallel)
    _assert_has_validation_artifact(parallel_summaries)

    print("[gate] PASS")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--rollout-url", default="")
    parser.add_argument("--op", default="polar_smoke_identity")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--session-timeout", type=float, default=1800.0)
    parser.add_argument("--wait-timeout", type=float, default=2400.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run(args)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
