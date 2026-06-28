#!/usr/bin/env python3
"""Probe that a Polar gateway can start a minimal DockerRuntime session."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


TERMINAL_STATUSES = {"COMPLETED", "ERROR", "TIMEOUT"}
DEFAULT_PROBE_WORKDIR = "/polar/session"


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


def _runtime_spec(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.pool:
        kwargs["ascend"] = {
            "pool": args.pool,
            "lock_dir": args.lock_dir,
            "lease_at_start": False,
        }
    if args.skills_dir:
        kwargs["volumes"] = [f"{args.skills_dir}:/opt/canonical:ro"]
    return {
        "backend": "docker",
        "image": args.image,
        "network": "host",
        "workdir": args.workdir,
        "kwargs": kwargs,
        "env": {},
        "prepare": [],
    }


def _dispatch_payload(args: argparse.Namespace, session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "task_id": args.task_id,
        "instruction": "Polar gateway DockerRuntime preflight probe.",
        "remaining_timeout_seconds": args.session_timeout,
        "runtime": _runtime_spec(args),
        "agent": {
            "harness": "shell",
            "custom_shell": {
                "command": args.command,
                "cwd": args.workdir,
            },
        },
        "builder": {"strategy": "per_request"},
        "evaluator": None,
        "metadata": {"probe": "polar_gateway_runtime"},
    }


def _poll_session(args: argparse.Namespace, session_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout
    session_url = f"{args.gateway_url}/sessions/{urllib.parse.quote(session_id, safe='')}"
    last: dict[str, Any] = {}
    while True:
        last = _request_json("GET", session_url, timeout=args.request_timeout)
        status = str(last.get("status") or "")
        if status in TERMINAL_STATUSES:
            return last
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"session {session_id} did not reach a terminal status within "
                f"{args.timeout:.1f}s; last status={status!r}"
            )
        time.sleep(args.poll_interval)


def _delete_session(args: argparse.Namespace, session_id: str) -> None:
    session_url = f"{args.gateway_url}/sessions/{urllib.parse.quote(session_id, safe='')}"
    try:
        _request_json("DELETE", session_url, timeout=args.request_timeout)
    except Exception:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.gateway_url = args.gateway_url.rstrip("/")
    session_id = args.session_id or f"polar-runtime-probe-{uuid.uuid4().hex[:12]}"
    _request_json("GET", f"{args.gateway_url}/health", timeout=args.request_timeout)
    _request_json(
        "POST",
        f"{args.gateway_url}/sessions",
        payload=_dispatch_payload(args, session_id),
        timeout=args.request_timeout,
    )
    result: dict[str, Any] | None = None
    try:
        result = _poll_session(args, session_id)
    except Exception:
        if not args.keep_session and args.delete_on_failure:
            _delete_session(args, session_id)
        raise

    status = str(result.get("status") or "")
    session_result = result.get("result") if isinstance(result.get("result"), dict) else {}
    error = result.get("error") or session_result.get("error")
    if status != "COMPLETED":
        if not args.keep_session and args.delete_on_failure:
            _delete_session(args, session_id)
        raise RuntimeError(
            f"gateway runtime probe failed: session={session_id} "
            f"status={status} error={error!r}"
        )
    if not args.keep_session:
        _delete_session(args, session_id)
    return {
        "gateway_url": args.gateway_url,
        "session_id": session_id,
        "status": status,
        "image": args.image,
        "pool": args.pool,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--image", default="sandbox:v1")
    parser.add_argument("--pool", default="")
    parser.add_argument("--lock-dir", default="/dev/shm/polar-npu-locks")
    parser.add_argument("--skills-dir", default="")
    parser.add_argument("--command", default="true")
    parser.add_argument("--workdir", default=DEFAULT_PROBE_WORKDIR)
    parser.add_argument("--task-id", default="polar-gateway-runtime-probe")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--session-timeout", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--keep-session", action="store_true")
    parser.add_argument("--delete-on-failure", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    else:
        print(
            "[check] Polar gateway runtime probe completed: "
            f"session={result['session_id']} image={result['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
