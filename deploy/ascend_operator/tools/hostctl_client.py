#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_token(root: Path) -> str:
    token_path = root / "hostctl" / "token"
    if not token_path.exists():
        raise RuntimeError(f"hostctl token not found: {token_path}; start hostctl on the host first")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"empty hostctl token: {token_path}")
    return token


def _build_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.ports:
        payload["ports"] = args.ports
    if args.no_cleanup:
        payload["cleanup_ports"] = False
    if args.action == "tail_logs":
        payload["name"] = args.log_name
        payload["lines"] = args.lines
    return payload


def request(root: Path, action: str, args: dict[str, Any], timeout: float) -> dict[str, Any]:
    request_id = f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(4)}"
    req_path = root / "hostctl" / "requests" / f"{request_id}.json"
    result_path = root / "hostctl" / "results" / f"{request_id}.json"
    _json_dump(req_path, {
        "id": request_id,
        "token": _load_token(root),
        "action": action,
        "args": args,
        "created_at_unix": time.time(),
    })

    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_path.exists():
            return _read_json(result_path)
        time.sleep(0.25)
    raise TimeoutError(f"hostctl request timed out after {timeout:.1f}s: {request_id}")


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "output" / "ascend_operator"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a Polar host control request through /home/docker.")
    parser.add_argument(
        "action",
        choices=[
            "status",
            "cleanup_ports",
            "restart_polar_gateway",
            "restart_observer",
            "restart_polar_stack",
            "tail_logs",
        ],
    )
    parser.add_argument("--root", default=str(_default_root()))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--port", dest="ports", action="append", help="Allowed port name or number.")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--log-name", default="gateway", choices=["gateway", "rollout", "observer", "watcher"])
    parser.add_argument("--lines", type=int, default=120)
    parsed = parser.parse_args(argv)

    root = Path(parsed.root).resolve()
    try:
        result = request(root, parsed.action, _build_args(parsed), parsed.timeout)
    except Exception as exc:  # noqa: BLE001
        print("RESULT=FAIL")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("RESULT=PASS" if result.get("ok") else "RESULT=FAIL")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
