#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ports_from_env() -> dict[str, int]:
    """Ports come from the profile via load_polar_profile.py's exports.

    以前这里写死四个端口，与 profile 漂移后 cleanup/status 会打到不存在的端口上
    （observer 就漂过：这里 18088，profile 里 18189）。start_hostctl.sh 已 source
    loader，所以正常路径下环境变量都在；下面的字面量只是脱离该路径时的兜底。
    """
    ports = {
        "polar_rollout": int(os.environ.get("POLAR_ROLLOUT_PORT") or 8080),
        "polar_gateway": int(os.environ.get("POLAR_GATEWAY_PORT") or 8100),
        "observer": int(os.environ.get("POLAR_OBSERVER_PORT") or 18088),
    }
    # stale_gateway 可以为空（同机有别的 polar 时 profile 里写 []），空则不注册该名字。
    stale = [
        p.strip()
        for p in (os.environ.get("POLAR_EXTRA_STALE_GATEWAY_PORTS") or "").split(",")
        if p.strip()
    ]
    if stale:
        ports["stale_gateway"] = int(stale[0])
    return ports


PORTS = _ports_from_env()
DEFAULT_SAFE_CMD_PATTERNS = (
    "polar_rollout_observer.py",
    "serve_gateway",
    "serve_rollout",
    "polar.cli",
)
POLL_SECONDS = 0.5


@dataclass
class ProcInfo:
    pid: int
    cmdline: str


def _now() -> float:
    return time.time()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_token(root: Path) -> str:
    token_path = root / "hostctl" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    return token


def _proc_cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _tcp_listen_inodes_for_port(port: int) -> set[str]:
    want = f"{port:04X}"
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.exists():
            continue
        for line in table.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            local = fields[1]
            state = fields[3]
            inode = fields[9]
            if state == "0A" and local.rsplit(":", 1)[-1].upper() == want:
                inodes.add(inode)
    return inodes


def _pids_for_inodes(inodes: set[str]) -> list[int]:
    if not inodes:
        return []
    pids: set[int] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fd_dir = proc / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                pids.add(int(proc.name))
                break
    return sorted(pids)


def _listeners(port: int) -> list[ProcInfo]:
    infos = []
    for pid in _pids_for_inodes(_tcp_listen_inodes_for_port(port)):
        infos.append(ProcInfo(pid=pid, cmdline=_proc_cmdline(pid)))
    return infos


def _is_safe_process(info: ProcInfo, safe_patterns: tuple[str, ...]) -> bool:
    return any(pattern in info.cmdline for pattern in safe_patterns)


def _normalize_ports(args: dict[str, Any] | None) -> list[tuple[str, int]]:
    raw = (args or {}).get("ports")
    if raw is None:
        names = ["polar_rollout", "polar_gateway", "observer", "stale_gateway"]
    elif isinstance(raw, str):
        names = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, list):
        names = [str(p).strip() for p in raw if str(p).strip()]
    else:
        raise ValueError("ports must be omitted, a comma string, or a list")
    out = []
    for name in names:
        if name in PORTS:
            out.append((name, PORTS[name]))
        elif name.isdigit() and int(name) in PORTS.values():
            out.append((name, int(name)))
        else:
            raise ValueError(f"port is not allowed: {name}")
    return out


def _cleanup_ports(
    args: dict[str, Any] | None = None,
    *,
    safe_patterns: tuple[str, ...] = DEFAULT_SAFE_CMD_PATTERNS,
) -> dict[str, Any]:
    results = []
    for name, port in _normalize_ports(args):
        before = _listeners(port)
        unsafe = [p for p in before if not _is_safe_process(p, safe_patterns)]
        if unsafe:
            results.append({
                "name": name,
                "port": port,
                "status": "refused_unsafe_process",
                "listeners": [p.__dict__ for p in unsafe],
            })
            continue

        for proc in before:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if before:
            time.sleep(float((args or {}).get("term_wait_seconds", 3)))
        after_term = _listeners(port)
        for proc in after_term:
            if _is_safe_process(proc, safe_patterns):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.2)
        after = _listeners(port)
        results.append({
            "name": name,
            "port": port,
            "status": "free" if not after else "still_occupied",
            "terminated": [p.__dict__ for p in before],
            "remaining": [p.__dict__ for p in after],
        })
    return {"ports": results}


def _run(
    root: Path,
    cmd: list[str],
    *,
    timeout: int = 120,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = _now()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "elapsed_seconds": _now() - started,
    }


def _status(root: Path) -> dict[str, Any]:
    pid_files = {}
    for name in ("rollout", "gateway", "observer", "pipeline_budget_watcher"):
        path = root / f"{name}.pid"
        pid = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        pid_files[name] = {
            "pid": int(pid) if pid.isdigit() else None,
            "alive": bool(pid.isdigit() and Path(f"/proc/{pid}").exists()),
        }
    return {
        "pid_files": pid_files,
        "ports": {
            name: [p.__dict__ for p in _listeners(port)]
            for name, port in PORTS.items()
        },
    }


def _restart_polar_gateway(
    root: Path,
    deploy_dir: Path,
    args: dict[str, Any] | None,
    safe_patterns: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if (args or {}).get("cleanup_ports", True):
        payload["cleanup"] = _cleanup_ports(
            {"ports": ["polar_rollout", "polar_gateway", "stale_gateway"]},
            safe_patterns=safe_patterns,
        )
    payload["restart"] = _run(
        deploy_dir,
        ["bash", str(deploy_dir / "restart_polar_host.sh")],
        timeout=180,
        extra_env={"POLAR_SKIP_INTERNAL_PORT_CLEANUP": "1"},
    )
    return payload


def _restart_observer(
    root: Path,
    deploy_dir: Path,
    args: dict[str, Any] | None,
    safe_patterns: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if (args or {}).get("cleanup_ports", True):
        payload["cleanup"] = _cleanup_ports({"ports": ["observer"]}, safe_patterns=safe_patterns)
    payload["stop"] = _run(deploy_dir, ["bash", str(deploy_dir / "stop_observer.sh")], timeout=60)
    payload["start"] = _run(deploy_dir, ["bash", str(deploy_dir / "start_observer.sh")], timeout=60)
    return payload


def _restart_stack(
    root: Path,
    deploy_dir: Path,
    args: dict[str, Any] | None,
    safe_patterns: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if (args or {}).get("cleanup_ports", True):
        cleanup_args = {"ports": (args or {}).get("ports")} if (args or {}).get("ports") else None
        payload["cleanup"] = _cleanup_ports(cleanup_args, safe_patterns=safe_patterns)
    payload["restart"] = _run(
        deploy_dir,
        ["bash", str(deploy_dir / "restart_polar_host.sh")],
        timeout=180,
        extra_env={"POLAR_SKIP_INTERNAL_PORT_CLEANUP": "1"},
    )
    return payload


def _tail_logs(root: Path, args: dict[str, Any] | None) -> dict[str, Any]:
    name = str((args or {}).get("name", "gateway"))
    allowed = {
        "gateway": root / "logs" / "gateway.log",
        "rollout": root / "logs" / "rollout.log",
        "observer": root / "logs" / "observer.nohup.log",
        "watcher": root / "logs" / "pipeline_budget_watcher.log",
    }
    if name not in allowed:
        raise ValueError(f"log is not allowed: {name}")
    lines = int((args or {}).get("lines", 120))
    lines = max(1, min(lines, 1000))
    path = allowed[name]
    if not path.exists():
        return {"path": str(path), "text": ""}
    return {"path": str(path), "text": "\n".join(path.read_text(errors="replace").splitlines()[-lines:])}


def _execute(
    root: Path,
    deploy_dir: Path,
    request: dict[str, Any],
    safe_patterns: tuple[str, ...],
) -> dict[str, Any]:
    action = request.get("action")
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    if action == "status":
        return _status(root)
    if action == "cleanup_ports":
        return _cleanup_ports(args, safe_patterns=safe_patterns)
    if action == "restart_polar_gateway":
        return _restart_polar_gateway(root, deploy_dir, args, safe_patterns)
    if action == "restart_observer":
        return _restart_observer(root, deploy_dir, args, safe_patterns)
    if action == "restart_polar_stack":
        return _restart_stack(root, deploy_dir, args, safe_patterns)
    if action == "tail_logs":
        return _tail_logs(root, args)
    raise ValueError(f"action is not allowed: {action}")


def _handle_request(
    root: Path,
    deploy_dir: Path,
    token: str,
    path: Path,
    safe_patterns: tuple[str, ...],
) -> None:
    request_id = path.stem
    result_path = root / "hostctl" / "results" / f"{request_id}.json"
    lock_path = root / "hostctl" / "hostctl.lock"
    started = _now()
    try:
        request = _read_json(path)
        if request.get("token") != token:
            raise PermissionError("invalid token")
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            output = _execute(root, deploy_dir, request, safe_patterns)
        payload = {
            "id": request_id,
            "ok": True,
            "action": request.get("action"),
            "started_at_unix": started,
            "ended_at_unix": _now(),
            "result": output,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "id": request_id,
            "ok": False,
            "started_at_unix": started,
            "ended_at_unix": _now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    _json_dump(result_path, payload)
    done_dir = root / "hostctl" / "processed"
    done_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.replace(done_dir / path.name)
    except OSError:
        path.unlink(missing_ok=True)


def serve(root: Path, deploy_dir: Path, safe_patterns: tuple[str, ...]) -> int:
    token = _ensure_token(root)
    request_dir = root / "hostctl" / "requests"
    for rel in ("requests", "results", "processed", "logs"):
        (root / "hostctl" / rel).mkdir(parents=True, exist_ok=True)
    log_path = root / "hostctl" / "logs" / "hostctl.log"
    with log_path.open("a", encoding="utf-8") as log:
        print(f"hostctl_server pid={os.getpid()} root={root}", file=log, flush=True)
        print(f"Hostctl server ready. root={root} token={root / 'hostctl' / 'token'}")
        while True:
            for path in sorted(request_dir.glob("*.json")):
                print(f"handling {path.name}", file=log, flush=True)
                _handle_request(root, deploy_dir, token, path, safe_patterns)
            time.sleep(POLL_SECONDS)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "output" / "ascend_operator"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(_default_root()))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--deploy-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    repo_root = Path(args.repo_root).resolve()
    deploy_dir = Path(args.deploy_dir).resolve()
    safe_patterns = DEFAULT_SAFE_CMD_PATTERNS + (str(repo_root), str(deploy_dir), str(root))
    return serve(root, deploy_dir, safe_patterns)


if __name__ == "__main__":
    raise SystemExit(main())
