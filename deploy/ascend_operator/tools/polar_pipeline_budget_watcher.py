#!/usr/bin/env python3
"""Cancel Polar sessions that keep running the operator eval pipeline too long.

The watcher intentionally stays outside Polar's critical path. It reads the same
gateway completion stream as the rollout observer and uses DELETE /sessions/{id}
only after the actual tools/triton_eval_pipeline.sh call count exceeds the
configured budget.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GATEWAY = "http://127.0.0.1:8100"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "output" / "ascend_operator"
DEFAULT_SESSION_BASE_DIR = Path(os.environ.get("POLAR_SESSION_BASE_DIR", DEFAULT_ROOT / "polar_sessions"))
PIPELINE_MARKER = "tools/triton_eval_pipeline.sh"
PIPELINE_STATUS_NAME = "pipeline_budget_status.json"
SUCCESS_RE = re.compile(
    r"(\[triton-eval\]\s+done\s+.*success=true|verdict\s+.*success=True|cached verdict\s+.*success=True)",
    re.IGNORECASE,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(message: str, log_file: Path | None = None) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _gateway_request(
    base_url: str,
    method: str,
    path: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> Any:
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _gateway_get(base_url: str, path: str, timeout: float) -> Any:
    return _gateway_request(base_url, "GET", path, timeout)


def _gateway_delete(base_url: str, path: str, timeout: float) -> Any:
    return _gateway_request(base_url, "DELETE", path, timeout)


def _text_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                typ = item.get("type")
                if typ == "text":
                    parts.append(str(item.get("text", "")))
                elif typ == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif typ == "tool_use":
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _response_message(record: dict[str, Any]) -> dict[str, Any] | None:
    resp = record.get("response")
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = (choices[0] or {}).get("message")
    return msg if isinstance(msg, dict) else None


def _tool_uses(content: Any) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input") or {},
                    }
                )
    return uses


def _tool_results(content: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(
                    {
                        "tool_use_id": block.get("tool_use_id"),
                        "content": str(block.get("content", "")),
                    }
                )
    return results


def _response_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in msg.get("tool_calls") or []:
        fn = item.get("function") or {}
        args: Any = fn.get("arguments") or item.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        out.append({"id": item.get("id"), "name": fn.get("name") or item.get("name"), "input": args})
    return out


def _tool_command(tool: dict[str, Any]) -> str:
    if str(tool.get("name") or "") != "Bash":
        return ""
    inp = tool.get("input") if isinstance(tool.get("input"), dict) else {}
    return str(inp.get("command") or "")


def _is_pipeline_command(command: str) -> bool:
    for segment in re.split(r"\s*(?:\||&&|\|\||;)\s*", command.strip()):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if not parts:
            continue
        executable = parts[0].replace("\\", "/")
        if executable in {"bash", "sh", "/bin/bash", "/bin/sh"} and len(parts) > 1:
            script = parts[1].replace("\\", "/")
        else:
            script = executable
        if script.endswith(PIPELINE_MARKER) or script.endswith("/triton_eval_pipeline.sh"):
            return True
    return False


def _is_pipeline_feedback(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return (
        "[pipeline-budget]" in low
        or "[triton-eval]" in low
        or "完整错误已写入" in text
        or "judge_out/metrics_error.log" in low
        or "success=true" in low
    )


@dataclass
class PipelineCall:
    index: int
    turn: int
    command: str
    result: str
    success: bool
    completed: bool


@dataclass
class BudgetState:
    session_id: str
    pipeline_calls: list[PipelineCall]
    generation_calls: int
    optimization_calls: int
    first_success_index: int | None


def _extract_pipeline_calls(record: dict[str, Any]) -> list[PipelineCall]:
    req = record.get("original_request") if isinstance(record.get("original_request"), dict) else {}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    current = _response_message(record)
    calls: list[PipelineCall] = []
    pending_tools: list[tuple[int, dict[str, Any]]] = []
    turn = 0

    def append_matching_results(results: list[dict[str, Any]]) -> None:
        nonlocal pending_tools
        result_by_id = {
            item.get("tool_use_id"): str(item.get("content") or "")
            for item in results
            if item.get("tool_use_id")
        }
        remaining: list[tuple[int, dict[str, Any]]] = []
        for tool_turn, tool in pending_tools:
            command = _tool_command(tool)
            if not _is_pipeline_command(command):
                remaining.append((tool_turn, tool))
                continue
            result = result_by_id.get(tool.get("id"))
            if result is None:
                remaining.append((tool_turn, tool))
                continue
            if not _is_pipeline_feedback(result):
                continue
            calls.append(
                PipelineCall(
                    index=len(calls) + 1,
                    turn=tool_turn,
                    command=command,
                    result=result,
                    success=bool(SUCCESS_RE.search(result)),
                    completed=True,
                )
            )
        pending_tools = remaining

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            turn += 1
            pending_tools = [(turn, tool) for tool in _tool_uses(content)]
        elif role == "user":
            append_matching_results(_tool_results(content))

    if current:
        turn += 1
        for tool in _response_tool_calls(current):
            command = _tool_command(tool)
            if _is_pipeline_command(command):
                calls.append(
                    PipelineCall(
                        index=len(calls) + 1,
                        turn=turn,
                        command=command,
                        result="",
                        success=False,
                        completed=False,
                    )
                )
    return calls


def analyze_budget(session_id: str, record: dict[str, Any]) -> BudgetState:
    calls = _extract_pipeline_calls(record)
    first_success = next((call.index for call in calls if call.completed and call.success), None)
    if first_success is None:
        gen = len(calls)
        opt = 0
    else:
        gen = first_success
        opt = sum(1 for call in calls if call.index > first_success)
    return BudgetState(
        session_id=session_id,
        pipeline_calls=calls,
        generation_calls=gen,
        optimization_calls=opt,
        first_success_index=first_success,
    )


def _active_session_ids(gateway: str, timeout: float) -> list[str]:
    ids: set[str] = set()
    for path in ("/health", "/sessions"):
        try:
            data = _gateway_get(gateway, path, timeout)
        except Exception:
            continue
        if isinstance(data, dict):
            items = []
            if isinstance(data.get("active_sessions"), list):
                items.extend(data["active_sessions"])
            if isinstance(data.get("sessions"), list):
                items.extend(data["sessions"])
            for item in items:
                if isinstance(item, dict) and item.get("session_id"):
                    status = str(item.get("status") or "")
                    if status in {"", "RUNNING", "INITIALIZING", "READY", "BUILDING", "EVALUATING"}:
                        ids.add(str(item["session_id"]))
    return sorted(ids)


def _gateway_completions(gateway: str, session_id: str, timeout: float) -> list[dict[str, Any]]:
    data = _gateway_get(
        gateway,
        f"/sessions/{urllib.parse.quote(session_id, safe='')}/completions",
        timeout,
    )
    records: Any
    if isinstance(data, dict):
        records = data.get("completions") or data.get("records") or []
    else:
        records = data
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict)]


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _pipeline_status_path(
    root: Path,
    session_id: str,
    session_base_dir: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    results_dir = root / "rollout_results"
    if results_dir.is_dir():
        candidates.extend(
            results_dir.glob(
                f"task_*/sessions/{session_id}/artifacts/{PIPELINE_STATUS_NAME}"
            )
        )
        candidates.extend(
            results_dir.glob(
                f"run_*/task_*/sessions/{session_id}/artifacts/{PIPELINE_STATUS_NAME}"
            )
        )

    if session_base_dir is not None and session_base_dir.is_dir():
        for path in list(session_base_dir.glob(f"session-*/artifacts/{PIPELINE_STATUS_NAME}")) + list(
            session_base_dir.glob(f"run_*/session-*/artifacts/{PIPELINE_STATUS_NAME}")
        ):
            data = _safe_load_json(path)
            if data is not None and str(data.get("session_id") or "") == session_id:
                candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_pipeline_status(
    root: Path,
    session_id: str,
    session_base_dir: Path | None = None,
) -> dict[str, Any] | None:
    path = _pipeline_status_path(root, session_id, session_base_dir)
    if path is None:
        return None
    data = _safe_load_json(path)
    if data is None:
        return None
    data["_status_path"] = str(path)
    return data


def _latest_disk_completion(root: Path, session_id: str) -> dict[str, Any] | None:
    results_dir = root / "rollout_results"
    if not results_dir.is_dir():
        return None
    candidates = sorted(results_dir.glob(f"task_*/sessions/{session_id}/completions/*.json"))
    candidates.extend(
        sorted(results_dir.glob(f"run_*/task_*/sessions/{session_id}/completions/*.json"))
    )
    if not candidates:
        return None
    return _safe_load_json(candidates[-1])


def latest_completion_record(gateway: str, root: Path, session_id: str, timeout: float) -> dict[str, Any] | None:
    try:
        records = _gateway_completions(gateway, session_id, timeout)
    except Exception:
        records = []
    if records:
        return records[-1]
    return _latest_disk_completion(root, session_id)


def should_cancel(state: BudgetState, gen_max: int, opt_max: int) -> tuple[bool, str]:
    if state.first_success_index is None:
        if state.generation_calls > gen_max:
            return True, f"generation pipeline calls {state.generation_calls}>{gen_max}"
        return False, ""
    if state.optimization_calls > opt_max:
        return True, f"optimization pipeline calls {state.optimization_calls}>{opt_max}"
    return False, ""


def should_cancel_from_status(status: dict[str, Any] | None) -> tuple[bool, str]:
    if not status:
        return False, "pipeline status missing"
    try:
        attempt = int(status.get("attempt") or 0)
        limit = int(status.get("limit") or 0)
    except Exception:
        return False, "pipeline status has invalid attempt/limit"
    if not (status.get("limit_exhausted") is True and limit > 0 and attempt > limit):
        return False, (
            f"pipeline status within budget phase={status.get('phase')} "
            f"attempt={attempt}/{limit}"
        )
    return True, (
        f"pipeline budget exceeded phase={status.get('phase')} "
        f"attempt={attempt}>{limit} status={status.get('_status_path')}"
    )


def run_once(args: argparse.Namespace, cancelled: set[str]) -> None:
    ids = _active_session_ids(args.gateway, args.timeout)
    for session_id in ids:
        if session_id in cancelled:
            continue
        status = _load_pipeline_status(args.root, session_id, args.session_base_dir)
        cancel, reason = should_cancel_from_status(status)
        if not cancel:
            if args.verbose:
                _log(f"ok {session_id}: {reason}", args.log_file)
            continue
        message = (
            f"cancel {session_id}: {reason}"
        )
        if args.dry_run:
            _log("DRY-RUN " + message, args.log_file)
            continue
        try:
            _gateway_delete(
                args.gateway,
                f"/sessions/{urllib.parse.quote(session_id, safe='')}?reason=pipeline_budget_exceeded",
                args.timeout,
            )
            cancelled.add(session_id)
            _log(message, args.log_file)
        except urllib.error.HTTPError as exc:
            _log(f"delete failed {session_id}: HTTP {exc.code} {exc.reason}", args.log_file)
        except Exception as exc:
            _log(f"delete failed {session_id}: {exc}", args.log_file)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--session-base-dir", type=Path, default=DEFAULT_SESSION_BASE_DIR)
    parser.add_argument("--gen-max", type=int, default=6)
    parser.add_argument("--opt-max", type=int, default=3)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cancelled: set[str] = set()
    _log(
        f"pipeline budget watcher started gateway={args.gateway} gen_max={args.gen_max} "
        f"opt_max={args.opt_max} session_base_dir={args.session_base_dir} dry_run={args.dry_run}",
        args.log_file,
    )
    while True:
        try:
            run_once(args, cancelled)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            _log(f"watch loop error: {exc}", args.log_file)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
