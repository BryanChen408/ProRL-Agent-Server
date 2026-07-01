#!/usr/bin/env python3
"""Local Polar rollout observer.

Reads Polar gateway health plus on-disk completion records and serves a small
web UI for watching active rollout sessions.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from polar.run_namespace import LEGACY_RUN_ID, run_id_from_task


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "output" / "ascend_operator"
DEFAULT_GATEWAY = "http://127.0.0.1:8100"
RECENT_STATE_HOURS = 168


def _now() -> float:
    return time.time()


def _read_text(path: Path, limit: int | None = None) -> str:
    try:
        if limit is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > limit:
                f.seek(max(0, size - limit))
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


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


def _assistant_content_text(value: Any) -> str:
    """Visible assistant text only; structured tool/thinking blocks are not leaks."""
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
                elif typ in {"tool_use", "tool_result", "thinking"}:
                    continue
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text") or "")
        return ""
    return str(value)


def _assistant_reasoning_text(msg: dict[str, Any] | None) -> str:
    if not isinstance(msg, dict):
        return ""
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    content = msg.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"thinking", "reasoning"}:
                text = item.get("thinking") or item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
    return "\n".join(part for part in parts if part)


def _snippet(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _format_time(ts: float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _latest_activity_mtime(active: dict[str, Any], default: float | None = None) -> float | None:
    idle = active.get("idle_seconds")
    if isinstance(idle, (int, float)):
        return max(0.0, _now() - float(idle))
    age = active.get("age_seconds")
    if isinstance(age, (int, float)):
        return max(0.0, _now() - float(age))
    return default


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_shallow_json_value(text: str, key: str) -> str | None:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"')
    match = pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1)


def _extract_shallow_json_number(text: str, key: str) -> int | float | None:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)')
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return None


def _extract_shallow_json_bool(text: str, key: str) -> bool | None:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*(true|false)')
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1) == "true"


def derive_run_id(task_id: str | None) -> str:
    """Return the run namespace encoded in a Polar task id."""
    return run_id_from_task(task_id)


def build_session_groups(sessions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    groups: dict[str, dict[str, Any]] = {}
    for session in sessions:
        run_id = str(session.get("run_id") or LEGACY_RUN_ID)
        group = groups.setdefault(
            run_id,
            {
                "run_id": run_id,
                "sessions": 0,
                "running": 0,
                "latest_mtime": None,
                "latest_time": "",
            },
        )
        group["sessions"] += 1
        if str(session.get("status") or "").startswith("RUNNING"):
            group["running"] += 1
        mtime = session.get("latest_mtime")
        if isinstance(mtime, (int, float)) and (group["latest_mtime"] is None or mtime > group["latest_mtime"]):
            group["latest_mtime"] = mtime
            group["latest_time"] = _format_time(float(mtime))

    ordered = sorted(groups.values(), key=lambda item: (item.get("latest_mtime") or 0), reverse=True)
    latest = next((item["run_id"] for item in ordered if item["run_id"] != LEGACY_RUN_ID), None)
    if latest is None and ordered:
        latest = str(ordered[0]["run_id"])
    return ordered, latest


def _gateway_get(base_url: str, path: str, timeout: float = 2.0) -> Any:
    url = base_url.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc), "url": url}


@dataclass(frozen=True)
class SessionPath:
    session_id: str
    task_id: str
    path: Path
    run_id: str = LEGACY_RUN_ID


class ObserverStore:
    def __init__(self, root: Path, gateway: str, results_dir: Path | None = None) -> None:
        self.root = root
        self.gateway = gateway.rstrip("/")
        self.results_dir = results_dir if results_dir is not None else self.root / "rollout_results"
        self.logs_dir = self.root / "logs"
        self._json_cache: dict[str, tuple[int, int, dict[str, Any] | None]] = {}
        self._summary_cache: dict[str, tuple[int, int, dict[str, Any] | None]] = {}

    def gateway_health(self) -> dict[str, Any]:
        data = _gateway_get(self.gateway, "/health")
        return data if isinstance(data, dict) else {"error": "unexpected health payload"}

    def gateway_sessions(self) -> list[dict[str, Any]]:
        data = _gateway_get(self.gateway, "/sessions")
        if isinstance(data, dict) and isinstance(data.get("sessions"), list):
            return [x for x in data["sessions"] if isinstance(x, dict)]
        return []

    def gateway_completion_metrics(self) -> dict[str, dict[str, Any]]:
        data = _gateway_get(self.gateway, "/completion_metrics", timeout=5.0)
        sessions = data.get("sessions") if isinstance(data, dict) else None
        if not isinstance(sessions, list):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in sessions:
            if isinstance(row, dict) and row.get("session_id"):
                out[str(row["session_id"])] = row
        return out

    def device_map(self) -> dict[str, dict[str, str]]:
        text = _read_text(self.logs_dir / "gateway.log", limit=512_000)
        out: dict[str, dict[str, str]] = {}
        pattern = re.compile(r"ascend:\s+polar-(sk-polar-[0-9a-fA-F-]+)(-eval)?\s+->\s+physical card\s+(\d+)")
        for session_id, eval_suffix, card in pattern.findall(text):
            out.setdefault(session_id, {})["eval_card" if eval_suffix else "runtime_card"] = card
        return out

    def gateway_timeout_map(self) -> dict[str, dict[str, Any]]:
        text = _read_text(self.logs_dir / "gateway.log", limit=20_000_000)
        out: dict[str, dict[str, Any]] = {}
        pattern = re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
            r"session (?P<session_id>sk-polar-[0-9a-fA-F-]+): Upstream request timed out",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            session_id = match.group("session_id")
            row = out.setdefault(session_id, {"count": 0, "last_time": "", "last_line": ""})
            row["count"] += 1
            row["last_time"] = match.group("ts")
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            row["last_line"] = text[line_start:line_end].strip()
        return out

    def session_paths(
        self,
        *,
        recent_since: float | None = None,
        include_session_ids: set[str] | None = None,
    ) -> list[SessionPath]:
        out: list[SessionPath] = []
        if not self.results_dir.is_dir():
            return out
        task_roots: list[tuple[Path, str]] = [(self.results_dir, LEGACY_RUN_ID)]
        for run_dir in sorted(self.results_dir.glob("run_*")):
            if run_dir.is_dir():
                task_roots.append((run_dir, run_dir.name.removeprefix("run_") or LEGACY_RUN_ID))
        for task_root, run_id in task_roots:
            for task_dir in sorted(task_root.glob("task_*")):
                sessions_dir = task_dir / "sessions"
                if not sessions_dir.is_dir():
                    continue
                task_id = task_dir.name.removeprefix("task_")
                for session_dir in sorted(sessions_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue
                    session_id = session_dir.name
                    if recent_since is not None and session_id not in (include_session_ids or set()):
                        try:
                            if session_dir.stat().st_mtime < recent_since:
                                continue
                        except OSError:
                            continue
                    out.append(SessionPath(session_id, task_id, session_dir, run_id))
        return out

    def completion_files(self, session: SessionPath) -> list[Path]:
        comp_dir = session.path / "completions"
        if not comp_dir.is_dir():
            return []
        return sorted(comp_dir.glob("*.json"))

    def session_result_path(self, session: SessionPath) -> Path:
        task_dir = session.path.parent.parent
        return task_dir / f"ses_{session.session_id}.json"

    def session_result(self, session: SessionPath) -> dict[str, Any] | None:
        return self.load_json(self.session_result_path(session))

    def session_result_shallow(self, session: SessionPath) -> dict[str, Any] | None:
        path = self.session_result_path(session)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = f"shallow:{path}"
        cached = self._json_cache.get(key)
        sig = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[0] == sig[0] and cached[1] == sig[1]:
            return cached[2]
        head = _read_text(path, limit=256_000)
        result = {
            "status": _extract_shallow_json_value(head, "status"),
            "error": _extract_shallow_json_value(head, "error"),
            "model_requested": _extract_shallow_json_value(head, "model_requested"),
            "model_used": _extract_shallow_json_value(head, "model_used"),
            "outcome_reward": _extract_shallow_json_number(head, "outcome_reward"),
            "success": _extract_shallow_json_bool(head, "success"),
        }
        self._json_cache[key] = (sig[0], sig[1], result)
        return result

    @staticmethod
    def classify_session_status(
        active_status: Any,
        *,
        has_files: bool,
        result: dict[str, Any] | None,
        timeout_info: dict[str, Any] | None,
        summary: dict[str, Any] | None = None,
    ) -> str:
        active_text = str(active_status or "").upper()
        has_timeout = bool((timeout_info or {}).get("count"))
        has_abnormal = bool((summary or {}).get("abnormal_termination"))
        if active_text:
            if active_text == "RUNNING" and has_timeout:
                return "RUNNING_TIMEOUT"
            if has_abnormal and active_text in {"COMPLETED", "DONE", "FINISHED", "STOPPED"}:
                return "ABNORMAL_ON_DISK" if has_files else "ABNORMAL"
            return active_text

        result_status = str((result or {}).get("status") or "").upper()
        result_error = (result or {}).get("error")
        if has_timeout or result_status == "TIMEOUT":
            return "TIMEOUT_ON_DISK"
        if result_status in {"ERROR", "FAILED"} or result_error:
            return "ERROR_ON_DISK"
        if has_abnormal:
            return "ABNORMAL_ON_DISK" if has_files or result_status == "COMPLETED" else "ABNORMAL"
        if result_status == "COMPLETED":
            return "COMPLETED_ON_DISK"
        if has_files:
            return "ON_DISK"
        return "EMPTY"

    def load_json(self, path: Path) -> dict[str, Any] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(path)
        cached = self._json_cache.get(key)
        sig = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[0] == sig[0] and cached[1] == sig[1]:
            return cached[2]
        data = _safe_json_load(path)
        self._json_cache[key] = (sig[0], sig[1], data)
        if len(self._json_cache) > 5000:
            for old in list(self._json_cache)[:500]:
                self._json_cache.pop(old, None)
        return data

    def load_summary(self, path: Path) -> dict[str, Any] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(path)
        cached = self._summary_cache.get(key)
        sig = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[0] == sig[0] and cached[1] == sig[1]:
            return cached[2]
        data = self.load_json(path)
        summary = analyze_session_messages(data) if data is not None else None
        self._summary_cache[key] = (sig[0], sig[1], summary)
        if len(self._summary_cache) > 5000:
            for old in list(self._summary_cache)[:500]:
                self._summary_cache.pop(old, None)
        return summary

    def aggregate_completion_summary(
        self,
        files: list[Path],
        extra_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for path in files:
            summary = self.load_summary(path)
            if summary is not None:
                summaries.append(summary)
        for record in extra_records or []:
            summaries.append(analyze_session_messages(record))
        return aggregate_session_summaries(summaries)

    def latest_completion(self, session: SessionPath) -> tuple[Path | None, dict[str, Any] | None]:
        files = self.completion_files(session)
        if not files:
            return None, None
        latest = files[-1]
        return latest, self.load_json(latest)

    def gateway_completions(self, session_id: str) -> list[dict[str, Any]]:
        data = _gateway_get(
            self.gateway,
            f"/sessions/{urllib.parse.quote(session_id, safe='')}/completions",
            timeout=10.0,
        )
        records: Any
        if isinstance(data, dict):
            records = data.get("completions") or data.get("records") or []
        else:
            records = data
        if not isinstance(records, list):
            return []
        return [item for item in records if isinstance(item, dict)]

    def session_summary(self) -> dict[str, Any]:
        health = self.gateway_health()
        active_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(health.get("active_sessions"), list):
            for item in health["active_sessions"]:
                if isinstance(item, dict) and item.get("session_id"):
                    active_by_id[str(item["session_id"])] = item
        for item in self.gateway_sessions():
            if item.get("session_id"):
                active_by_id.setdefault(str(item["session_id"]), {}).update(item)

        metrics_by_id = self.gateway_completion_metrics()
        device_map = self.device_map()
        timeout_by_id = self.gateway_timeout_map()
        sessions: list[dict[str, Any]] = []
        seen_session_ids: set[str] = set()
        recent_since = _now() - RECENT_STATE_HOURS * 3600
        for session in self.session_paths(recent_since=recent_since, include_session_ids=set(active_by_id)):
            seen_session_ids.add(session.session_id)
            files = self.completion_files(session)
            latest_file = files[-1] if files else None
            latest_data = self.load_json(latest_file) if latest_file else None
            stat = latest_file.stat() if latest_file else None
            active = active_by_id.get(session.session_id, {})
            result = self.session_result_shallow(session)
            timeout_info = timeout_by_id.get(session.session_id, {})
            latest_mtime = stat.st_mtime if stat else _latest_activity_mtime(active)
            live_metrics = metrics_by_id.get(session.session_id, {})
            summary = self.aggregate_completion_summary(files[-2:]) if files else analyze_session_messages({})
            status = self.classify_session_status(
                active.get("status"),
                has_files=bool(files),
                result=result,
                timeout_info=timeout_info,
                summary=summary,
            )
            task_id = str(active.get("task_id") or session.task_id)
            run_id = str(
                active.get("run_id")
                or (session.run_id if session.run_id != LEGACY_RUN_ID else "")
                or derive_run_id(task_id)
            )
            completion_metrics = (
                active.get("completion_metrics")
                or live_metrics.get("completion_metrics")
                or {}
            )
            sessions.append(
                {
                    "session_id": session.session_id,
                    "task_id": task_id,
                    "run_id": run_id,
                    "status": status,
                    "active": bool(active),
                    "age_seconds": active.get("age_seconds"),
                    "idle_seconds": active.get("idle_seconds"),
                    "completion_count": active.get("completion_count") or len(files),
                    "completion_files": len(files),
                    "latest_completion": latest_file.name if latest_file else None,
                    "latest_mtime": latest_mtime,
                    "latest_time": _format_time(latest_mtime),
                    "latest_size": stat.st_size if stat else 0,
                    "model_requested": active.get("model_requested") or (latest_data or {}).get("model_requested") or (result or {}).get("model_requested"),
                    "model_used": active.get("model_used") or (latest_data or {}).get("model_used") or (result or {}).get("model_used"),
                    "completion_metrics": completion_metrics,
                    "result_status": (result or {}).get("status"),
                    "result_error": (result or {}).get("error"),
                    "timeout_count": timeout_info.get("count", 0),
                    "timeout_last_time": timeout_info.get("last_time"),
                    "timeout_last_line": timeout_info.get("last_line"),
                    "runtime_card": device_map.get(session.session_id, {}).get("runtime_card"),
                    "eval_card": device_map.get(session.session_id, {}).get("eval_card"),
                    "quality": {
                        "tool_counts": summary["tool_counts"],
                        "pipeline_runs": summary["pipeline_runs"],
                        "pipeline_stage_counts": summary["pipeline_stage_counts"],
                        "validation": summary["validation"],
                        "submission_writes": summary["submission_writes"],
                        "writes_before_first_pipeline": summary["writes_before_first_pipeline"],
                        "first_submission_write_turn": summary["first_submission_write_turn"],
                        "first_pipeline_turn": summary["first_pipeline_turn"],
                        "workflow_violation": summary["workflow_violation"],
                        "doc_drift_count": summary["doc_drift_count"],
                        "doc_drift_turns": summary["doc_drift_turns"],
                        "forbidden_writes": len(summary["forbidden_writes"]),
                        "readonly_mutation_attempts": len(summary["readonly_mutation_attempts"]),
                        "abnormal_termination": summary["abnormal_termination"],
                        "abnormal_reasons": summary["abnormal_reasons"],
                        "abnormal_events": summary["abnormal_events"],
                    },
                }
            )
        for session_id, active in active_by_id.items():
            if session_id in seen_session_ids:
                continue
            live_metrics = metrics_by_id.get(session_id, {})
            timeout_info = timeout_by_id.get(session_id, {})
            task_id = str(active.get("task_id") or live_metrics.get("task_id") or "")
            completion_metrics = (
                active.get("completion_metrics")
                or live_metrics.get("completion_metrics")
                or {}
            )
            latest_mtime = _latest_activity_mtime(active, _now())
            summary = self.aggregate_completion_summary([])
            status = self.classify_session_status(
                active.get("status") or "ACTIVE",
                has_files=False,
                result=None,
                timeout_info=timeout_info,
                summary=summary,
            )
            sessions.append(
                {
                    "session_id": session_id,
                    "task_id": task_id,
                    "run_id": derive_run_id(task_id),
                    "status": status,
                    "active": True,
                    "age_seconds": active.get("age_seconds"),
                    "idle_seconds": active.get("idle_seconds"),
                    "completion_count": active.get("completion_count") or completion_metrics.get("request_count") or 0,
                    "completion_files": 0,
                    "latest_completion": None,
                    "latest_mtime": latest_mtime,
                    "latest_time": _format_time(latest_mtime),
                    "latest_size": 0,
                    "model_requested": active.get("model_requested") or live_metrics.get("model_requested"),
                    "model_used": active.get("model_used") or live_metrics.get("model_used"),
                    "completion_metrics": completion_metrics,
                    "result_status": None,
                    "result_error": None,
                    "timeout_count": timeout_info.get("count", 0),
                    "timeout_last_time": timeout_info.get("last_time"),
                    "timeout_last_line": timeout_info.get("last_line"),
                    "runtime_card": device_map.get(session_id, {}).get("runtime_card"),
                    "eval_card": device_map.get(session_id, {}).get("eval_card"),
                    "quality": {
                        "tool_counts": summary["tool_counts"],
                        "pipeline_runs": summary["pipeline_runs"],
                        "pipeline_stage_counts": summary["pipeline_stage_counts"],
                        "validation": summary["validation"],
                        "forbidden_writes": len(summary["forbidden_writes"]),
                        "readonly_mutation_attempts": len(summary["readonly_mutation_attempts"]),
                        "abnormal_termination": summary["abnormal_termination"],
                        "abnormal_reasons": summary["abnormal_reasons"],
                        "abnormal_events": summary["abnormal_events"],
                    },
                }
            )
        sessions.sort(key=lambda x: (x.get("latest_mtime") or 0), reverse=True)
        session_groups, latest_run_id = build_session_groups(sessions)
        return {
            "root": str(self.root),
            "results_dir": str(self.results_dir),
            "gateway_url": self.gateway,
            "gateway_health": health,
            "now": _now(),
            "health": health,
            "session_groups": session_groups,
            "latest_run_id": latest_run_id,
            "sessions": sessions,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        found = next((s for s in self.session_paths() if s.session_id == session_id), None)
        active = next(
            (item for item in self.gateway_sessions() if str(item.get("session_id") or "") == session_id),
            None,
        )
        gateway_records = self.gateway_completions(session_id)
        live_metrics = self.gateway_completion_metrics().get(session_id, {})
        timeout_info = self.gateway_timeout_map().get(session_id, {})
        result = self.session_result(found) if found else None
        if not found and not active and not gateway_records and not live_metrics:
            return None
        files = self.completion_files(found) if found else []
        latest_file = files[-1] if files else None
        latest_disk_data = self.load_json(latest_file) if latest_file else None
        latest_data = gateway_records[-1] if gateway_records else latest_disk_data
        summary_source = "gateway_memory_aggregate" if gateway_records else "disk_aggregate"
        summary = self.aggregate_completion_summary(files, gateway_records if gateway_records else None)
        completions = []
        record_rows: list[tuple[str, float | None, int, dict[str, Any]]] = []
        if gateway_records:
            for index, data in enumerate(gateway_records, start=1):
                completion_id = str(data.get("completion_id") or f"completion-{index}")
                name = f"gateway:{index:04d}-{completion_id}"
                timestamp = data.get("timestamp")
                mtime = None
                if isinstance(timestamp, (int, float)):
                    mtime = float(timestamp)
                size = len(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8", errors="ignore"))
                record_rows.append((name, mtime, size, data))
        else:
            for path in files:
                stat = path.stat()
                record_rows.append((path.name, stat.st_mtime, stat.st_size, self.load_json(path) or {}))
        for name, mtime, size, data in record_rows:
            resp = data.get("response") if isinstance(data.get("response"), dict) else {}
            choices = resp.get("choices") if isinstance(resp, dict) else None
            finish = None
            tool_names: list[str] = []
            response_text = ""
            if choices and isinstance(choices, list):
                msg = (choices[0] or {}).get("message") or {}
                finish = (choices[0] or {}).get("finish_reason")
                response_text = _text_block(msg.get("content"))
                for tc in msg.get("tool_calls") or []:
                    fn = (tc.get("function") or {}).get("name") or tc.get("name")
                    if fn:
                        tool_names.append(str(fn))
            completions.append(
                {
                    "file": name,
                    "mtime": mtime,
                    "time": _format_time(mtime),
                    "size": size,
                    "completion_id": data.get("completion_id"),
                    "timestamp": data.get("timestamp"),
                    "finish_reason": finish,
                    "response_truncated": bool((resp or {}).get("__truncated")),
                    "response_snippet": _snippet(response_text),
                    "tool_names": tool_names,
                }
            )
        task_id = (
            found.task_id
            if found
            else str((active or {}).get("task_id") or live_metrics.get("task_id") or "")
        )
        return {
            "session_id": session_id,
            "results_dir": str(self.results_dir),
            "task_id": task_id,
            "run_id": str(
                (active or {}).get("run_id")
                or (found.run_id if found and found.run_id != LEGACY_RUN_ID else "")
                or derive_run_id(task_id)
            ),
            "path": str(found.path) if found else "",
            "latest_file": latest_file.name if latest_file else None,
            "summary_source": summary_source,
            "gateway_completion_count": len(gateway_records),
            "disk_completion_count": len(files),
            "status": self.classify_session_status(
                (active or {}).get("status"),
                has_files=bool(files),
                result=result,
                timeout_info=timeout_info,
                summary=summary,
            ),
            "result_status": (result or {}).get("status"),
            "result_error": (result or {}).get("error"),
            "timeout_count": timeout_info.get("count", 0),
            "timeout_last_time": timeout_info.get("last_time"),
            "timeout_last_line": timeout_info.get("last_line"),
            "summary": summary,
            "completion_metrics": (
                (active or {}).get("completion_metrics")
                or live_metrics.get("completion_metrics")
                or {}
            ),
            "completions": completions,
            "log_tail": {
                "gateway": self.tail_log("gateway", 120),
                "rollout": self.tail_log("rollout", 80),
            },
        }

    def get_completion(self, session_id: str, file_name: str) -> dict[str, Any] | None:
        found = next((s for s in self.session_paths() if s.session_id == session_id), None)
        if not found:
            return None
        target = found.path / "completions" / file_name
        try:
            target.resolve().relative_to((found.path / "completions").resolve())
        except Exception:
            return None
        data = self.load_json(target)
        if data is None:
            return None
        return summarize_completion_payload(data)

    def tail_log(self, name: str, lines: int = 200) -> str:
        mapping = {
            "gateway": self.logs_dir / "gateway.log",
            "rollout": self.logs_dir / "rollout.log",
        }
        path = mapping.get(name)
        if not path:
            return ""
        text = _read_text(path, limit=300_000)
        return "\n".join(text.splitlines()[-lines:])


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
                        "is_error": bool(block.get("is_error")),
                        "content": str(block.get("content", "")),
                    }
                )
    return results


def _is_operator_task_text(text: str) -> bool:
    if "Implement a Triton operator" in text:
        return True
    if "reference task is at src/" in text and "output/submission/" in text:
        return True
    if "Write your implementation as class ModelNew" in text and "tools/triton_eval_pipeline.sh" in text:
        return True
    return False


def _is_skill_reference_text(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("# Triton Ascend 基础知识参考手册"):
        return True
    if stripped.startswith("# ") and "Triton Ascend" in stripped and "参考" in stripped[:300]:
        return True
    return False


def _select_task_prompt(prompt_blocks: list[dict[str, Any]]) -> str:
    for block in prompt_blocks:
        text = str(block.get("text") or "")
        if _is_operator_task_text(text):
            return text
    for block in prompt_blocks:
        text = str(block.get("text") or "")
        if text and not _is_skill_reference_text(text):
            return text
    return prompt_blocks[-1]["text"] if prompt_blocks else ""


def _content_text_blocks(content: Any) -> list[dict[str, Any]]:
    def label_block(index: int, text: str) -> str:
        if "The following skills are available" in text:
            return "Available Skills"
        if "CLAUDE.md" in text or "# claudeMd" in text:
            return "Project Workflow (CLAUDE.md)"
        if _is_operator_task_text(text):
            return "Operator Task"
        if _is_skill_reference_text(text):
            return "Skill Reference"
        return f"Injected Context {index + 1}"

    blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        total = len(content)
        for idx, item in enumerate(content):
            if isinstance(item, dict):
                typ = str(item.get("type") or "object")
                text = _text_block([item])
            else:
                typ = type(item).__name__
                text = str(item)
            blocks.append(
                {
                    "index": idx,
                    "label": label_block(idx, text),
                    "type": typ,
                    "chars": len(text),
                    "snippet": _snippet(text, 700),
                    "text": text,
                }
            )
    else:
        text = _text_block(content)
        blocks.append(
            {
                "index": 0,
                "label": label_block(0, text),
                "type": type(content).__name__,
                "chars": len(text),
                "snippet": _snippet(text, 700),
                "text": text,
            }
        )
    return blocks


def _response_message(data: dict[str, Any]) -> dict[str, Any] | None:
    resp = data.get("response")
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = (choices[0] or {}).get("message")
    return msg if isinstance(msg, dict) else None


def _response_finish_reason(data: dict[str, Any]) -> str | None:
    resp = data.get("response")
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    reason = (choices[0] or {}).get("finish_reason")
    return str(reason) if reason is not None else None


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


def _extract_turns(messages: list[dict[str, Any]], current_response: dict[str, Any] | None) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            assistant_text = _assistant_content_text(content)
            assistant_reasoning = _assistant_reasoning_text(msg)
            turns.append(
                {
                    "index": len(turns) + 1,
                    "source": "history",
                    "assistant_text": assistant_text,
                    "assistant_reasoning": assistant_reasoning,
                    "assistant_snippet": _snippet(assistant_text or assistant_reasoning),
                    "tool_uses": _tool_uses(content),
                    "tool_results": [],
                }
            )
            pending_results = turns[-1]["tool_results"]
        elif role == "user":
            results = _tool_results(content)
            if results and pending_results is not None:
                pending_results.extend(results)
    if current_response:
        assistant_text = _assistant_content_text(current_response.get("content"))
        assistant_reasoning = _assistant_reasoning_text(current_response)
        turns.append(
            {
                "index": len(turns) + 1,
                "source": "current_response",
                "assistant_text": assistant_text,
                "assistant_reasoning": assistant_reasoning,
                "assistant_snippet": _snippet(assistant_text or assistant_reasoning),
                "tool_uses": _response_tool_calls(current_response),
                "tool_results": [],
            }
        )
    return turns


def _tool_name_and_target(tool: dict[str, Any]) -> tuple[str, str]:
    name = str(tool.get("name") or "")
    inp = tool.get("input") or {}
    target = ""
    if isinstance(inp, dict):
        if name == "Read":
            target = str(inp.get("file_path") or "")
        elif name in {"Write", "Edit"}:
            target = str(inp.get("file_path") or "")
        elif name == "Bash":
            target = str(inp.get("command") or "")
        elif name == "Skill":
            target = str(inp.get("skill") or inp.get("name") or inp.get("args") or "")[:240]
        else:
            target = json.dumps(inp, ensure_ascii=False)[:240]
    return name, target


def _classify_tool_result(text: str) -> dict[str, Any]:
    low = text.lower()
    labels = []
    if "success=true" in low or "[triton-eval] done" in low:
        labels.append("success")
    if "ast failed" in low or "ast_check_failed" in low:
        labels.append("ast_fail")
    if "verify failed" in low or "数值验证失败" in text:
        labels.append("verify_fail")
    if "benchmark failed" in low or "性能测试失败" in text:
        labels.append("benchmark_fail")
    if "unsupportedlanguageconstruct" in low:
        labels.append("unsupported_construct")
    if "cbuf" in low:
        labels.append("cbuf")
    if "coredim" in low:
        labels.append("core_dim")
    return {"labels": labels, "snippet": _snippet(text, 900)}


PLAN_TOOL_NAMES = {"TaskCreate", "TaskUpdate", "TodoWrite", "EnterPlanMode", "ExitPlanMode"}
PROTECTED_PATH_PREFIXES = ("tools/", ".agents/skills/")
PROTECTED_EXACT_PATHS = ("CLAUDE.md", "./CLAUDE.md")
SUBMISSION_PATH_RE = re.compile(r"(?:^|/)output/submission/[^/\s]+_impl\.py$")
DOC_DRIFT_TERMS = ("总结", "点评", "评价", "改写", "重写", "完整", "全面", "实用", "详细", "涵盖", "包括", "提供", "介绍")
RAW_TOOL_CALL_TEXT_RE = re.compile(
    r'(<\s*/?\s*(?:tool_use|tool_call|toolcall|tool_use_error)\b|'
    r'"type"\s*:\s*"(?:tool_use|tool_result)"|'
    r'"tool_calls"\s*:\s*\[)',
    re.IGNORECASE,
)


def _looks_like_raw_tool_call_text(text: str) -> bool:
    if not text:
        return False
    if not RAW_TOOL_CALL_TEXT_RE.search(text):
        return False
    if re.search(r'"type"\s*:\s*"tool_use"', text, re.IGNORECASE) and re.search(
        r'"(?:name|input|id)"\s*:',
        text,
    ):
        return True
    if re.search(r'"tool_calls"\s*:\s*\[', text, re.IGNORECASE):
        return True
    if re.search(r"<\s*/?\s*(?:tool_use|tool_call|toolcall|tool_use_error)\b", text, re.IGNORECASE):
        return True
    if re.search(r'"type"\s*:\s*"tool_result"', text, re.IGNORECASE) and re.search(
        r'"(?:tool_use_id|content)"\s*:',
        text,
    ):
        return True
    return False


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
        if script.endswith("tools/triton_eval_pipeline.sh") or script.endswith("/triton_eval_pipeline.sh"):
            return True
    return False


def _is_submission_path(path: str) -> bool:
    normalized = _normalize_observed_path(path)
    return bool(SUBMISSION_PATH_RE.search(normalized))


def _is_submission_write_tool(name: str, tool: dict[str, Any]) -> bool:
    if name not in {"Write", "Edit", "MultiEdit"}:
        return False
    inp = tool.get("input") if isinstance(tool.get("input"), dict) else {}
    return _is_submission_path(str(inp.get("file_path") or ""))


def _is_doc_drift_text(text: str) -> bool:
    if not text:
        return False
    if "reference" not in text.lower() and "文档" not in text and "手册" not in text and "参考" not in text:
        return False
    return any(term in text for term in DOC_DRIFT_TERMS)


def _normalize_observed_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if "/agent_workdir/" in value:
        value = value.split("/agent_workdir/", 1)[1]
    if value.startswith("/opt/workspace/agent_workdir/"):
        value = value.removeprefix("/opt/workspace/agent_workdir/")
    return value


def _is_protected_path(path: str) -> bool:
    normalized = _normalize_observed_path(path)
    return normalized in PROTECTED_EXACT_PATHS or normalized.startswith(PROTECTED_PATH_PREFIXES)


def _protected_path_from_tool(tool: dict[str, Any]) -> str:
    name = str(tool.get("name") or "")
    inp = tool.get("input") if isinstance(tool.get("input"), dict) else {}
    if name in {"Write", "Edit", "MultiEdit"}:
        return str(inp.get("file_path") or "")
    return ""


def _bash_readonly_mutation(command: str) -> dict[str, str] | None:
    protected = r"(?:\.?/)?(?:tools/|\.agents/skills/|CLAUDE\.md\b)"
    patterns: list[tuple[str, str]] = [
        ("redirect", rf"(?:^|[\s;&|])(?:[12]?>|>>|&>)\s*['\"]?({protected}[^'\"\s;&|]*)"),
        ("tee", rf"(?:^|[\s;&|])tee(?:\s+-a)?\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("sed_i", rf"(?:^|[\s;&|])sed\s+[^;&|]*-i[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("perl_pi", rf"(?:^|[\s;&|])perl\s+[^;&|]*-p(?:i|[^;&|]*\s+-i)[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("chmod", rf"(?:^|[\s;&|])chmod\s+[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("rm", rf"(?:^|[\s;&|])rm\s+[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("mv", rf"(?:^|[\s;&|])mv\s+[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
        ("cp", rf"(?:^|[\s;&|])cp\s+[^;&|]*\s+['\"]?({protected}[^'\"\s;&|]*)"),
    ]
    for kind, pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return {"kind": kind, "path": _normalize_observed_path(match.group(1))}
    return None


def _pipeline_status(labels: list[str], text: str) -> str:
    if "success" in labels and "verify_fail" not in labels and "ast_fail" not in labels and "benchmark_fail" not in labels:
        return "success"
    if "ast_fail" in labels:
        return "ast_fail"
    if "verify_fail" in labels:
        return "verify_fail"
    if "benchmark_fail" in labels:
        return "benchmark_fail"
    if "CompilationError" in text or "CompilationError:" in text:
        return "compile_fail"
    if "Exit code 1" in text:
        return "failed"
    return "unknown"


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


def _pipeline_stage_status(labels: list[str], text: str) -> dict[str, str]:
    low = text.lower()
    precision_started = (
        "[triton-eval] step2 verify" in low
        or "verify failed" in low
        or "verify_fail" in low
        or "数值验证失败" in text
    )
    profiling_started = (
        "[triton-eval] step3" in low
        or "benchmark" in low
        or "性能测试" in text
    )

    if "ast_fail" in labels:
        precision = "not_reached"
    elif "verify_fail" in labels:
        precision = "fail"
    elif profiling_started or "success" in labels or "benchmark_fail" in labels:
        precision = "pass"
    elif precision_started:
        precision = "unknown"
    else:
        precision = "not_reached"

    if "ast_fail" in labels or "verify_fail" in labels:
        profiling = "not_reached"
    elif "benchmark_fail" in labels:
        profiling = "fail"
    elif "success" in labels:
        profiling = "pass"
    elif profiling_started:
        profiling = "unknown"
    else:
        profiling = "not_reached"

    return {"precision": precision, "profiling": profiling}


def _empty_pipeline_stage_counts() -> dict[str, dict[str, int]]:
    return {
        "precision": {"attempts": 0, "pass": 0, "fail": 0, "unknown": 0, "not_reached": 0},
        "profiling": {"attempts": 0, "pass": 0, "fail": 0, "unknown": 0, "not_reached": 0},
    }


def _update_pipeline_stage_counts(counts: dict[str, dict[str, int]], stage: dict[str, str]) -> None:
    for key in ("precision", "profiling"):
        status = stage.get(key, "unknown")
        if status != "not_reached":
            counts[key]["attempts"] += 1
        counts[key][status] = counts[key].get(status, 0) + 1


def analyze_session_messages(data: dict[str, Any]) -> dict[str, Any]:
    req = data.get("original_request") if isinstance(data.get("original_request"), dict) else {}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    current = _response_message(data)
    finish_reason = _response_finish_reason(data)
    turns = _extract_turns(messages, current)
    first_user = next((msg for msg in messages if msg.get("role") == "user"), {})
    prompt_blocks = _content_text_blocks(first_user.get("content")) if first_user else []
    task_prompt = _select_task_prompt(prompt_blocks)
    current_text = _text_block(current.get("content")) if current else ""
    current_reasoning = _assistant_reasoning_text(current)
    current_tool_calls = _response_tool_calls(current) if current else []
    current_tool_summaries = []
    for tool in current_tool_calls:
        name, target = _tool_name_and_target(tool)
        current_tool_summaries.append({"name": name, "target": target, "input": tool.get("input") or {}})
    tool_counts: dict[str, int] = {}
    plan_tool_counts: dict[str, int] = {}
    skill_calls: list[dict[str, Any]] = []
    skill_reference_reads: list[dict[str, Any]] = []
    forbidden_writes: list[dict[str, Any]] = []
    readonly_mutation_attempts: list[dict[str, Any]] = []
    pipeline_runs = 0
    pipeline_commands: list[dict[str, Any]] = []
    pipeline_runs_detail: list[dict[str, Any]] = []
    pipeline_command_errors: list[dict[str, Any]] = []
    submission_writes = 0
    submission_write_turns: list[int] = []
    first_submission_write_turn: int | None = None
    first_pipeline_turn: int | None = None
    doc_drift_turns: list[int] = []
    abnormal_events: list[dict[str, Any]] = []
    validation = {"success": 0, "ast_fail": 0, "verify_fail": 0, "benchmark_fail": 0, "core_dim": 0, "cbuf": 0}
    pipeline_stage_counts = _empty_pipeline_stage_counts()
    for turn in turns:
        result_by_id = {r.get("tool_use_id"): r for r in turn.get("tool_results", []) if r.get("tool_use_id")}
        assistant_text = str(turn.get("assistant_text") or "")
        if _looks_like_raw_tool_call_text(assistant_text):
            abnormal_events.append(
                {
                    "reason": "raw_tool_call_text",
                    "turn": turn["index"],
                    "source": turn.get("source"),
                    "snippet": _snippet(assistant_text, 1200),
                }
            )
        if _is_doc_drift_text(str(turn.get("assistant_text") or "")):
            doc_drift_turns.append(turn["index"])
        for tool in turn.get("tool_uses", []):
            name, target = _tool_name_and_target(tool)
            if name in PLAN_TOOL_NAMES:
                plan_tool_counts[name] = plan_tool_counts.get(name, 0) + 1
            elif name:
                tool_counts[name] = tool_counts.get(name, 0) + 1
            if name == "Skill":
                inp = tool.get("input") if isinstance(tool.get("input"), dict) else {}
                skill_calls.append(
                    {
                        "turn": turn["index"],
                        "skill": inp.get("skill") or inp.get("name") or "",
                        "args": _snippet(str(inp.get("args") or ""), 500),
                    }
                )
            if name == "Read" and ("/.claude/skills/" in target or "/skills/" in target):
                skill_reference_reads.append({"turn": turn["index"], "path": target})
            protected_path = _protected_path_from_tool(tool)
            if protected_path and _is_protected_path(protected_path):
                forbidden_writes.append(
                    {
                        "turn": turn["index"],
                        "tool": name,
                        "path": _normalize_observed_path(protected_path),
                    }
                )
            if _is_submission_write_tool(name, tool):
                submission_writes += 1
                submission_write_turns.append(turn["index"])
                if first_submission_write_turn is None:
                    first_submission_write_turn = turn["index"]
            if name == "Bash":
                mutation = _bash_readonly_mutation(target)
                if mutation is not None:
                    readonly_mutation_attempts.append(
                        {
                            "turn": turn["index"],
                            "tool": name,
                            "kind": mutation["kind"],
                            "path": mutation["path"],
                            "command": _snippet(target, 800),
                        }
                    )
            if name == "Bash" and _is_pipeline_command(target):
                result = result_by_id.get(tool.get("id")) or {}
                result_text = str(result.get("content") or "")
                if not _is_pipeline_feedback(result_text):
                    pipeline_command_errors.append(
                        {
                            "turn": turn["index"],
                            "tool_id": tool.get("id"),
                            "command": target,
                            "is_error": bool(result.get("is_error")),
                            "result_chars": len(result_text),
                            "result_snippet": _snippet(result_text, 1200),
                        }
                    )
                    continue
                pipeline_runs += 1
                if first_pipeline_turn is None:
                    first_pipeline_turn = turn["index"]
                pipeline_commands.append({"turn": turn["index"], "command": target})
                cls = _classify_tool_result(result_text)
                stage = _pipeline_stage_status(cls["labels"], result_text)
                _update_pipeline_stage_counts(pipeline_stage_counts, stage)
                for label in cls["labels"]:
                    if label in validation:
                        validation[label] += 1
                pipeline_runs_detail.append(
                    {
                        "index": pipeline_runs,
                        "turn": turn["index"],
                        "tool_id": tool.get("id"),
                        "command": target,
                        "labels": cls["labels"],
                        "status": _pipeline_status(cls["labels"], result_text),
                        "precision_status": stage["precision"],
                        "profiling_status": stage["profiling"],
                        "is_error": bool(result.get("is_error")),
                        "result_chars": len(result_text),
                        "result_snippet": _snippet(result_text, 1200),
                        "result": result_text,
                    }
                )
    writes_before_first_pipeline = sum(
        1
        for turn_index in submission_write_turns
        if first_pipeline_turn is None or turn_index < first_pipeline_turn
    )
    workflow_violation = submission_writes > 0 and (
        pipeline_runs == 0 or writes_before_first_pipeline > 1
    )
    response_truncated = bool(((data.get("response") or {}) if isinstance(data.get("response"), dict) else {}).get("__truncated"))
    finish_reason_text = str(finish_reason or "").lower()
    if (
        current is not None
        and not response_truncated
        and not current_text.strip()
        and not current_reasoning.strip()
        and not current_tool_calls
        and finish_reason_text in {"stop", "eos", "eos_token", "end_turn"}
    ):
        abnormal_events.append(
            {
                "reason": "empty_stop_response",
                "turn": turns[-1]["index"] if turns else None,
                "source": "current_response",
                "finish_reason": finish_reason,
                "snippet": "",
            }
        )
    abnormal_events = _unique_dicts(abnormal_events, limit=50)
    abnormal_reasons = sorted({str(event.get("reason") or "") for event in abnormal_events if event.get("reason")})
    return {
        "messages_count": len(messages),
        "turns_count": len(turns),
        "request": {
            "model": req.get("model"),
            "max_tokens": req.get("max_tokens"),
            "thinking": req.get("thinking"),
            "context_management": req.get("context_management"),
            "prompt_blocks": prompt_blocks,
            "task_prompt": task_prompt,
            "task_prompt_is_operator": _is_operator_task_text(task_prompt),
            "current_response_text": current_text,
            "current_response_reasoning": current_reasoning,
            "current_response_snippet": _snippet(current_text, 1200),
            "current_tool_calls": current_tool_summaries,
            "finish_reason": finish_reason,
        },
        "turns": turns[-80:],
        "tool_counts": tool_counts,
        "action_tool_counts": tool_counts,
        "plan_tool_counts": plan_tool_counts,
        "skill_calls": skill_calls,
        "skill_reference_reads": skill_reference_reads,
        "forbidden_writes": forbidden_writes,
        "readonly_mutation_attempts": readonly_mutation_attempts,
        "pipeline_runs": pipeline_runs,
        "pipeline_commands": pipeline_commands[-20:],
        "pipeline_runs_detail": pipeline_runs_detail[-30:],
        "pipeline_command_errors": pipeline_command_errors[-30:],
        "pipeline_stage_counts": pipeline_stage_counts,
        "submission_writes": submission_writes,
        "submission_write_turns": submission_write_turns[-30:],
        "first_submission_write_turn": first_submission_write_turn,
        "first_pipeline_turn": first_pipeline_turn,
        "writes_before_first_pipeline": writes_before_first_pipeline,
        "workflow_violation": workflow_violation,
        "doc_drift_turns": doc_drift_turns[-30:],
        "doc_drift_count": len(doc_drift_turns),
        "validation": validation,
        "abnormal_termination": bool(abnormal_events),
        "abnormal_reasons": abnormal_reasons,
        "abnormal_events": abnormal_events,
        "current_response_available": current is not None,
        "response_truncated": response_truncated,
    }


def _summary_score(summary: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    stages = summary.get("pipeline_stage_counts") if isinstance(summary.get("pipeline_stage_counts"), dict) else {}
    precision = stages.get("precision") if isinstance(stages.get("precision"), dict) else {}
    profiling = stages.get("profiling") if isinstance(stages.get("profiling"), dict) else {}
    tool_counts = summary.get("tool_counts") if isinstance(summary.get("tool_counts"), dict) else {}
    return (
        int(bool(summary.get("abnormal_termination"))),
        int(summary.get("pipeline_runs") or 0),
        int(summary.get("submission_writes") or 0),
        int(precision.get("attempts") or 0),
        int(profiling.get("attempts") or 0),
        sum(int(v) for v in tool_counts.values() if isinstance(v, int)),
        int(summary.get("turns_count") or 0),
        int(summary.get("messages_count") or 0),
    )


def _max_int_dict(summaries: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for summary in summaries:
        values = summary.get(key)
        if not isinstance(values, dict):
            continue
        for item_key, value in values.items():
            if isinstance(value, int):
                merged[str(item_key)] = max(merged.get(str(item_key), 0), value)
    return merged


def _max_pipeline_stage_counts(summaries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged = _empty_pipeline_stage_counts()
    for summary in summaries:
        stages = summary.get("pipeline_stage_counts")
        if not isinstance(stages, dict):
            continue
        for stage_name, counts in stages.items():
            if stage_name not in merged or not isinstance(counts, dict):
                continue
            for status, value in counts.items():
                if isinstance(value, int):
                    merged[stage_name][str(status)] = max(merged[stage_name].get(str(status), 0), value)
    return merged


def _unique_dicts(items: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[-limit:]


def aggregate_session_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return stable cumulative metrics from chronological completion summaries.

    A Polar completion may contain the full conversation so far, only the latest
    response, or a truncated payload while it is being written.  For cumulative
    observer counters, use maxima across historical summaries instead of the
    newest record alone; this avoids transient zeroing without double-counting
    repeated conversation history.
    """
    valid = [summary for summary in summaries if isinstance(summary, dict)]
    if not valid:
        return analyze_session_messages({})

    latest = valid[-1]
    richest = max(valid, key=_summary_score)
    merged = dict(latest)

    latest_request = latest.get("request") if isinstance(latest.get("request"), dict) else {}
    if not latest_request.get("task_prompt_is_operator"):
        for summary in reversed(valid):
            request = summary.get("request") if isinstance(summary.get("request"), dict) else {}
            if request.get("task_prompt_is_operator"):
                merged["request"] = dict(request)
                break

    merged["messages_count"] = max(int(s.get("messages_count") or 0) for s in valid)
    merged["turns_count"] = max(int(s.get("turns_count") or 0) for s in valid)
    merged["tool_counts"] = _max_int_dict(valid, "tool_counts")
    merged["action_tool_counts"] = dict(merged["tool_counts"])
    merged["plan_tool_counts"] = _max_int_dict(valid, "plan_tool_counts")
    merged["pipeline_runs"] = max(int(s.get("pipeline_runs") or 0) for s in valid)
    merged["pipeline_stage_counts"] = _max_pipeline_stage_counts(valid)
    merged["validation"] = _max_int_dict(valid, "validation")
    merged["submission_writes"] = max(int(s.get("submission_writes") or 0) for s in valid)
    merged["writes_before_first_pipeline"] = max(int(s.get("writes_before_first_pipeline") or 0) for s in valid)
    merged["doc_drift_count"] = max(int(s.get("doc_drift_count") or 0) for s in valid)
    merged["workflow_violation"] = any(bool(s.get("workflow_violation")) for s in valid)
    merged["abnormal_termination"] = any(bool(s.get("abnormal_termination")) for s in valid)
    abnormal_events: list[dict[str, Any]] = []
    for summary in valid:
        value = summary.get("abnormal_events")
        if isinstance(value, list):
            abnormal_events.extend(item for item in value if isinstance(item, dict))
    merged["abnormal_events"] = _unique_dicts(abnormal_events, limit=50)
    merged["abnormal_reasons"] = sorted(
        {
            str(event.get("reason") or "")
            for event in merged["abnormal_events"]
            if event.get("reason")
        }
    )
    first_write_values = [
        int(s["first_submission_write_turn"])
        for s in valid
        if isinstance(s.get("first_submission_write_turn"), int)
    ]
    first_pipeline_values = [
        int(s["first_pipeline_turn"])
        for s in valid
        if isinstance(s.get("first_pipeline_turn"), int)
    ]
    merged["first_submission_write_turn"] = min(first_write_values) if first_write_values else None
    merged["first_pipeline_turn"] = min(first_pipeline_values) if first_pipeline_values else None

    for key in (
        "pipeline_commands",
        "pipeline_runs_detail",
        "pipeline_command_errors",
        "turns",
    ):
        value = richest.get(key)
        if value is not None:
            merged[key] = value

    for key in (
        "skill_calls",
        "skill_reference_reads",
        "forbidden_writes",
        "readonly_mutation_attempts",
    ):
        items: list[dict[str, Any]] = []
        for summary in valid:
            value = summary.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
        merged[key] = _unique_dicts(items)
    for key in ("submission_write_turns", "doc_drift_turns"):
        values: set[int] = set()
        for summary in valid:
            raw = summary.get(key)
            if isinstance(raw, list):
                values.update(int(item) for item in raw if isinstance(item, int))
        merged[key] = sorted(values)[-30:]

    merged["response_truncated"] = any(bool(s.get("response_truncated")) for s in valid)
    merged["current_response_available"] = any(bool(s.get("current_response_available")) for s in valid)
    return merged


def summarize_completion_payload(data: dict[str, Any]) -> dict[str, Any]:
    req = data.get("original_request") if isinstance(data.get("original_request"), dict) else {}
    transformed = data.get("transformed_request") if isinstance(data.get("transformed_request"), dict) else {}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    current = _response_message(data)
    turns = _extract_turns(messages, current)
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    return {
        "completion_id": data.get("completion_id"),
        "timestamp": data.get("timestamp"),
        "session_id": data.get("session_id"),
        "task_id": data.get("task_id"),
        "model_requested": data.get("model_requested"),
        "model_used": data.get("model_used"),
        "response_truncated": bool(response.get("__truncated")),
        "original_messages_count": len(messages),
        "transformed_messages_count": len(transformed.get("messages") or []),
        "turns": turns[-20:],
        "latest_original_messages": messages[-8:],
        "response_message": current,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Polar Rollout Observer</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --panel-alt: #fbfcfe;
      --line: #d9dde3;
      --text: #1f2328;
      --muted: #6b7280;
      --header: #ffffff;
      --button: #ffffff;
      --button-hover: #f0f3f7;
      --session-hover: #eef4ff;
      --pre-bg: #0b1020;
      --pre-text: #d8e2ff;
      --green: #107c41;
      --red: #b42318;
      --amber: #a15c00;
      --blue: #155eef;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --sidebar-width: 430px;
    }
    body.dark {
      --bg: #0f1319;
      --panel: #171c24;
      --panel-alt: #121720;
      --line: #303845;
      --text: #e6edf3;
      --muted: #9aa7b5;
      --header: #151a22;
      --button: #202733;
      --button-hover: #283241;
      --session-hover: #1d2a3d;
      --pre-bg: #060912;
      --pre-text: #dbe7ff;
      --green: #5fc782;
      --red: #ff8b7f;
      --amber: #f3bd5e;
      --blue: #8bb4ff;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; overflow:hidden; }
    header { height: 54px; display:flex; align-items:center; gap:16px; padding:0 18px; border-bottom:1px solid var(--line); background:var(--header); position:sticky; top:0; z-index:5; }
    header h1 { font-size: 16px; margin:0; font-weight:650; }
    .header-left { display:flex; align-items:center; gap:14px; min-width:0; }
    .header-controls { margin-left:auto; display:flex; align-items:center; gap:14px; }
    .switch { display:inline-flex; align-items:center; gap:7px; color:var(--muted); font-size:12px; cursor:pointer; user-select:none; }
    .switch input { position:absolute; opacity:0; pointer-events:none; }
    .switch-track { width:34px; height:19px; border-radius:999px; background:#b8c0cc; position:relative; transition:background .16s ease; flex:0 0 auto; }
    .switch-track::after { content:""; position:absolute; width:15px; height:15px; left:2px; top:2px; border-radius:50%; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.25); transition:transform .16s ease; }
    .switch input:checked + .switch-track { background:var(--blue); }
    .switch input:checked + .switch-track::after { transform:translateX(15px); }
    .icon-button { width:34px; height:34px; display:inline-flex; align-items:center; justify-content:center; border-radius:50%; padding:0; font-size:16px; }
    .sub { color:var(--muted); font-size:12px; }
    .layout { display:grid; grid-template-columns: var(--sidebar-width) 6px minmax(0, 1fr); height: calc(100vh - 54px); min-height: 0; }
    aside { background:var(--panel); overflow:hidden; display:flex; flex-direction:column; min-height:0; }
    .resizer { background:#edf0f4; border-left:1px solid var(--line); border-right:1px solid var(--line); cursor:col-resize; position:relative; }
    .resizer::after { content:""; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:2px; height:38px; border-left:1px solid #aeb6c2; border-right:1px solid #aeb6c2; }
    .resizer:hover, body.resizing .resizer { background:#dfe7ff; }
    body.resizing { cursor:col-resize; user-select:none; }
    main { overflow:auto; padding:16px; }
    .toolbar { display:flex; align-items:center; gap:8px; padding:12px; border-bottom:1px solid var(--line); background:var(--panel); flex-shrink:0; }
    .toolbar.secondary { padding-top:0; }
    .filter-tabs { display:flex; align-items:center; gap:6px; padding:8px 12px; border-bottom:1px solid var(--line); background:var(--panel-alt); flex-shrink:0; }
    .filter-tabs button { padding:5px 8px; font-size:12px; }
    .filter-tabs button.active { border-color:#9db8ff; color:var(--blue); background:#eff4ff; }
    #sessionSummary { margin-left:auto; }
    #sessions { overflow:auto; min-height:0; flex:1; }
    input[type="search"], select { width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:6px; font:inherit; background:var(--panel); color:var(--text); min-width:0; }
    select { font-size:12px; }
    button { border:1px solid var(--line); background:var(--button); color:var(--text); border-radius:6px; padding:7px 10px; cursor:pointer; }
    button:hover { background:var(--button-hover); }
    button.active { border-color:#9db8ff; color:var(--blue); background:#eff4ff; }
    .session { padding:12px; border-bottom:1px solid var(--line); cursor:pointer; }
    .session:hover, .session.active { background:var(--session-hover); }
    .row { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .sid { font-family:var(--mono); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .task { font-weight:650; }
    .badge { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:2px 7px; font-size:12px; background:var(--panel); color:var(--muted); white-space:nowrap; }
    .badge.green { color:var(--green); border-color:#b7e2c8; background:#effaf3; }
    .badge.red { color:var(--red); border-color:#ffd0cc; background:#fff3f1; }
    .badge.amber { color:var(--amber); border-color:#f5d08a; background:#fff8e8; }
    .badge.blue { color:var(--blue); border-color:#c9d8ff; background:#eff4ff; }
    body.dark .badge.green { background:#10291a; border-color:#265c39; }
    body.dark .badge.red { background:#321916; border-color:#68322c; }
    body.dark .badge.amber { background:#332511; border-color:#6b501f; }
    body.dark .badge.blue { background:#14233d; border-color:#315487; }
    .grid { display:grid; gap:12px; }
    .metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin-bottom:14px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
    .metric .label { color:var(--muted); font-size:12px; }
    .metric .value { font-family:var(--mono); font-size:16px; margin-top:4px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:14px; }
    .panel h2 { margin:0; padding:12px 14px; font-size:14px; border-bottom:1px solid var(--line); }
    .panel h3 { margin:14px 0 6px; font-size:13px; color:var(--text); }
    .panel h3:first-child { margin-top:0; }
    .panel-body { padding:12px 14px; }
    .hint { color:var(--muted); font-size:12px; margin:0 0 10px; line-height:1.45; }
    .turn { border-top:1px solid var(--line); padding:12px 14px; }
    .turn:first-child { border-top:0; }
    .turn-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
    .text { white-space:pre-wrap; line-height:1.45; }
    .mono { font-family:var(--mono); }
    pre { margin:8px 0 0; padding:10px; background:var(--pre-bg); color:var(--pre-text); border-radius:6px; overflow:auto; max-height:420px; line-height:1.45; font-family:var(--mono); font-size:12px; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
    body.no-wrap pre { white-space:pre; overflow-wrap:normal; word-break:normal; }
    details { margin-top:8px; }
    summary { cursor:pointer; color:var(--blue); }
    .tool { border-left:3px solid #c9d8ff; padding:8px 10px; background:var(--panel-alt); margin:6px 0; }
    .result { border-left:3px solid var(--line); padding:8px 10px; background:var(--panel-alt); margin:6px 0; }
    .danger { border-left-color:#ffaaa3; background:var(--panel-alt); }
    .timeline-empty { color:var(--muted); padding:20px; }
    .files { max-height:240px; overflow:auto; }
    .file-row { display:grid; grid-template-columns: 1fr auto auto; gap:8px; padding:6px 0; border-top:1px solid var(--line); align-items:center; }
    .file-row:first-child { border-top:0; }
    .small { font-size:12px; color:var(--muted); }
    .split { display:grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 420px); gap:14px; align-items:start; }
    @media (max-width: 980px) {
      body { overflow:auto; }
      header { height:auto; min-height:54px; flex-wrap:wrap; padding:10px 12px; }
      .header-controls { margin-left:0; width:100%; justify-content:flex-end; }
      .layout { grid-template-columns: 1fr; height:auto; }
      aside { height:45vh; border-right:0; border-bottom:1px solid var(--line); }
      .resizer { display:none; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <h1>Polar Rollout Observer</h1>
      <span id="health" class="badge">loading</span>
      <span id="root" class="sub"></span>
      <span id="updated" class="sub"></span>
    </div>
    <div class="header-controls">
      <label class="switch" title="Refresh only the session list automatically">
        <span>Auto</span>
        <input id="autoRefresh" type="checkbox" checked />
        <span class="switch-track"></span>
      </label>
      <label class="switch" title="Wrap long prompt, command, and log lines">
        <span>Wrap</span>
        <input id="wrapText" type="checkbox" checked />
        <span class="switch-track"></span>
      </label>
      <button id="themeToggle" class="icon-button" title="Toggle dark mode">☾</button>
    </div>
  </header>
  <div class="layout">
    <aside>
      <div class="toolbar">
        <input id="filter" type="search" placeholder="Filter session/task/run/status" />
        <button id="refresh">Refresh</button>
      </div>
      <div class="toolbar secondary">
        <select id="runFilter" title="Filter by run id">
          <option value="ALL">All runs</option>
        </select>
        <select id="timeFilter" title="Filter by latest completion time">
          <option value="ALL">All time</option>
          <option value="1">Last 1h</option>
          <option value="6">Last 6h</option>
          <option value="24">Last 24h</option>
          <option value="168">Last 7d</option>
        </select>
      </div>
      <div class="filter-tabs">
        <button class="status-filter active" data-status="ALL">All</button>
        <button class="status-filter" data-status="RUNNING">Running</button>
        <button class="status-filter" data-status="ABNORMAL">Abnormal</button>
        <button class="status-filter" data-status="TIMEOUT">Timeout</button>
        <button class="status-filter" data-status="ON_DISK">On disk</button>
        <span id="sessionSummary" class="small"></span>
      </div>
      <div id="sessions"></div>
    </aside>
    <div id="resizer" class="resizer" title="Drag to resize sidebar"></div>
    <main>
      <div id="detail" class="timeline-empty">Select a session.</div>
    </main>
  </div>
  <script>
    let state = null;
    let selected = null;
    let detail = null;
    let statusFilter = 'ALL';
    let runFilter = localStorage.getItem('polarObserverRunFilter') || 'ALL';
    let timeFilter = localStorage.getItem('polarObserverTimeFilter') || 'ALL';
    let autoRefresh = true;
    let wrapText = true;
    let theme = 'light';
    let refreshTimer = null;
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmtAge = (v) => v == null ? '' : `${Math.round(v)}s`;
    const fmtNum = (v, digits=0) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return '-';
      return n.toLocaleString(undefined, {maximumFractionDigits: digits});
    };
    const fmtRate = (v) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return '-';
      return n.toFixed(n >= 100 ? 0 : 1);
    };
    const isTimeoutStatus = (s) => String(s || '').includes('TIMEOUT');
    const isOnDiskStatus = (s) => String(s || '').includes('ON_DISK');
    const isAbnormalStatus = (s) => String(s || '').includes('ABNORMAL');
    const clsStatus = (s) => s === 'RUNNING' ? 'green' : (isAbnormalStatus(s) || isTimeoutStatus(s) || s === 'ERROR' || s === 'ERROR_ON_DISK' ? 'red' : (isOnDiskStatus(s) ? '' : 'blue'));
    const statusRank = (s) => s === 'RUNNING' ? 0 : (isAbnormalStatus(s) || isTimeoutStatus(s) || s === 'ERROR' || s === 'ERROR_ON_DISK' ? 1 : (isOnDiskStatus(s) ? 3 : 2));
    function logicalTaskKey(session) {
      const task = String(session?.task_id || '');
      const run = String(session?.run_id || '');
      const pool = task.match(/--g(\d+)-sp(\d+)$/);
      if (pool) {
        return [run, Number(pool[1]), Number(pool[2]), task, String(session?.session_id || '')];
      }
      const op = task.match(/-polar-op-(\d+)-(\d+)(?:$|--)/);
      if (op) {
        return [run, Number(op[1]), Number(op[2]), task, String(session?.session_id || '')];
      }
      return [run, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, task, String(session?.session_id || '')];
    }
    function compareLogicalTask(a, b) {
      const ak = logicalTaskKey(a);
      const bk = logicalTaskKey(b);
      for (let i = 0; i < ak.length; i += 1) {
        if (typeof ak[i] === 'number' || typeof bk[i] === 'number') {
          const av = Number(ak[i]);
          const bv = Number(bk[i]);
          if (av !== bv) return av - bv;
        } else {
          const cmp = String(ak[i]).localeCompare(String(bk[i]), undefined, {numeric: true});
          if (cmp) return cmp;
        }
      }
      return 0;
    }
    function clampSidebarWidth(value) {
      const max = Math.max(360, Math.floor(window.innerWidth * 0.72));
      return Math.min(Math.max(value, 280), max);
    }
    function setSidebarWidth(value, persist=false) {
      const width = clampSidebarWidth(value);
      document.documentElement.style.setProperty('--sidebar-width', `${width}px`);
      if (persist) localStorage.setItem('polarObserverSidebarWidth', String(width));
    }
    function initResizer() {
      const saved = Number(localStorage.getItem('polarObserverSidebarWidth'));
      if (Number.isFinite(saved) && saved > 0) setSidebarWidth(saved);
      const handle = $('resizer');
      let dragging = false;
      handle.addEventListener('pointerdown', (event) => {
        dragging = true;
        document.body.classList.add('resizing');
        handle.setPointerCapture(event.pointerId);
      });
      handle.addEventListener('pointermove', (event) => {
        if (!dragging) return;
        setSidebarWidth(event.clientX);
      });
      function finishDrag(event) {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove('resizing');
        setSidebarWidth(event.clientX, true);
        try { handle.releasePointerCapture(event.pointerId); } catch (_) {}
      }
      handle.addEventListener('pointerup', finishDrag);
      handle.addEventListener('pointercancel', finishDrag);
      window.addEventListener('resize', () => {
        const current = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width'), 10);
        if (Number.isFinite(current)) setSidebarWidth(current, true);
      });
    }
    function initWrapToggle() {
      const saved = localStorage.getItem('polarObserverWrapText');
      wrapText = saved == null ? true : saved !== 'false';
      document.body.classList.toggle('no-wrap', !wrapText);
      $('wrapText').checked = wrapText;
      $('wrapText').onchange = () => {
        wrapText = $('wrapText').checked;
        localStorage.setItem('polarObserverWrapText', String(wrapText));
        document.body.classList.toggle('no-wrap', !wrapText);
      };
    }
    function initThemeToggle() {
      const saved = localStorage.getItem('polarObserverTheme');
      theme = saved === 'dark' ? 'dark' : 'light';
      document.body.classList.toggle('dark', theme === 'dark');
      $('themeToggle').textContent = theme === 'dark' ? '☀' : '☾';
      $('themeToggle').title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
      $('themeToggle').onclick = () => {
        theme = theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('polarObserverTheme', theme);
        document.body.classList.toggle('dark', theme === 'dark');
        $('themeToggle').textContent = theme === 'dark' ? '☀' : '☾';
        $('themeToggle').title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
      };
    }
    function detailOpenState() {
      const out = new Set();
      document.querySelectorAll('#detail details[data-key]').forEach(el => {
        if (el.open) out.add(el.dataset.key);
      });
      return out;
    }
    function restoreDetailOpenState(openSet) {
      document.querySelectorAll('#detail details[data-key]').forEach(el => {
        if (openSet.has(el.dataset.key)) el.open = true;
      });
    }
    async function fetchJson(url) {
      const r = await fetch(url, {cache:'no-store'});
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return await r.json();
    }
    async function loadState(options = {}) {
      try {
        const updateDetail = options.updateDetail ?? true;
        state = await fetchJson('/api/state');
        renderHeader();
        renderSessions();
        if (!selected && state.sessions.length) selected = state.sessions[0].session_id;
        if (selected && updateDetail) await loadDetail(selected);
      } catch (e) {
        $('health').className = 'badge red';
        $('health').textContent = `error ${e}`;
      }
    }
    async function loadDetail(sid) {
      try {
        const openSet = detailOpenState();
        detail = await fetchJson(`/api/session/${encodeURIComponent(sid)}`);
        selected = sid;
        renderSessions();
        renderDetail();
        restoreDetailOpenState(openSet);
      } catch (e) {
        $('detail').innerHTML = `<div class="panel"><div class="panel-body">Failed to load session: ${esc(e)}</div></div>`;
      }
    }
    function renderHeader() {
      const h = state.gateway_health || state.health || {};
      const ok = h.status === 'ok';
      $('health').className = `badge ${ok ? 'green':'red'}`;
      const counts = h.active_status_counts || {};
      $('health').textContent = ok ? `gateway ok · running ${counts.RUNNING || 0}` : `gateway error`;
      $('root').textContent = state.root;
      $('updated').textContent = `updated ${new Date().toLocaleTimeString()}`;
    }
    function runLabel(group) {
      const base = `${group.run_id} (${group.sessions})`;
      return group.running ? `${base} · ${group.running} running` : base;
    }
    function renderRunFilter() {
      const select = $('runFilter');
      const groups = state?.session_groups || [];
      const latest = state?.latest_run_id || '';
      const values = new Set(['ALL']);
      if (latest) values.add('LATEST');
      groups.forEach(group => values.add(String(group.run_id)));
      if (!values.has(runFilter)) runFilter = 'ALL';
      if (!['ALL', '1', '6', '24', '168'].includes(timeFilter)) timeFilter = 'ALL';
      select.innerHTML = [
        `<option value="ALL">All runs</option>`,
        latest ? `<option value="LATEST">Latest run · ${esc(latest)}</option>` : '',
        ...groups.map(group => `<option value="${esc(group.run_id)}">${esc(runLabel(group))}</option>`),
      ].join('');
      select.value = runFilter;
      $('timeFilter').value = timeFilter;
    }
    function selectedRunId() {
      if (runFilter === 'LATEST') return state?.latest_run_id || null;
      if (runFilter === 'ALL') return null;
      return runFilter;
    }
    function passesTimeFilter(session) {
      if (timeFilter === 'ALL') return true;
      if (session.status === 'RUNNING') return true;
      const hours = Number(timeFilter);
      const latest = Number(session.latest_mtime || 0);
      const now = Number(state?.now || Date.now() / 1000);
      if (!Number.isFinite(hours) || !latest) return false;
      return latest >= now - hours * 3600;
    }
    function renderSessions() {
      renderRunFilter();
      const q = $('filter').value.toLowerCase();
      const all = (state?.sessions || []).slice().sort((a,b) => {
        const lr = compareLogicalTask(a, b);
        if (lr) return lr;
        return statusRank(a.status) - statusRank(b.status);
      });
      const running = all.filter(s => s.status === 'RUNNING').length;
      const selectedRun = selectedRunId();
      const sessions = all.filter(s => {
        if (statusFilter === 'ABNORMAL' && !isAbnormalStatus(s.status)) return false;
        if (statusFilter === 'TIMEOUT' && !isTimeoutStatus(s.status)) return false;
        if (statusFilter === 'ON_DISK' && !isOnDiskStatus(s.status)) return false;
        if (!['ALL', 'ABNORMAL', 'TIMEOUT', 'ON_DISK'].includes(statusFilter) && s.status !== statusFilter) return false;
        if (selectedRun && s.run_id !== selectedRun) return false;
        if (!passesTimeFilter(s)) return false;
        return JSON.stringify(s).toLowerCase().includes(q);
      });
      $('sessionSummary').textContent = `${running} running / ${sessions.length} shown / ${all.length} total`;
      $('sessions').innerHTML = sessions.map(s => {
        const val = s.quality?.validation || {};
	        const forbidden = s.quality?.forbidden_writes || 0;
	        const readonlyMut = s.quality?.readonly_mutation_attempts || 0;
	        const submissionWrites = s.quality?.submission_writes || 0;
	        const writesBeforePipeline = s.quality?.writes_before_first_pipeline || 0;
	        const workflowViolation = !!s.quality?.workflow_violation;
	        const docDrift = s.quality?.doc_drift_count || 0;
	        const stages = s.quality?.pipeline_stage_counts || {};
        const precision = stages.precision || {};
        const profiling = stages.profiling || {};
        const cm = s.completion_metrics || {};
        const timeoutCount = Number(s.timeout_count || 0);
        const timeoutBadge = timeoutCount ? `<span class="badge red">timeouts ${esc(timeoutCount)}${s.timeout_last_time ? ' · ' + esc(s.timeout_last_time) : ''}</span>` : '';
        const abnormalReasons = s.quality?.abnormal_reasons || [];
        const abnormalBadge = s.quality?.abnormal_termination ? `<span class="badge red">abnormal ${esc(abnormalReasons.join(', ') || 'yes')}</span>` : '';
        return `<div class="session ${s.session_id===selected?'active':''}" data-sid="${esc(s.session_id)}">
          <div class="row"><span class="task">${esc(s.task_id)}</span><span class="badge ${clsStatus(s.status)}">${esc(s.status)}</span></div>
          <div class="sid">${esc(s.session_id)}</div>
          <div class="row small"><span>turns/files ${esc(s.completion_count)} / ${esc(s.completion_files)}</span><span>idle ${esc(fmtAge(s.idle_seconds))}</span></div>
          <div class="row small"><span>req ${esc(fmtNum(cm.request_count))} · prompt ${esc(fmtNum(cm.prompt_tokens))} · decode ${esc(fmtNum(cm.completion_tokens))}</span><span>${esc(fmtRate(cm.completion_tokens_per_second))} tok/s</span></div>
          <div class="row small"><span>runtime ${esc(s.runtime_card ?? '-')} · eval ${esc(s.eval_card ?? '-')}</span><span>${esc(s.latest_time || '')}</span></div>
          <div style="margin-top:6px; display:flex; gap:5px; flex-wrap:wrap">
	            <span class="badge blue">run ${esc(s.run_id || 'legacy')}</span>
	            <span class="badge blue">pipeline ${esc(s.quality?.pipeline_runs || 0)}</span>
	            <span class="badge ${workflowViolation ? 'red':(writesBeforePipeline > 1 ? 'amber':'')}">writes ${esc(submissionWrites)}</span>
	            <span class="badge ${docDrift ? 'amber':''}">doc drift ${esc(docDrift)}</span>
	            <span class="badge ${precision.pass ? 'green':(precision.fail ? 'amber':'')}">precision ${esc(precision.pass || 0)}/${esc(precision.attempts || 0)}</span>
            <span class="badge ${profiling.pass ? 'green':(profiling.fail ? 'amber':'')}">profile ${esc(profiling.pass || 0)}/${esc(profiling.attempts || 0)}</span>
            <span class="badge ${val.verify_fail ? 'amber':''}">verify ${esc(val.verify_fail || 0)}</span>
            <span class="badge ${forbidden ? 'red':''}">forbidden writes ${esc(forbidden)}</span>
            <span class="badge ${readonlyMut ? 'red':''}">readonly mutations ${esc(readonlyMut)}</span>
            ${abnormalBadge}
            ${timeoutBadge}
          </div>
        </div>`;
      }).join('') || `<div class="timeline-empty">No sessions found.</div>`;
      document.querySelectorAll('.session').forEach(el => el.onclick = () => loadDetail(el.dataset.sid));
    }
    function renderDetail() {
      if (!detail) return;
      const sum = detail.summary || {};
      const req = sum.request || {};
      const v = sum.validation || {};
      const tc = sum.action_tool_counts || sum.tool_counts || {};
      const planTc = sum.plan_tool_counts || {};
      const forbiddenWrites = sum.forbidden_writes || [];
      const readonlyMutations = sum.readonly_mutation_attempts || [];
      const abnormalEvents = sum.abnormal_events || [];
	      const stages = sum.pipeline_stage_counts || {};
	      const precision = stages.precision || {};
	      const profiling = stages.profiling || {};
	      const workflowViolation = !!sum.workflow_violation;
	      const firstWrite = sum.first_submission_write_turn || '-';
	      const firstPipeline = sum.first_pipeline_turn || '-';
	      const cm = detail.completion_metrics || {};
      const metric = (label, value, extra='') => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>${extra ? `<div class="small">${esc(extra)}</div>`:''}</div>`;
      const toolChips = Object.entries(tc).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `<span class="badge">${esc(k)} ${esc(v)}</span>`).join(' ');
      const planToolChips = Object.entries(planTc).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `<span class="badge">${esc(k)} ${esc(v)}</span>`).join(' ');
      const skillCalls = (sum.skill_calls || []).map(c => `<div class="result">
        <span class="badge blue">turn ${esc(c.turn)}</span> <span class="mono">${esc(c.skill || '(unknown skill)')}</span>
        ${c.args ? `<pre>${esc(c.args)}</pre>` : ''}
      </div>`).join('');
      const skillReads = (sum.skill_reference_reads || []).map(r => `<div class="file-row">
        <span class="mono">${esc(r.path)}</span><span class="small">turn ${esc(r.turn)}</span><span></span>
      </div>`).join('');
      const forbiddenWriteRows = forbiddenWrites.map(item => `<div class="result danger">
        <span class="badge red">turn ${esc(item.turn)}</span> <span class="badge">${esc(item.tool || '')}</span>
        <pre>${esc(item.path || '')}</pre>
      </div>`).join('');
      const readonlyMutationRows = readonlyMutations.map(item => `<div class="result danger">
        <span class="badge red">turn ${esc(item.turn)}</span> <span class="badge">${esc(item.kind || '')}</span>
        <div class="mono">${esc(item.path || '')}</div>
        <pre>${esc(item.command || '')}</pre>
      </div>`).join('');
      const abnormalRows = abnormalEvents.map(item => `<div class="result danger">
        <span class="badge red">${esc(item.reason || 'abnormal')}</span>
        ${item.turn ? `<span class="badge">turn ${esc(item.turn)}</span>` : ''}
        ${item.finish_reason ? `<span class="badge">finish ${esc(item.finish_reason)}</span>` : ''}
        ${item.source ? `<span class="badge">${esc(item.source)}</span>` : ''}
        ${item.snippet ? `<pre>${esc(item.snippet)}</pre>` : ''}
      </div>`).join('');
      const promptBlocks = (req.prompt_blocks || []).map(b => `<details data-key="prompt-block-${esc(b.index)}">
        <summary>${esc(b.label || ('Context ' + b.index))} · ${esc(b.type)} · ${esc(b.chars)} chars</summary>
        <pre>${esc(b.text || '')}</pre>
      </details>`).join('');
      const currentTools = (req.current_tool_calls || []).map(t => `<div class="tool"><span class="badge blue">${esc(t.name || 'tool')}</span><pre>${esc(t.target || JSON.stringify(t.input || {}, null, 2))}</pre></div>`).join('');
      const files = (detail.completions || []).slice(-35).reverse().map(f => `<div class="file-row">
        <span class="mono">${esc(f.file)}</span><span class="small">${esc(f.time)}</span><span class="small">${Math.round((f.size||0)/1024)}KB</span>
      </div>`).join('');
      const pipelineDetails = (sum.pipeline_runs_detail || []).slice().reverse().map((run, idx) => {
        const cls = run.status === 'success' ? 'green' : (run.status === 'unknown' ? '' : 'red');
        const shouldOpen = idx === 0 && run.status !== 'success';
        const labels = (run.labels || []).map(l => `<span class="badge ${l==='success'?'green':(l.includes('fail')?'amber':'blue')}">${esc(l)}</span>`).join(' ');
        return `<details data-key="pipeline-run-${esc(run.index)}" ${shouldOpen ? 'open' : ''}>
          <summary><span class="badge ${cls}">#${esc(run.index)} ${esc(run.status)}</span> <span class="small">precision ${esc(run.precision_status || '-')} · profile ${esc(run.profiling_status || '-')} · turn ${esc(run.turn)} · ${esc(run.result_chars || 0)} chars</span></summary>
          <div class="result">
            <div>${labels || '<span class="badge">no labels</span>'}</div>
            <h3>Command</h3>
            <pre>${esc(run.command || '')}</pre>
            <h3>Full Feedback</h3>
            <pre>${esc(run.result || '(no tool result captured yet)')}</pre>
          </div>
        </details>`;
      }).join('');
      const pipelineCommandErrors = (sum.pipeline_command_errors || []).slice().reverse().map((run, idx) => {
        const shouldOpen = idx === 0;
        return `<details data-key="pipeline-command-error-${esc(idx)}" ${shouldOpen ? 'open' : ''}>
          <summary><span class="badge red">command error</span> <span class="small">turn ${esc(run.turn)} · ${esc(run.result_chars || 0)} chars</span></summary>
          <div class="result">
            <h3>Command</h3>
            <pre>${esc(run.command || '')}</pre>
            <h3>Tool Result</h3>
            <pre>${esc(run.result_snippet || '(no tool result captured yet)')}</pre>
          </div>
        </details>`;
      }).join('');
      const turns = (sum.turns || []).slice().reverse().map(t => {
        const tools = (t.tool_uses || []).map(tool => {
          const name = tool.name || '';
          const target = (tool.input && (tool.input.file_path || tool.input.command || tool.input.skill || tool.input.args)) || JSON.stringify(tool.input || {});
          return `<div class="tool"><span class="badge blue">${esc(name)}</span><pre>${esc(typeof target === 'string' ? target : JSON.stringify(target, null, 2))}</pre></div>`;
        }).join('');
        const results = (t.tool_results || []).map((r, idx) => `<details data-key="turn-${esc(t.index)}-result-${idx}"><summary>tool result ${r.is_error ? '(error)' : ''}</summary><pre>${esc(r.content || '')}</pre></details>`).join('');
        const reasoning = t.assistant_reasoning ? `<details data-key="turn-${esc(t.index)}-reasoning"><summary>reasoning</summary><pre>${esc(t.assistant_reasoning || '')}</pre></details>` : '';
        const textBody = t.assistant_text || (t.assistant_reasoning ? '' : '(no assistant text)');
        return `<div class="turn">
          <div class="turn-head"><div><span class="badge">turn ${esc(t.index)}</span> <span class="badge">${esc(t.source)}</span></div><span class="small">${esc((t.tool_uses||[]).length)} tools</span></div>
          <div class="text">${esc(t.assistant_snippet || '(no assistant text)')}</div>
          ${tools}
          ${results}
          ${reasoning}
          <details data-key="turn-${esc(t.index)}-assistant"><summary>full assistant text</summary><pre>${esc(textBody)}</pre></details>
        </div>`;
      }).join('');
      $('detail').innerHTML = `
        <div class="metrics">
          ${metric('session', detail.session_id, detail.task_id)}
          ${metric('messages / turns', `${sum.messages_count || 0} / ${sum.turns_count || 0}`)}
          ${metric('LLM requests', fmtNum(cm.request_count), `latest #${cm.latest?.sequence || '-'}`)}
          ${metric('prompt / decode tokens', `${fmtNum(cm.prompt_tokens)} / ${fmtNum(cm.completion_tokens)}`, `cached prompt ${fmtNum(cm.cached_prompt_tokens)}`)}
	          ${metric('decode throughput', `${fmtRate(cm.completion_tokens_per_second)} tok/s`, `mean latency ${fmtNum(cm.latency_ms_mean, 1)} ms`)}
	          ${metric('pipeline runs', sum.pipeline_runs || 0, 'budget-counted tools/triton_eval_pipeline.sh feedbacks')}
	          ${metric('submission writes', sum.submission_writes || 0, `before first pipeline ${sum.writes_before_first_pipeline || 0}`)}
	          ${metric('workflow', workflowViolation ? 'violation' : 'ok', `first write turn ${firstWrite} · first pipeline turn ${firstPipeline}`)}
	          ${metric('abnormal', sum.abnormal_termination ? 'yes' : 'no', (sum.abnormal_reasons || []).join(', ') || '-')}
	          ${metric('doc drift', sum.doc_drift_count || 0, `turns ${(sum.doc_drift_turns || []).join(', ') || '-'}`)}
	          ${metric('precision verify', `${precision.pass || 0} / ${precision.attempts || 0}`, `${precision.fail || 0} failed · ${precision.unknown || 0} unknown`)}
          ${metric('profiling', `${profiling.pass || 0} / ${profiling.attempts || 0}`, `${profiling.fail || 0} failed · ${profiling.unknown || 0} unknown`)}
          ${metric('forbidden writes', forbiddenWrites.length || 0, 'Write/Edit/MultiEdit to protected paths')}
          ${metric('readonly mutations', readonlyMutations.length || 0, 'Bash write attempts against readonly paths')}
        </div>
        <div class="panel"><h2>Prompt / Latest Model Step</h2><div class="panel-body">
          <p class="hint">本块展示 Polar/Claude Code 当前这一轮真正送给模型的上下文：固定 skills、CLAUDE.md 工作流、真实算子任务，以及最新模型回复。</p>
          <div class="row small"><span>model ${esc(req.model || '')}</span><span>max_tokens ${esc(req.max_tokens || '')}</span></div>
          <h3>Task Prompt</h3>
          <pre>${esc(req.task_prompt || '')}</pre>
          <h3>Injected User Message</h3>
          <p class="hint">这里是完整首轮 user message 的各段。Operator Task、skill 文档和 CLAUDE.md 会分开标记；上面的 Task Prompt 优先展示真实算子任务。</p>
          ${promptBlocks || '<span class="small">No user prompt parsed</span>'}
	          <h3>Latest Assistant Text</h3>
	          ${req.current_response_text ? `<pre>${esc(req.current_response_text || '')}</pre>` : '<div class="result"><span class="small">No text in latest assistant message; this step is likely tool-call only.</span></div>'}
	          ${req.current_response_reasoning ? `<h3>Latest Reasoning</h3><pre>${esc(req.current_response_reasoning || '')}</pre>` : ''}
	          <h3>Latest Tool Calls</h3>
          ${currentTools || '<span class="small">No tool calls in latest assistant message</span>'}
        </div></div>
        <div class="split">
          <div>
            <div class="panel"><h2>Action Tools</h2><div class="panel-body"><p class="hint">真正会读写文件、调用 skill 或执行命令的工具。TaskCreate/TaskUpdate 这类计划工具不混在这里。</p>${toolChips || '<span class="small">No action tools yet</span>'}</div></div>
            <div class="panel"><h2>Plan Tools</h2><div class="panel-body"><p class="hint">Claude Code 自己维护待办列表用的工具，不代表 sub-agent 或 Polar task 调用。</p>${planToolChips || '<span class="small">No plan tools yet</span>'}</div></div>
            <div class="panel"><h2>Readonly Audit</h2><div class="panel-body"><p class="hint">标记模型尝试写入 tools/、.agents/skills/ 或 CLAUDE.md 的行为；只读 mount 是硬边界，这里只做审计展示。</p><h3>Forbidden Writes</h3>${forbiddenWriteRows || '<span class="small">No forbidden write tool call found</span>'}<h3>Readonly Mutation Attempts</h3>${readonlyMutationRows || '<span class="small">No readonly mutation bash command found</span>'}</div></div>
            <div class="panel"><h2>Abnormal Termination</h2><div class="panel-body"><p class="hint">Observer-only 标记：这些 session 仍可能已进入训练，但不应在审查界面里显示成正常 completed。</p>${abnormalRows || '<span class="small">No abnormal termination marker found</span>'}</div></div>
            <div class="panel"><h2>Skill Usage</h2><div class="panel-body"><p class="hint">列出实际调用的 Skill 工具和读取过的 skill reference。正常轨迹通常至少会看到 designer/coding/verifier，优化阶段才会看到 optimizer。</p><h3>Skill Calls</h3>${skillCalls || '<span class="small">No Skill tool call found</span>'}<h3>Reference Reads</h3><div class="files">${skillReads || '<span class="small">No skill reference read found</span>'}</div></div></div>
	            <div class="panel"><h2>Pipeline Runs</h2><div class="panel-body"><p class="hint">只统计产生 pipeline feedback 的 tools/triton_eval_pipeline.sh 调用；路径错误等 Bash 失败单独列在 Command Errors。</p>${pipelineDetails || '<span class="small">No pipeline run found yet</span>'}<h3>Command Errors</h3>${pipelineCommandErrors || '<span class="small">No pipeline command error found</span>'}</div></div>
            <div class="panel"><h2>Timeline</h2><div class="panel-body"><p class="hint">按倒序展示每一轮模型输出、工具调用和工具返回。这里用来看 CC 是否按设计、编码、验证、迭代的流程推进。</p></div>${turns || '<div class="timeline-empty">No turns parsed yet.</div>'}</div>
          </div>
          <div>
            <div class="panel"><h2>Completion Files</h2><div class="panel-body files"><p class="hint">Polar 每次模型 completion 落盘的 JSON。数量增长说明 session 还在继续推进。</p>${files || '<span class="small">No completion files yet</span>'}</div></div>
            <div class="panel"><h2>Gateway Tail</h2><div class="panel-body"><p class="hint">Polar gateway 日志尾部，主要用于确认 session、容器、runtime/eval 卡和请求状态。</p><pre>${esc(detail.log_tail?.gateway || '')}</pre></div></div>
          </div>
        </div>`;
    }
    $('refresh').onclick = () => loadState({updateDetail: true});
    $('filter').oninput = renderSessions;
    $('runFilter').onchange = () => {
      runFilter = $('runFilter').value || 'ALL';
      localStorage.setItem('polarObserverRunFilter', runFilter);
      renderSessions();
    };
    $('timeFilter').onchange = () => {
      timeFilter = $('timeFilter').value || 'ALL';
      localStorage.setItem('polarObserverTimeFilter', timeFilter);
      renderSessions();
    };
    document.querySelectorAll('.status-filter').forEach(btn => {
      btn.onclick = () => {
        statusFilter = btn.dataset.status || 'ALL';
        document.querySelectorAll('.status-filter').forEach(b => b.classList.toggle('active', b === btn));
        renderSessions();
      };
    });
    initResizer();
    initWrapToggle();
    initThemeToggle();
    loadState();
    $('autoRefresh').onchange = () => {
      autoRefresh = $('autoRefresh').checked;
    };
    refreshTimer = setInterval(() => {
      if (autoRefresh) loadState({updateDetail: false});
    }, 5000);
  </script>
</body>
</html>
"""


class ObserverHandler(BaseHTTPRequestHandler):
    store: ObserverStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[observer] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            _json_response(self, self.store.session_summary())
            return
        if path.startswith("/api/session/"):
            session_id = urllib.parse.unquote(path.removeprefix("/api/session/"))
            payload = self.store.get_session(session_id)
            if payload is None:
                _json_response(self, {"error": "session not found"}, 404)
                return
            _json_response(self, payload)
            return
        if path.startswith("/api/completion/"):
            parts = path.removeprefix("/api/completion/").split("/", 1)
            if len(parts) != 2:
                _json_response(self, {"error": "expected /api/completion/{session}/{file}"}, 400)
                return
            session_id = urllib.parse.unquote(parts[0])
            file_name = urllib.parse.unquote(parts[1])
            payload = self.store.get_completion(session_id, file_name)
            if payload is None:
                _json_response(self, {"error": "completion not found"}, 404)
                return
            _json_response(self, payload)
            return
        if path.startswith("/api/log/"):
            name = urllib.parse.unquote(path.removeprefix("/api/log/"))
            qs = urllib.parse.parse_qs(parsed.query)
            lines = int(qs.get("lines", ["200"])[0])
            _json_response(self, {"name": name, "text": self.store.tail_log(name, lines)})
            return
        _json_response(self, {"error": "not found"}, 404)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override rollout result directory (default: ROOT/rollout_results).",
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)

    store = ObserverStore(args.root, args.gateway, args.results_dir)
    ObserverHandler.store = store
    server = ReusableThreadingHTTPServer((args.host, args.port), ObserverHandler)
    print(f"Polar rollout observer: http://{args.host}:{args.port}")
    print(f"root={args.root} results_dir={store.results_dir} gateway={args.gateway}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
