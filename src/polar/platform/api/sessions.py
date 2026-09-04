"""Session detail routes — pulls timing, trajectory, completions, evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _load_session_file(state, session_id: str) -> tuple[dict[str, Any] | None, Path | None]:
    path = state.fs_index.session_file_for(session_id)
    if path is None:
        return None, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), path
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None, path


def _session_trace_root(state) -> Path | None:
    configured = state.config.topology.gateway.persist_traces_dir
    if not configured:
        return None
    root = Path(configured)
    if not root.is_absolute():
        root = state.config.save_dir / root
    return root.resolve()


def _session_trace_file(state, session_id: str) -> Path | None:
    """Resolve an exact trace filename under the configured, bounded trace root."""
    if not session_id or Path(session_id).name != session_id:
        return None
    root = _session_trace_root(state)
    if root is None:
        return None
    candidate = (root / f"{session_id}.json").resolve()
    if candidate.parent != root or not candidate.is_file():
        return None
    return candidate


def _load_artifact_manifest(state, session_id: str) -> tuple[dict[str, Any], Path] | None:
    if not session_id or Path(session_id).name != session_id:
        return None
    root = _session_trace_root(state)
    if root is None:
        return None
    path = (root / f"{session_id}.artifacts.json").resolve()
    if path.parent != root or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        return None
    return payload, root


def _load_trace_document(path: Path) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read trace: {exc}") from exc
    if isinstance(payload, list):
        return 1, {}, [event for event in payload if isinstance(event, dict)]
    if not isinstance(payload, dict) or not isinstance(payload.get("traceEvents"), list):
        raise HTTPException(status_code=500, detail="Invalid trace document")
    return (
        int(payload.get("schemaVersion", 1)),
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        [event for event in payload["traceEvents"] if isinstance(event, dict)],
    )


async def _find_gateway_for_session(state, session_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Probe each gateway for the session; returns (node_id, payload) on the first hit."""
    for node_id, client in state.gateway_clients.items():
        try:
            payload = await client.get_json(f"/sessions/{session_id}")
        except Exception:
            continue
        if payload:
            return node_id, payload
    return None, None


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    data, path = _load_session_file(state, session_id)
    node_id, live = await _find_gateway_for_session(state, session_id)
    if data is None and live is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result: dict[str, Any] = {}
    if data is not None:
        result.update({
            "session_id": data.get("session_id", session_id),
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "node_id": data.get("node_id"),
            "error": data.get("error"),
            "timing": data.get("timing") or {},
            "metadata": data.get("metadata") or {},
            "file_path": str(path) if path else None,
            "source": "filesystem",
        })
    if live is not None:
        result.update({
            "session_id": live.get("session_id", session_id),
            "task_id": live.get("task_id") or result.get("task_id"),
            "status": live.get("status") or result.get("status"),
            "completion_count": live.get("completion_count"),
            "created_at": live.get("created_at"),
            "node_id": node_id or result.get("node_id"),
            "source": "live",
        })
        if isinstance(live.get("result"), dict) and not result.get("error"):
            result["error"] = live["result"].get("error")
    return result


@router.get("/sessions/{session_id}/trajectory")
async def get_session_trajectory(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    data, _path = _load_session_file(state, session_id)
    trajectory: dict[str, Any] | None = None
    if data is not None:
        trajectory = data.get("trajectory")

    if trajectory is None:
        # Try the live gateway
        _node_id, live = await _find_gateway_for_session(state, session_id)
        if live and isinstance(live.get("result"), dict):
            trajectory = live["result"].get("trajectory")

    if not isinstance(trajectory, dict):
        return {"session_id": session_id, "traces": [], "metadata": {}, "status": None}
    return {
        "session_id": session_id,
        "status": trajectory.get("status"),
        "metadata": trajectory.get("metadata") or {},
        "traces": trajectory.get("traces") or [],
        "error": trajectory.get("error"),
    }


@router.get("/sessions/{session_id}/evaluation")
async def get_session_evaluation(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    data, _path = _load_session_file(state, session_id)
    if data is None:
        _node, live = await _find_gateway_for_session(state, session_id)
        if live and isinstance(live.get("result"), dict):
            data = live["result"]
    trajectory = (data or {}).get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    traces = trajectory.get("traces") or []
    trace_rewards = [
        trace.get("reward") if isinstance(trace, dict) else None for trace in traces
    ]
    return {
        "session_id": session_id,
        "outcome_reward": evaluation.get("outcome_reward"),
        "strategy": evaluation.get("strategy"),
        "report": evaluation.get("report"),
        "patch_path": evaluation.get("patch_path"),
        "trace_rewards": evaluation.get("trace_rewards") or trace_rewards,
        "raw": evaluation,
    }


def _read_completion_files_for(state, session_id: str) -> list[dict[str, Any]]:
    """Read on-disk completion records from `<save_dir>/.../sessions/<sid>/completions/`."""
    save_dir: Path = state.config.save_dir
    matches: list[dict[str, Any]] = []
    for task_dir in save_dir.glob("task_*"):
        candidate = task_dir / "sessions" / session_id / "completions"
        if candidate.is_dir():
            for path in sorted(candidate.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        record = json.load(handle)
                except (OSError, ValueError):
                    continue
                if isinstance(record, dict):
                    matches.append(record)
            if matches:
                return matches
    return matches


@router.get("/sessions/{session_id}/completions")
async def get_session_completions(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    # Try live gateway first.
    for client in state.gateway_clients.values():
        payload = await client.safe_get_json(
            f"/sessions/{session_id}/completions", default=None
        )
        if isinstance(payload, dict) and payload.get("completions"):
            return {
                "session_id": session_id,
                "completions": payload.get("completions") or [],
                "source": "gateway",
            }
    # Fallback: scan disk.
    completions = _read_completion_files_for(state, session_id)
    return {
        "session_id": session_id,
        "completions": completions,
        "source": "filesystem" if completions else "none",
    }


@router.get("/sessions/{session_id}/raw")
async def get_session_raw(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    data, path = _load_session_file(state, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session file not found")
    return {"session_id": session_id, "file_path": str(path), "data": data}


@router.get("/sessions/{session_id}/trace")
async def get_session_trace(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    path = _session_trace_file(state, session_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Session trace not found")
    schema_version, metadata, events = _load_trace_document(path)
    return {
        "session_id": session_id,
        "schema_version": schema_version,
        "metadata": metadata,
        "trace_events": events,
        "event_count": len(events),
        "download_url": f"/api/sessions/{session_id}/trace/download",
    }


@router.get("/sessions/{session_id}/trace/download")
async def download_session_trace(request: Request, session_id: str) -> FileResponse:
    state = request.app.state.platform
    path = _session_trace_file(state, session_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Session trace not found")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{session_id}.trace.json",
    )


@router.get("/sessions/{session_id}/artifacts")
async def get_session_artifacts(request: Request, session_id: str) -> dict[str, Any]:
    loaded = _load_artifact_manifest(request.app.state.platform, session_id)
    if loaded is None:
        return {
            "session_id": session_id,
            "artifact_count": 0,
            "total_bytes": 0,
            "skipped_bytes": 0,
            "artifacts": [],
        }
    manifest, _root = loaded
    artifacts = []
    for item in manifest["artifacts"]:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        artifacts.append({
            **item,
            "download_url": f"/api/sessions/{session_id}/artifacts/{item['id']}",
        })
    return {**manifest, "artifacts": artifacts}


@router.get("/sessions/{session_id}/artifacts/{artifact_id}")
async def download_session_artifact(
    request: Request,
    session_id: str,
    artifact_id: str,
) -> FileResponse:
    loaded = _load_artifact_manifest(request.app.state.platform, session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Session artifact not found")
    manifest, root = loaded
    entry = next(
        (
            item
            for item in manifest["artifacts"]
            if isinstance(item, dict) and str(item.get("id")) == artifact_id
        ),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="Session artifact not found")
    artifact_root = (root / f"{session_id}.artifacts").resolve()
    path = (artifact_root / str(entry.get("relative_path", ""))).resolve()
    if not path.is_relative_to(artifact_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Session artifact not found")
    return FileResponse(
        path,
        media_type=str(entry.get("media_type") or "application/octet-stream"),
        filename=str(entry.get("name") or path.name),
    )


@router.delete("/sessions/{session_id}")
async def cancel_session(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    last_status: int | None = None
    last_body: Any = None
    last_node: str | None = None
    for node_id, client in state.gateway_clients.items():
        try:
            status_code, body = await client.request_json(
                "DELETE", f"/sessions/{session_id}"
            )
        except Exception as exc:
            logger.debug("cancel: gateway %s unreachable: %s", node_id, exc)
            continue
        last_status = status_code
        last_body = body
        last_node = node_id
        if status_code < 500 and status_code != 404:
            return {
                "session_id": session_id,
                "node_id": node_id,
                "status_code": status_code,
                "body": body,
            }
    if last_status is None:
        raise HTTPException(status_code=503, detail="No gateway reachable to cancel session")
    raise HTTPException(
        status_code=last_status if last_status == 404 else 502,
        detail={
            "session_id": session_id,
            "node_id": last_node,
            "status_code": last_status,
            "body": last_body,
        },
    )


__all__ = ["router"]
