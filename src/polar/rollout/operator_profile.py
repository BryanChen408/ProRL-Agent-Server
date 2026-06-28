"""Expand thin operator sample requests with Polar-owned rollout profiles."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Any

from polar.config import RolloutServiceConfig
from polar.rollout.models import OperatorSampleRequest, TaskRequest

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")


def expand_operator_sample_request(
    request: OperatorSampleRequest,
    rollout: RolloutServiceConfig,
) -> TaskRequest:
    profile_name = _resolve_profile_name(request, rollout)
    profile_model = rollout.operator_profiles.get(profile_name)
    if profile_model is None:
        raise ValueError(f"unknown operator profile: {profile_name}")

    profile = profile_model.model_dump(mode="python")
    context = _build_context(request, profile_name, profile)
    rendered = _render_value(profile, context)
    if not isinstance(rendered, dict):
        raise ValueError(f"operator profile {profile_name!r} must render to a mapping")

    payload: dict[str, Any] = {
        "task_id": request.task_id,
        "instruction": request.instruction,
        "num_samples": request.num_samples,
        "agent": rendered.get("agent"),
        "metadata": _merge_metadata(request, rendered, profile_name),
    }
    timeout_seconds = (
        request.timeout_seconds
        if request.timeout_seconds is not None
        else rendered.get("timeout_seconds")
    )
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds
    for key in ("runtime", "builder", "evaluator", "callback_url"):
        if key in rendered and rendered[key] is not None:
            payload[key] = rendered[key]

    manifest_hash = _operator_runtime_manifest_sha256(rendered)
    if manifest_hash:
        payload["metadata"].setdefault("operator_runtime_manifest_sha256", manifest_hash)

    return TaskRequest.model_validate(payload)


def _resolve_profile_name(
    request: OperatorSampleRequest,
    rollout: RolloutServiceConfig,
) -> str:
    profile_name = (request.profile or rollout.default_operator_profile or "").strip()
    if not profile_name:
        raise ValueError(
            "operator profile is required: pass request.profile or set rollout.default_operator_profile"
        )
    return profile_name


def _build_context(
    request: OperatorSampleRequest,
    profile_name: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    sample = request.sample.model_dump(mode="python")
    return {
        "task_id": request.task_id,
        "instruction": request.instruction,
        "num_samples": request.num_samples,
        "op_name": request.sample.op_name,
        "profile_name": profile_name,
        "profile": profile,
        "sample": sample,
        "metadata": request.metadata,
    }


def _merge_metadata(
    request: OperatorSampleRequest,
    rendered_profile: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    profile_metadata = rendered_profile.get("metadata")
    if isinstance(profile_metadata, dict):
        metadata.update(profile_metadata)
    metadata.update(request.metadata)
    metadata.setdefault("operator_profile", profile_name)
    metadata.setdefault("op_name", request.sample.op_name)
    if request.sample.group_index is not None:
        metadata.setdefault("group_index", request.sample.group_index)
    if request.sample.index is not None:
        metadata.setdefault("sample_index", request.sample.index)
    metadata.setdefault("sample_metadata", deepcopy(request.sample.metadata))
    return metadata


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if match := re.fullmatch(r"{([^{}]+)}", value):
            return deepcopy(_resolve_path(context, match.group(1)))

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): _render_value(item, context) for key, item in value.items()}
    return value


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise ValueError(f"unknown operator profile template variable: {path}")
    return current


def _operator_runtime_manifest_sha256(rendered_profile: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    runtime_dir = rendered_profile.get("operator_runtime_dir")
    if isinstance(runtime_dir, str) and runtime_dir.strip():
        candidates.append(runtime_dir.strip())
    for volume in _iter_profile_volumes(rendered_profile):
        if ":/opt/canonical" in volume:
            candidates.append(volume.split(":", 1)[0])

    for candidate in candidates:
        manifest = Path(candidate) / "MANIFEST.json"
        if manifest.is_file():
            return hashlib.sha256(manifest.read_bytes()).hexdigest()
    return None


def _iter_profile_volumes(rendered_profile: dict[str, Any]) -> list[str]:
    volumes: list[str] = []
    for runtime in (
        rendered_profile.get("runtime"),
        (rendered_profile.get("evaluator") or {}).get("runtime")
        if isinstance(rendered_profile.get("evaluator"), dict)
        else None,
    ):
        if not isinstance(runtime, dict):
            continue
        kwargs = runtime.get("kwargs")
        if not isinstance(kwargs, dict):
            continue
        raw_volumes = kwargs.get("volumes")
        if isinstance(raw_volumes, list):
            volumes.extend(str(volume) for volume in raw_volumes)
    return volumes

