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
    rendered = _attach_task_source(request, rollout, rendered)

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


def _attach_task_source(
    request: OperatorSampleRequest,
    rollout: RolloutServiceConfig,
    rendered_profile: dict[str, Any],
) -> dict[str, Any]:
    task_source = request.sample.task_source
    if task_source is None:
        return rendered_profile

    source_hash = hashlib.sha256(task_source.encode("utf-8")).hexdigest()
    expected_hash = request.sample.task_source_sha256
    if expected_hash and expected_hash != source_hash:
        raise ValueError(
            "sample.task_source_sha256 does not match sample.task_source"
        )

    cache_dir = _operator_task_cache_dir(rendered_profile, rollout)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source_hash}.py"
    if not cache_path.exists() or cache_path.read_text(encoding="utf-8") != task_source:
        cache_path.write_text(task_source, encoding="utf-8")

    rendered = deepcopy(rendered_profile)
    replaced = _rewrite_operator_task_upload_sources(
        rendered,
        str(cache_path),
        op_name=request.sample.op_name,
    )
    if replaced == 0:
        raise ValueError(
            "sample.task_source was provided, but the operator profile has no "
            "upload_file source for the operator task"
        )
    return rendered


def _operator_task_cache_dir(
    rendered_profile: dict[str, Any],
    rollout: RolloutServiceConfig,
) -> Path:
    configured = rendered_profile.get("operator_task_cache_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser()

    save_dir = rollout.save_dir
    if save_dir:
        return Path(save_dir).expanduser().parent / "asset_cache" / "op_tasks"
    return Path("output") / "ascend_operator" / "asset_cache" / "op_tasks"


def _rewrite_operator_task_upload_sources(
    value: Any,
    source_path: str,
    *,
    op_name: str,
) -> int:
    replaced = 0
    if isinstance(value, dict):
        if _is_operator_task_upload_action(value, op_name=op_name):
            value["source"] = source_path
            replaced += 1
        for item in value.values():
            replaced += _rewrite_operator_task_upload_sources(
                item,
                source_path,
                op_name=op_name,
            )
    elif isinstance(value, list):
        for item in value:
            replaced += _rewrite_operator_task_upload_sources(
                item,
                source_path,
                op_name=op_name,
            )
    return replaced


def _is_operator_task_upload_action(value: dict[str, Any], *, op_name: str) -> bool:
    if value.get("type") != "upload_file":
        return False
    source = value.get("source")
    target = value.get("target")
    if not isinstance(source, str) or not isinstance(target, str):
        return False
    return source.endswith(f"/{op_name}.py") and (
        target.endswith(f"/src/{op_name}.py")
        or target.endswith(f"/input/{op_name}.py")
    )


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
