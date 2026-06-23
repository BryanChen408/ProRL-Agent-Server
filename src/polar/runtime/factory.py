"""Runtime factory with built-in backend map and import_path support."""

from __future__ import annotations

from pathlib import Path

from polar._imports import import_subclass
from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.base import BaseRuntime
from polar.runtime.docker import DockerRuntime
from polar.runtime.models import RuntimeSpec

_BUILTIN_BACKENDS: dict[str, type[BaseRuntime]] = {
    "docker": DockerRuntime,
    "apptainer": ApptainerRuntime,
}


def create_runtime(
    spec: RuntimeSpec, session_id: str, session_dir: Path
) -> BaseRuntime:
    """Instantiate a runtime from a RuntimeSpec.

    Uses the built-in backend map for ``docker`` and ``apptainer``.
    Falls back to ``spec.import_path`` for plugin runtimes.
    """
    if spec.import_path:
        cls = _import_runtime_class(spec.import_path)
        runtime = cls(spec, session_id, session_dir)
        _validate_runtime_capabilities(runtime)
        return runtime
    cls = _BUILTIN_BACKENDS.get(spec.backend)
    if cls is None:
        raise ValueError(f"Unsupported runtime backend: {spec.backend}")
    runtime = cls(spec, session_id, session_dir)
    _validate_runtime_capabilities(runtime)
    return runtime


def _import_runtime_class(import_path: str) -> type[BaseRuntime]:
    return import_subclass(import_path, BaseRuntime, kind="runtime import path")


def _validate_runtime_capabilities(runtime: BaseRuntime) -> None:
    spec = runtime.spec
    backend = spec.backend
    if spec.gpus > 0 and not runtime.supports_gpus:
        raise ValueError(f"runtime backend {backend!r} does not support GPUs")
    if spec.cpus is not None and not runtime.supports_cpu_limits:
        raise ValueError(f"runtime backend {backend!r} does not support CPU limits")
    if spec.memory_mb is not None and not runtime.supports_memory_limits:
        raise ValueError(f"runtime backend {backend!r} does not support memory limits")
    if spec.storage_mb is not None and not runtime.supports_storage_limits:
        raise ValueError(f"runtime backend {backend!r} does not support storage limits")
    if spec.kwargs.get("ascend") and not runtime.supports_ascend:
        raise ValueError(
            f"runtime backend {backend!r} does not implement Ascend NPU passthrough "
            "(kwargs.ascend); only the 'docker' backend exposes the required driver/device mounts "
            "and optional runtime lifetime card lease. Running an operator rollout on this backend "
            "would silently miss the required Ascend environment."
        )
    if not spec.allow_internet:
        if not runtime.can_disable_internet:
            raise ValueError(
                f"runtime backend {backend!r} cannot disable internet access"
            )
        if spec.network not in (None, "", "host", "none"):
            raise ValueError(
                "runtime.network must be unset, 'host', or 'none' when "
                "allow_internet=false"
            )
