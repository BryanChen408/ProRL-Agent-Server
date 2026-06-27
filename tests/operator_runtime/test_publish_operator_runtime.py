from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publish_operator_runtime.py"

spec = importlib.util.spec_from_file_location("publish_operator_runtime", SCRIPT)
assert spec is not None and spec.loader is not None
publish_operator_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_operator_runtime)


def _write(path: Path, text: str = "x\n", mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)


def _source(root: Path) -> Path:
    source = root / "skills-rl-deploy"
    _write(source / "CLAUDE.md", "claude\n")
    verifier = source / ".agents/skills/triton-op-verifier/scripts"
    _write(verifier / "validate_triton_impl.py", "validate\n")
    _write(verifier / "verify.py", "verify\n")
    _write(verifier / "benchmark.py", "benchmark\n")
    _write(verifier / "_common_utils.py", "common\n")
    _write(verifier / "_log_utils.py", "log\n")
    _write(verifier / "__pycache__/benchmark.cpython-311.pyc", "cache\n")
    _write(source / "skills/triton-op-designer/SKILL.md", "designer\n")
    _write(source / "skills/triton-op-coding/SKILL.md", "coding\n")
    _write(source / "skills/npu-arch/references/npu-arch-guide-triton.md", "guide\n")
    _write(source / "skills/npu-arch/references/npu-hardware-params.md", "params\n")
    _write(source / "tools/triton_eval_pipeline.sh", "#!/usr/bin/env bash\n", 0o755)
    _write(source / "tools/env.sh", "export X=1\n")
    _write(source / "runtime/prepare_operator_workdir.py", "#!/usr/bin/env python3\n", 0o755)
    _write(source / "runtime/__pycache__/prepare_operator_workdir.cpython-311.pyc", "cache\n")
    return source


def test_publish_operator_runtime_creates_manifest_and_preserves_assets(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "operator_runtime"

    manifest = publish_operator_runtime.publish(source, output)

    manifest_path = output / "MANIFEST.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == manifest
    assert loaded["schema_version"] == 1
    assert loaded["source"] == str(source.resolve())
    assert loaded["output"] == str(output.resolve())
    assert loaded["file_count"] == len(loaded["files"])
    by_path = {item["path"]: item for item in loaded["files"]}
    assert "MANIFEST.json" not in by_path
    assert by_path["CLAUDE.md"]["sha256"]
    assert by_path["tools/triton_eval_pipeline.sh"]["mode"] == "0o755"
    assert by_path[".agents/skills/triton-op-verifier/scripts/benchmark.py"]["sha256"]
    assert "runtime/__pycache__/prepare_operator_workdir.cpython-311.pyc" not in by_path
    assert ".agents/skills/triton-op-verifier/scripts/__pycache__/benchmark.cpython-311.pyc" not in by_path
    assert not (output / "runtime/__pycache__").exists()
    assert not (output / ".agents/skills/triton-op-verifier/scripts/__pycache__").exists()
    assert (output / "skills/npu-arch/references/npu-arch-guide-triton.md").read_text(
        encoding="utf-8"
    ) == "guide\n"


def test_publish_operator_runtime_removes_stale_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "operator_runtime"
    _write(output / "stale.txt", "stale\n")

    publish_operator_runtime.publish(source, output)

    assert not (output / "stale.txt").exists()
    assert (output / "CLAUDE.md").is_file()


def test_publish_operator_runtime_requires_core_assets(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "skills/npu-arch/references/npu-hardware-params.md").unlink()

    with pytest.raises(FileNotFoundError, match="npu-hardware-params.md"):
        publish_operator_runtime.publish(source, tmp_path / "operator_runtime")


def test_publish_operator_runtime_requires_legacy_verifier_scripts(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / ".agents/skills/triton-op-verifier/scripts/benchmark.py").unlink()

    with pytest.raises(FileNotFoundError, match="benchmark.py"):
        publish_operator_runtime.publish(source, tmp_path / "operator_runtime")
