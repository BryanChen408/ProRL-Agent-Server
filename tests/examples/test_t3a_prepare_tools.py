"""CPU-only regression for T3A agent and fresh-judge tools preparation."""
import runpy
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "deploy/ascend_operator/t3a/runtime/prepare_operator_workdir.py"
RUNTIME = ROOT / "operator_runtime_t3a"


def test_t3a_prepare_replica_matches_source():
    assert SOURCE.read_bytes() == (RUNTIME / "runtime/prepare_operator_workdir.py").read_bytes()


@pytest.mark.parametrize("value", ["237568genzhi", "0", "-1", "", "１２３"])
def test_t3a_invalid_context_fails_before_preparing(tmp_path, monkeypatch, value):
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", value)
    workdir = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--op-name", "OP", "--workdir", str(workdir)])
    with pytest.raises(ValueError, match="positive integer"):
        runpy.run_path(str(SOURCE))["main"]()
    assert not workdir.exists()


@pytest.mark.parametrize("agent", [True, False], ids=["agent", "judge"])
@pytest.mark.parametrize("mode", ["copy", "readonly", "missing", "nonexec"])
def test_t3a_prepare_tools(tmp_path, monkeypatch, agent, mode):
    canonical = tmp_path / "canonical"
    shutil.copytree(RUNTIME, canonical)
    workdir = tmp_path / "workdir"
    (workdir / "input").mkdir(parents=True)
    (workdir / "input/OP.py").write_text("# CPU-only task placeholder\n")
    readonly = mode != "copy"
    if readonly:
        # Same inode through two paths, just like the production bind mounts.
        (workdir / "tools").symlink_to(canonical / "tools", target_is_directory=True)
    if mode == "missing":
        (canonical / "tools/npu_lease_exec.py").unlink()
    if mode == "nonexec":
        (canonical / "tools/npu_wrap.sh").chmod(0o644)

    original_copy = shutil.copy2
    original_chmod = Path.chmod

    def guarded_copy(src, dst, *args, **kwargs):
        if readonly:
            assert not Path(dst).resolve().is_relative_to(canonical / "tools"), "write to mounted tools"
        return original_copy(src, dst, *args, **kwargs)

    def guarded_chmod(path, *args, **kwargs):
        if readonly:
            assert not path.resolve().is_relative_to(canonical / "tools"), "chmod mounted tools"
        return original_chmod(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", guarded_copy)
    monkeypatch.setattr(Path, "chmod", guarded_chmod)
    monkeypatch.setattr(shutil, "which", lambda name: "/mock/claude")
    argv = [str(SOURCE), "--op-name", "OP", "--canonical-root", str(canonical),
            "--workdir", str(workdir)]
    if readonly:
        argv.append("--readonly-tools")
    if agent:
        argv.append("--require-claude")
    monkeypatch.setattr(sys, "argv", argv)
    main = runpy.run_path(str(SOURCE))["main"]
    if mode == "missing":
        with pytest.raises(FileNotFoundError, match="npu_lease_exec.py"):
            main()
    elif mode == "nonexec":
        with pytest.raises(PermissionError, match="npu_wrap.sh"):
            main()
    else:
        for _ in range(2):  # Re-preparing must also be safe.
            assert main() == 0
        assert (workdir / "tools/npu_lease_exec.py").is_file()
        assert bool((workdir / "tools/npu_wrap.sh").stat().st_mode & 0o111)
        assert (workdir / "judge/npu_lease_exec.py").is_file() is (not agent)
