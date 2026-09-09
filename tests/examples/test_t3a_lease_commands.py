"""Exercise the real lease wrapper with CPU-only scripts and isolated lock files."""
import fcntl
import runpy
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "operator_runtime_t3a"


@pytest.mark.parametrize("prefix", ["", "cd . && ", "CASE_ENV=value ", "export CASE_ENV=value && "])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_shell_semantics_and_lease_release(tmp_path, monkeypatch, prefix, exit_code):
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(RUNTIME / "tools/npu_lease_exec.py", tools)
    locks = tmp_path / "locks"
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("POLAR_NPU_LEASE_POOL", "0")
    monkeypatch.setenv("POLAR_NPU_LOCK_DIR", str(locks))
    # The fake verifier confirms that even compound commands run inside the lease.
    script = tmp_path / "verification_ascendc.py"
    script.write_text(
        "import fcntl, os, pathlib, sys\n"
        "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '0'\n"
        "p = pathlib.Path(os.environ['POLAR_NPU_LOCK_DIR']) / 'npu0.lock'\n"
        "with p.open('r+') as f:\n"
        "    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    except BlockingIOError: pass\n"
        "    else: raise AssertionError('child is outside the lease')\n"
        "print('verifier-out')\n"
        "print('verifier-err', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n"
    )
    module = runpy.run_path(str(RUNTIME / "hooks/skill_script_hook.py"))
    command = prefix + "python3 " + shlex.quote(str(script))
    assert module["should_intercept"](command)
    wrapped = module["_lease_wrap_command"](command)
    result = subprocess.run(["bash", "-c", wrapped], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.strip() == "verifier-out"
    assert result.stderr.strip() == "verifier-err"
    with (locks / "npu0.lock").open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cpu_commands_passthrough_and_missing_executor_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLAR_NPU_LEASE_POOL", "0")
    wrap = runpy.run_path(str(RUNTIME / "hooks/skill_script_hook.py"))["_lease_wrap_command"]
    for command in ("python3 build_ascendc.py OP", "python3 validate_ascendc_impl.py OP",
                    "bash evaluate_ascendc.sh OP"):
        assert wrap(command) == command
    with pytest.raises(FileNotFoundError, match="lease configured"):
        wrap("python3 verification_ascendc.py OP")
    monkeypatch.delenv("POLAR_NPU_LEASE_POOL")
    assert wrap("python3 verification_ascendc.py OP") == "python3 verification_ascendc.py OP"


def test_build_script_preserves_wrapper_fix():
    build = (ROOT / "deploy/ascend_operator/build_t3a_replica.sh").read_text()
    assert '-- bash -c {_shlex.quote(command)}' in build
    assert 'NPU lease configured but tools/npu_lease_exec.py is missing' in build
