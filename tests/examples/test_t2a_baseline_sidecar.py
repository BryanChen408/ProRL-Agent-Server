"""Run the actual baseline injection without invoking compilation or an NPU."""
import os
from pathlib import Path
import re
import runpy
import subprocess
import tempfile


def test_trusted_baseline_sidecars():
    script = Path(__file__).resolve().parents[2] / "operator_runtime_t2a/tools/ascendc_eval_pipeline.sh"
    function = re.search(r"inject_baseline\(\) \{.*?\n\}", script.read_text(), re.S).group()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work = root / "evaluation"
        work.mkdir()
        source = root / "op.py"
        source.write_text(
            "from pathlib import Path\n"
            "fixed = Path(__file__).with_name('op.json').read_bytes()\n"
            "dynamic = Path(__file__).with_suffix('.json').read_bytes()\n"
        )
        cases = root / "op.json"
        cases.write_bytes(b'{"trusted": true}\n')
        outside = root / "outside"
        outside.write_text("must not change")
        env = dict(os.environ, TASK_SRC=str(source), JSON_SRC=str(cases),
                   TASK_DIR=str(work), OP_NAME="op")

        def inject():
            return subprocess.run(["bash", "-c", function + "\ninject_baseline"], env=env)

        for kind in ("absent", "file", "symlink", "dangling"):
            for name in ("model.py", "op.json", "model.json"):
                dest = work / name
                dest.unlink(missing_ok=True)
                if kind == "file":
                    dest.write_text("untrusted")
                    dest.chmod(0o444)
                elif kind in ("symlink", "dangling"):
                    dest.symlink_to(outside if kind == "symlink" else root / "missing")
            assert inject().returncode == 0
            result = runpy.run_path(str(work / "model.py"))
            assert result["fixed"] == result["dynamic"] == cases.read_bytes()
            assert outside.read_text() == "must not change"
            assert not (root / "missing").exists()
            assert all(not (work / name).is_symlink() for name in ("model.py", "op.json", "model.json"))
            assert inject().returncode == 0  # Reinjection after compilation.

        cases.unlink()
        assert inject().returncode == 0
        assert not (work / "model.json").exists()
        assert not (work / "op.json").exists()
        (work / "model.json").mkdir()
        assert inject().returncode != 0  # Do not silently accept an invalid layout.


if __name__ == "__main__":
    test_trusted_baseline_sidecars()
