"""AscendC 固定评测入口的任务路径解析回归。"""

from __future__ import annotations

import subprocess
from pathlib import Path


PIPELINE = (
    Path(__file__).resolve().parents[2]
    / "operator_runtime_t2a"
    / "tools"
    / "ascendc_eval_pipeline.sh"
)


def _resolve_task_path(*, cwd: Path, work_root: Path, task_file: str = "") -> str:
    source = PIPELINE.read_text(encoding="utf-8")
    start = source.index('TASK_SRC="${TASK_FILE:-input/${OP_NAME}.py}"')
    end = source.index('if [[ ! -f "$TASK_SRC" ]]', start)
    path_setup = source[start:end]
    script = "\n".join(
        (
            f"WORK_ROOT={work_root}",
            "OP_NAME=demo",
            f"TASK_FILE={task_file}",
            path_setup,
            'printf "%s" "$TASK_SRC"',
        )
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_default_task_path_is_anchored_to_work_root_from_nested_cwd(tmp_path):
    work_root = tmp_path / "workdir"
    nested = work_root / "demo" / "kernel" / "build"
    task = work_root / "input" / "demo.py"
    nested.mkdir(parents=True)
    task.parent.mkdir(parents=True)
    task.touch()

    assert _resolve_task_path(cwd=nested, work_root=work_root) == str(task)


def test_absolute_task_path_is_unchanged(tmp_path):
    work_root = tmp_path / "workdir"
    nested = work_root / "demo" / "kernel" / "build"
    task = tmp_path / "external" / "demo.py"
    nested.mkdir(parents=True)
    task.parent.mkdir(parents=True)
    task.touch()

    assert _resolve_task_path(
        cwd=nested,
        work_root=work_root,
        task_file=str(task),
    ) == str(task)
