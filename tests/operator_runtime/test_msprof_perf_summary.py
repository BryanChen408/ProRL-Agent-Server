from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MSPROF_SUMMARY_PATH = (
    ROOT
    / "operator_runtime_t2a"
    / "skills"
    / "ops-profiling"
    / "scripts"
    / "msprof_perf_summary.py"
)


def load_msprof_summary():
    spec = importlib.util.spec_from_file_location("t2a_msprof_perf_summary", MSPROF_SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_parser_counts_non_meta_device_tasks(tmp_path):
    module = load_msprof_summary()
    csv_dir = tmp_path / "mindstudio_profiler_output"
    csv_dir.mkdir()
    (csv_dir / "task_time_1.csv").write_text(
        "kernel_type,task_time(us),kernel_name\n"
        "PROFILING_ENABLE,2.0,enable\n"
        "AI_VECTOR_CORE,18.4,log\n"
        "MEMCPY,11.0,d2h\n"
        "AICPU,7.0,other_device_task\n"
        "PROFILING_DISABLE,3.0,disable\n"
        "TASK_TIMEOUT_SET,4.0,timeout\n",
        encoding="utf-8",
    )

    duration, kernel_name, error = module._parse_msprof_duration_quick(str(tmp_path))

    assert duration == 36.4
    assert kernel_name == "multiple_kernels"
    assert error is None


def test_quick_parser_ignores_meta_events(tmp_path):
    module = load_msprof_summary()
    csv_dir = tmp_path / "mindstudio_profiler_output"
    csv_dir.mkdir()
    (csv_dir / "task_time_1.csv").write_text(
        "kernel_type,task_time(us),kernel_name\n"
        "PROFILING_ENABLE,2.0,enable\n"
        "PROFILING_DISABLE,3.0,disable\n"
        "TASK_TIMEOUT_SET,4.0,timeout\n",
        encoding="utf-8",
    )

    duration, kernel_name, error = module._parse_msprof_duration_quick(str(tmp_path))

    assert duration is None
    assert kernel_name is None
    assert error == "no task_time or api_statistic csv found"
