"""msprof_perf_summary 墙钟兜底:csv 切不出 device 任务时用进程内 wall 计时出数。

覆盖三条路径:grouped 窗口空(纯 view/零 device 任务 reference)、marker 计数
不符(实测 3x)、task_time.csv 整体缺失;以及正常 csv 优先(口径不变)。
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[2]
          / "operator_runtime_t2a/skills/ops-profiling/scripts/msprof_perf_summary.py")
spec = importlib.util.spec_from_file_location("msprof_perf_summary", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules.setdefault("msprof_perf_summary", m)
spec.loader.exec_module(m)


def _write_csv(prof_dir, rows):
    out = Path(prof_dir) / "mindstudio_profiler_output"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "task_time_0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_start(us)", "task_time(us)", "kernel_type", "kernel_name"])
        w.writerows(rows)


def _write_manifest(path, blocks):
    Path(path).write_text(json.dumps({"blocks": blocks}), encoding="utf-8")


def _block(case, impl, wall_us, ok=True):
    return {"case": case, "impl": impl, "ok": ok,
            "has_markers": True, "error": None, "wall_us": wall_us}


def test_empty_window_falls_back_to_wall(tmp_path):
    # 一个块窗口内有 kernel,一个块窗口空(reference 纯 view)
    _write_csv(tmp_path, [
        (1.0, 0.5, "EVENT_RECORD", ""),   # block0 start
        (2.0, 100.0, "AI_VECTOR_CORE", "KernelA"),
        (3.0, 0.5, "EVENT_RECORD", ""),   # block0 end / block1 相邻
        (4.0, 0.5, "EVENT_RECORD", ""),   # block1 start
        (5.0, 0.5, "EVENT_RECORD", ""),   # block1 end(窗口空)
    ])
    _write_manifest(tmp_path / "m.json", [
        _block(0, "reference", 800.0),   # 有 kernel
        _block(0, "ascendc", 1234.0),    # 窗口空
    ])
    # 4 个 marker = 2 块 × 2,计数匹配
    got, err = m._parse_msprof_grouped(str(tmp_path), tmp_path / "m.json", 1)
    assert err is None
    # 同 case 同钟:ascendc 侧落了 wall,reference 侧也从 csv 换成自己的 wall,
    # 避免「含发射开销 vs 不含」的跨口径比值
    assert got[(0, "reference")]["duration_us"] == 800.0
    assert got[(0, "reference")]["timing"] == "wall_clock"
    assert got[(0, "ascendc")]["duration_us"] == 1234.0
    assert got[(0, "ascendc")]["timing"] == "wall_clock"


def test_all_csv_keeps_task_time_untouched(tmp_path):
    # 两侧 csv 都切得出:口径完全不变,wall 不介入
    _write_csv(tmp_path, [
        (1.0, 0.5, "EVENT_RECORD", ""),
        (2.0, 100.0, "AI_VECTOR_CORE", "K"),
        (3.0, 0.5, "EVENT_RECORD", ""),
        (4.0, 0.5, "EVENT_RECORD", ""),
        (5.0, 250.0, "AI_VECTOR_CORE", "K2"),
        (6.0, 0.5, "EVENT_RECORD", ""),
    ])
    _write_manifest(tmp_path / "m.json", [_block(0, "reference", 800.0),
                                          _block(0, "ascendc", 1234.0)])
    got, err = m._parse_msprof_grouped(str(tmp_path), tmp_path / "m.json", 1)
    assert err is None
    assert got[(0, "reference")]["duration_us"] == 100.0
    assert "timing" not in got[(0, "reference")]
    assert got[(0, "ascendc")]["duration_us"] == 250.0
    assert "timing" not in got[(0, "ascendc")]


def test_marker_mismatch_uses_wall_instead_of_discarding(tmp_path):
    # marker 是预期 3 倍(实测形态):以前整组作废,现在 wall 出全套
    _write_csv(tmp_path, [(float(i), 0.5, "EVENT_RECORD", "") for i in range(12)])
    _write_manifest(tmp_path / "m.json", [_block(0, "reference", 500.0),
                                          _block(0, "ascendc", 250.0)])
    got, err = m._parse_msprof_grouped(str(tmp_path), tmp_path / "m.json", 1)
    assert err is None
    assert got[(0, "reference")]["duration_us"] == 500.0
    assert got[(0, "ascendc")]["duration_us"] == 250.0


def test_missing_csv_uses_wall(tmp_path):
    _write_manifest(tmp_path / "m.json", [_block(0, "reference", 42.0),
                                          _block(0, "ascendc", 84.0)])
    got, err = m._parse_msprof_grouped(str(tmp_path), tmp_path / "m.json", 1)
    assert err is None and got[(0, "reference")]["duration_us"] == 42.0


def test_failed_block_still_errors(tmp_path):
    # 块本身没跑成(ok=False)不给 wall,维持失败语义
    bad = _block(0, "reference", 0.0, ok=False)
    bad["error"] = "RuntimeError: boom"
    _write_manifest(tmp_path / "m.json", [bad])
    got, err = m._parse_msprof_grouped(str(tmp_path), tmp_path / "m.json", 1)
    assert got is None or got[(0, "reference")]["duration_us"] is None


def test_wall_marker_parsed_from_app_log(tmp_path):
    (tmp_path / "app_output.log").write_text(
        "noise\nPOLAR_WALL_US=123.456\nmore noise\n", encoding="utf-8")
    assert m._wall_us_from_app_log(str(tmp_path)) == 123.456
    (tmp_path / "app_output.log").write_text("nothing here", encoding="utf-8")
    assert m._wall_us_from_app_log(str(tmp_path)) is None
