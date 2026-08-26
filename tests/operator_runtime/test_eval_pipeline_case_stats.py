"""case 统计链路回归:Step2b --json-file 产出 → extract_case_stats → metrics 字段。

背景:correctness 失败档 reward 改为 0.3 + α×通过率(operator_reward.reward_from_metrics),
分子分母来自 verify_report.json 的 case_oks。本文件不需要 NPU:
- 静态契约:管线确实给 verification_ascendc.py 传了 --json-file、两个 write_metrics
  调用点前都有 extract_case_stats、且 verification 脚本的 --json-file 分支仍打印
  人类可读报告(否则 verify.log 失去 case[N]: 行,classify/错误分类全挂);
- 动态:把管线里 extract_case_stats 的 python 片段原样抽出来,对各种形状(含异常)
  的 verify_report.json 跑一遍,校验「passed total」/空输出语义。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

PIPELINE = (
    Path(__file__).resolve().parents[2]
    / "operator_runtime_t2a"
    / "tools"
    / "ascendc_eval_pipeline.sh"
)
VERIFICATION = (
    Path(__file__).resolve().parents[2]
    / "operator_runtime_t2a"
    / "skills"
    / "tilelang2ascend-translator"
    / "scripts"
    / "verification_ascendc.py"
)


def _extract_stats_snippet() -> str:
    """抽出管线 extract_case_stats 里真实的 python 片段(不把逻辑在测试里复制一份)。"""
    src = PIPELINE.read_text(encoding="utf-8")
    m = re.search(r'stats=\$\(\s*"\$PY_BIN"\s+-c\s+"(.*?)"\s*2>/dev/null', src, re.S)
    assert m, "extract_case_stats 的 stats=$(...) 片段在管线里没找到,契约变了"
    return m.group(1)


def _run_snippet(tmp_path: Path, report: dict | None, snippet: str) -> str:
    if report is not None:
        (tmp_path / "verify_report.json").write_text(json.dumps(report), encoding="utf-8")
    # 片段里 '$OUT_DIR/verify_report.json' 由 shell 双引号展开;这里直接替换模拟
    code = snippet.replace("$OUT_DIR", str(tmp_path))
    out = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
    return out.stdout.strip()


def test_pipeline_static_contract():
    src = PIPELINE.read_text(encoding="utf-8")
    # Step2b 调用带 --json-file
    assert '--json-file "$OUT_DIR/verify_report.json"' in src
    # 两个 case 统计消费点(verify 失败分支 / 最终成功)都先 extract_case_stats
    assert re.search(r"extract_case_stats\n\s+write_metrics true false false", src)
    assert re.search(r"extract_case_stats\nwrite_metrics true true true", src)
    # metrics.json 落两个可空字段
    assert '"cases_passed": i(os.environ.get("CASES_PASSED",""))' in src
    assert '"cases_total": i(os.environ.get("CASES_TOTAL",""))' in src


def test_verification_json_file_mode_keeps_human_report():
    """--json-file 分支必须仍调 _print_report:verify.log 的 case[N]: 行是分类判据。"""
    src = VERIFICATION.read_text(encoding="utf-8")
    m = re.search(r"if args\.json_file:(.*?)raise SystemExit", src, re.S)
    assert m and "_print_report(report" in m.group(1)


def test_extract_stats_normal(tmp_path):
    snippet = _extract_stats_snippet()
    oks = [True] * 9 + [False]
    assert _run_snippet(tmp_path, {"ok": False, "case_oks": oks}, snippet) == "9 10"
    # 全过(success 路径也会落统计,便于观测)
    (tmp_path / "verify_report.json").unlink()
    assert _run_snippet(tmp_path, {"ok": True, "case_oks": [True, True]}, snippet) == "2 2"


def test_extract_stats_degenerate_inputs_yield_empty(tmp_path):
    snippet = _extract_stats_snippet()
    assert _run_snippet(tmp_path, None, snippet) == ""                       # 文件不存在
    assert _run_snippet(tmp_path, {"ok": False}, snippet) == ""               # 无 case_oks
    assert _run_snippet(tmp_path, {"ok": False, "case_oks": []}, snippet) == ""  # 空列表
    assert _run_snippet(tmp_path, {"ok": False, "case_oks": None}, snippet) == ""
    (tmp_path / "verify_report.json").write_text("{ not json", encoding="utf-8")
    assert _run_snippet(tmp_path, None, snippet) == ""                       # 坏 json(不再重写)


def test_metrics_fields_nullable_shape():
    """write_metrics 内嵌 python 的可空 int 语义:i() 对空串/垃圾/正常值的行为。"""
    src = PIPELINE.read_text(encoding="utf-8")
    m = re.search(r"def i\(x\):.*?except \(TypeError, ValueError\): return None", src, re.S)
    assert m
    ns: dict = {}
    exec(m.group(0), ns)  # noqa: S102 - 抽自本仓管线脚本
    i = ns["i"]
    assert i("") is None and i(None) is None and i("junk") is None
    assert i("9") == 9
