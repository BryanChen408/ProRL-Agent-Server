"""error_type 判定回归:verify 阶段失败不得被 classify() 的编译关键词分支抢走。

背景(实测 run polar_20260806_195351):verification_ascendc.py 的输出开头会 dump
PATH 环境变量,里面含 `.../ccec_compiler/bin`。classify() 的
`"ccec" in l or ... or "compil" in l` 分支排在 对拍 分支之前 ——> 13/38 个 verify
失败(9 个真精度错 + 4 个崩溃)被错标成 ascendc_compile_failed,reward 从
0.35/0.30 掉到 0.25。修复:verify/compile 两个 write_metrics 调用都显式传
force_type,不再走文本推断。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PIPELINE = (
    Path(__file__).resolve().parents[2]
    / "operator_runtime_t2a"
    / "tools"
    / "ascendc_eval_pipeline.sh"
)

# 实测 verify.log 的形状:PATH dump 带 ccec_compiler,后面才是对拍报告。
_ENV_DUMP = (
    "INFO: get env PATH = /usr/local/Ascend/cann-9.0.0/bin:"
    "/usr/local/Ascend/cann-9.0.0/tools/ccec_compiler/bin:/usr/bin\n"
    "AscendC Verification Report\nStatus       : FAIL\n"
)
LOG_PRECISION = _ENV_DUMP + (
    "case[0]: output[0]: max_abs_diff=2.12883, mean_abs_diff=0.388091, "
    "matched_ratio=0.000029, MERE=6.29178, allclose_ok=False, passed=False\n"
    "Result: fail\n"
)
LOG_CRASH = _ENV_DUMP + (
    "Traceback (most recent call last):\n"
    "RuntimeError: ACL stream synchronize failed\nResult: fail\n"
)
# 第二个碰撞词(实测 session-sk-polar-894vxu62):torch_npu 的例行 warning 里有
# "performance degradation",撞上 classify() 的 AST 分支("degrad"),同样排在
# 对拍分支之前 —> 被标 ast_check_failed(0.20 地板),而同一份 metrics 里
# ast_check_ok=true,自相矛盾。证明这不是 ccec 专属,是「上游任意措辞都可能
# 撞上排在前面的关键词」这一类问题。
LOG_DEGRAD_WARNING = (
    "数值对拍失败(Result: fail;完整对拍输出如下)\n"
    "Warning: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
    "resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING.\n"
    "Traceback (most recent call last):\nRuntimeError: kernel launch failed\n"
    "Result: fail\n"
)

# 形状不符 / NaN 不符:上游走前置检查早退,算不出逐元素差,但对拍确实给了结论。
# 原话取自实测日志(195351 的 cumsum_exclusive / Matmul_with_transposed_A)。
LOG_SHAPE_MISMATCH = _ENV_DUMP + (
    "case[0]: output[0]: shape mismatch: ref=(32767, 32769), cand=(32768, 32768)\n"
    "Result: fail\n"
)
LOG_NAN_MISMATCH = _ENV_DUMP + (
    "case[0]: output[0]: NaN mask mismatch: ref=0/8388608, cand=325632/8388608\n"
    "Result: fail\n"
)

# 与 pipeline 里 verify 分支同一组判据。外层看「对拍给出结论没有」(case[N]: 行),
# 内层再分数值差异 / 前置检查不通过。
_CASE_RE = r"case\[[0-9]+\]:"
_NUM_RE = r"(max_abs_diff|mere|matched_ratio)[[:space:]]*="


def _decide(log_text: str, tmp_path: Path) -> str:
    """跑 pipeline 里那两条真实的 grep,复现 bash 侧的三分支判定。"""
    p = tmp_path / "verify.log"
    p.write_text(log_text)

    def _grep(pat: str, ci: bool = False) -> bool:
        flags = "-qEi" if ci else "-qE"
        return subprocess.run([("grep"), flags, pat, str(p)], check=False).returncode == 0

    if not _grep(_CASE_RE):
        return "ascendc_run_crashed"
    return "correctness_failed" if _grep(_NUM_RE, ci=True) else "output_precheck_failed"


def test_precision_failure_not_labeled_compile(tmp_path):
    """比较跑完 + 日志里有 ccec ==> 真精度错(D类 0.35),不是编译失败。"""
    assert _decide(LOG_PRECISION, tmp_path) == "correctness_failed"


def test_crash_before_comparison_labeled_crash(tmp_path):
    """比较没跑完 + 日志里有 ccec ==> 崩溃(A类 0.30),不是编译失败。"""
    assert _decide(LOG_CRASH, tmp_path) == "ascendc_run_crashed"


def test_shape_mismatch_is_precheck_not_crash(tmp_path):
    """形状不符:对拍跑完并给了结论,不是崩溃。

    旧判据只看 max_abs_diff,形状不一致时上游算不出逐元素差 -> 被误判成崩溃(0.30),
    agent 于是被告知「查越界/非法访存」,而真正该查的是输出 shape 推导。
    """
    assert _decide(LOG_SHAPE_MISMATCH, tmp_path) == "output_precheck_failed"


def test_nan_mismatch_is_precheck_not_crash(tmp_path):
    """NaN 掩码不符:同上 —— 对拍比了 838 万个元素才得出结论,不是中途崩了。"""
    assert _decide(LOG_NAN_MISMATCH, tmp_path) == "output_precheck_failed"


def test_degradation_warning_not_labeled_ast_failure(tmp_path):
    """上游 warning 里的 "degradation" 不得把 verify 失败拖进 ast_check_failed。

    判据只看比较有没有跑完,与上游措辞无关 —— 所以换一个碰撞词也不会误判。
    """
    assert _decide(LOG_DEGRAD_WARNING, tmp_path) == "ascendc_run_crashed"


def test_verify_and_compile_call_sites_pass_explicit_force_type():
    """两个 write_metrics 调用都要显式给 error_type,不能回退到文本推断。"""
    src = PIPELINE.read_text()
    compile_call = re.search(
        r'write_metrics true false false[^\n]*compile\.log"\s*\\?\s*\n?\s*'
        r'"ascendc_compile_failed"',
        src,
    )
    assert compile_call, "compile 失败没有显式传 ascendc_compile_failed"
    verify_call = re.search(
        r'write_metrics true false false[^\n]*verify\.log"\s*\\?\s*\n?\s*"\$_VER_TYPE"',
        src,
    )
    assert verify_call, "verify 失败没有显式传 $_VER_TYPE"
    # 判据必须先于 write_metrics 出现,且外层用 case[N] 行、内层才用数值字段。
    assert "_VER_TYPE=" in src and "matched_ratio" in src
    assert r"case\[[0-9]+\]:" in src, "verify 判据没用 case[N] 行(会把形状/NaN 误判成崩溃)"
    assert "output_precheck_failed" in src, "缺 output_precheck_failed 档"


def test_fallback_classify_agrees_with_explicit_path():
    """classify() 兜底路径必须和显式路径判得一样,否则两条路会给出不同标签。"""
    src = PIPELINE.read_text()
    # classify() 内部也要先看 case[N] 行,再看数值字段。
    assert re.search(r're\.search\(r"case\\\[\\d\+\\\]:", l\)', src), \
        "classify() 的对拍分支没换成 case[N] 判据"
    # fail_hint 的 crash_failure 同理。
    assert re.search(r'crash_failure = not bool\(re\.search\(r"case\\\[\\d\+\\\]:"', src), \
        "fail_hint 的 crash_failure 还在用旧的数值字段判据"
