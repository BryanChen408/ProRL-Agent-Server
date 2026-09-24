"""error_type 判定回归:verify 阶段失败不得被 classify() 的编译关键词分支抢走。

背景(实测 run polar_20260806_195351):verification_ascendc.py 的输出开头会 dump
PATH 环境变量,里面含 `.../ccec_compiler/bin`。classify() 的
`"ccec" in l or ... or "compil" in l` 分支排在 对拍 分支之前 ——> 13/38 个 verify
失败(9 个真精度错 + 4 个崩溃)被错标成 ascendc_compile_failed,reward 从
0.35/0.30 掉到 0.25。修复:verify/compile 两个 write_metrics 调用都显式传
force_type,不再走文本推断。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
LOG_LOAD = _ENV_DUMP + (
    "Traceback (most recent call last):\n"
    "AttributeError: '_OpNamespace' 'npu' object has no attribute 'demo'\nResult: fail\n"
)
LOG_TIMEOUT = _ENV_DUMP + "RuntimeError: vector core timeout, error code 507034\nResult: fail\n"
LOG_LAUNCH = _ENV_DUMP + "RuntimeError: kernel launch failed, ACL call failed\nResult: fail\n"
LOG_LAUNCH_507035_WITH_TIMEOUT_NAME = _ENV_DUMP + (
    "RuntimeError: rtDeviceSynchronizeWithTimeout failed, error code 507035\nResult: fail\n"
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

def _decide(log_text: str, tmp_path: Path) -> str:
    """执行实际 Bash 分类分支，不在测试里复制规则。"""
    (tmp_path / "verify.log").write_text(log_text)
    source = PIPELINE.read_text()
    start = source.index('  if grep -qEi "')
    end = source.index('\n  extract_case_stats', start)
    result = subprocess.run(
        ["bash", "-c", 'OUT_DIR="$1"\n' + source[start:end]
         + '\nprintf "%s" "$_VER_TYPE"', "classify", str(tmp_path)],
        text=True, capture_output=True, check=True,
    )
    return result.stdout


@pytest.mark.parametrize("signal", [
    "Segmentation fault (core dumped)",
    "/bin/sh: helper.sh: No such file or directory",
    "error: invalid argument",
])
def test_cannbot_a_signals_precede_numeric_diff(tmp_path, signal):
    log = LOG_PRECISION + signal + "\n"
    assert _decide(log, tmp_path) == "ascendc_run_crashed"
    import ast
    source = PIPELINE.read_text().split("python3 - <<'PY'", 1)[1].split("\nPY", 1)[0]
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "classify")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "classify", "exec"), namespace)
    assert namespace["classify"](log) == "ascendc_run_crashed"


def test_matched_case_does_not_override_numeric_failure(tmp_path):
    assert _decide("case[1]: output[1]: matched\n" + LOG_PRECISION, tmp_path) == "correctness_failed"


def test_precision_failure_not_labeled_compile(tmp_path):
    """比较跑完 + 日志里有 ccec ==> 真精度错(D类 0.35),不是编译失败。"""
    assert _decide(LOG_PRECISION, tmp_path) == "correctness_failed"


def test_crash_before_comparison_labeled_crash(tmp_path):
    """比较没跑完 + 日志里有 ccec ==> 崩溃(A类 0.30),不是编译失败。"""
    assert _decide(LOG_CRASH, tmp_path) == "ascendc_run_crashed"


def test_incomplete_verify_is_split_into_load_timeout_launch(tmp_path):
    assert _decide(LOG_LOAD, tmp_path) == "ascendc_load_failed"
    assert _decide(LOG_TIMEOUT, tmp_path) == "ascendc_run_timeout"
    assert _decide(LOG_LAUNCH, tmp_path) == "ascendc_launch_failed"
    assert _decide(LOG_LAUNCH_507035_WITH_TIMEOUT_NAME, tmp_path) == "ascendc_launch_failed"


def test_explicit_acl_codes_precede_ambiguous_timeout_words():
    src = PIPELINE.read_text(encoding="utf-8")
    explicit = src.index('elif grep -qE "507035"')
    fallback = src.index('elif grep -qEi "timed?[[:space:]]*out|timeout')
    assert explicit < fallback


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

    比较没有形成 case 结论后再按真实故障细分；warning 不能抢走 launch 分类。
    """
    assert _decide(LOG_DEGRAD_WARNING, tmp_path) == "ascendc_launch_failed"


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


def test_cannbot_class_boundaries_and_explicit_infra_types_are_locked():
    src = PIPELINE.read_text(encoding="utf-8")
    # CANNBot:shape/dtype/不可比较输出属于 A；D 只留给已有数值比较字段的精度失败。
    assert 'label = "A类-输出契约/有效性错误' in src
    assert 'D类-输出不可比较' not in src
    assert 'label = "D类-精度不匹配"' in src
    # 已知阶段不得再靠错误文案猜。
    assert re.search(
        r'"对拍结果不可信\(缓存/常量输出\).*?"stateful_impl_detected"',
        src,
        re.S,
    )
    assert re.search(r'"profiler_unavailable"\s*\n\s*echo "\[ascendc-eval\] msprof', src)
    # 未覆盖类型是 judge 分类缺口，不得默认伪装成 agent 编译错误。
    assert 'return "judge_classification_failed"' in src
    assert "B类-INFRA-分类器未覆盖该error_type" in src
    # 粗粒度 error_type 不能充当探索终止条件；停止只服从固定入口总预算。
    assert "update_conductor_state" not in src
    assert "C类-同一A类子类型连续失败" not in src


def _run_fail_hint(tmp_path, error_type, log_text="", *, success=False):
    """Run the real feedback code from a nested cwd, without invoking NPU evaluation."""
    source = PIPELINE.read_text(encoding="utf-8")
    snippet = source.split("<<'CLASSIFY'", 1)[1].split("\nCLASSIFY", 1)[0]
    snippet = snippet.split("\n", 1)[1]
    work_root = tmp_path / "work dir"
    nested = work_root / "demo" / "kernel"
    nested.mkdir(parents=True)
    metrics = work_root / "metrics.json"
    metrics.write_text(json.dumps({"success": success, "error_type": error_type}))
    before = metrics.read_bytes()
    log = work_root / "metrics_error.log"
    log.write_text(log_text)
    result = subprocess.run(
        [sys.executable, "-c", snippet, str(metrics), str(log), str(work_root)],
        cwd=nested, text=True, capture_output=True, check=True,
    )
    assert metrics.read_bytes() == before  # Guidance must not alter the judgement.
    return result.stdout, work_root


@pytest.mark.parametrize("error_type,log_text,topic,reference", [
    ("ascendc_compile_failed", "", "编译/链接", "调用 Skill ascendc-docs-search"),
    ("ast_check_failed", "", "AST退化", "调用 Skill tilelang2ascend-translator"),
    ("op_not_registered", LOG_LOAD, "加载失败", "调用 Skill tilelang2ascend-translator"),
    ("ascendc_load_failed", LOG_LOAD, "加载失败", "ascendc-runtime-debug/references/kernel_binary_debug.md"),
    ("ascendc_run_crashed", LOG_CRASH, "崩溃", "ascendc-crash-debug/references/crash_workflow.md"),
    ("ascendc_run_timeout", LOG_TIMEOUT, "超时", "ascendc-crash-debug/references/crash_workflow.md"),
    ("ascendc_launch_failed", LOG_LAUNCH, "启动失败", "ascendc-runtime-debug/references/error_codes.md"),
    ("output_precheck_failed", LOG_SHAPE_MISMATCH, "输出契约", "调用 Skill tilelang2ascend-translator"),
    ("output_precheck_failed", LOG_NAN_MISMATCH, "输出契约", "调用 Skill ascendc-precision-debug"),
    ("correctness_failed", LOG_PRECISION, "D类-精度不匹配", "调用 Skill tilelang2ascend-precision-tuning"),
    ("correctness_failed", LOG_CRASH, "A类-kernel崩溃", "ascendc-crash-debug/references/crash_workflow.md"),
    ("stateful_impl_detected", "", "状态化", "调用 Skill tilelang2ascend-translator"),
    ("benchmark_failed", "", "benchmark执行失败", "调用 Skill ops-profiling"),
])
def test_fail_hint_links_existing_references_from_any_cwd(
    tmp_path, error_type, log_text, topic, reference,
):
    output, work_root = _run_fail_hint(tmp_path, error_type, log_text)
    assert topic in output
    expected = reference if reference.startswith("调用 Skill ") else f"{work_root}/.claude/skills/{reference}"
    assert expected in output
    paths = re.findall(re.escape(str(work_root)) + r"/\.claude/skills/([\w./-]+\.md)", output)
    skills = re.findall(r"调用 Skill ([\w-]+)", output)
    assert paths or skills
    for name in skills:
        assert (PIPELINE.parent.parent / "skills" / name / "SKILL.md").is_file(), name
    for path in paths:
        assert (PIPELINE.parent.parent / "skills" / path).is_file(), path
    assert "Read .claude/" not in output
    assert "已读内容可复用" in output
    assert "Glob" in output
    if error_type == "ast_check_failed":
        assert "ascendc-docs-search" not in output
    if error_type == "benchmark_failed":
        assert "测速未完成" in output and "perf.log" in output
        assert "ascendc-crash-debug/references/crash_workflow.md" in output


@pytest.mark.parametrize("error_type,success", [
    ("submission_missing", False), ("input_load_failed", False),
    ("npu_runtime_unavailable", False), ("unknown_error", False), (None, True),
])
def test_fail_hint_does_not_route_non_repair_results_to_skills(tmp_path, error_type, success):
    output, _ = _run_fail_hint(tmp_path, error_type, success=success)
    assert ".claude/skills/" not in output
    assert "资料读取:" not in output


def test_reference_paths_survive_real_session_prepare(tmp_path):
    output, work_root = _run_fail_hint(tmp_path, "correctness_failed", LOG_PRECISION)
    runtime = PIPELINE.parent.parent
    task = tmp_path / "demo.py"
    task.write_text(
        "import torch\nclass Model(torch.nn.Module):\n"
        "    def forward(self, x: torch.Tensor): return x + 1\n"
        "def get_inputs(): return [torch.ones(8)]\n"
    )
    subprocess.run(
        [sys.executable, str(runtime / "runtime/prepare_operator_workdir.py"),
         "--backend", "ascendc", "--op-name", "demo", "--task-path", str(task),
         "--workdir", str(work_root), "--canonical-root", str(runtime)],
        cwd=work_root / "demo/kernel", capture_output=True, text=True, check=True,
    )
    source = PIPELINE.read_text().split("<<'CLASSIFY'", 1)[1].split("\nCLASSIFY", 1)[0]
    paths = set(re.findall(r"\.claude/skills/([\w./-]+\.md)", source))
    paths.update(f"{name}/SKILL.md" for name in re.findall(r"调用 Skill ([\w-]+)", source))
    for path in paths:
        installed = work_root / ".claude/skills" / path
        assert installed.read_bytes() == (runtime / "skills" / path).read_bytes()
        # Follow local markdown links and the short skill's backtick reference paths.
        links = re.findall(r"\]\(([^)]+)\)|`(references/[\w./-]+\.md)`", installed.read_text())
        for markdown, backtick in links:
            target = (markdown or backtick).split("#", 1)[0]
            if target and "://" not in target:
                assert (installed.parent / target).exists(), (path, target)
    complex_guide = work_root / (
        ".claude/skills/tilelang2ascend-precision-tuning/references/debug-workflow-complex.md"
    )
    text = complex_guide.read_text()
    assert "lingxi-ascendc" not in text and ".lingxi_verify_logs" not in text
    assert "evaluation_results.json" not in text
    assert all(name in text for name in ("verify_report.json", "verify.log", "metrics.json"))
    assert "ascendc_eval_pipeline.sh" in text
    assert "固定评测入口为准" in output
