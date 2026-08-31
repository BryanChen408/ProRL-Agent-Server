#!/usr/bin/env bash
set -uo pipefail

WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-50}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9382}"
# 先 readlink -f 解掉软链接再取 dirname —— 直接 dirname 在脚本被软链接调用时会拿到
# 软链接的目录而不是真脚本目录,导致 _SCRIPT_DIR/WORK_ROOT 全错。readlink -f 对非软链
# 按原样返回,所以两种情况都安全。
_SRC="${BASH_SOURCE[0]}"
command -v readlink >/dev/null 2>&1 && _SRC="$(readlink -f "$_SRC")"
_SCRIPT_DIR="$(cd "$(dirname "$_SRC")" && pwd)"
# WORK_ROOT = agent 的 workdir(tools/ 的上一级)。打包/找工程/judge_out 一律用它,
# 不用调用时的 $PWD —— agent 常在 <op>/ 或 <op>/kernel/build/ 里跑评测,用 $PWD 会
# 找不到 {op_name}/ → 报 submission missing。脚本固定在 <workdir>/tools/,所以它的
# 上一级就是 workdir,与调用目录无关。
WORK_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
if [[ -z "${ASCENDC_SKILLS_SRC:-}" ]]; then
  if [[ -d /opt/canonical/skills ]]; then ASCENDC_SKILLS_SRC=/opt/canonical/skills
  else ASCENDC_SKILLS_SRC="${_SCRIPT_DIR}/../skills"; fi
fi
TRANS_SKILL="tilelang2ascend-translator"
PERF_SKILL="ops-profiling"

[[ -f "${_SCRIPT_DIR}/env.sh" ]] && source "${_SCRIPT_DIR}/env.sh"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"
PY_BIN="${PY_BIN:-$OPERATOR_PYTHON}"

POLAR_NPU_LOCK_DIR="${POLAR_NPU_LOCK_DIR:-/tmp/npu-locks}"
NPU_LEASE_EXEC="${NPU_LEASE_EXEC:-${_SCRIPT_DIR}/npu_lease_exec.py}"

run_npu_phase() {
  local phase="$1"; shift
  if [[ -n "${POLAR_NPU_LEASE_POOL:-}" ]]; then
    "$AST_CHECK_PYTHON" "$NPU_LEASE_EXEC" \
      --pool "$POLAR_NPU_LEASE_POOL" \
      --lock-dir "$POLAR_NPU_LOCK_DIR" \
      --status-file "$OUT_DIR/npu_lease_status.${phase}.json" \
      -- "$@"
  else
    "$@"
  fi
}

OP_NAME="" IMPL_FILE="" TASK_FILE="" OUT_DIR="" INCREMENTAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --op_name)     OP_NAME="$2"; shift 2;;
    --impl)        IMPL_FILE="$2"; shift 2;;
    --task)        TASK_FILE="$2"; shift 2;;
    --out_dir)     OUT_DIR="$2"; shift 2;;
    --incremental) INCREMENTAL=1; shift;;
    *) echo "[ascendc-eval] unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -z "$OP_NAME" ]] && { echo "[ascendc-eval] --op_name required" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-judge_out}"; case "$OUT_DIR" in /*) ;; *) OUT_DIR="$WORK_ROOT/$OUT_DIR";; esac
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"
IMPL_FILE="${IMPL_FILE:-output/submission/${OP_NAME}_impl.tar.gz}"
# 相对路径的 --impl / --out_dir 一律相对 WORK_ROOT 解析(而不是调用目录),
# 这样从 <op>/ 或 <op>/kernel/build/ 里跑也能定位到 workdir 下的工程与提交物。
case "$IMPL_FILE" in /*) ;; *) IMPL_FILE="$WORK_ROOT/$IMPL_FILE";; esac

SRC_DIR="$WORK_ROOT/$OP_NAME"
AGENT_SIDE=0; [[ -d "$SRC_DIR" ]] && AGENT_SIDE=1
STATE_DIR="$WORK_ROOT/output/.selfcheck"
PACK_SH="${_SCRIPT_DIR}/pack_submission.sh"

write_metrics() {  # ast_ok corr_ok success fw impl speedup error [完整日志文件] [强制 error_type]
  # $9 (force_type) 覆盖文本推断:classify() 按关键词猜,而部分失败的措辞里天然带
  # "对拍"/"编译" 之类的词,会被归错档(例如"对拍前的注册冒烟检查失败"被判成
  # correctness_failed)。调用方已经确知类型时,直接指定。
  local ast_ok="$1" corr_ok="$2" success="$3" fw="$4" impl="$5" sp="$6" error="$7" log_src="${8:-}" force_type="${9:-}"
  : > "$OUT_DIR/metrics_error.log"
  [[ -n "$error" ]] && printf "%s\n" "$error" >> "$OUT_DIR/metrics_error.log"
  [[ -n "$log_src" && -f "$log_src" ]] && cat "$log_src" >> "$OUT_DIR/metrics_error.log"
  AST_OK="$ast_ok" CORR_OK="$corr_ok" SUCCESS="$success" FW="$fw" IMPL="$impl" SP="$sp" \
  CASES_PASSED="${CASES_PASSED:-}" CASES_TOTAL="${CASES_TOTAL:-}" \
  FORCE_TYPE="$force_type" \
  ERR_FILE="$OUT_DIR/metrics_error.log" OUT_DIR="$OUT_DIR" OP="$OP_NAME" python3 - <<'PY'
import json, os, hashlib, re
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
def b(x): return str(x).lower() in ("1","true","yes")
def i(x):
    try: return int(x)
    except (TypeError, ValueError): return None
fw, impl, sp = os.environ.get("FW",""), os.environ.get("IMPL",""), os.environ.get("SP","")
ef = Path(os.environ.get("ERR_FILE","")); full = ef.read_text(errors="replace") if ef.exists() else ""
def classify(t):
    if not t: return None
    l = t.lower(); c = l.replace(" ","")
    if "submission" in l and ("missing" in l or "untar" in l or "缺" in t or "布局" in t): return "submission_missing"
    # get_input 必须搭配失败上下文才算 input_load_failed —— benchmark/verify 日志里
    # 常出现 get_inputs/get_input_groups(取输入的函数名),裸匹配会把 benchmark_failed
    # 错标成 input_load_failed(本该得 T2 0.4 的被判成 infra 重试)。
    if "判分基准" in t \
        or ("get_input" in l and any(k in t for k in ("无法加载","无法写入","未就位","注入失败","缺失"))) \
        or ("filenotfounderror" in l and ".json" in l):
        return "input_load_failed"
    if "aclinit" in c and ("invaliddeviceid" in c or "getdevicecntfailed" in c) or "invalid device id" in l: return "npu_runtime_unavailable"
    if "找不到msprof" in c or "msprof:notfound" in c: return "profiler_unavailable"
    # CANNBot 的 D 类入口必须同时满足:编译通过、正常运行完成、输出契约正确、已有
    # 数值差异字段。对拍日志可能带 ccec_compiler/PATH、degradation warning 等无关词，
    # 所以对拍分支必须先于 AST/编译关键词判定。
    if "对拍" in t or "mare" in l or "mere" in l or "correctness" in l or "result: fail" in l:
        # 有结论再分「数值差异」和「输出契约/前置检查不通过」。后者虽然已经跑到
        # comparator,但不满足 D 类的 shape/可比较前提,粗分类仍是 A 类。
        if re.search(r"case\[\d+\]:", l):
            if re.search(r"(max_abs_diff|mere|matched_ratio)\s*=", l):
                return "correctness_failed"
            return "output_precheck_failed"
        if re.search(r"modulenotfounderror|importerror|_opnamespace|has no attribute|cannot open shared object|undefined symbol", l):
            return "ascendc_load_failed"
        # 明确 ACL 错误码优先于函数名/描述关键词。507035 的常见日志包含
        # rtDeviceSynchronizeWithTimeout，若先搜 timeout 会被错误路由到超时。
        if "507035" in l:
            return "ascendc_launch_failed"
        if "507034" in l:
            return "ascendc_run_timeout"
        if re.search(r"kernel launch failed|aclrtlaunch\w*.*failed|vector core exception|aic error", l):
            return "ascendc_launch_failed"
        if re.search(r"timed?\s*out|timeout|超时|kernel hang|vector core timeout", l):
            return "ascendc_run_timeout"
        return "ascendc_run_crashed"
    if "ast退化" in l or "ast_check" in l or "ast check" in l or "退化" in t or "degrad" in l:
        return "ast_check_failed"
    if "cmake" in l or "make error" in l or "ccec" in l or "bisheng" in l or "编译" in t or "compil" in l: return "ascendc_compile_failed"
    if "speedup" in l or "performance" in l or "性能" in t: return "benchmark_failed"
    # 无法建立责任边界时不能默认为代码/编译错并向训练注入假负样本。
    return "judge_classification_failed"
perf = None
if fw and impl and sp:
    perf = {"framework_latency_ms": float(fw), "impl_latency_ms": float(impl), "speedup_vs_torch": float(sp)}
elif sp:
    perf = {"framework_latency_ms": None, "impl_latency_ms": None, "speedup_vs_torch": float(sp)}
metrics = {
    "schema_version": 2, "op_name": os.environ.get("OP",""),
    "success": b(os.environ["SUCCESS"]), "ast_check_ok": b(os.environ["AST_OK"]),
    "correctness_ok": b(os.environ["CORR_OK"]), "perf_data": perf,
    # 对拍 case 统计(Step2b --json-file 产出;缺失为 null,reward 侧回退固定档)
    "cases_passed": i(os.environ.get("CASES_PASSED","")),
    "cases_total": i(os.environ.get("CASES_TOTAL","")),
    "error": None, "error_type": (os.environ.get("FORCE_TYPE") or "").strip() or classify(full),
    "error_file": "metrics_error.log" if full else None,
    "error_bytes": len(full.encode("utf-8")) if full else 0,
    "error_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest() if full else None,
    "error_truncated": False,
}
_CAP = int(os.environ.get("ASCENDC_ERRLOG_CAP_BYTES", "2097152"))
_raw = full.encode("utf-8")
if len(_raw) > _CAP:
    _head = _raw[: _CAP // 2].decode("utf-8", errors="replace")
    _tail = _raw[-(_CAP // 2):].decode("utf-8", errors="replace")
    ef.write_text(
        f"{_head}\n\n...[truncated {len(_raw) - _CAP} bytes; 完整错误已超过 {_CAP} 字节上限]...\n\n{_tail}",
        encoding="utf-8",
    )
    metrics["error_truncated"] = True
(out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  # 过程事件采集(process reward 数据源,dev_05 M2):agent 侧 only —— judge 侧
  # (AGENT_SIDE=0)的评测是打分动作,不是被评分的过程。从刚落盘的 metrics.json 原样
  # 读四参数,error_type 与 metrics.json 逐字节一致(judge 侧 V3 校验依赖)。
  # || true 兜底:采集失败绝不影响评测本身。
  if [[ "$AGENT_SIDE" == "1" ]]; then
    "$PY_BIN" "$_SCRIPT_DIR/process_track.py" record-eval \
      --metrics "$OUT_DIR/metrics.json" \
      --file "$STATE_DIR/process_info.json" \
      --mirror "${ARTIFACTS_DIR:-}/process_info.json" \
      --op-name "$OP_NAME" \
      >>"$OUT_DIR/process_track.log" 2>&1 || true
  fi
}

# 从 verify_report.json(Step2b --json-file 产出)提取对拍 case 统计到
# CASES_PASSED/CASES_TOTAL,供 write_metrics 落进 metrics.json。
# 缺失/异常(脚本被杀、report 未写、无 case_oks 字段)→ 置空 → metrics 里为 null,
# reward 侧回退固定档,不影响旧行为。
extract_case_stats() {
  CASES_PASSED=""; CASES_TOTAL=""
  [[ -f "$OUT_DIR/verify_report.json" ]] || return 0
  local stats
  stats=$("$PY_BIN" -c "
import json
try:
    oks = (json.load(open('$OUT_DIR/verify_report.json')) or {}).get('case_oks')
except Exception:
    oks = None
print('%d %d' % (sum(1 for x in oks if x), len(oks)) if isinstance(oks, list) and oks else '')
" 2>/dev/null || true)
  [[ -n "$stats" ]] && { CASES_PASSED="${stats%% *}"; CASES_TOTAL="${stats##* }"; }
  return 0
}

fail_hint() {
  # success 是 judge/reward 的历史字段,语义只能保持「实现正确且完成性能测量」；不能再把它
  # 原样展示成 agent 的任务完成信号。agent 侧用 operator_valid/task_complete 两个正交状态，
  # judge 侧没有 task_complete 时明确打印 None，不凭空声称任务已经结束。
  python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print('[ascendc-eval] verdict — operator_valid=%s task_complete=%s ast_check_ok=%s correctness_ok=%s error_type=%s speedup_vs_torch=%s'%(d.get('operator_valid',d.get('success')),d.get('task_complete'),d.get('ast_check_ok'),d.get('correctness_ok'),d.get('error_type'),p.get('speedup_vs_torch')))" 2>/dev/null || true
  python3 - "$OUT_DIR/metrics.json" "$OUT_DIR/metrics_error.log" <<'CLASSIFY' 2>/dev/null || true
import json, re, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if d.get("success"):
    print("[ascendc-eval] 错误分类: 通过（仅表示 operator_valid=true；是否允许结束只看 task_complete）")
    sys.exit(0)
et = str(d.get("error_type") or "")

# Pull the first real exception line out of metrics_error.log. Without this the
# agent only ever sees the coarse label and has to guess; a load/registration
# failure looks exactly like a numerical one.
EXC_RE = re.compile(
    r"^\s*(?:\w+\.)*(?:AttributeError|ImportError|ModuleNotFoundError|RuntimeError|"
    r"TypeError|ValueError|KeyError|IndexError|OSError|AssertionError|"
    r"NameError|SyntaxError|Error|Exception)\b.*",
)
first_exc = ""
try:
    for line in open(sys.argv[2], encoding="utf-8", errors="replace"):
        s = line.rstrip()
        if EXC_RE.match(s):
            first_exc = s.strip()[:400]
            break
except Exception:
    pass

# A "correctness_failed" whose log shows a load/registration exception never
# compared a single element -- calling it a precision mismatch sends the agent
# to tune numerics when the real fix is packaging/registration.
LOAD_EXC = ("AttributeError", "ImportError", "ModuleNotFoundError", "OSError")
load_failure = bool(first_exc) and any(k in first_exc for k in LOAD_EXC)
if "_OpNamespace" in first_exc or "has no attribute" in first_exc:
    load_failure = True

# 崩溃/异常 vs 对拍给出了结论:不看错误文案(关键词白名单补不完),看有没有 case[N]: 行。
# 有 → 对拍跑完并给了结论(数值差异 或 形状/NaN/dtype 前置检查不通过);没有 → 中途崩了。
# 不用「有没有 max_abs_diff」:形状/NaN 不符时上游前置检查早退、算不出逐元素差,那批
# 会被误判成崩溃(实测 9/38),给 agent 的方向也就跟着错。
try:
    _log_low = open(sys.argv[2], encoding="utf-8", errors="replace").read().lower()
except Exception:
    _log_low = ""
crash_failure = not bool(re.search(r"case\[\d+\]:", _log_low))

# 错误分类行内嵌路由:agent 必读此行,把「下一步去哪查」直接写在这里。
# 目标全部是现有 skill 的现有小文档,无新增内容;词表与分类行原话对齐(agent 能逐字命中)。
_SUBMISSION_ROUTE = (";下一步:检查工程顶层、model_new_ascendc.py、kernel/ 与 tarball 布局,"
                     "不需要读取调试 Skill")
_COMPILE_ROUTE = (";下一步:先 Read .claude/skills/tilelang2ascend-translator/SKILL.md;"
                  "符号/API 错再 Read .claude/skills/ascendc-docs-search/SKILL.md,"
                  "按错误符号查 $ASC_DEVKIT_DIR 文档")
_LOAD_ROUTE = (";下一步:先 Read .claude/skills/ascendc-runtime-debug/SKILL.md 和 "
               ".claude/skills/ascendc-runtime-debug/references/kernel_binary_debug.md")
_CRASH_ROUTE = (";下一步:先 Read .claude/skills/ascendc-crash-debug/SKILL.md 和 "
                ".claude/skills/ascendc-crash-debug/references/crash_workflow.md;"
                "ACL错误码再 Read .claude/skills/ascendc-runtime-debug/references/error_codes.md")
_OUTPUT_ROUTE = (";下一步:shape/dtype/输出数量错误先 Read "
                 ".claude/skills/tilelang2ascend-translator/SKILL.md;"
                 "NaN/Inf/全零输出再 Read .claude/skills/ascendc-precision-debug/SKILL.md")
_PRECISION_ROUTE = (";下一步:先 Read .claude/skills/ops-precision-standard/SKILL.md 对容差表,"
                    "再按 .claude/skills/ascendc-precision-debug/SKILL.md 的指引修")
_STATEFUL_ROUTE = (";下一步:先 Read .claude/skills/tilelang2ascend-translator/SKILL.md,"
                   "删除跨调用缓存、常量输出或输入无关捷径")
_BENCHMARK_ROUTE = (";下一步:先 Read .claude/skills/ops-profiling/SKILL.md;"
                    "若 perf.log 是 kernel/ACL 崩溃,再按运行期错误路线处理")

INFRA = {"npu_runtime_unavailable", "input_load_failed", "judge_container_failed",
         "judge_metrics_unreadable", "judge_no_metrics", "task_missing",
         "submission_fetch_failed", "profiler_unavailable",
         "judge_classification_failed"}
if et in INFRA:
    label = "B类-INFRA-环境/基础设施故障(不是你的代码问题,不要改kernel或读取修复文档)"
elif et == "submission_missing":
    label = "A类-提交物/工程布局错误" + _SUBMISSION_ROUTE
elif et == "ast_check_failed":
    label = "A类-AST退化/实现不合规" + _COMPILE_ROUTE
elif et == "ascendc_compile_failed":
    label = "A类-编译/链接错误" + _COMPILE_ROUTE
elif et in ("op_not_registered", "ascendc_load_failed"):
    label = "A类-算子未注册/加载失败(不是精度问题)" + _LOAD_ROUTE
elif et in ("ascendc_run_crashed", "ascendc_run_timeout", "ascendc_launch_failed"):
    label = "A类-kernel崩溃/超时/启动失败(不是精度问题)" + _CRASH_ROUTE
elif et == "output_precheck_failed":
    label = "A类-输出契约/有效性错误(shape/dtype/输出数量/NaN前置检查未通过,不满足D类入口)" + _OUTPUT_ROUTE
elif et == "stateful_impl_detected":
    label = "A类-状态化/缓存/常量输出退化(对拍结果不可信)" + _STATEFUL_ROUTE
elif et == "correctness_failed" and crash_failure:
    # 兜底:error_type 没被 classify 拆出 ascendc_run_crashed 时
    label = "A类-kernel崩溃/运行期错误(不是精度问题)" + _CRASH_ROUTE
elif et == "correctness_failed" and load_failure:
    label = "A类-算子未注册/加载失败(对拍未比较任何元素,不是精度问题)" + _LOAD_ROUTE
elif et == "correctness_failed":
    label = "D类-精度不匹配" + _PRECISION_ROUTE
elif et == "benchmark_failed":
    label = "A类-benchmark执行失败(正确性已通过,但没有形成有效性能结果)" + _BENCHMARK_ROUTE
else:
    # 新增 error_type 未进入映射属于 judge 分类器缺口,不能假装是代码错误。
    label = "B类-INFRA-分类器未覆盖该error_type(停止并上报,不要猜测修复):" + et
print(f"[ascendc-eval] 错误分类: {label}")
if first_exc:
    print(f"[ascendc-eval] 首个异常: {first_exc}")
CLASSIFY
  if ! python3 -c "import json;raise SystemExit(0 if json.load(open('$OUT_DIR/metrics.json')).get('success') else 1)" 2>/dev/null; then
    echo "  ↳ 完整错误在 $OUT_DIR/metrics_error.log;只改 {op}/ 下实现、重打 tarball、重跑本固定入口。"
  fi
}

PIPELINE_GEN_MAX="${POLAR_GEN_PIPELINE_MAX:-6}"
PIPELINE_OPT_MAX="${POLAR_OPT_PIPELINE_MAX:-3}"
PIPELINE_PHASE="generation"; PIPELINE_LIMIT="$PIPELINE_GEN_MAX"; PIPELINE_ATTEMPT=1
PIPELINE_GEN_COUNT=0; PIPELINE_OPT_COUNT=0; PIPELINE_FIRST_SUCCESS=0
BEST_META="$WORK_ROOT/output/submission/.${OP_NAME}_impl.best.meta.json"
TASK_STATE_FILE="$OUT_DIR/task_state.json"
CUR_HASH=""
# 性能目标线。reward = 0.75 + 0.25*tanh(ln speedup)(operator_reward.reward_from_metrics):
# 1.0x 只拿 0.75,低于 1.0x 反而往 0.5 掉。CLAUDE.md 4-S.4 的达标判定必须同步这个数。
PERF_TARGET="${POLAR_PERF_TARGET:-1.1}"

# 预算状态写进 $ARTIFACTS_DIR(gateway 侧 session 目录),不是 workdir —— 逐字对齐 triton 侧的
# pipeline_status_write();ascendc 移植时整段漏了,导致 pipeline_budget_status.json 从来没落过盘,
# watcher 的 should_cancel_from_status 分支对 ascendc 一直是空跑。
# 注意口径:workdir 里的 .selfcheck 计数器 agent 删得掉,这份状态文件也在同一个 session bind mount 里,
# 两者都只是「第二信号 + 遥测」。真正的强制层是 watcher 从 gateway completion 流里数固定入口调用次数
# (polar_pipeline_budget_watcher.analyze_budget / should_cancel),那条链路 agent 改不到。
pipeline_status_write() {
  [[ -n "${ARTIFACTS_DIR:-}" ]] || return 0
  mkdir -p "${ARTIFACTS_DIR}" 2>/dev/null || return 0
  PIPELINE_STATUS_FILE="${ARTIFACTS_DIR}/pipeline_budget_status.json" \
  SESSION_ID="${SESSION_ID:-}" TASK_ID="${TASK_ID:-}" OP_NAME="${OP_NAME:-}" \
  PIPELINE_PHASE="$PIPELINE_PHASE" PIPELINE_ATTEMPT="$PIPELINE_ATTEMPT" PIPELINE_LIMIT="$PIPELINE_LIMIT" \
  PIPELINE_GEN_COUNT="$PIPELINE_GEN_COUNT" PIPELINE_OPT_COUNT="$PIPELINE_OPT_COUNT" \
  PIPELINE_FIRST_SUCCESS="$PIPELINE_FIRST_SUCCESS" python3 - <<'PY' 2>/dev/null || true
import json, os, time
from pathlib import Path

def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

attempt = to_int(os.environ.get("PIPELINE_ATTEMPT"), 0)
limit = to_int(os.environ.get("PIPELINE_LIMIT"), 0)
payload = {
    "schema_version": 1,
    "session_id": os.environ.get("SESSION_ID") or None,
    "task_id": os.environ.get("TASK_ID") or None,
    "op_name": os.environ.get("OP_NAME") or None,
    "phase": os.environ.get("PIPELINE_PHASE") or None,
    "attempt": attempt,
    "limit": limit,
    "gen_count": to_int(os.environ.get("PIPELINE_GEN_COUNT"), 0),
    "opt_count": to_int(os.environ.get("PIPELINE_OPT_COUNT"), 0),
    "first_success": str(os.environ.get("PIPELINE_FIRST_SUCCESS") or "").lower() in {"1", "true", "yes"},
    "limit_exhausted": bool(limit and attempt > limit),
    "updated_at_unix": time.time(),
}
path = Path(os.environ["PIPELINE_STATUS_FILE"])
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

# 正确性一过就收工是当前最大的分数漏点(实测 179 个成功 session 只有 2 个继续优化,
# speedup 中位数 0.859x、58.8% 慢于 torch)。这里把「实现有效」和「任务完成」拆开:
# operator_valid 只表示正确性/测速通过；task_complete 才是 Stop hook 的唯一放行信号。
# 目标未达且仍有 optimization 预算时 task_complete=false，不能再被 success=true 误导结束。
emit_optimization_prompt() {  # $1=speedup
  [[ "$AGENT_SIDE" == "1" ]] || return 0
  local sp="${1:-}" remain=$(( PIPELINE_OPT_MAX - PIPELINE_OPT_COUNT ))
  [[ "$remain" -ge 0 ]] || remain=0
  local hit task_complete completion_reason phase_next
  hit=$(SP="$sp" TARGET="$PERF_TARGET" python3 -c '
import os
try:
    print("1" if float(os.environ["SP"]) >= float(os.environ["TARGET"]) else "0")
except Exception:
    print("0")' 2>/dev/null || echo 0)
  if [[ "$hit" == "1" ]]; then
    task_complete="1"; completion_reason="target_met"; phase_next="complete"
  elif [[ "$remain" -le 0 ]]; then
    task_complete="1"; completion_reason="budget_exhausted"; phase_next="complete"
  else
    task_complete="0"; completion_reason="pending_optimization"; phase_next="optimization"
  fi

  # 先落权威状态再回显。Stop hook 只读这份文件；stdout 即使因 Bash 自动转后台而丢失，
  # 也不会把「目标未达」误当成已完成。只在 agent 侧写，judge 的 metrics schema/奖励字段不变。
  if [[ -n "${OUT_DIR:-}" && -f "${OUT_DIR}/metrics.json" ]]; then
    SP="$sp" HIT="$hit" COMPLETE="$task_complete" REASON="$completion_reason" \
    PHASE_NEXT="$phase_next" TARGET="$PERF_TARGET" OPT_MAX="$PIPELINE_OPT_MAX" \
    OPT_USED="$PIPELINE_OPT_COUNT" REMAIN="$remain" MJ="$OUT_DIR/metrics.json" \
    TS="$TASK_STATE_FILE" python3 - <<'PY' 2>/dev/null || true
import json, os
from pathlib import Path
p = Path(os.environ["MJ"])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
hit = os.environ.get("HIT") == "1"
complete = os.environ.get("COMPLETE") == "1"
target, remain = os.environ["TARGET"], int(os.environ["REMAIN"])
reason = os.environ["REASON"]
if hit:
    action = (f"正确性已通过,speedup={os.environ.get('SP','')}x ≥ 目标线 {target}x —— "
              "已达标，停止性能迭代并提交最佳版本。")
elif complete:
    action = (f"正确性已通过,但 speedup={os.environ.get('SP','')}x < 目标线 {target}x；"
              "optimization 预算已耗尽，停止调用工具并提交 .best 最佳版本。")
else:
    action = (f"正确性已通过,但 speedup={os.environ.get('SP','')}x < 目标线 {target}x —— "
              f"未达标,不要结束任务。下一次调用本固定入口进入 optimization 阶段，还剩 {remain} 次预算。"
              " 先 Read .claude/skills/ops-profiling/SKILL.md，再读取真实逐 case 结果并修改 kernel；"
              "源码变化后再重跑。"
              " .best.tar.gz 只在 speedup 更高时才替换。")
d["operator_valid"] = bool(d.get("success") and d.get("correctness_ok"))
d["task_complete"] = complete
d["completion_reason"] = reason
d["next_step"] = {
    "perf_target_speedup": float(target),
    "target_met": hit,
    "phase_next": os.environ["PHASE_NEXT"],
    "optimization_budget": int(os.environ["OPT_MAX"]),
    "optimization_used": int(os.environ["OPT_USED"]),
    "optimization_remaining": remain,
    "action": action,
}
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
state = {
    "schema_version": 1,
    "op_name": d.get("op_name"),
    "operator_valid": d["operator_valid"],
    "task_complete": complete,
    "completion_reason": reason,
    "perf_data": d.get("perf_data"),
    "next_step": d["next_step"],
}
ts = Path(os.environ["TS"])
tmp = ts.with_suffix(ts.suffix + ".tmp")
tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(ts)
PY
  fi

  if [[ "$hit" == "1" ]]; then
    echo "[ascendc-eval] task status — operator_valid=true task_complete=true completion_reason=target_met"
    echo "[ascendc-eval] 正确性已通过,speedup=${sp}x ≥ 目标线 ${PERF_TARGET}x —— 已达标,停止性能迭代并提交最佳版本。"
  elif [[ "$remain" -le 0 ]]; then
    echo "[ascendc-eval] task status — operator_valid=true task_complete=true completion_reason=budget_exhausted"
    echo "[ascendc-eval] 正确性已通过,但 speedup=${sp}x < 目标线 ${PERF_TARGET}x；optimization 预算已耗尽,提交 .best 最佳版本。"
  else
    echo "[ascendc-eval] task status — operator_valid=true task_complete=false completion_reason=pending_optimization"
    echo "[ascendc-eval] 正确性已通过,但 speedup=${sp}x < 目标线 ${PERF_TARGET}x —— 未达标,不要结束任务。"
    echo "[ascendc-eval] 下一次调用本固定入口进入 optimization 阶段:预算 ${PIPELINE_OPT_MAX} 次,已用 ${PIPELINE_OPT_COUNT} 次,剩 ${remain} 次。"
    echo "[ascendc-eval] 先 Read .claude/skills/ops-profiling/SKILL.md,再读取真实逐 case 结果并修改 kernel;源码变化后重跑本入口。"
    echo "[ascendc-eval] .best.tar.gz 只在 speedup 更高时才替换 —— 优化失败不会覆盖已知最佳版本。"
  fi

}

# task_state.json 与当前候选 metrics 分离。进入 optimization 后，即使新候选编译/运行失败、
# metrics.json 被失败结果覆盖，历史正确实现仍然存在，Stop 门禁也不能忘掉尚未用完的预算。
# 每次实际 optimization 调用一开始就同步剩余预算；最后一次即使失败也按 best 放行。
sync_task_state_budget() {
  [[ "$AGENT_SIDE" == "1" && "$PIPELINE_PHASE" == "optimization" \
      && -f "$TASK_STATE_FILE" ]] || return 0
  TS="$TASK_STATE_FILE" USED="$PIPELINE_OPT_COUNT" LIMIT="$PIPELINE_OPT_MAX" \
  TARGET="$PERF_TARGET" python3 - <<'PY' 2>/dev/null || true
import json, os
from pathlib import Path

p = Path(os.environ["TS"])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if not isinstance(d, dict) or d.get("operator_valid") is not True:
    raise SystemExit(0)
used = max(0, int(os.environ["USED"]))
limit = max(0, int(os.environ["LIMIT"]))
remaining = max(0, limit - used)
ns = d.get("next_step") if isinstance(d.get("next_step"), dict) else {}
ns["perf_target_speedup"] = float(os.environ["TARGET"])
ns["optimization_budget"] = limit
ns["optimization_used"] = used
ns["optimization_remaining"] = remaining
if d.get("task_complete") is not True:
    if remaining == 0:
        d["task_complete"] = True
        d["completion_reason"] = "budget_exhausted"
        ns["phase_next"] = "complete"
        ns["action"] = "optimization 预算已耗尽；停止调用工具并提交 .best 历史最佳正确版本。"
    else:
        d["task_complete"] = False
        d["completion_reason"] = "pending_optimization"
        ns["phase_next"] = "optimization"
d["next_step"] = ns
tmp = p.with_suffix(p.suffix + ".tmp")
tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(p)
PY
}

pipeline_over_limit() {
  [[ "$PIPELINE_ATTEMPT" -gt "$PIPELINE_LIMIT" ]] && return 0
  return 1
}

pipeline_at_limit() {
  [[ "$PIPELINE_ATTEMPT" -ge "$PIPELINE_LIMIT" ]] && return 0
  return 1
}

pack_best() {  # $1=verified?  $2=speedup?
  [[ -x "$PACK_SH" || -f "$PACK_SH" ]] || return 0
  local args=("$OP_NAME"); [[ -n "${1:-}" ]] && args+=(--verified)
  [[ -n "${2:-}" ]] && args+=(--speedup "$2")
  # 打包目录用 WORK_ROOT(脚本位置反推的 workdir),不用调用时的 $PWD —— 否则在
  # <op>/ 或 <op>/kernel/build/ 里跑时 pack 找不到 {op_name}/,submission missing。
  WORKDIR="$WORK_ROOT" bash "$PACK_SH" "${args[@]}" || true
}

if [[ "$AGENT_SIDE" == "1" ]]; then
  mkdir -p "$STATE_DIR"
  CUR_HASH=$(cd "$SRC_DIR" && find . \
      \( -name build -o -name dist -o -name '*.egg-info' -o -name '__pycache__' \) -prune -o \
      -type f ! -name '*.so' ! -name '*.a' ! -name '*.o' ! -name '*.whl' \
      ! -name '.eval_last.log' ! -name 'performance.json' ! -name 'preformance.json' \
      -print0 2>/dev/null | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
  _HASH_FILE="$STATE_DIR/.${OP_NAME}_last.hash"
  if [[ -n "$CUR_HASH" && -f "$_HASH_FILE" && -f "$OUT_DIR/metrics.json" \
        && "$CUR_HASH" == "$(cat "$_HASH_FILE" 2>/dev/null)" ]]; then
    echo "[ascendc-eval] {op}/ 源码与上次评测完全一致 → 复用上次结论,跳过编译/对拍/性能(不消耗预算)"
    python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print('[ascendc-eval] cached evaluation — operator_valid=%s task_complete=%s ast_check_ok=%s correctness_ok=%s speedup_vs_torch=%s'%(d.get('operator_valid',d.get('success')),d.get('task_complete'),d.get('ast_check_ok'),d.get('correctness_ok'),p.get('speedup_vs_torch')))" 2>/dev/null || true
    exit 0
  fi
  if [[ -f "$BEST_META" ]] && python3 -c "
import json,sys;d=json.load(open('$BEST_META'));sys.exit(0 if int(d.get('tier') or 0)>=3 else 1)" 2>/dev/null; then
    PIPELINE_PHASE="optimization"; PIPELINE_LIMIT="$PIPELINE_OPT_MAX"; PIPELINE_FIRST_SUCCESS=1
  fi
  read -r PIPELINE_GEN_COUNT PIPELINE_OPT_COUNT <<<"$(PIPELINE_STATE_FILE="$STATE_DIR/.${OP_NAME}_budget.json" PIPELINE_PHASE="$PIPELINE_PHASE" python3 -c '
import json, os
from pathlib import Path
path = Path(os.environ["PIPELINE_STATE_FILE"]); data = {}
if path.exists():
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict): data = loaded
    except Exception: data = {}
key = "opt_count" if os.environ["PIPELINE_PHASE"] == "optimization" else "gen_count"
data[key] = int(data.get(key) or 0) + 1
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(int(data.get("gen_count") or 0), int(data.get("opt_count") or 0))' 2>/dev/null || echo "1 0")"
  PIPELINE_GEN_COUNT="${PIPELINE_GEN_COUNT:-0}"; PIPELINE_OPT_COUNT="${PIPELINE_OPT_COUNT:-0}"
  if [[ "$PIPELINE_PHASE" == "optimization" ]]; then
    PIPELINE_ATTEMPT="${PIPELINE_OPT_COUNT:-1}"
  else
    PIPELINE_ATTEMPT="${PIPELINE_GEN_COUNT:-1}"
  fi
  pipeline_status_write
  sync_task_state_budget
  echo "[pipeline-budget] phase=$PIPELINE_PHASE attempt=$PIPELINE_ATTEMPT/$PIPELINE_LIMIT"
  # 只有预算内的候选才能打包并参与 best 比较。第 limit+1 次之后的源码
  # 未经评测，无论当前目录里还留有什么旧日志/性能文件，都不得影响 .best。
  if ! pipeline_over_limit; then
    pack_best
  fi
fi

_on_exit() {
  local rc=$?
  [[ "$AGENT_SIDE" == "1" ]] || return 0
  # 只有本次确实在预算内执行了评测，才能用 metrics 认证当前源码。
  # 超限分支在 Step0 前退出，metrics.json 仍属于上一版源码；若在此读取，
  # 会把旧的 correctness/speedup 错贴到未评测的当前代码上。
  if [[ "$PIPELINE_ATTEMPT" =~ ^[0-9]+$ ]] && ! pipeline_over_limit; then
    local corr sp
    corr=$(python3 -c "import json;print('1' if json.load(open('$OUT_DIR/metrics.json')).get('correctness_ok') else '')" 2>/dev/null || echo "")
    sp=$(python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print(p.get('speedup_vs_torch') or '')" 2>/dev/null || echo "")
    pack_best "$corr" "$sp"
  fi
  if [[ -n "$CUR_HASH" && "$PIPELINE_ATTEMPT" -le "$PIPELINE_LIMIT" ]]; then
    printf "%s" "$CUR_HASH" > "$STATE_DIR/.${OP_NAME}_last.hash"
  fi
  if [[ "$PIPELINE_ATTEMPT" =~ ^[0-9]+$ ]] && pipeline_at_limit; then
    if [[ "$PIPELINE_ATTEMPT" -gt "$PIPELINE_LIMIT" || "$PIPELINE_PHASE" != "generation" || "$rc" != "0" ]]; then
      echo "[pipeline-budget] LIMIT_EXHAUSTED phase=$PIPELINE_PHASE attempt=$PIPELINE_ATTEMPT/$PIPELINE_LIMIT"
      echo "本阶段固定评测入口调用次数已经用完。必须立即停止当前任务。"
      echo "禁止继续分析错误、禁止总结修复方案、禁止恢复文件、禁止再次验证。"
      echo "**提交物已自动打包并留存历史最优版本(.best.tar.gz),当前工作文件状态不重要。**"
      echo "禁止再调用本固定入口;禁止再调用 Bash/Skill/Read/Write/Edit 等任何工具。"
      echo "现在只能输出最终简短结论并结束。"
    fi
  fi
}
trap _on_exit EXIT

# 超限调用只落状态并保留最佳提交物，不能再进入解包、编译、对拍和上板。之前第 N+1 次
# 仍会把整条 pipeline 跑完，watcher 虽最终能取消 session，昂贵工作已经发生。
if [[ "$AGENT_SIDE" == "1" ]] && pipeline_over_limit; then
  echo "[pipeline-budget] budget already exhausted; skip evaluation work"
  exit 0
fi

# 走到这里说明本次要对当前源码开始一次新评测。哈希缓存分支已在上方
# 返回，因此下列文件都是上一版源码的遗留结果，不能再被 EXIT trap、case 统计
# 或下一轮 Agent 当成当前结果。本轮各阶段会按实际进展重新产生它们。
rm -f "$OUT_DIR/metrics.json" "$OUT_DIR/metrics_error.log" \
      "$OUT_DIR/verify_report.json" "$OUT_DIR/performance.json"

# Step0
WORK="$OUT_DIR/work"
[[ "$INCREMENTAL" == "1" ]] || rm -rf "$WORK"
mkdir -p "$WORK"
if [[ ! -f "$IMPL_FILE" ]]; then
  if [[ ! -d "$SRC_DIR" ]]; then
    # 工程目录压根不存在:agent 在没建 {op_name}/ 工程时就跑了评测(逻辑错,不是 infra)。
    # 明确告诉它先建工程,而不是一句 "submission missing" 让它猜。
    write_metrics false false false "" "" "" \
      "工程目录不存在: $SRC_DIR —— 先创建 $OP_NAME/ 工程(kernel/ + model_new_ascendc.py),再跑固定评测入口;不要在没建工程时跑评测"
  else
    # 工程在,但提交物没出来(pack 失败或 judge 侧没就位)。
    write_metrics false false false "" "" "" "submission missing: $IMPL_FILE (工程目录 $SRC_DIR 存在但打包未产出 tarball)"
  fi
  echo "[ascendc-eval] submission missing"; fail_hint; exit 1
fi
tar xzf "$IMPL_FILE" -C "$WORK" 2>/dev/null || { write_metrics false false false "" "" "" "cannot untar submission: $IMPL_FILE"; echo "[ascendc-eval] untar failed"; fail_hint; exit 1; }
MNA="$(find "$WORK" -maxdepth 3 -name model_new_ascendc.py | head -1)"
TASK_DIR="$(dirname "$MNA" 2>/dev/null)"
if [[ -z "$TASK_DIR" || ! -d "$TASK_DIR/kernel" ]]; then
  write_metrics false false false "" "" "" "submission tarball 缺 {op}/kernel 或 model_new_ascendc.py"
  echo "[ascendc-eval] bad submission layout"; fail_hint; exit 1
fi
if [[ "$(realpath "$TASK_DIR")" == "$(realpath "$WORK")" ]]; then
  _NORM="$WORK/.__norm__/$OP_NAME"
  mkdir -p "$_NORM"
  find "$WORK" -mindepth 1 -maxdepth 1 ! -name '.__norm__' -exec mv {} "$_NORM/" \; 2>/dev/null || true
  mv "$_NORM" "$WORK/$OP_NAME" && rmdir "$WORK/.__norm__" 2>/dev/null
  TASK_DIR="$WORK/$OP_NAME"
  if [[ ! -d "$TASK_DIR/kernel" ]]; then
    write_metrics false false false "" "" "" "submission tarball 布局异常:归一化后仍缺 $OP_NAME/kernel"
    echo "[ascendc-eval] bad submission layout after normalize"; fail_hint; exit 1
  fi
  echo "[ascendc-eval] normalized flat tarball -> $OP_NAME/"
fi
OP_DIR_NAME="$OP_NAME"
if [[ "$(basename "$TASK_DIR")" != "$OP_NAME" ]]; then
  write_metrics false false false "" "" "" \
    "submission tarball 顶层目录名 $(basename "$TASK_DIR") 与 --op_name $OP_NAME 不一致"
  echo "[ascendc-eval] op dir name mismatch"; fail_hint; exit 1
fi

_TOP_REL="$(realpath --relative-to="$WORK" "$TASK_DIR" 2>/dev/null | cut -d/ -f1)"
if [[ -n "$_TOP_REL" && "$_TOP_REL" != "." && "$_TOP_REL" != ".." ]]; then
  find "$WORK" -mindepth 1 -maxdepth 1 ! -name "$_TOP_REL" -exec rm -rf {} + 2>/dev/null || true
fi

find "$TASK_DIR" \( -name '*.so' -o -name '*.a' -o -name '*.o' -o -name '*.whl' \
     -o -name 'build' -o -name 'dist' -o -name '*.egg-info' -o -name '__pycache__' \) \
     -exec rm -rf {} + 2>/dev/null || true

TASK_SRC="${TASK_FILE:-input/${OP_NAME}.py}"
# 与 --impl / --out_dir 保持一致:相对 --task 固定从 workdir 解析，避免 agent
# 在 <op>/kernel/build/ 等子目录调用时把 input/ 错当成当前目录的子目录。
case "$TASK_SRC" in /*) ;; *) TASK_SRC="$WORK_ROOT/$TASK_SRC";; esac
JSON_SRC="$(dirname "$TASK_SRC")/${OP_NAME}.json"
TASK_SRC="$(realpath "$TASK_SRC" 2>/dev/null || echo "$TASK_SRC")"
JSON_SRC="$(realpath "$JSON_SRC" 2>/dev/null || echo "$JSON_SRC")"
if [[ ! -f "$TASK_SRC" ]]; then
  write_metrics false false false "" "" "" \
    "判分基准缺失(get_input 无法加载): $TASK_SRC —— judge 侧 input/ 未就位,非算子问题"
  echo "[ascendc-eval] golden missing"; fail_hint; exit 1
fi
if [[ ! -f "$JSON_SRC" ]] && grep -q "get_input_groups" "$TASK_SRC" 2>/dev/null; then
  write_metrics false false false "" "" "" \
    "判分基准缺失(get_input_groups 需同名 .json): $JSON_SRC —— judge 侧 input/ 未就位,非算子问题"
  echo "[ascendc-eval] case spec missing"; fail_hint; exit 1
fi

inject_baseline() {
  cp -f "$TASK_SRC" "$TASK_DIR/model.py" || return 1
  [[ -f "$JSON_SRC" ]] && { cp -f "$JSON_SRC" "$TASK_DIR/${OP_NAME}.json" || return 1; }
  chmod 444 "$TASK_DIR/model.py" "$TASK_DIR/${OP_NAME}.json" 2>/dev/null || true
  return 0
}
if ! inject_baseline; then
  write_metrics false false false "" "" "" "判分基准注入失败(get_input):无法写入 $TASK_DIR"
  echo "[ascendc-eval] baseline inject failed"; fail_hint; exit 1
fi

SK="$WORK/.claude/skills"
rm -rf "$SK"; mkdir -p "$SK"
if ! cp -r "$ASCENDC_SKILLS_SRC/$TRANS_SKILL" "$SK/$TRANS_SKILL" \
   || ! cp -r "$ASCENDC_SKILLS_SRC/$PERF_SKILL" "$SK/$PERF_SKILL"; then
  write_metrics false false false "" "" "" \
    "judge 环境异常(get_input 之前):无法从 $ASCENDC_SKILLS_SRC 铺设评测脚本"
  echo "[ascendc-eval] skills setup FAILED"; fail_hint; exit 1
fi

# Step1
echo "[ascendc-eval] Step1 anti-degradation (AST)"
VALIDATOR="$SK/$TRANS_SKILL/scripts/validate_ascendc_impl.py"
if [[ -f "$VALIDATOR" ]]; then
  if ! AST_OUT=$("$AST_CHECK_PYTHON" "$VALIDATOR" "$TASK_DIR/model_new_ascendc.py" 2>&1); then
    write_metrics false false false "" "" "" "AST退化检查失败: $AST_OUT"
    echo "[ascendc-eval] AST FAILED — 退化检测未通过(error_type=ast_check_failed)"; fail_hint; exit 1
  fi
elif ! grep -q "torch.ops.npu" "$TASK_DIR/model_new_ascendc.py" 2>/dev/null; then
  write_metrics false false false "" "" "" "退化: model_new_ascendc.py 未调用 torch.ops.npu.<op>(疑似纯 torch)"
  echo "[ascendc-eval] degradation FAILED"; fail_hint; exit 1
else
  echo "[ascendc-eval] WARN: validate_ascendc_impl.py 缺失,已回退到 grep 判据(护栏变弱)"
fi

# Step2
echo "[ascendc-eval] Step2 compile (no NPU)"
KERNEL_DIR="$TASK_DIR/kernel"
BUILDER="$SK/$TRANS_SKILL/scripts/build_ascendc.py"
_CLEAN=(--clean); [[ "$INCREMENTAL" == "1" ]] && _CLEAN=()
if ! (
  # 不要用 set -e:在 `if ! ( ... )` 里 `!` 会禁用子 shell 的 errexit,build_ascendc
  # 失败时 set -e 不退出,继续跑 setup.py(被 || echo 兜底成 exit 0),子 shell 退出码变 0,
  # `if !` 看不到失败 → 编译失败被错标成后续的 op_not_registered(.so NONE)。
  # 改为 build 失败后 `|| exit $?` 显式传播退出码。
  cd "$WORK"
  rm -rf "$KERNEL_DIR/dist"
  WORKDIR="$WORK" ASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
    "$PY_BIN" "$BUILDER" "$TASK_DIR" -v "$SOC_VERSION" --build-type "$BUILD_TYPE" "${_CLEAN[@]}" \
    || exit $?
  if [[ -f "$KERNEL_DIR/setup.py" ]]; then
    cd "$KERNEL_DIR"
    "$PY_BIN" setup.py bdist_wheel && "$PY_BIN" -m pip install dist/*.whl --force-reinstall \
      || echo "[ascendc-eval] wheel 安装失败,已忽略(对拍从 kernel/build/ 直接 import)"
  fi
) >"$OUT_DIR/compile.log" 2>&1; then
  write_metrics true false false "" "" "" "AscendC 编译失败(完整构建日志如下)" "$OUT_DIR/compile.log" \
    "ascendc_compile_failed"
  # 改动3: 编译失败时把首个错误 ±上下文直接打到 stderr(进工具结果),否则 agent 只能拿到
  # "完整错误在 metrics_error.log",38% 的 session 会跑去自编译 cmake 反推错误、白烧轮数。
  echo "--- compile.log 首个错误上下文(完整日志见 $OUT_DIR/compile.log) ---" >&2
  # 单进程 awk 取代 grep|cut+sed 管道:① grep -E 只有三种错误形态,ccec/ld/Traceback 会漏到
  #   tail 兜底拿错段;② grep -n|cut 行号解析脆。这里一次扫描、定位首个错误、打印前后文。
  awk -v before=5 -v after=25 '
    { lines[NR] = $0 }
    !found && /error:|CMake Error|undefined reference|fatal error|FAILED:|ld: |Traceback|\[ERROR\]/ {
      found = NR
    }
    END {
      if (found) {
        s = found - before; if (s < 1) s = 1
        e = found + after;  if (e > NR) e = NR
        for (i = s; i <= e; i++) print lines[i]
      } else {
        s = NR - 29; if (s < 1) s = 1
        for (i = s; i <= NR; i++) print lines[i]
      }
    }' "$OUT_DIR/compile.log" >&2
  echo "[ascendc-eval] compile FAILED"; fail_hint; exit 1
fi

if ! inject_baseline; then
  write_metrics false false false "" "" "" "判分基准注入失败(get_input):对拍前无法复位 $TASK_DIR"
  echo "[ascendc-eval] baseline re-inject failed"; fail_hint; exit 1
fi

# Step2a: is the operator actually reachable via torch.ops.npu? Packaging mistakes
# (nested NpuExtension name, .so built where the submission's loader never globs,
# silently skipped wheel install) survive compile and only blow up inside Step2b
# as an AttributeError, which then gets reported as correctness_failed ->
# "D类-精度不匹配". Catch it here, needs no NPU, and give it its own error_type.
SMOKE="${_SCRIPT_DIR}/check_op_registered.py"
if [[ -f "$SMOKE" ]]; then
  echo "[ascendc-eval] Step2a op registration smoke (no NPU)"
  if ! SMOKE_OUT=$(cd "$WORK" && WORKDIR="$WORK" "$PY_BIN" "$SMOKE" "$TASK_DIR" --workdir "$WORK" 2>&1); then
    printf "%s\n" "$SMOKE_OUT" > "$OUT_DIR/op_smoke.log"
    write_metrics true false false "" "" "" \
      "算子未注册/加载失败(对拍前冒烟检查;完整输出如下)" "$OUT_DIR/op_smoke.log" "op_not_registered"
    echo "[ascendc-eval] op registration FAILED"; fail_hint; exit 1
  fi
  printf "%s\n" "$SMOKE_OUT" > "$OUT_DIR/op_smoke.log"
fi

# Step2b
echo "[ascendc-eval] Step2b verify (NPU lease)"
VER="$SK/$TRANS_SKILL/scripts/verification_ascendc.py"
if ! VER_OUT=$(cd "$WORK" && export WORKDIR="$WORK" PYTHONPATH="$SK/$TRANS_SKILL/scripts:${PYTHONPATH:-}" \
      && run_npu_phase verify "$PY_BIN" "$VER" "$OP_DIR_NAME" --json-file "$OUT_DIR/verify_report.json" 2>&1); then
  printf "%s\n" "$VER_OUT" > "$OUT_DIR/verify.log"
  # 在这里就判定,不交给 classify() 的文本推断 —— verify.log 里带 PATH 环境 dump
  # (含 ccec_compiler),classify() 的 "ccec"/"compil" 分支在 对拍 分支之前命中,
  # 会把对拍阶段的失败一律错标成 ascendc_compile_failed(0.25)。
  #
  # 判据是「对拍有没有给出结论」,看有没有 case[N]: 行 —— 上游无论用什么措辞报结论
  # (数值差异/形状不符/NaN 不符/dtype 不符/...)都会打 case[N]: output[M]: 前缀。
  # 不用「有没有 max_abs_diff」:形状或 NaN 掩码不一致时上游走前置检查早退,压根算不出
  # 逐元素差,那批本是「对拍跑完了、结果不对」,却会被误判成「对拍没跑完(崩溃)」
  # (实测 195351:9/38 中招 —— 6 个形状不符 + 3 个 NaN 不符)。
  #
  #   有 case[N] 行 → 对拍给出结论了
  #        ├─ 有 max_abs_diff/MERE/matched_ratio → 数值差异   correctness_failed
  #        └─ 没有(形状/NaN/dtype 前置检查不通过) → output_precheck_failed
  #   无 case[N] 行 → 对拍未形成有效结论,再按加载/超时/启动/其他崩溃细分(A类)
  if grep -qE "case\[[0-9]+\]:" "$OUT_DIR/verify.log"; then
    if grep -qEi "(max_abs_diff|mere|matched_ratio)[[:space:]]*=" "$OUT_DIR/verify.log"; then
      _VER_TYPE="correctness_failed"
    else
      _VER_TYPE="output_precheck_failed"
    fi
  elif grep -qEi "ModuleNotFoundError|ImportError|_OpNamespace|has no attribute|cannot open shared object|undefined symbol" "$OUT_DIR/verify.log"; then
    _VER_TYPE="ascendc_load_failed"
  # 错误码必须先于文本关键词。507035 日志中的 WithTimeout 是同步 API 名，不代表 507034 超时。
  elif grep -qE "507035" "$OUT_DIR/verify.log"; then
    _VER_TYPE="ascendc_launch_failed"
  elif grep -qE "507034" "$OUT_DIR/verify.log"; then
    _VER_TYPE="ascendc_run_timeout"
  elif grep -qEi "kernel launch failed|aclrtlaunch[a-zA-Z0-9_]*.*failed|vector core exception|aic error" "$OUT_DIR/verify.log"; then
    _VER_TYPE="ascendc_launch_failed"
  elif grep -qEi "timed?[[:space:]]*out|timeout|超时|kernel hang|vector core timeout" "$OUT_DIR/verify.log"; then
    _VER_TYPE="ascendc_run_timeout"
  else
    _VER_TYPE="ascendc_run_crashed"
  fi
  extract_case_stats
  write_metrics true false false "" "" "" "数值对拍失败(Result: fail;完整对拍输出如下)" "$OUT_DIR/verify.log" \
    "$_VER_TYPE"
  # 改动3扩展: 对拍失败把 Comparison 段直接打到 stderr(进工具结果)。否则 agent 只拿到
  # "D类-精度不匹配"一句,没有 max_abs_diff/tolerance/失配元素数,只能盲调数值。
  echo "--- verify 对拍失败详情(完整日志见 $OUT_DIR/verify.log) ---" >&2
  if grep -q "Comparison" "$OUT_DIR/verify.log"; then
    sed -n '/Comparison/,$p' "$OUT_DIR/verify.log" | head -60 >&2
  else
    tail -40 "$OUT_DIR/verify.log" >&2
  fi
  echo "[ascendc-eval] verify FAILED"; fail_hint; exit 1
fi
printf "%s\n" "$VER_OUT" > "$OUT_DIR/verify.log"

# Step2c
DETECT="${_SCRIPT_DIR}/detect_stateful_impl.py"
if [[ -f "$DETECT" ]]; then
  echo "[ascendc-eval] Step2c stateful/cache detection (NPU lease)"
  DET_OUT=$(cd "$WORK" && run_npu_phase detect "$PY_BIN" "$DETECT" "$TASK_DIR" 2>&1); DET_RC=$?
  printf "%s\n" "$DET_OUT" > "$OUT_DIR/detect.log"
  if [[ "$DET_RC" == "1" ]]; then
    write_metrics true false false "" "" "" "对拍结果不可信(缓存/常量输出): $DET_OUT" \
      "$OUT_DIR/detect.log" "stateful_impl_detected"
    echo "[ascendc-eval] stateful/cache DETECTED"; fail_hint; exit 1
  fi
  [[ "$DET_RC" == "2" ]] && echo "  ↳ ${DET_OUT}"
fi

# Step3
echo "[ascendc-eval] Step3 performance (msprof --quick, NPU lease)"
PERF="$SK/$PERF_SKILL/scripts/msprof_perf_summary.py"
MSPROF_BIN="${MSPROF_BIN:-}"
if [[ -z "$MSPROF_BIN" ]]; then
  if [[ -x "${ASCEND_HOME_PATH}/bin/msprof" ]]; then MSPROF_BIN="${ASCEND_HOME_PATH}/bin/msprof"
  else MSPROF_BIN="$(command -v msprof 2>/dev/null || true)"; fi
fi
if [[ -z "$MSPROF_BIN" ]]; then
  write_metrics true true false "" "" "" \
    "judge 环境异常:找不到 msprof(ASCEND_HOME_PATH=${ASCEND_HOME_PATH})" "" \
    "profiler_unavailable"
  echo "[ascendc-eval] msprof NOT FOUND"; fail_hint; exit 1
fi
export PATH="$(dirname "$MSPROF_BIN"):$PATH"
MSPROF_WARMUP="${MSPROF_WARMUP:-3}"
PERF_JSON="$TASK_DIR/performance.json"
rm -f "$PERF_JSON"
( export PYTHONPATH="$SK/$PERF_SKILL/scripts:${PYTHONPATH:-}" \
  && run_npu_phase benchmark "$PY_BIN" "$PERF" --quick --output-dir "$TASK_DIR" \
       --warmup "$MSPROF_WARMUP" --repeats 1 ) >"$OUT_DIR/perf.log" 2>&1
[[ -f "$PERF_JSON" ]] && cp -f "$PERF_JSON" "$OUT_DIR/performance.json"
SP=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));print(d.get('geomean_speedup') or d.get('mean_speedup') or '')" 2>/dev/null || echo "")
FW=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_ref_us') or d.get('mean_ref_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
IMPL=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_asc_us') or d.get('mean_asc_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
if [[ -z "$SP" ]]; then
  write_metrics true true false "" "" "" "性能测试失败(无 geomean_speedup;完整日志如下)" "$OUT_DIR/perf.log"
  echo "[ascendc-eval] benchmark FAILED"; fail_hint; exit 1
fi

extract_case_stats
write_metrics true true true "$FW" "$IMPL" "$SP" ""
emit_optimization_prompt "$SP"
fail_hint
