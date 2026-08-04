#!/usr/bin/env bash
set -uo pipefail

WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-50}"
export SOC_VERSION="${SOC_VERSION:-ascend910b1}"
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
  FORCE_TYPE="$force_type" \
  ERR_FILE="$OUT_DIR/metrics_error.log" OUT_DIR="$OUT_DIR" OP="$OP_NAME" python3 - <<'PY'
import json, os, hashlib
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
def b(x): return str(x).lower() in ("1","true","yes")
fw, impl, sp = os.environ.get("FW",""), os.environ.get("IMPL",""), os.environ.get("SP","")
ef = Path(os.environ.get("ERR_FILE","")); full = ef.read_text(errors="replace") if ef.exists() else ""
def classify(t):
    if not t: return None
    l = t.lower(); c = l.replace(" ","")
    if "submission" in l and ("missing" in l or "untar" in l or "缺" in t or "布局" in t): return "submission_missing"
    if "判分基准" in t or "get_input" in l or ("filenotfounderror" in l and ".json" in l):
        return "input_load_failed"
    if "aclinit" in c and ("invaliddeviceid" in c or "getdevicecntfailed" in c) or "invalid device id" in l: return "npu_runtime_unavailable"
    if "ast退化" in l or "ast_check" in l or "ast check" in l or "退化" in t or "degrad" in l:
        return "ast_check_failed"
    if "cmake" in l or "make error" in l or "ccec" in l or "bisheng" in l or "编译" in t or "compil" in l: return "ascendc_compile_failed"
    if "对拍" in t or "mare" in l or "mere" in l or "correctness" in l or "result: fail" in l: return "correctness_failed"
    if "speedup" in l or "performance" in l or "性能" in t: return "benchmark_failed"
    return "unknown"
perf = None
if fw and impl and sp:
    perf = {"framework_latency_ms": float(fw), "impl_latency_ms": float(impl), "speedup_vs_torch": float(sp)}
elif sp:
    perf = {"framework_latency_ms": None, "impl_latency_ms": None, "speedup_vs_torch": float(sp)}
metrics = {
    "schema_version": 2, "op_name": os.environ.get("OP",""),
    "success": b(os.environ["SUCCESS"]), "ast_check_ok": b(os.environ["AST_OK"]),
    "correctness_ok": b(os.environ["CORR_OK"]), "perf_data": perf,
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
}

fail_hint() {
  python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print('[ascendc-eval] verdict — success=%s ast_check_ok=%s correctness_ok=%s error_type=%s speedup_vs_torch=%s'%(d.get('success'),d.get('ast_check_ok'),d.get('correctness_ok'),d.get('error_type'),p.get('speedup_vs_torch')))" 2>/dev/null || true
  python3 - "$OUT_DIR/metrics.json" "$OUT_DIR/metrics_error.log" <<'CLASSIFY' 2>/dev/null || true
import json, re, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if d.get("success"):
    print("[ascendc-eval] 错误分类: 通过"); sys.exit(0)
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

INFRA = {"npu_runtime_unavailable", "input_load_failed", "judge_container_failed",
         "judge_metrics_unreadable", "judge_no_metrics", "task_missing",
         "submission_fetch_failed"}
if et in INFRA:
    label = "INFRA-环境故障(不是你的代码问题,不要迭代修复)"
elif et in ("op_not_registered", "ascendc_load_failed"):
    label = "A类-算子未注册/加载失败(不是精度问题:改 setup.py 打包与 import,别调数值)"
elif et == "correctness_failed" and load_failure:
    label = "A类-算子未注册/加载失败(对拍未比较任何元素,不是精度问题:改 setup.py 打包与 import)"
elif et == "correctness_failed":
    label = "D类-精度不匹配"
elif et in ("ascendc_compile_failed", "ast_check_failed", "submission_missing",
            "benchmark_failed"):
    label = "A类-代码/编译错误"
else:
    label = "A类-代码/编译错误"
print(f"[ascendc-eval] 错误分类: {label}")
if first_exc:
    print(f"[ascendc-eval] 首个异常: {first_exc}")
CLASSIFY
  echo "  ↳ 完整错误在 $OUT_DIR/metrics_error.log;只改 {op}/ 下实现、重打 tarball、重跑本固定入口。"
}

PIPELINE_GEN_MAX="${POLAR_GEN_PIPELINE_MAX:-6}"
PIPELINE_OPT_MAX="${POLAR_OPT_PIPELINE_MAX:-3}"
PIPELINE_PHASE="generation"; PIPELINE_LIMIT="$PIPELINE_GEN_MAX"; PIPELINE_ATTEMPT=1
BEST_META="$WORK_ROOT/output/submission/.${OP_NAME}_impl.best.meta.json"
CUR_HASH=""

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
    python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print('[ascendc-eval] cached verdict — success=%s ast_check_ok=%s correctness_ok=%s speedup_vs_torch=%s'%(d.get('success'),d.get('ast_check_ok'),d.get('correctness_ok'),p.get('speedup_vs_torch')))" 2>/dev/null || true
    exit 0
  fi
  if [[ -f "$BEST_META" ]] && python3 -c "
import json,sys;d=json.load(open('$BEST_META'));sys.exit(0 if int(d.get('tier') or 0)>=3 else 1)" 2>/dev/null; then
    PIPELINE_PHASE="optimization"; PIPELINE_LIMIT="$PIPELINE_OPT_MAX"
  fi
  PIPELINE_ATTEMPT=$(PIPELINE_STATE_FILE="$STATE_DIR/.${OP_NAME}_budget.json" PIPELINE_PHASE="$PIPELINE_PHASE" python3 -c '
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
print(data[key])' 2>/dev/null || echo 1)
  echo "[pipeline-budget] phase=$PIPELINE_PHASE attempt=$PIPELINE_ATTEMPT/$PIPELINE_LIMIT"
  pack_best
fi

_on_exit() {
  local rc=$?
  [[ "$AGENT_SIDE" == "1" ]] || return 0
  local corr sp
  corr=$(python3 -c "import json;print('1' if json.load(open('$OUT_DIR/metrics.json')).get('correctness_ok') else '')" 2>/dev/null || echo "")
  sp=$(python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print(p.get('speedup_vs_torch') or '')" 2>/dev/null || echo "")
  pack_best "$corr" "$sp"
  [[ -n "$CUR_HASH" ]] && printf "%s" "$CUR_HASH" > "$STATE_DIR/.${OP_NAME}_last.hash"
  if [[ "$PIPELINE_ATTEMPT" =~ ^[0-9]+$ && "$PIPELINE_ATTEMPT" -ge "$PIPELINE_LIMIT" ]]; then
    if [[ "$PIPELINE_PHASE" != "generation" || "$rc" != "0" ]]; then
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
  write_metrics true false false "" "" "" "AscendC 编译失败(完整构建日志如下)" "$OUT_DIR/compile.log"
  # 改动3: 编译失败时把首个错误 ±上下文直接打到 stderr(进工具结果),否则 agent 只能拿到
  # "完整错误在 metrics_error.log",38% 的 session 会跑去自编译 cmake 反推错误、白烧轮数。
  echo "--- compile.log 首个错误上下文(完整日志见 $OUT_DIR/compile.log) ---" >&2
  _ERR_LINE=$(grep -n -m1 -E "error:|CMake Error|undefined reference" "$OUT_DIR/compile.log" | cut -d: -f1)
  if [[ -n "$_ERR_LINE" ]]; then
    sed -n "$(( _ERR_LINE > 5 ? _ERR_LINE - 5 : 1 )),$(( _ERR_LINE + 25 ))p" "$OUT_DIR/compile.log" >&2
  else
    tail -30 "$OUT_DIR/compile.log" >&2
  fi
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
      && run_npu_phase verify "$PY_BIN" "$VER" "$OP_DIR_NAME" 2>&1); then
  printf "%s\n" "$VER_OUT" > "$OUT_DIR/verify.log"
  write_metrics true false false "" "" "" "数值对拍失败(Result: fail;完整对拍输出如下)" "$OUT_DIR/verify.log"
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
    write_metrics true false false "" "" "" "对拍结果不可信(缓存/常量输出): $DET_OUT"
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
  write_metrics true true false "" "" "" "judge 环境异常:找不到 msprof(ASCEND_HOME_PATH=${ASCEND_HOME_PATH})"
  echo "[ascendc-eval] msprof NOT FOUND"; fail_hint; exit 1
fi
export PATH="$(dirname "$MSPROF_BIN"):$PATH"
MSPROF_WARMUP="${MSPROF_WARMUP:-3}"
PERF_JSON="$TASK_DIR/performance.json"
rm -f "$PERF_JSON"
( export PYTHONPATH="$SK/$PERF_SKILL/scripts:${PYTHONPATH:-}" \
  && run_npu_phase benchmark "$PY_BIN" "$PERF" --quick --output-dir "$TASK_DIR" \
       --warmup "$MSPROF_WARMUP" --repeats 1 ) >"$OUT_DIR/perf.log" 2>&1
SP=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));print(d.get('geomean_speedup') or d.get('mean_speedup') or '')" 2>/dev/null || echo "")
FW=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_ref_us') or d.get('mean_ref_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
IMPL=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_asc_us') or d.get('mean_asc_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
if [[ -z "$SP" ]]; then
  write_metrics true true false "" "" "" "性能测试失败(无 geomean_speedup;完整日志如下)" "$OUT_DIR/perf.log"
  echo "[ascendc-eval] benchmark FAILED"; fail_hint; exit 1
fi

write_metrics true true true "$FW" "$IMPL" "$SP" ""
echo "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=$SP"
fail_hint
