#!/usr/bin/env bash
# =============================================================================
# ascendc_eval_pipeline.sh — AscendC 固定评测入口(polar judge_command)
#
# 逐字派生 triton_eval_pipeline.sh 的护栏(schema_version 2 metrics + error 分类 +
# fail_hint verdict),3 个核心步骤换成 AscendC:
#   Step0 解 tarball → {op}/(model.py + model_new_ascendc.py + kernel/)
#   Step1 反退化(model_new_ascendc.py 必须真调 torch.ops.npu.<op>)
#   Step2 编 kernel/(自包含 CMakeLists) + verification_ascendc.py 对拍 model.py
#   Step3 ops-profiling msprof --quick 出 geomean_speedup(t2a:上游 Phase 5 已从 performance.py 换成它)
# 复用上游的 verification_ascendc.py / msprof_perf_summary.py(不重造)。
#
# 用法: bash ascendc_eval_pipeline.sh --op_name <op> --impl <{op}.tar.gz> --task <model.py> --out_dir judge_out
# 环境: ASCENDC_SKILLS_SRC(eval 脚本来源,B2 后=canonical) SOC_VERSION WARMUP REPEATS
# =============================================================================
set -uo pipefail

WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-50}"
export SOC_VERSION="${SOC_VERSION:-ascend910b1}"
# eval skills 默认从本 canonical 目录自带的 skills/ 取(tools/../skills);
# 也可用 ASCENDC_SKILLS_SRC 覆盖(须是含 tilelang2ascend-translator/ops-profiling 的扁平目录)。
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# judge 里 canonical 恒挂在 /opt/canonical;优先它,回落脚本同级 ../skills(B1/B2 本地测)。
if [[ -z "${ASCENDC_SKILLS_SRC:-}" ]]; then
  if [[ -d /opt/canonical/skills ]]; then ASCENDC_SKILLS_SRC=/opt/canonical/skills
  else ASCENDC_SKILLS_SRC="${_SCRIPT_DIR}/../skills"; fi
fi
TRANS_SKILL="tilelang2ascend-translator"
PERF_SKILL="ops-profiling"

# 解释器与工具链:沿用 env.sh(AST_CHECK_PYTHON 免 NPU;OPERATOR_PYTHON 上 NPU),
# 缺省回落 python3,便于无 env.sh 的环境 —— 逐字对齐 triton_eval_pipeline.sh:28-33。
[[ -f "${_SCRIPT_DIR}/env.sh" ]] && source "${_SCRIPT_DIR}/env.sh"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"
PY_BIN="${PY_BIN:-$OPERATOR_PYTHON}"

# ---- NPU lease(逐字派生 triton_eval_pipeline.sh:37-54,npu_lease_exec.py 是同一份拷贝)----
# 设计:agent 与 judge 都只在预留卡池(POLAR_NPU_LEASE_POOL)里跑;judge 用时去抢,
# 池里哪张空就占哪张(flock ${lock_dir}/npu{N}.lock,非阻塞轮询=排队等待),用完释放,
# 不存在"某张卡属于某个 session"。锁文件命名与 polar DockerRuntime.acquire_card() 一致,
# 锁目录 bind mount 进容器,故 agent 容器 / judge 子进程 / polar 三方共用同一套锁。
# 未配置 POLAR_NPU_LEASE_POOL 时原样直跑(与 triton 同样保持向后兼容)。
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
    --incremental) INCREMENTAL=1; shift;;   # agent 侧迭代用:复用 $WORK 与 build/,走增量编译
    *) echo "[ascendc-eval] unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -z "$OP_NAME" ]] && { echo "[ascendc-eval] --op_name required" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-judge_out}"; mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"
IMPL_FILE="${IMPL_FILE:-output/submission/${OP_NAME}_impl.tar.gz}"

# =============================================================================
# agent 侧 / judge 侧的自动判别 —— 两边跑**同一条命令**,行为按上下文自适应。
# 判据:workdir 顶层有没有 {op}/ 源目录。agent 容器里有(它就在那儿写代码);
# judge 容器里没有(judge 只收到 output/submission/{op}_impl.tar.gz + input/ 基准)。
# agent 侧额外做:自动打包 → 预算计数 → 内容哈希短路 → 结束时更新 .best;
# judge 侧一律不做这些(一个 session 只判一次,在那儿计数/留 best 没有意义)。
# =============================================================================
SRC_DIR="$PWD/$OP_NAME"
AGENT_SIDE=0; [[ -d "$SRC_DIR" ]] && AGENT_SIDE=1
STATE_DIR="$PWD/output/.selfcheck"
PACK_SH="${_SCRIPT_DIR}/pack_submission.sh"

# ---- write_metrics(schema_version 2,字段与 triton_eval_pipeline.sh 逐字一致)----
write_metrics() {  # ast_ok corr_ok success fw impl speedup error [完整日志文件]
  # 【第 8 个参数:完整日志路径】polar 会把 judge_out/metrics_error.log **整份下载**回去,
  # 交给 operator_reward.classify_infra_error_text 做 infra 二次分类(operator_judge.py:257-263→:525)。
  # 那些签名(aclInit / InvalidDeviceId / getDeviceCntFailed)出现在栈的**开头**,而早先这里用
  # `tail -60` 传参 —— 头部被砍掉,polar 判不出 infra,本该 retry 不计分的环境故障被当成算子
  # 失败给 0.2/0.3,直接毒化 GRPO。triton 那边三个失败点传的都是完整文本,所以它硬编码
  # error_truncated=False 是诚实的;我们截断了还照抄 False,那就是撒谎。故:日志整份进文件。
  local ast_ok="$1" corr_ok="$2" success="$3" fw="$4" impl="$5" sp="$6" error="$7" log_src="${8:-}"
  : > "$OUT_DIR/metrics_error.log"
  [[ -n "$error" ]] && printf "%s\n" "$error" >> "$OUT_DIR/metrics_error.log"
  [[ -n "$log_src" && -f "$log_src" ]] && cat "$log_src" >> "$OUT_DIR/metrics_error.log"
  AST_OK="$ast_ok" CORR_OK="$corr_ok" SUCCESS="$success" FW="$fw" IMPL="$impl" SP="$sp" \
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
    if "submission" in l and "missing" in l: return "submission_missing"
    # judge 侧基准/用例加载不了 = 环境问题,不是算子写错(INFRA → retry 不计分,别毒化 GRPO)
    # 与 triton_eval_pipeline.sh 的 input_load_failed 规则对齐
    if "判分基准" in t or "get_input" in l or ("filenotfounderror" in l and ".json" in l):
        return "input_load_failed"
    if "aclinit" in c and ("invaliddeviceid" in c or "getdevicecntfailed" in c) or "invalid device id" in l: return "npu_runtime_unavailable"
    # 退化必须排在编译之前(逐字对齐 triton_eval_pipeline.sh:191 早于 :205):AST 检查器的
    # 输出里常带"编译/compile"等字眼(修复建议),排在后面会被 compile 规则抢先命中而误分类。
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
    "error": None, "error_type": classify(full),
    "error_file": "metrics_error.log" if full else None,
    "error_bytes": len(full.encode("utf-8")) if full else 0,      # 原始完整错误的大小
    "error_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest() if full else None,  # 原始完整错误的哈希
    "error_truncated": False,   # 下面按实际情况改写
}
# 超上限才截断,且**保头 + 保尾**:infra 签名(aclInit/InvalidDeviceId)在头,真实报错在尾;
# 并如实置 error_truncated=true。error_bytes/error_sha256 恒描述**原始完整**错误。
_CAP = int(os.environ.get("ASCENDC_ERRLOG_CAP_BYTES", "2097152"))   # 2 MiB
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
  echo "  ↳ 完整错误在 $OUT_DIR/metrics_error.log;只改 {op}/ 下实现、重打 tarball、重跑本固定入口。"
}

# =============================================================================
# agent 侧专属:预算 / 哈希短路 / 自动打包 / 结束时更新 best
# 预算状态机逐字派生 triton_eval_pipeline.sh:56-170(triton 的 agent 与 judge 也是同一个
# 脚本,那些护栏本就长在里面)。judge 侧全部跳过。
# =============================================================================
PIPELINE_GEN_MAX="${POLAR_GEN_PIPELINE_MAX:-6}"
PIPELINE_OPT_MAX="${POLAR_OPT_PIPELINE_MAX:-3}"
PIPELINE_PHASE="generation"; PIPELINE_LIMIT="$PIPELINE_GEN_MAX"; PIPELINE_ATTEMPT=1
BEST_META="$PWD/output/submission/.${OP_NAME}_impl.best.meta.json"
CUR_HASH=""

pack_best() {  # $1=verified?  $2=speedup?  —— 交给 pack_submission 做档位比较,只升不降
  [[ -x "$PACK_SH" || -f "$PACK_SH" ]] || return 0
  local args=("$OP_NAME"); [[ -n "${1:-}" ]] && args+=(--verified)
  [[ -n "${2:-}" ]] && args+=(--speedup "$2")
  WORKDIR="$PWD" bash "$PACK_SH" "${args[@]}" || true
}

if [[ "$AGENT_SIDE" == "1" ]]; then
  mkdir -p "$STATE_DIR"
  # --- 内容哈希短路:{op}/ 源码没变就复用上次结论,不烧预算、不占卡 ---
  # 只对源码取哈希(build/dist/egg-info/__pycache__ 整棵剪掉,并排除本入口自己写进 {op}/ 的产物)
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
  # --- 预算计数(与 triton 同义:best 已达"成功"则切到 optimization 阶段)---
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
  # --- 自动打包:agent 不需要记得打包,评测前先从 {op}/ 生成提交物 ---
  pack_best
fi

# --- 结束钩子:一个 trap 覆盖全部退出点(agent 侧才生效)---
_on_exit() {
  local rc=$?
  [[ "$AGENT_SIDE" == "1" ]] || return 0
  local corr sp
  corr=$(python3 -c "import json;print('1' if json.load(open('$OUT_DIR/metrics.json')).get('correctness_ok') else '')" 2>/dev/null || echo "")
  sp=$(python3 -c "import json;d=json.load(open('$OUT_DIR/metrics.json'));p=d.get('perf_data') or {};print(p.get('speedup_vs_torch') or '')" 2>/dev/null || echo "")
  pack_best "$corr" "$sp"                      # 按本次真实档位更新 .best(只升不降)
  [[ -n "$CUR_HASH" ]] && printf "%s" "$CUR_HASH" > "$STATE_DIR/.${OP_NAME}_last.hash"
  # LIMIT_EXHAUSTED 收尾话术(逐字派生 triton_eval_pipeline.sh:154-168)
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

# ---- Step0: 解 tarball ----
WORK="$OUT_DIR/work"
[[ "$INCREMENTAL" == "1" ]] || rm -rf "$WORK"    # --incremental:复用上次解包目录,make 走增量
mkdir -p "$WORK"
if [[ ! -f "$IMPL_FILE" ]]; then write_metrics false false false "" "" "" "submission missing: $IMPL_FILE"; echo "[ascendc-eval] submission missing"; fail_hint; exit 1; fi
tar xzf "$IMPL_FILE" -C "$WORK" 2>/dev/null || { write_metrics false false false "" "" "" "cannot untar submission: $IMPL_FILE"; echo "[ascendc-eval] untar failed"; fail_hint; exit 1; }
MNA="$(find "$WORK" -maxdepth 3 -name model_new_ascendc.py | head -1)"
TASK_DIR="$(dirname "$MNA" 2>/dev/null)"
if [[ -z "$TASK_DIR" || ! -d "$TASK_DIR/kernel" ]]; then
  write_metrics false false false "" "" "" "submission tarball 缺 {op}/kernel 或 model_new_ascendc.py"
  echo "[ascendc-eval] bad submission layout"; fail_hint; exit 1
fi
OP_DIR_NAME="$(basename "$TASK_DIR")"

# ---- 隔离:tarball 里除了 {op}/ 这棵树,其它一律删掉 ----
# 【为什么】$WORK 是**完全由 agent 控制**的解包目录,而下面 SK="$WORK/.claude/skills"
# 正是 judge 自己跑 verification/performance 的地方。`cp -r SRC DEST` 在 DEST 已存在时是
# **嵌套**而不是覆盖(cp -r a b → b/a),所以只要 tarball 里带一份
# .claude/skills/ascendc-translator/scripts/verification_ascendc.py(内容 `print("Result: pass")`),
# judge 执行的就是它自己 —— 判分链被提交物整个接管。非恶意也可达:agent 用 `tar czf x.tar.gz .`
# 会顺手把项目级 .claude 打进去。故解包后先把 {op}/ 之外的一切铲掉。
_TOP_REL="$(realpath --relative-to="$WORK" "$TASK_DIR" 2>/dev/null | cut -d/ -f1)"
if [[ -n "$_TOP_REL" && "$_TOP_REL" != "." && "$_TOP_REL" != ".." ]]; then
  find "$WORK" -mindepth 1 -maxdepth 1 ! -name "$_TOP_REL" -exec rm -rf {} + 2>/dev/null || true
fi

# ---- purge:agent 夹带的预编译产物一律删掉,judge 只跑自己从源码编出来的东西 ----
# 提交包里可能带着 agent 容器里编好的 .so/.a/.whl(实测 18_Index 那个包就有)。这些二进制
# 与它交的源码**没有任何保证的对应关系** —— 极端情况源码是漂亮的 AscendC、.so 里却是 torch
# 算的,而退化检测只看 model_new_ascendc.py 有没有 torch.ops.npu,查不出来。RL 不需要恶意:
# 只要某次夹带碰巧拿了高分,梯度就会强化这个行为。故解包后强制只留源码。
find "$TASK_DIR" \( -name '*.so' -o -name '*.a' -o -name '*.o' -o -name '*.whl' \
     -o -name 'build' -o -name 'dist' -o -name '*.egg-info' -o -name '__pycache__' \) \
     -exec rm -rf {} + 2>/dev/null || true

# ---- 判分基准一律以数据集为准(agent 交什么都不算)----
# judge 容器的 eval_prepare 已把数据集原版放到 input/{op}.py + input/{op}.json。
# golden(model.py)和用例(.json)**无条件从这里注入并覆盖** tarball 里的同名文件:
#   ① agent 改 golden 让参考实现迁就自己的错 kernel → 无效;
#   ② agent 只交精简后的用例(CLAUDE.md Phase 2 是原版流程,给它自己迭代提速用的,
#      Phase 6 才恢复全量;被 abort 时交上来的就是砍过的)→ 判分仍跑全量;
#   ③ agent 根本不需要把 model.py/{op}.json 打进 tarball(契约已相应放宽)。
# 31/31 个数据集算子的 get_input_groups() 都是硬编码同名 json(无 __file__ 推导),
# 故改名成 model.py 后仍能正确读到同目录的 {op}.json。
# --task 仅保留给本地/离线测试(judge_command 不传)。
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

# 注入要做**两次**:第一次在这里,第二次在编译/安装之后、对拍之前。
# 【为什么两次】Step2 会执行 agent 提供的代码(setup.py、pip install 触发的 import、
# 编译期脚本),它完全可以在那时把 model.py 或 {op}.json 改回自己那份 —— 那样第一次注入就白做了。
# 第二次注入后立即 chmod 444,再由对拍读取。
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

# ---- 准备 eval 脚本(.claude/skills 布局,verification/performance 就位)----
SK="$WORK/.claude/skills"
rm -rf "$SK"; mkdir -p "$SK"     # 先铲后铺:cp -r 到已存在目录会嵌套而非覆盖(见上面的隔离说明)
if ! cp -r "$ASCENDC_SKILLS_SRC/$TRANS_SKILL" "$SK/$TRANS_SKILL" \
   || ! cp -r "$ASCENDC_SKILLS_SRC/$PERF_SKILL" "$SK/$PERF_SKILL"; then
  # 静默吞掉 cp 失败 = 后面执行的可能是别人的脚本;这属于 judge 环境问题,判 infra 不计分
  write_metrics false false false "" "" "" \
    "judge 环境异常(get_input 之前):无法从 $ASCENDC_SKILLS_SRC 铺设评测脚本"
  echo "[ascendc-eval] skills setup FAILED"; fail_hint; exit 1
fi

# ---- Step1: 反退化(AST 检测,逐字对应 triton_eval_pipeline.sh:309-315)----
# 【为什么不能只 grep】原来这里是 `grep -q "torch.ops.npu"`,只要文件任意位置(注释、docstring、
# 未被调用的 helper、if False 分支)出现这个字符串就算过 —— 于是 forward 里写纯 torch、
# 注释里提一句 torch.ops.npu,就能编过→对拍必过(和 golden 逐位相同)→ speedup≈1.0 → reward 0.75。
# 这是全流程投入产出比最高的策略,GRPO 会几十步内收敛到它,整批训练信号作废。
# validate_ascendc_impl.py 是 AST 检测(Type1 无扩展导入 / Type2 forward 未调 kernel /
# Type3 forward 里仍有 torch 计算 / Type4 逐元素 for 循环),skill 里本来就带,judge 已经把它
# cp 进 $SK 却没用 —— 而 agent 侧 selfcheck 用的正是它。judge 比 agent 自检还松是不能接受的。
echo "[ascendc-eval] Step1 anti-degradation (AST)"
VALIDATOR="$SK/$TRANS_SKILL/scripts/validate_ascendc_impl.py"
if [[ -f "$VALIDATOR" ]]; then
  if ! AST_OUT=$("$AST_CHECK_PYTHON" "$VALIDATOR" "$TASK_DIR/model_new_ascendc.py" 2>&1); then
    write_metrics false false false "" "" "" "AST退化检查失败: $AST_OUT"
    echo "[ascendc-eval] AST FAILED — 退化检测未通过(error_type=ast_check_failed)"; fail_hint; exit 1
  fi
elif ! grep -q "torch.ops.npu" "$TASK_DIR/model_new_ascendc.py" 2>/dev/null; then
  # 检查器缺失时的保守回退(判据同 grep);检查器本该存在,缺了说明 judge 环境不完整
  write_metrics false false false "" "" "" "退化: model_new_ascendc.py 未调用 torch.ops.npu.<op>(疑似纯 torch)"
  echo "[ascendc-eval] degradation FAILED"; fail_hint; exit 1
else
  echo "[ascendc-eval] WARN: validate_ascendc_impl.py 缺失,已回退到 grep 判据(护栏变弱)"
fi

# ---- Step2: 编译 + 打 whl + 安装(**不占卡**)----
# 逐字对应 cannbot skills/ascendc-translator/scripts/evaluate_ascendc.sh:110-126
#   rm -rf build → cmake(-DSOC_VERSION/-DASCEND_CANN_PACKAGE_PATH/-DCMAKE_BUILD_TYPE)
#   → make -j → setup.py bdist_wheel → pip install dist/*.whl --force-reinstall
# **改这段必须同步那份**。bdist_wheel+install 不是可选优化:AscendC 算子靠 .so 被 import 时
# 执行 TORCH_LIBRARY_IMPL 才注册出 torch.ops.npu.<op>,只 make 不装 = 对拍时算子不存在
# (曾因漏这两步把写对的 18_Index 误判成 correctness_failed)。
# 编译不需要 NPU,故**不套 run_npu_phase**:AscendC 编译要数分钟,占着卡编译会堵死整个卡池。
# 与官方的 2 处有意 delta:①连 dist/ 一起清(保证装的是本次编出来的 whl,而不是 agent 打进
# tarball 的旧 whl);②用 $PY_BIN -m pip(镜像里 python/pip 不一定在 PATH 上)。
echo "[ascendc-eval] Step2 compile + install (no NPU)"
KERNEL_DIR="$TASK_DIR/kernel"
if ! (
  set -e
  cd "$KERNEL_DIR"
  # judge 恒 clean build(必须从源码重来);--incremental 只给 agent 迭代用,保留 build/ 走增量。
  # dist/ 无论如何都清:保证 pip 装的是本次编出来的 whl,而不是 agent 打进 tarball 的旧 whl。
  [[ "$INCREMENTAL" == "1" ]] || rm -rf build
  rm -rf dist
  mkdir -p build && cd build
  cmake "$KERNEL_DIR" \
    -DSOC_VERSION="$SOC_VERSION" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  make -j"$(nproc)"
  cd "$KERNEL_DIR"
  "$PY_BIN" setup.py bdist_wheel
  "$PY_BIN" -m pip install dist/*.whl --force-reinstall
) >"$OUT_DIR/compile.log" 2>&1; then
  write_metrics true false false "" "" "" "AscendC 编译/安装失败(完整 cmake/make/pip 日志如下)" "$OUT_DIR/compile.log"
  echo "[ascendc-eval] compile FAILED"; fail_hint; exit 1
fi

# ---- 第二次注入:Step2 刚刚执行过 agent 的 setup.py / import,基准可能已被改回 ----
if ! inject_baseline; then
  write_metrics false false false "" "" "" "判分基准注入失败(get_input):对拍前无法复位 $TASK_DIR"
  echo "[ascendc-eval] baseline re-inject failed"; fail_hint; exit 1
fi

# ---- Step2b: 数值对拍(**占卡**,池内抢一张)----
echo "[ascendc-eval] Step2b verify (NPU lease)"
VER="$SK/$TRANS_SKILL/scripts/verification_ascendc.py"
if ! VER_OUT=$(cd "$WORK" && export WORKDIR="$WORK" PYTHONPATH="$SK/$TRANS_SKILL/scripts:${PYTHONPATH:-}" \
      && run_npu_phase verify "$PY_BIN" "$VER" "$OP_DIR_NAME" 2>&1); then
  printf "%s\n" "$VER_OUT" > "$OUT_DIR/verify.log"
  write_metrics true false false "" "" "" "数值对拍失败(Result: fail;完整对拍输出如下)" "$OUT_DIR/verify.log"
  echo "[ascendc-eval] verify FAILED"; fail_hint; exit 1
fi
printf "%s\n" "$VER_OUT" > "$OUT_DIR/verify.log"

# ---- Step2c: 缓存/常量输出探测(**占卡**,很快)----
# C 方案:测速对同一输入连调 56 次,隐含假设"每次都真算"。缓存实现能拿虚高 speedup(→满分),
# 而对拍(只调一次)和退化检测都拦不住。详见 tools/detect_stateful_impl.py 顶部说明。
DETECT="${_SCRIPT_DIR}/detect_stateful_impl.py"
if [[ -f "$DETECT" ]]; then
  echo "[ascendc-eval] Step2c stateful/cache detection (NPU lease)"
  DET_OUT=$(cd "$WORK" && run_npu_phase detect "$PY_BIN" "$DETECT" "$TASK_DIR" 2>&1); DET_RC=$?
  printf "%s\n" "$DET_OUT" > "$OUT_DIR/detect.log"
  if [[ "$DET_RC" == "1" ]]; then
    # 判定"不真算":对拍虽过但性能不可信 → 按对拍失败处理,不让它拿 0.4/0.75
    write_metrics true false false "" "" "" "对拍结果不可信(缓存/常量输出): $DET_OUT"
    echo "[ascendc-eval] stateful/cache DETECTED"; fail_hint; exit 1
  fi
  [[ "$DET_RC" == "2" ]] && echo "  ↳ ${DET_OUT}"
fi

# ---- Step3: 性能(**占卡**,池内再抢一张;与对拍分开租,期间不长占)----
# t2a:上游 Phase 5 从 performance.py(wall-clock,56 次取平均)换成了
# ops-profiling 的 msprof --quick(采 device 侧 kernel 时间)。跟着换,保持与 agent 自检同口径。
#
# 【为什么钉死 --quick --repeats 1,且永不用 --compare】——— 这是防测速作弊的关键,别改:
#   msprof_perf_summary.py:1346-1348  quick 模式用 `repeats - 1` 作为 wrapper 的**内部** warmup;
#   :926-955  外部 warmup 跑在**独立子进程**里,msprof 只 profile 最后新起的那一个进程。
#   ⇒ repeats=1 时内部 warmup=0,被 profile 的进程里 `model = cls(...)` 全新构造、只调用一次,
#     那一次必然是**冷调用** —— Python 对象级的 `self._cache` 跨不过进程边界,缓存作弊无效。
#   ⇒ 反之 repeats>1(内部 warmup=repeats-1)或 --compare(内部 warmup=args.warmup,默认 3)
#     都会在同一进程内先把缓存喂饱、再计时那一次,speedup 可以虚高到任意大。
#   (pin 版 performance.py 是 56 次同输入取平均,所以那边需要 ASCENDC_PERF_INPUT_VARIANTS
#    轮换输入来堵;换到 quick 之后该补丁不再需要 —— 但 detect_stateful_impl.py 仍要留,
#    它防的是"输出不跟输入变"这类正确性问题,与计时无关。)
echo "[ascendc-eval] Step3 performance (msprof --quick, NPU lease)"
PERF="$SK/$PERF_SKILL/scripts/msprof_perf_summary.py"
# msprof 必须显式解析:env.sh 恰好把 BISHENGIR_BIN(=$ASCEND_HOME_PATH/bin)加进 PATH,
# 而 msprof 正好同目录 —— 那是巧合(该变量本意给 bisheng 编译器)。不靠巧合,按优先级定位。
# msprof_perf_summary.py:948 是裸调 "msprof",所以必须保证它在 PATH 上。
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
# 卡由 run_npu_phase 经 npu_lease_exec.py 注入 ASCEND_RT_VISIBLE_DEVICES=<物理卡>;
# msprof_perf_summary.py:1284-1285 会从该 env 读设备(source=env),故**不要**传 --device,
# 否则会与租约打架。
PERF_JSON="$TASK_DIR/performance.json"   # :1499 固定写 <output-dir>/performance.json
rm -f "$PERF_JSON"                        # 防止读到上一轮的陈旧结果
( export PYTHONPATH="$SK/$PERF_SKILL/scripts:${PYTHONPATH:-}" \
  && run_npu_phase benchmark "$PY_BIN" "$PERF" --quick --output-dir "$TASK_DIR" \
       --warmup "$WARMUP" --repeats 1 ) >"$OUT_DIR/perf.log" 2>&1
# geomean 而非 mean:上游自己的日志把 geomean 标为"主指标"(:1226),且对单个异常 case 不敏感。
SP=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));print(d.get('geomean_speedup') or d.get('mean_speedup') or '')" 2>/dev/null || echo "")
# 单位换算:新工具出的是**微秒**(geomean_ref_us / geomean_asc_us),
# metrics.json 的 framework_latency_ms / impl_latency_ms 是**毫秒**。
FW=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_ref_us') or d.get('mean_ref_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
IMPL=$(python3 -c "import json;d=json.load(open('$PERF_JSON'));v=d.get('geomean_asc_us') or d.get('mean_asc_us');print(round(v/1000.0,6) if v else '')" 2>/dev/null || echo "")
if [[ -z "$SP" ]]; then
  write_metrics true true false "" "" "" "性能测试失败(无 geomean_speedup;完整日志如下)" "$OUT_DIR/perf.log"
  echo "[ascendc-eval] benchmark FAILED"; fail_hint; exit 1
fi

# ---- 成功 ----
write_metrics true true true "$FW" "$IMPL" "$SP" ""
echo "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=$SP"
fail_hint
