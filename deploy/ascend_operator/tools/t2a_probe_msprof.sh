#!/usr/bin/env bash
# =============================================================================
# t2a 迁移 —— msprof 可用性探测  v2
#
# 【v1 的教训】v1 在宿主机上跑了,而宿主机没装 torch —— 3/4/5 全挂是环境不对,
# 不是 msprof 的问题。**必须在 ascendc-sandbox:v1 容器里跑**,因为 agent 和 judge
# 都在那个镜像里,只有它的答案算数。
#
# 【怎么跑】—— 在宿主机执行这一条,它会自己进容器:
#
#     bash /home/docker/t2a_probe_msprof.sh
#
#   指定卡:  NPU_ID=2 bash /home/docker/t2a_probe_msprof.sh
#   已经在容器里了(不想再套一层): IN_CONTAINER=1 bash /home/docker/t2a_probe_msprof.sh
#
# 容器参数逐字取自 polar 起 agent 容器的配方
# (src/polar/runtime/ascend.py:_DRIVER_MOUNTS + ascend_mount_create_args):
#   --privileged --ipc host --network host
#   -v /dev:/dev
#   -v /usr/local/Ascend/driver:...:ro   -v /usr/local/Ascend/firmware:...:ro
#   -v /usr/local/dcmi:...:ro            -v /usr/local/bin/npu-smi:...:ro
#   -v /etc/ascend_install.info:...:ro   -v /usr/local/sbin:...:ro
#
# 【安全性】只读探测 + /tmp 临时目录;容器用 --rm 起完即删;不动任何现有文件、不装包。
#           唯一副作用是短暂占用一张 NPU 卡(几秒)。
#
# 【v1 修了什么】
#   ① CANN 版本选错 —— v1 用 `find|head -1` 挑到了 cann-8.5.1(字典序),
#      而实际该用的是 9.0.0(session 日志里全是 cann-9.0.0 的路径)。改为优先
#      ASCEND_HOME_PATH,其次按版本号取最新,并把找到的所有 msprof 都列出来。
#   ② `msprof --version` 不被支持(v1 报 unrecognized option)。改用 --help 探活。
#   ③ 加了"你是不是在宿主机上"的判定,不对就直接告诉你怎么进容器。
# =============================================================================

set +e
IMAGE="${IMAGE:-ascendc-sandbox:v1}"
NPU_ID="${NPU_ID:-0}"

# ---------------------------------------------------------------------------
# 外层:如果不在容器里,自己套一层 docker run 再进来
# ---------------------------------------------------------------------------
if [[ "${IN_CONTAINER:-0}" != "1" && ! -f /.dockerenv ]]; then
  echo "============================================================"
  echo " 检测到当前在宿主机 —— 自动进入 $IMAGE 容器再探测"
  echo "============================================================"
  if ! command -v docker >/dev/null 2>&1; then
    echo "❌ 宿主机没有 docker,无法自动进容器。"
    echo "   请手动到 ascendc-sandbox:v1 里执行: IN_CONTAINER=1 bash /home/docker/t2a_probe_msprof.sh"
    exit 1
  fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "❌ 找不到镜像 $IMAGE。现有的 ascendc/sandbox 相关镜像:"
    docker images 2>/dev/null | grep -iE "ascendc|sandbox" | head -10
    echo "   可用 IMAGE=<镜像名> 覆盖,例如: IMAGE=sandbox:v1 bash $0"
    exit 1
  fi
  exec docker run --rm --privileged --ipc host --network host \
    -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /home/docker:/home/docker:ro \
    -e IN_CONTAINER=1 -e NPU_ID="$NPU_ID" \
    "$IMAGE" bash /home/docker/t2a_probe_msprof.sh
fi

# ---------------------------------------------------------------------------
# 内层:真正的探测
# ---------------------------------------------------------------------------
WORK="$(mktemp -d /tmp/t2a_msprof_probe.XXXXXX)"
RESULTS=()
log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
rec()  { RESULTS+=("$1|$2|$3"); }

echo "============================================================"
echo " t2a msprof 可用性探测 v2"
echo " 时间: $(date '+%F %T')   主机: $(hostname)   NPU_ID=${NPU_ID}"
echo " 容器内: $([ -f /.dockerenv ] && echo yes || echo 'no(IN_CONTAINER 强制)')   镜像: $IMAGE"
echo "============================================================"

# ---------------------------------------------------------------------------
log "1/8  基础环境 + CANN 版本选择"
# ---------------------------------------------------------------------------
note "python3         : $(command -v python3 || echo '未找到')  $(python3 -V 2>&1)"
note "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-<未设置>}"

note "机器上所有 CANN:"
ls -d /usr/local/Ascend/cann-* 2>/dev/null | sed 's/^/     /' || note "     <一个都没有>"

# 优先 ASCEND_HOME_PATH;否则按版本号取最新(v1 的 bug:用字典序挑到了 8.5.1)
CANN_ROOT="${ASCEND_HOME_PATH:-}"
if [[ -z "$CANN_ROOT" || ! -d "$CANN_ROOT" ]]; then
  CANN_ROOT="$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1)"
  note "ASCEND_HOME_PATH 未设/无效 → 按版本号取最新: ${CANN_ROOT:-<找不到>}"
fi
note "选定 CANN_ROOT  : ${CANN_ROOT:-<找不到>}"
[[ -n "$CANN_ROOT" ]] && rec "CANN 安装" "PASS" "$CANN_ROOT" || rec "CANN 安装" "FAIL" "找不到"

SET_ENV="$CANN_ROOT/set_env.sh"
if [[ -f "$SET_ENV" ]]; then
  note "source $SET_ENV"
  source "$SET_ENV" >/dev/null 2>&1
  note "  → ASCEND_HOME_PATH 现在 = ${ASCEND_HOME_PATH:-<仍未设置>}"
  rec "set_env.sh" "PASS" "$SET_ENV"
else
  note "⚠️ 没有 $SET_ENV"
  rec "set_env.sh" "FAIL" "不存在"
fi

# ---------------------------------------------------------------------------
log "2/8  msprof 可执行文件"
# ---------------------------------------------------------------------------
note "机器上所有 msprof:"
find /usr/local/Ascend -maxdepth 6 -name msprof -type f 2>/dev/null | sed 's/^/     /' || true

MSPROF="$(command -v msprof 2>/dev/null)"
IN_PATH=1
if [[ -z "$MSPROF" ]]; then
  IN_PATH=0
  # 优先选定 CANN_ROOT 下的那个
  MSPROF="$(find "$CANN_ROOT" -maxdepth 5 -name msprof -type f -perm -u+x 2>/dev/null | head -1)"
  [[ -z "$MSPROF" ]] && MSPROF="$(find /usr/local/Ascend -maxdepth 6 -name msprof -type f -perm -u+x 2>/dev/null | sort -V | tail -1)"
fi

if [[ -n "$MSPROF" ]]; then
  note "选用: $MSPROF"
  note "探活(--help 前 5 行;v1 用 --version 是错的,该选项不存在):"
  "$MSPROF" --help 2>&1 | head -5 | sed 's/^/     /'
  rec "msprof 存在" "PASS" "$MSPROF"
  [[ $IN_PATH -eq 1 ]] && rec "msprof 在 PATH" "PASS" "-" || rec "msprof 在 PATH" "WARN" "source set_env.sh 后仍不在 PATH"
else
  rec "msprof 存在" "FAIL" "找不到"; rec "msprof 在 PATH" "SKIP" "-"
fi

# ---------------------------------------------------------------------------
log "3/8  torch / torch_npu"
# ---------------------------------------------------------------------------
python3 - <<'PY' 2>&1 | sed 's/^/   /'
import sys
try:
    import torch; print(f"torch     : {torch.__version__}")
except Exception as e:
    print(f"torch     : 导入失败 {e}")
    print(">>> 如果这里失败,说明不在 ascendc-sandbox:v1 里,后面的结果都不算数")
    sys.exit(0)
try:
    import torch_npu
    print(f"torch_npu : {torch_npu.__version__}")
    print(f"npu 可用  : {torch.npu.is_available()}   设备数: {torch.npu.device_count()}")
except Exception as e:
    print(f"torch_npu : 导入失败 {e}")
PY
python3 -c "import torch" 2>/dev/null || {
  echo
  echo "   ⚠️⚠️ torch 都没有 —— 这个环境不是 agent/judge 实际运行的环境。"
  echo "        请确认镜像名对不对(当前 IMAGE=$IMAGE),或用 IMAGE=<正确镜像> 重跑。"
}
python3 -c "import torch,torch_npu;exit(0 if torch.npu.is_available() else 1)" 2>/dev/null \
  && rec "torch_npu 可用" "PASS" "-" || rec "torch_npu 可用" "FAIL" "见上"

# ---------------------------------------------------------------------------
log "4/8  裸跑最小 NPU 负载"
# ---------------------------------------------------------------------------
cat > "$WORK/tiny.py" <<'PY'
import torch, torch_npu
dev = torch.device("npu")
x = torch.randn(1024, 1024).to(dev); y = torch.randn(1024, 1024).to(dev)
for _ in range(3):
    z = torch.add(x, y)
torch.npu.synchronize()
print("tiny workload OK, sum =", float(z.sum().cpu()))
PY
ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" python3 "$WORK/tiny.py" 2>&1 | tail -4 | sed 's/^/   /'
ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" python3 "$WORK/tiny.py" >/dev/null 2>&1 \
  && rec "裸 NPU 负载" "PASS" "device $NPU_ID" || rec "裸 NPU 负载" "FAIL" "见上"

# ---------------------------------------------------------------------------
log "5/8  msprof 采集(核心)"
# ---------------------------------------------------------------------------
if [[ -n "$MSPROF" ]] && python3 -c "import torch,torch_npu" 2>/dev/null; then
  OUT="$WORK/prof_out"; mkdir -p "$OUT"
  note "msprof --output=$OUT --application=\"python3 tiny.py\"  (device $NPU_ID)"
  ( cd "$WORK" && ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" \
      "$MSPROF" --output="$OUT" --application="python3 $WORK/tiny.py" ) > "$WORK/msprof.log" 2>&1
  RC=$?
  note "退出码: $RC"; note "日志尾部:"; tail -12 "$WORK/msprof.log" | sed 's/^/     /'
  [[ $RC -eq 0 ]] && rec "msprof 执行" "PASS" "rc=0" || rec "msprof 执行" "FAIL" "rc=$RC"
  note ""; note "关键 csv:"
  for f in op_summary task_time api_statistic; do
    hit="$(find "$OUT" -name "${f}*.csv" 2>/dev/null | head -1)"
    if [[ -n "$hit" ]]; then note "  ✓ ${f}: $(basename "$hit") ($(wc -l < "$hit") 行)"; rec "csv:${f}" "PASS" "-"
    else note "  ✗ ${f}: 没有"; rec "csv:${f}" "FAIL" "未生成"; fi
  done
else
  rec "msprof 执行" "SKIP" "前置条件不满足"
  for f in op_summary task_time api_statistic; do rec "csv:${f}" "SKIP" "-"; done
fi

# ---------------------------------------------------------------------------
log "6/8  权限信号"
# ---------------------------------------------------------------------------
note "/dev/davinci* : $(ls /dev/davinci* 2>/dev/null | wc -l) 个节点"
note "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '<读不到>')"
if [[ -f "$WORK/msprof.log" ]] && grep -qiE "permission|denied|not permitted|EACCES|privilege" "$WORK/msprof.log"; then
  grep -iE "permission|denied|not permitted|EACCES|privilege" "$WORK/msprof.log" | head -5 | sed 's/^/     /'
  rec "权限信号" "FAIL" "日志含权限关键词"
else
  rec "权限信号" "PASS" "无权限类报错"
fi

# ---------------------------------------------------------------------------
log "7/8  cmake / 编译链(顺带确认阶段 B 的编译步骤)"
# ---------------------------------------------------------------------------
note "cmake : $(command -v cmake || echo '未找到')  $(cmake --version 2>/dev/null | head -1)"
note "make  : $(command -v make || echo '未找到')"
note "关键头文件:"
for h in kernel_operator.h kernel_tpipe.h; do
  p="$(find /usr/local/Ascend -name "$h" 2>/dev/null | head -1)"
  note "  $h → ${p:-★找不到}"
done
command -v cmake >/dev/null && rec "cmake" "PASS" "$(cmake --version 2>/dev/null|head -1|awk '{print $3}')" || rec "cmake" "FAIL" "-"

# ---------------------------------------------------------------------------
log "8/8  解析脚本依赖"
# ---------------------------------------------------------------------------
python3 -c "import csv,json,statistics,subprocess,pathlib,argparse,logging;print('标准库齐全')" 2>&1 | sed 's/^/   /'
rec "解析脚本依赖" "PASS" "仅需标准库"

# ---------------------------------------------------------------------------
echo; echo "============================================================"; echo " 判定汇总"; echo "============================================================"
printf ' %-22s %-6s %s\n' "检查项" "结果" "详情"
printf ' %-22s %-6s %s\n' "----------------------" "------" "----------------------------------------"
FAILED=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r n s d <<< "$r"; printf ' %-22s %-6s %s\n' "$n" "$s" "$d"
  [[ "$s" == "FAIL" ]] && FAILED=$((FAILED+1))
done
echo
if [[ $FAILED -eq 0 ]]; then
  echo " ✅ 全部通过 —— msprof 路线可行,阶段 B 按计划做"
else
  echo " ❌ $FAILED 项失败"
  echo "    若 torch/torch_npu 那两项失败 → 环境不对,先确认 IMAGE 是不是 agent 实际用的镜像"
  echo "    若只有 msprof 相关失败      → 走备选:继续用 performance.py"
  echo "      (官方仓 collaborative-agent-kernel-evolution/skills/ascendc-evaluation/scripts/)"
fi
echo; echo " 完整日志(容器内,--rm 退出即删,需要就现在拷走): $WORK"
echo "============================================================"
