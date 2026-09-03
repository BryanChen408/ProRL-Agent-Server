#!/usr/bin/env bash
# =============================================================================
# t2a 阶段 B 闸门 —— 判分链在新基座上能不能出真 metrics
#
# 【怎么跑】宿主机执行,会自己进 ascendc-sandbox:v1:
#     bash /home/docker/t2a_gate_b.sh
#     NPU_ID=2 bash /home/docker/t2a_gate_b.sh          # 指定卡
#     ONLY=B1 bash /home/docker/t2a_gate_b.sh           # 只跑某一项(B1/B2/B3)
#
# 【三个独立检查】故意拆开,因为 B1 是我改动最大、风险最高的部分,
#                 它不依赖"我手写的 kernel 对不对",单独成立:
#
#   B1  msprof 路径          直接对一个**平凡 torch 实现**跑 msprof_perf_summary.py --quick。
#                            验证:wrapper 生成 → msprof 采集 → csv 解析 → performance.json
#                            字段(geomean_speedup / geomean_ref_us / geomean_asc_us)
#                            → 我们的提取逻辑与单位换算。**预期 speedup ≈ 1**。
#                            ⭐ 这项过了,阶段 B 的核心改动就验证了。
#
#   B2  固定入口全链路        用模板派生的 3_Add kernel 打 tarball,走 ascendc_eval_pipeline.sh,
#                            **逐阶段报告走到哪一步**(解包/注入/AST/编译/对拍/测速/metrics)。
#                            ⚠️ 那个 kernel 是我照 helloworld 模板改的(half→float、加 alpha),
#                            没能编译验证过。**编不过或对拍失败不推翻 B1**,只说明 kernel 要修,
#                            报告会指出卡在哪一阶段。
#
#   B3  AST 退化闸门反例      交一个纯 torch 的 model_new_ascendc.py,**预期被拦**。
#                            顺带验证阶段 D 的一道护栏在新基座上仍有效。
#
# 【安全性】全部在容器 /tmp 下操作,--rm 起容器,不动仓库、不动数据集(只读拷贝)。
#           占卡:B1 和 B2 的测速/对拍阶段会短暂占用一张 NPU。
# =============================================================================
set +e
IMAGE="${IMAGE:-ascendc-sandbox:v1}"
NPU_ID="${NPU_ID:-0}"
ONLY="${ONLY:-}"
CANON_SRC="${CANON_SRC:-/home/docker/cannbot_debug/ProRL-Agent-Server/operator_runtime_t2a}"
DATASET="${DATASET:-/home/docker/datasets/op_tasks/npukernelbench_level1_ascendc/op_tasks}"
OP="3_Add"

# ---------------------------------------------------------------------------
if [[ "${IN_CONTAINER:-0}" != "1" && ! -f /.dockerenv ]]; then
  echo "== 当前在宿主机,自动进入 $IMAGE =="
  command -v docker >/dev/null || { echo "❌ 没有 docker"; exit 1; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "❌ 找不到镜像 $IMAGE(可用 IMAGE=... 覆盖)"; exit 1; }
  exec docker run --rm --privileged --ipc host --network host \
    -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /home/docker:/home/docker:ro \
    -e IN_CONTAINER=1 -e NPU_ID="$NPU_ID" -e ONLY="$ONLY" -e IMAGE="$IMAGE" \
    "$IMAGE" bash /home/docker/t2a_gate_b.sh
fi

# ---------------------------------------------------------------------------
W="$(mktemp -d /tmp/t2a_gate_b.XXXXXX)"
RESULTS=()
log()  { printf '\n\033[1m========== %s ==========\033[0m\n' "$*"; }
sub()  { printf '\n-- %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
rec()  { RESULTS+=("$1|$2|$3"); }
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

echo "============================================================"
echo " t2a 阶段 B 闸门   $(date '+%F %T')   NPU_ID=$NPU_ID"
echo " 工作目录 $W"
echo "============================================================"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

[[ -d "$CANON_SRC" ]] || { echo "❌ 找不到 canonical: $CANON_SRC"; exit 1; }
[[ -f "$DATASET/$OP.py" ]] || { echo "❌ 找不到数据集: $DATASET/$OP.py"; exit 1; }
cp -r "$CANON_SRC" "$W/canonical"
note "canonical → $W/canonical   skills: $(ls "$W/canonical/skills" | wc -l) 个"

# ===========================================================================
# B1 —— msprof 路径(不依赖任何 AscendC kernel)
# ===========================================================================
if want B1; then
log "B1  msprof 路径(平凡 torch 实现,预期 speedup ≈ 1)"
D="$W/b1/$OP"; mkdir -p "$D"
cp "$DATASET/$OP.py" "$D/model.py"
cp "$DATASET/$OP.json" "$D/$OP.json"
# ModelNew 用纯 torch —— B1 只验 msprof 链路,不验 AscendC
cat > "$D/model_new_ascendc.py" <<'PY'
import torch, torch.nn as nn
class ModelNew(nn.Module):
    def forward(self, x, y, alpha=1.0):
        return torch.add(x, y, alpha=alpha)
PY
PERF="$W/canonical/skills/ops-profiling/scripts/msprof_perf_summary.py"
note "调用: msprof_perf_summary.py --quick --output-dir $D --warmup 3 --repeats 1"
note "(不传 --device —— 让它从 ASCEND_RT_VISIBLE_DEVICES 读,与 npu_lease 的行为一致)"
ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" \
  python3 "$PERF" --quick --output-dir "$D" --warmup 3 --repeats 1 > "$W/b1.log" 2>&1
B1RC=$?
note "退出码 $B1RC;日志尾部:"; tail -15 "$W/b1.log" | sed 's/^/     /'

if [[ -f "$D/performance.json" ]]; then
  note ""; note "performance.json 顶层字段:"
  python3 - "$D/performance.json" <<'PY' 2>&1 | sed 's/^/     /'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("n_cases_total","n_cases_valid","geomean_speedup","mean_speedup",
          "geomean_ref_us","geomean_asc_us","mean_ref_us","mean_asc_us"):
    print(f"{k:20s} = {d.get(k)}")
print(f"{'per_case 条数':20s} = {len(d.get('per_case') or [])}")
PY
  note ""; note "按 pipeline 的提取逻辑算出来的值(应与上面一致,延迟单位 us→ms):"
  python3 - "$D/performance.json" <<'PY' 2>&1 | sed 's/^/     /'
import json, sys
d = json.load(open(sys.argv[1]))
sp = d.get('geomean_speedup') or d.get('mean_speedup')
fw = d.get('geomean_ref_us') or d.get('mean_ref_us')
im = d.get('geomean_asc_us') or d.get('mean_asc_us')
print(f"SP   (speedup_vs_torch)      = {sp}")
print(f"FW   (framework_latency_ms)  = {round(fw/1000.0,6) if fw else None}")
print(f"IMPL (impl_latency_ms)       = {round(im/1000.0,6) if im else None}")
PY
  SP1=$(python3 -c "import json;d=json.load(open('$D/performance.json'));print(d.get('geomean_speedup') or d.get('mean_speedup') or '')" 2>/dev/null)
  if [[ -n "$SP1" ]]; then
    rec "B1 performance.json" "PASS" "geomean_speedup=$SP1"
    # 平凡实现两边同代码,speedup 应接近 1(0.3~3 之间都算合理,msprof 单次采样有噪声)
    python3 -c "import sys;v=float('$SP1');sys.exit(0 if 0.3<v<3.0 else 1)" 2>/dev/null \
      && rec "B1 speedup 合理性" "PASS" "≈1(实测 $SP1)" \
      || rec "B1 speedup 合理性" "WARN" "$SP1 偏离 1 较多,看是否单次采样噪声"
  else
    rec "B1 performance.json" "FAIL" "有文件但取不到 speedup 字段"
  fi
else
  note "✗ 没有生成 $D/performance.json"
  rec "B1 performance.json" "FAIL" "未生成,见 $W/b1.log"
fi
fi

# ===========================================================================
# B2 —— 固定入口全链路
# ===========================================================================
if want B2; then
log "B2  固定入口全链路(模板派生的 3_Add kernel)"
P="$W/b2"; mkdir -p "$P/$OP/kernel" "$P/input" "$P/output/submission" "$P/judge_out"
cp "$DATASET/$OP.py"   "$P/input/$OP.py";   cp "$DATASET/$OP.py"   "$P/$OP/model.py"
cp "$DATASET/$OP.json" "$P/input/$OP.json"; cp "$DATASET/$OP.json" "$P/$OP/$OP.json"
TPL="$W/canonical/workflows/templates/kernel_skeleton/kernel"
mkdir -p "$P/$OP/kernel/op_host" "$P/$OP/kernel/op_kernel" "$P/$OP/kernel/utils"
cp "$TPL/utils/torch_kernel_helper.h" "$P/$OP/kernel/utils/"
sed 's/{op_name}/add_alpha/g' "$TPL/CMakeLists.txt" > "$P/$OP/kernel/CMakeLists.txt"
sed 's/{op_name}/add_alpha/g' "$TPL/setup.py" > "$P/$OP/kernel/setup.py"

# ---- kernel:照 helloworld 改(half→float,加 alpha) --------------------
# 单核 + TILE=128 float(512B,32B 对齐)。数据集 5 个 case 的元素数
# 128/256/512/32768/131072 都是 128 的整数倍,故无尾块。
cat > "$P/$OP/kernel/op_kernel/add_alpha_kernel.cpp" <<'CPP'
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE = 128;

class KernelAddAlpha {
public:
    __aicore__ inline KernelAddAlpha() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, float alpha)
    {
        this->total = totalLength;
        this->alpha = alpha;
        this->tileNum = totalLength / TILE;
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ float *)z, totalLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) { CopyIn(i); Compute(i); CopyOut(i); }
    }
private:
    __aicore__ inline void CopyIn(int32_t p)
    {
        AscendC::LocalTensor<float> xL = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yL = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xL, xGm[p * TILE], TILE);
        AscendC::DataCopy(yL, yGm[p * TILE], TILE);
        inQueueX.EnQue(xL); inQueueY.EnQue(yL);
    }
    __aicore__ inline void Compute(int32_t p)
    {
        AscendC::LocalTensor<float> xL = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yL = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zL = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> t  = tmpBuf.Get<float>();
        AscendC::Muls(t, yL, this->alpha, TILE);
        AscendC::Add(zL, xL, t, TILE);
        outQueueZ.EnQue<float>(zL);
        inQueueX.FreeTensor(xL); inQueueY.FreeTensor(yL);
    }
    __aicore__ inline void CopyOut(int32_t p)
    {
        AscendC::LocalTensor<float> zL = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[p * TILE], zL, TILE);
        outQueueZ.FreeTensor(zL);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;
    uint32_t total; int32_t tileNum; float alpha;
};

extern "C" __global__ __aicore__ void add_alpha(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                uint32_t totalLength, float alpha)
{
    KernelAddAlpha op;
    op.Init(x, y, z, totalLength, alpha);
    op.Process();
}
CPP

cat > "$P/$OP/kernel/op_host/add_alpha.cpp" <<'CPP'
#include "torch_kernel_helper.h"
#include "aclrtlaunch_add_alpha.h"
namespace ascend_kernel {
at::Tensor add_alpha(const at::Tensor &x, const at::Tensor &y, double alpha)
{
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 1;                 // 单核:避免 GetBlockNum 分块与对齐问题
    uint32_t totalLength = 1;
    for (uint32_t s : x.sizes()) { totalLength *= s; }
    float a = static_cast<float>(alpha);
    EXEC_KERNEL_CMD(add_alpha, blockDim, x, y, z, totalLength, a);
    return z;
}
}  // namespace ascend_kernel
CPP

cat > "$P/$OP/kernel/ops.h" <<'CPP'
#ifndef OPS_H
#define OPS_H
#include <torch/extension.h>
namespace ascend_kernel {
at::Tensor add_alpha(const at::Tensor &x, const at::Tensor &y, double alpha);
}
#endif
CPP

cat > "$P/$OP/kernel/register.cpp" <<'CPP'
#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"
namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("add_alpha(Tensor x, Tensor y, float alpha) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("add_alpha", TORCH_FN(ascend_kernel::add_alpha));
}
}  // namespace
CPP

cat > "$P/$OP/model_new_ascendc.py" <<'PY'
import torch, torch.nn as nn
class ModelNew(nn.Module):
    def forward(self, x, y, alpha=1.0):
        return torch.ops.npu.add_alpha(x, y, float(alpha))
PY

sub "打包"
( cd "$P" && tar czf "output/submission/${OP}_impl.tar.gz" "$OP" ) \
  && note "✓ $(cd "$P" && tar tzf output/submission/${OP}_impl.tar.gz | wc -l) 个条目" \
  || note "✗ 打包失败"

sub "跑固定入口(judge 侧:workdir 顶层没有 $OP/,故 AGENT_SIDE=0)"
J="$W/b2run"; mkdir -p "$J/input" "$J/output/submission" "$J/judge_out"
cp "$P/input/$OP.py" "$P/input/$OP.json" "$J/input/"
cp "$P/output/submission/${OP}_impl.tar.gz" "$J/output/submission/"
( cd "$J" && ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" \
    ASCENDC_SKILLS_SRC="$W/canonical/skills" \
    bash "$W/canonical/tools/ascendc_eval_pipeline.sh" \
      --op_name "$OP" --impl "output/submission/${OP}_impl.tar.gz" --out_dir judge_out
) > "$W/b2.log" 2>&1
B2RC=$?
note "退出码 $B2RC"
note "阶段推进:"
grep -oE "\[ascendc-eval\] (Step[0-9a-z]*|done|verdict).*" "$W/b2.log" | sed 's/^/     /' || note "     (无阶段输出)"
note "日志尾部:"; tail -12 "$W/b2.log" | sed 's/^/     /'

if [[ -f "$J/judge_out/metrics.json" ]]; then
  note ""; note "metrics.json:"; cat "$J/judge_out/metrics.json" | sed 's/^/     /'
  B2EL="$J/judge_out/metrics_error.log"
  B2CRASH=0
  [[ -f "$B2EL" ]] && grep -qE "Traceback|ModuleNotFoundError|ImportError|SyntaxError" "$B2EL" && B2CRASH=1
  for f in ast_check_ok correctness_ok; do
    v=$(python3 -c "import json;print(json.load(open('$J/judge_out/metrics.json')).get('$f'))" 2>/dev/null)
    if [[ "$v" == "True" ]]; then rec "B2 $f" "PASS" "-"
    elif [[ $B2CRASH -eq 1 ]]; then rec "B2 $f" "FAIL" "$v(工具链崩溃,非实现问题)"
    else rec "B2 $f" "FAIL" "$v"; fi
  done
  SP2=$(python3 -c "import json;d=json.load(open('$J/judge_out/metrics.json'));p=d.get('perf_data') or {};print(p.get('speedup_vs_torch') or '')" 2>/dev/null)
  [[ -n "$SP2" ]] && rec "B2 speedup 落盘" "PASS" "$SP2" || rec "B2 speedup 落盘" "FAIL" "perf_data 无 speedup"
else
  rec "B2 metrics.json" "FAIL" "未生成"
fi
[[ -f "$J/judge_out/metrics_error.log" ]] && { note ""; note "metrics_error.log 头 20 行:"; head -20 "$J/judge_out/metrics_error.log" | sed 's/^/     /'; }
fi

# ===========================================================================
# B3 —— AST 退化闸门反例
# ===========================================================================
if want B3; then
log "B3  AST 退化闸门反例(纯 torch 实现,预期被拦)"
Q="$W/b3"; mkdir -p "$Q/$OP" "$Q/input" "$Q/output/submission" "$Q/judge_out"
cp "$DATASET/$OP.py" "$Q/input/$OP.py"; cp "$DATASET/$OP.json" "$Q/input/$OP.json"
cp -r "$P/$OP/kernel" "$Q/$OP/kernel" 2>/dev/null || mkdir -p "$Q/$OP/kernel"
cat > "$Q/$OP/model_new_ascendc.py" <<'PY'
import torch, torch.nn as nn
# 故意退化:纯 torch 实现,只在注释里提一句 torch.ops.npu.add_alpha
class ModelNew(nn.Module):
    def forward(self, x, y, alpha=1.0):
        return torch.add(x, y, alpha=alpha)
PY
( cd "$Q" && tar czf "output/submission/${OP}_impl.tar.gz" "$OP" ) >/dev/null 2>&1
( cd "$Q" && ASCENDC_SKILLS_SRC="$W/canonical/skills" \
    bash "$W/canonical/tools/ascendc_eval_pipeline.sh" \
      --op_name "$OP" --impl "output/submission/${OP}_impl.tar.gz" --out_dir judge_out
) > "$W/b3.log" 2>&1
ET=$(python3 -c "import json;print(json.load(open('$Q/judge_out/metrics.json')).get('error_type'))" 2>/dev/null)
note "error_type = $ET"
tail -6 "$W/b3.log" | sed 's/^/     /'
# 判据分两层:光看 error_type 不够 —— 检查器**自己崩掉**时也会报 ast_check_failed。
# 必须同时确认 metrics_error.log 里没有 Traceback / ModuleNotFoundError,
# 否则那是"闸门崩了"而不是"闸门拦住了"。
EL="$Q/judge_out/metrics_error.log"
CRASH=0
if [[ -f "$EL" ]] && grep -qE "Traceback|ModuleNotFoundError|ImportError|SyntaxError" "$EL"; then
  CRASH=1
  note "⚠️ metrics_error.log 里有异常栈 —— 检查器是崩了,不是拦住了:"
  grep -mE "Traceback|ModuleNotFoundError|ImportError|SyntaxError" -A 2 "$EL" 2>/dev/null | head -6 | sed "s/^/     /" \
    || grep -E "Traceback|ModuleNotFoundError|ImportError|SyntaxError" "$EL" | head -3 | sed "s/^/     /"
fi
if [[ "$ET" == "ast_check_failed" && $CRASH -eq 0 ]]; then
  rec "B3 退化闸门" "PASS" "如期拦下(且检查器未崩)"
elif [[ $CRASH -eq 1 ]]; then
  rec "B3 退化闸门" "FAIL" "检查器崩溃(非拦截)——见 metrics_error.log"
else
  rec "B3 退化闸门" "FAIL" "error_type=$ET(应为 ast_check_failed)"
fi
fi

# ===========================================================================
echo; echo "============================================================"; echo " 判定汇总"; echo "============================================================"
printf ' %-24s %-6s %s\n' "检查项" "结果" "详情"
F=0
for r in "${RESULTS[@]}"; do IFS='|' read -r n s d <<< "$r"; printf ' %-24s %-6s %s\n' "$n" "$s" "$d"; [[ "$s" == FAIL ]] && F=$((F+1)); done
echo
if [[ $F -eq 0 ]]; then
  echo " ✅ 闸门 B 通过 —— 判分链在新基座上出真 metrics,可以进阶段 C"
else
  echo " ❌ $F 项失败。判读顺序:"
  echo "    1) 先看 B1 —— 它不依赖我手写的 kernel。B1 过 = 阶段 B 的核心改动(msprof 接线)正确"
  echo "    2) B2 失败但 B1 过 → 多半是我那个 kernel 的问题,看它卡在哪一阶段:"
  echo "       解包/注入失败 = pipeline 接线问题;编译失败 = kernel 写错;对拍失败 = 数值不对"
  echo "    3) B3 失败 → 退化闸门在新基座上失效,要单独查 validate_ascendc_impl.py 的 108 行 diff"
fi
echo; echo " 日志: $W/{b1,b2,b3}.log   (容器 --rm,需要就现在拷走)"
echo "============================================================"
