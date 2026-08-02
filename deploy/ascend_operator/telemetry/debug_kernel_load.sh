#!/usr/bin/env bash
# 复现 AscendC kernel: build → 找.so → load_library → 检查 torch.ops.npu.<op> 注册。
# 目的:定位 0% 通过的根因 —— 编译产物加载了但自定义 op 没进 torch.ops.npu。
#
# 在【判分沙箱容器】里跑(容器有 CANN + torch + torch_npu)。宿主机示例:
#   TAR=<某个 {op}_impl.best.tar.gz 的宿主机路径>
#   RT=/mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/operator_runtime_t2a
#   docker run --rm -i \
#     -v "$RT":/opt/canonical:ro \
#     -v "$RT/tools":/opt/workspace/agent_workdir/tools:ro \
#     -v "$TAR":/tmp/impl.tar.gz:ro \
#     -v "$RT/../deploy/ascend_operator/telemetry/debug_kernel_load.sh":/tmp/dbg.sh:ro \
#     -e SOC_VERSION=ascend910b1 -e ASC_DEVKIT_DIR=/opt/asc-devkit \
#     ascendc-tilelang:v1-aarch64 bash /tmp/dbg.sh <OP_NAME> /tmp/impl.tar.gz
#   (注册检查不需要 NPU 设备;若还想真执行 op,加 --device 与判分同样的卡)
set -uo pipefail
OP="${1:?用法: dbg.sh <OP_NAME> <impl.tar.gz>}"
TAR="${2:?缺 tarball}"
SK="${ASCENDC_SKILLS_SRC:-/opt/canonical/skills}/tilelang2ascend-translator"
export SOC_VERSION="${SOC_VERSION:-ascend910b1}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

W=/tmp/dbgwork; rm -rf "$W"; mkdir -p "$W"; tar xzf "$TAR" -C "$W"
TASK="$(dirname "$(find "$W" -name model_new_ascendc.py | head -1)")"
echo "########## TASK_DIR=$TASK  OP=$OP  SK=$SK ##########"
echo "===== register.cpp / model_new 的 op 名与机制 ====="
grep -rnE "m\.def|TORCH_LIBRARY|PYBIND11_MODULE" "$TASK/kernel" 2>/dev/null | head
grep -nE "import|ops\.npu|load_library" "$TASK/model_new_ascendc.py" 2>/dev/null | head

echo "===== STEP1 build_ascendc.py（完整输出尾部）====="
python3 "$SK/scripts/build_ascendc.py" "$TASK" -v "$SOC_VERSION" --build-type Release 2>&1 | tail -50
echo "build 退出码=$?"

echo "===== STEP2 产物 .so ====="
find "$TASK/kernel" -name "*.so" 2>/dev/null
echo "--- pip 里的包 ---"; pip list 2>/dev/null | grep -iE "$(echo "$OP"|tr 'A-Z' 'a-z'|tr -d '_')|$OP" || echo "(无)"

echo "===== STEP3 load + register 检查 ====="
python3 - "$OP" "$TASK" <<'PY'
import sys, glob, os, re, importlib.util as U
op, task = sys.argv[1], sys.argv[2]
import torch
print("torch =", torch.__version__)
try:
    import torch_npu; print("torch_npu =", torch_npu.__version__)
except Exception as e:
    print("torch_npu 导入失败:", type(e).__name__, e)
regs = glob.glob(f"{task}/kernel/**/register.cpp", recursive=True)
opname = None
if regs:
    m = re.search(r'm\.def\("([A-Za-z_]\w*)', open(regs[0], errors="replace").read())
    opname = m.group(1) if m else None
print("register.cpp 定义 op =", opname)
sos = glob.glob(f"{task}/kernel/**/*.so", recursive=True)
print("找到 .so:", [os.path.relpath(s, task) for s in sos] or "无")
for so in sos:
    try:
        torch.ops.load_library(so); print("  load_library OK:", os.path.basename(so))
    except Exception as e:
        print("  load_library FAIL:", os.path.basename(so), "->", type(e).__name__, str(e)[:200])
if opname:
    print(f"load 后 hasattr(torch.ops.npu,'{opname}') =", hasattr(torch.ops.npu, opname))
    try:
        getattr(torch.ops.npu, opname); print("  getattr OK")
    except Exception as e:
        print("  getattr 真实报错:", type(e).__name__, str(e)[:200])
# 再走 model_new 自身的加载逻辑
sys.path.insert(0, task)
try:
    spec = U.spec_from_file_location("mnew", f"{task}/model_new_ascendc.py")
    mm = U.module_from_spec(spec); spec.loader.exec_module(mm)
    print("model_new import OK; ModelNew:", hasattr(mm, "ModelNew"))
    if opname:
        print(f"  model import 后 hasattr npu.{opname} =", hasattr(torch.ops.npu, opname))
except Exception as e:
    print("model_new import FAIL:", type(e).__name__, str(e)[:200])
PY
echo "########## DONE ##########"
