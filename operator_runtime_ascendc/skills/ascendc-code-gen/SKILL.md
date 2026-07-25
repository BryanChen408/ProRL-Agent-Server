---
name: ascendc-code-gen
description: >
  AscendC kernel 代码生成专家 Skill。读取设计文档（design.md 或 tile_level 设计），
  根据算子类型选择模板，生成 op_host/[op].cpp + op_kernel/[op].cpp +
  ops.h + register.cpp + setup.py，编译验证统一由 evaluate_ascendc.sh 完成。
  当设计文档已生成且需要生成 AscendC kernel 代码时，使用此 skill。
argument-hint: >
  输入：output_dir 目录路径（包含设计文档）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 代码生成 Skill

你是一名 AscendC kernel 代码生成专家。你的目标是根据设计文档生成完整的 AscendC kernel 代码、框架适配文件和构建脚本，最终编译安装为 whl 包。

## 前置条件

本阶段开始前，必须已存在以下之一：
- `{output_dir}/design/design.md` — 设计文档（简单算子路径）
- `{output_dir}/design/tile_level/` — TileLang tile-level 设计（复杂算子路径，供 ascendc-translator 使用）

注意：复杂算子路径使用 `ascendc-translator` skill 而非本 skill。本 skill 主要用于简单算子的模板驱动代码生成。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许张量创建、张量变换以及调用 `torch.ops.npu.<op_name>(...)`。
- 在 AscendC 实现中只能使用块级或向量化操作，不能用标量逐元素写法。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 即使测试用例中不包含某个功能或分支对应的 case，也要生成对应的 kernel 代码。

## 目标 kernel 目录结构

```
{output_dir}/kernel/
├── CMakeLists.txt           # CMake 编译配置
├── setup.py                 # whl 打包配置
├── ops.h                    # 算子函数声明 (namespace ascend_kernel)
├── register.cpp             # torch.ops.npu.* 注册（仅注册，不含 host 逻辑）
├── op_host/
│   └── <op_name>.cpp        # Host 端: tiling + EXEC_KERNEL_CMD 启动
├── op_kernel/
│   └── <op_name>.cpp        # Device 端: CopyIn → Compute → CopyOut
└── utils/                   # 固定工具文件（从模板复制，不生成）
    └── torch_kernel_helper.h   # EXEC_KERNEL_CMD 宏
```

## Skill 参考资料

| 文件 | 用途 |
|------|------|
| `references/GUIDE.md` | 场景化参考加载策略 |
| `references/data-copy-api.md` | DataCopy/DataCopyPad API |
| `references/vector-compute-api.md` | Vector 计算 API |
| `references/sync-control-api.md` | TQue/Pipe 同步控制 |
| `references/resource-management-api.md` | TPipe/TBuf 资源管理 |
| `references/basic-data-structures-api.md` | LocalTensor/GlobalTensor 等 |
| `references/kernel-constraints.md` | Kernel 编程约束 |
| `references/sort_topk-api.md` | Sort/TopK 专用 API |

### 模板文件

| 算子类型 | op_host 模板 | op_kernel 模板 |
|---------|-------------|---------------|
| Elementwise | `templates/elementwise_op_host.cpp` | `templates/elementwise_op_kernel.cpp` |
| 行处理 (Reduce/Norm) | `templates/row_op_host.cpp` | `templates/row_op_kernel.cpp` |
| Index (逐行索引) | `templates/index_op_host.cpp` | `templates/index_op_kernel.cpp` |
| Index (逐元素索引) | `templates/index_op_per_elem_host.cpp` | `templates/index_op_per_elem_kernel.cpp` |
| Sort | `templates/sort_op_host.cpp` | `templates/sort_op_kernel.cpp` |
| Pool | `templates/pool_ndhwc_op_host.cpp` | `templates/pool_ndhwc_op_kernel.cpp` |

## 流程

### 阶段 1: 加载参考文档 + 读取设计文档

**MANDATORY** — 先读取 `references/GUIDE.md`，按算子类型加载对应 reference。

从 `design/design.md` 提取：

| 提取项 | 设计文档章节 | 用途 |
|--------|------------|------|
| 函数签名 + 支持的 dtype | 算子接口定义 | op_host 函数原型、kernel 模板参数 |
| 算子类型 | Tiling 策略 | 选择模板 (elementwise / row / index / sort / pool) |
| UB 分配表 + bufferCoefficient | UB 分配表 | InitBuffer 大小、tileLength 计算 |
| AscendC API 调用伪代码 | 计算逻辑设计 | Compute 函数逻辑 |
| FP16/BF16 升精度流程 | Tiling 策略 | Cast 路径 |

### 阶段 2: 生成 op_kernel/<op_name>.cpp

**MANDATORY** — 读取模板文件完整内容，复制到目标路径后修改。

模板使用 `template <typename T>` 泛型 + `if constexpr` 分支：

```cpp
if constexpr (sizeof(T) == sizeof(float)) {
    // float32 直接计算
} else {
    // fp16/bf16: Cast 升精度 → fp32 计算 → Cast 降精度
}
```

**Kernel 类结构**:
- `Init(GM_ADDR ..., AscendC::TPipe *pipe)`: 读取 tiling 数据，设置 GlobalTensor，InitBuffer (BUFFER_NUM=2)
- `Process()`: 计算 tileNum，循环 CopyIn→Compute→CopyOut，处理尾 tile 对齐
- `CopyIn(progress, curTileLength)`: DataCopyPad GM→UB，EnQue
- `Compute(progress, curTileLength)`: DeQue 输入，计算，EnQue 输出
- `CopyOut(progress, curTileLength)`: DeQue 输出，DataCopyPad UB→GM

**MUST 遵守的反模式**:
- NEVER 让 FP16/BF16 直接参与复杂计算，必须先 Cast 到 FP32
- NEVER 在 EXEC_KERNEL_CMD 中传右值
- NEVER 对 GM↔UB 搬运使用 DataCopy，必须用 DataCopyPad
- NEVER 在 ReduceSum/ReduceMax 后直接复用源 tensor
- NEVER 在 kernel 中使用 `std::min/max/abs/sqrt/exp`
- NEVER 硬编码核数或 UB 大小
- NEVER 使用 bool 参数类型，用 int64_t 替代

**FP16/BF16 升精度模板**:
```cpp
// CopyIn 时 Cast 升精度
AscendC::DataCopyPad(xLocal, xGm[...], copyInParams, padParams);
AscendC::Cast(xLocalFp32, xLocal, CAST_NONE, curTileLength);
// FP32 计算
AscendC::Relu(yLocalFp32, xLocalFp32, curTileLength);
// CopyOut 时 Cast 降精度
AscendC::Cast(yLocal, yLocalFp32, CAST_ROUND, curTileLength);
AscendC::DataCopyPad(yGm[...], yLocal, copyOutParams);
```

### 阶段 3: 生成 op_host/<op_name>.cpp

**Host 端关键模式**（参考 helloworld 模板，使用 EXEC_KERNEL_CMD 启动 kernel）：

```cpp
#include "torch_kernel_helper.h"       // EXEC_KERNEL_CMD 宏
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<kernel_func>.h"  // cmake 自动生成，每个 kernel 入口一个

namespace ascend_kernel {
at::Tensor <op_name>(args) {
    // 输入校验
    // 创建输出 tensor
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
uint64_t ubSize;
ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
```

**Tiling 计算**:
1. Block 级: Cache Line 512B 对齐，formerNum/formerLength/tailNum/tailLength
2. UB 级: `bufferCoefficient` 从 UB 分配表推导，32B 对齐
3. `EXEC_KERNEL_CMD(<op_name>, blockDim, self, output, formerNum, formerLength, tailLength, tileLength, dtypeSize)`

**EXEC_KERNEL_CMD 硬约束**：所有参数必须是**左值**（具名变量），禁止传入临时变量/右值/字面量。`double` 先转 `float` 局部变量，`bool` 用 `int64_t` 替代，表达式先赋给局部变量再传入。

### 阶段 4: 生成框架适配文件

#### 4.1 `ops.h`
```cpp
namespace ascend_kernel {
at::Tensor <op_name>(<参数列表>);
} // namespace ascend_kernel
```

#### 4.2 `register.cpp`（仅含注册，不含 host 逻辑）

```cpp
#include "ops.h"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("<op_name>(<schema>) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
}
```

Schema 类型映射:
| C++ | Schema | 示例 |
|-----|--------|------|
| `const at::Tensor &` | `Tensor` | `Tensor self` |
| `at::IntArrayRef` | `int[]` | `int[] kernel_size` |
| `int64_t` | `int` | `int dim=-1` |
| `double` | `float` | `float eps=1e-5` |
| `bool` | `bool` | `bool flag=False` |

#### 4.3 `setup.py`

使用 `NpuExtension` (torch_npu 标准) + `cmake build_lib` 模式：

```python
import os, glob, subprocess
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"


class BuildExt(build_ext):
    def run(self):
        so_files = glob.glob(str(BUILD_DIR / "<op_name>_ext*.so"))
        if not so_files:
            os.makedirs(BUILD_DIR, exist_ok=True)
            soc_ver = os.environ.get("SOC_VERSION", "Ascend910B2")
            ascend = os.environ.get("ASCEND_HOME_PATH", "")
            subprocess.check_call(
                ["cmake", str(HERE),
                 f"-DSOC_VERSION={soc_ver}",
                 f"-DASCEND_CANN_PACKAGE_PATH={ascend}",
                 "-DCMAKE_BUILD_TYPE=Release"],
                cwd=BUILD_DIR)
            subprocess.check_call(
                ["make", f"-j{os.cpu_count()}"], cwd=BUILD_DIR)

        self.build_lib = str(BUILD_DIR)
        super().run()


setuptools.setup(
    name="<op_name>",
    version="0.1.0",
    description="<OperatorName> AscendC kernel",
    ext_modules=[NpuExtension("<op_name>_ext", sources=[])],
    cmdclass={"build_ext": BuildExt},
    license="BSD 3-Clause",
    python_requires=">=3.8",
)
```

关键机制：`build_lib` 指向 cmake 输出目录 `build/`，setuptools 直接从 cmake 产物打包 whl；`.so 不存在时自动触发 cmake + make`。

### 阶段 5: 生成 model_new_ascendc.py

采用**双路径加载**模式：优先 import（whl 安装后自动触发 TORCH_LIBRARY 注册），失败回退 `torch.ops.load_library` 直加载 `.so`：

```python
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "<op_name>_ext*")

# 优先 whl import（触发 TORCH_LIBRARY 注册）
try:
    import <op_name>_ext  # noqa: F401
except ImportError:
    # 兜底：直加载 .so
    if _LIB_PATTERN not in "".join(sys.path):
        import glob as _glob
        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor):
        # 预处理: 输入校验 + shape 处理
        assert x.dtype in (torch.float16, torch.float32)
        assert x.is_contiguous()
        # 调用注册的算子
        return torch.ops.npu.<op_name>(x)
```

**关键规则**: forward() 中只允许：张量创建、reshape/contiguous 等变换、`torch.ops.npu.<op>()` 调用。禁止任何 `torch.*` / `F.*` 计算。

## 生成后检查清单

### op_kernel
- [ ] 使用平台 API 获取硬件参数（不硬编码）
- [ ] BUFFER_NUM = 2 (double buffer)
- [ ] Init 整核/尾核偏移正确
- [ ] Process 尾 tile 对齐
- [ ] AllocTensor/FreeTensor 配对，EnQue/DeQue 配对
- [ ] FP16/BF16 升精度到 FP32 计算
- [ ] DataCopyPad 用于 GM↔UB 搬运

### op_host
- [ ] include torch_kernel_helper.h + platform_ascendc.h
- [ ] bufferCoefficient 与设计文档一致
- [ ] EXEC_KERNEL_CMD 参数均为左值

### 框架适配
- [ ] ops.h 声明与 op_host 一致
- [ ] register.cpp schema 参数类型/默认值正确

### model_new_ascendc.py
- [ ] 无 torch 计算算子
- [ ] 双路径加载: `try: import <op>_ext` 优先, `except: torch.ops.load_library` 兜底
- [ ] 通过 torch.ops.npu.<op>() 调用 kernel
