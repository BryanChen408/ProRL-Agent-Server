# CannBot 设计与实现参考

本文件中的链接相对 `.claude/workflows/`。用 Read 打开链接对应的真实路径；只读本题相关
资料，修复时复用已读内容。按 reference 的计算结构、shape、dtype 和索引语义选择模板，
不按数据集名或题号选择。cudaLLM 的 get_inputs 与 NPUKernelBench 的 get_input_groups
都以原始 Python 定义为准；只有原题提供 JSON 时才读取 JSON。

## 来源与边界

下列两组 references/templates 原样来自 Polar 的 `operator_runtime_ascendc/skills/`
（引入提交 `a8f802ed`），保留 cannbot 原文与版权声明，共 28 个文件。未复制旧 SKILL.md、
脚本或子代理；它们是只读使用的知识资料，不是另一套执行流程。
prepare 沿用现有 skills/workflows 复制流程，容器不需要访问宿主历史工作目录。

旧模板含 `<op_name>`、示例计算和硬件常量（如 MAX_CORES=40）；仅复用适合本题的
设计与代码片段，在当前预生成工程原位实现。实际核数、UB、API 重载和 dtype 支持须按
当前 SOC 与 `$ASC_DEVKIT_DIR` 核实；不得用模板覆盖 reference 语义、固定 pipeline、
attempt 预算、用例或独立 judge。模板示例是 gather 不代表可直接实现 scatter_add。

## 设计入口

- [原版设计模板](../skills/ascendc-design-doc-generator/templates/design-template.md)
- [通用 tiling 原则](../skills/ascendc-design-doc-generator/references/general-tiling-principles.md)
- [硬件设计参考](../skills/ascendc-design-doc-generator/references/hardware-architecture.md)
- [代码生成资料加载指南](../skills/ascendc-code-gen/references/GUIDE.md)

GUIDE 中的 `references/xxx.md` 相对 `.claude/skills/ascendc-code-gen/`，裸文件名相对该目录
的 `references/`。旧 pooling 文中的 `skills/ascendc-operator-code-gen/` 指本地
`.claude/skills/ascendc-code-gen/`；下表给出可直接读取的路径，无需寻找另一套 skill。

| 实际计算结构 | 设计资料 | Host / kernel 参考 |
| --- | --- | --- |
| 逐元素 | [elementwise](../skills/ascendc-design-doc-generator/references/elementwise-tiling.md) | [host](../skills/ascendc-code-gen/templates/elementwise_op_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/elementwise_op_kernel.cpp) |
| 一维共享索引，如 index_select | [index](../skills/ascendc-design-doc-generator/references/index-tiling.md) | [host](../skills/ascendc-code-gen/templates/index_op_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/index_op_kernel.cpp) |
| 每个输出位置独立索引，如 gather | [index](../skills/ascendc-design-doc-generator/references/index-tiling.md) | [host](../skills/ascendc-code-gen/templates/index_op_per_elem_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/index_op_per_elem_kernel.cpp) |
| 行归约/归一化 | [reduction](../skills/ascendc-design-doc-generator/references/reduction-tiling.md) | [host](../skills/ascendc-code-gen/templates/row_op_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/row_op_kernel.cpp) |
| 池化，先核对 NDHWC 布局 | [pooling](../skills/ascendc-design-doc-generator/references/pooling-tiling.md) | [host](../skills/ascendc-code-gen/templates/pool_ndhwc_op_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/pool_ndhwc_op_kernel.cpp) |
| 排序/TopK | [sort](../skills/ascendc-design-doc-generator/references/sort-tiling.md) | [host](../skills/ascendc-code-gen/templates/sort_op_host.cpp) / [kernel](../skills/ascendc-code-gen/templates/sort_op_kernel.cpp) |

## 已有工程的真实文件名

先读工程 model.py 与 host 的分发条件，再选 device 变体；不要从目录名猜 `_kernel.cpp`。

| 工程 | 入口 | Device 入口与选择依据 |
| --- | --- | --- |
| rms_norm | [model](templates/archive_tasks/rms_norm/model.py)、[host](templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp) | [rms_norm.cpp](templates/archive_tasks/rms_norm/kernel/op_kernel/rms_norm.cpp)；参考行归约及 launch 传参，不是通用索引模板 |
| gather_elements_v2 | [model](templates/archive_tasks/gather_elements_v2/model.py)、[host](templates/archive_tasks/gather_elements_v2/kernel/op_host/gather_elements_v2.cpp) | [last_dim_fp16](templates/archive_tasks/gather_elements_v2/kernel/op_kernel/gather_elements_v2_last_dim_fp16.cpp)、[last_dim_fp32](templates/archive_tasks/gather_elements_v2/kernel/op_kernel/gather_elements_v2_last_dim_fp32.cpp)；其他轴/布局继续按 host 分支查看同目录 scalar/transpose 变体 |
| matmul_leakyrelu | [model](templates/archive_tasks/matmul_leakyrelu/model.py)、[host](templates/archive_tasks/matmul_leakyrelu/kernel/op_host/matmul_leakyrelu.cpp) | [matmul_leakyrelu_fp16.cpp](templates/archive_tasks/matmul_leakyrelu/kernel/op_kernel/matmul_leakyrelu_fp16.cpp)；先核对输入精度、矩阵布局与激活语义 |

其他已有工程用 Glob `.claude/workflows/templates/archive_tasks/**/model.py` 列出，再按内容查找。
历史对话中的 `0718/...`、`0723/...` 等目录不在当前环境；不要反复寻找这些绝对路径。
没有完整原件的历史同题工程未恢复，不能把截断的 Read 输出当成可编译源码。

历史 `ascendc-operator-project-init` 中的工程文件，当前对应
[kernel_skeleton 的 CMakeLists.txt](templates/kernel_skeleton/kernel/CMakeLists.txt) 和本题预生成的 `kernel/`；helper 原件在
[torch_kernel_helper.h](templates/kernel_skeleton/kernel/utils/torch_kernel_helper.h)。
查旧 `ascendc-performance-analyzer/script/performance.py` 的目的是理解测速时，应读当前
[ops-profiling](../skills/ops-profiling/SKILL.md) 与
[msprof_perf_summary.py](../skills/ops-profiling/scripts/msprof_perf_summary.py) 的实际输入和计算逻辑；
不寻找旧脚本来另起测速，不复用旧 PASS 或历史耗时作为当前判分。

## 当前工具链源码

官方 OPP 实现和编译器头文件不一定在 devkit 中。需要时在当前 session 工作目录运行
下面的只读定位命令一次，复用固定 pipeline 的环境选择；不导入 TileLang、不运行 NPU：

```bash
(
source tools/env.sh
"$OPERATOR_PYTHON" - <<'PY'
import os
from importlib.util import find_spec
from pathlib import Path

for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
    value = os.environ.get(key)
    root = Path(value).resolve() if value else None
    print(f"{key}={root if root and root.is_dir() else '不可用或未设置'}")
spec = find_spec("tilelang")
package = Path(spec.origin).resolve().parent if spec and spec.origin else None
print(f"TILELANG_SOURCE={package if package and package.is_dir() else '未安装或无可读源码目录'}")
examples = package.parent / "examples" if package else None
print(f"TILELANG_EXAMPLES={examples if examples and examples.is_dir() else '当前安装不含仓内示例'}")
PY
)
```

输出的真实路径可直接用于 Glob/Grep/Read。`ASCEND_HOME_PATH` 是 CANN 源码检索根目录；
`ASCEND_OPP_PATH` 如已设置，也允许检索其中官方源码。OPP 变量未设置时先 Glob CANN 根目录
的 `opp/`；目录不存在即记录当前安装未提供，不搜索其他版本，也不把 `不可用` 当路径读取。
不要读历史的 `/usr/local/Ascend/cann-8.5.1/`、`/home/.../tilelang-ascend/` 等绝对路径。

| 想确认的信息 | 在已确认的根目录中检索 |
| --- | --- |
| BF16 类型、API 模板与函数签名 | CANN：`**/include/**/*.h`、`**/pkg_inc/**/*.h`、`**/impl/**/*.h`，先按符号或文件名缩小范围；rad2deg 轨迹读过的 `atvoss/util/vec.h` 也从这里定位 |
| 官方算子的访存和计算实现 | OPP：`**/interleave_rope/*.h`、`**/rotary_position_embedding/*.h`；找到文件后再读，不假定 `ops_transformer` 层级不变 |
| 编译选项和生成头文件机制 | CANN：按报错文件名定位 `**/function.cmake`、`**/ascendc.cmake` 等当前定义 |
| TileLang API 是否存在、参数及设计示例 | `TILELANG_SOURCE` 下按符号检索 `.py`；`TILELANG_EXAMPLES` 存在时只读相关源码，不导入或运行示例 |

只读取与本题有关的片段并复用已读信息。发现版本/API 差异，以当前编译工具链定义为准；
修改仅限本题工程。找不到完整官方实现时返回通用设计/API 资料，不补造“官方源码”，
不因缺少一个参考示例提前终止解题，也不复制旧二进制、旧 PASS 或耗时来判分。

## 找不到时

1. Read 失败后，Glob 文件所在目录；目录本身不存在就退到上述 skills/archive 根目录检索。
2. 按返回的真实文件名 Read；Gather 等多变体先读 host，不能任选一个变体代替不存在的名字。
3. API/示例缺失时按 [docs-search](../skills/ascendc-docs-search/SKILL.md) 检索 devkit；
   需要底层定义或官方 OPP 实现时，再用上方实际工具链入口。
4. 评测报错先读 `judge_out/metrics_error.log`；允许只读检查 `tools/` 和 skill scripts 的
   参数/输入处理，但禁止修改这些脚本或绕过固定 pipeline 执行验证、测速。
