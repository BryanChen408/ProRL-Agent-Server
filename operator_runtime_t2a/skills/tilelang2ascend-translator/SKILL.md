---
name: tilelang2ascend-translator
description: >
  AscendC kernel 转译与实现专家 Skill。在 Polar 预生成工程骨架上完成 AscendC
  kernel；简单算子直接读取 model.py，复杂算子同时参考 TileLang 设计。
  当需要实现、修复或接线 AscendC kernel 时，使用此 skill。
argument-hint: >
  输入：output_dir 目录路径（必含 model.py 与预生成骨架；复杂算子另含 TileLang 产物）。
  输出：原位补全后的 AscendC 实现与必要接口接线。
---

# AscendC Kernel 转译 Skill

你是一名 AscendC kernel 转译与实现专家。你的目标是在 Polar 已经为本题生成的工程骨架上完成 AscendC kernel，并通过 AscendC 验证。简单算子以 `model.py` 为语义输入；复杂算子还要参考 TileLang 设计。TileLang 是可选设计输入，不是 correctness gate，也不是工程初始化来源。

## 前置条件
本阶段开始前，以下产物必须已经存在：
- `{output_dir}/model.py` — 唯一语义参考
- `{output_dir}/kernel/` 与 `{output_dir}/model_new_ascendc.py` — Polar prepare 预生成骨架

若 `op_type == "complex"`，还必须存在：
- `{output_dir}/design/tile_level/` — TileLang tile-level 设计，作为转译输入
- `{output_dir}/model_new_tilelang.py` — TileLang 绑定层/设计表达，可参考但不作为正确性依据

若 `op_type == "simple"`，不要求也不要为了满足本 skill 而新建 TileLang 产物。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录。唯一例外是可以只读访问
  `$ASC_DEVKIT_DIR`，用于查阅当前 CANN 环境配套的官方文档、示例和实现参考。
  禁止读取其他工作区外路径。
- 禁止读取 `.claude/skills/tilelang2ascend-translator/references/TileLangAscendProgrammingGuide.md`；该文档是 TileLang 编程指南，仅供 TileLang 阶段使用，与本阶段无关。
- 预生成骨架是唯一工程契约。禁止调用项目初始化 skill、复制其他任务或模板、重建整个工程。
- `kernel/CMakeLists.txt`、`kernel/setup.py`、`kernel/utils/` 和 `model_new_ascendc.py` 的双路径 loader 属于机制件，默认原样保留。
- 如果 reference 的真实接口要求变化，允许同步修改 `model_new_ascendc.py` 调用、`register.cpp` schema、`ops.h` 声明和 `op_host` 实现；禁止只改其中一处造成签名断链。
- 严格按照算子描述生成kernel，ascend c kernel的功能应该和标杆完全一致，不能出现部分功能使用ascend c，部分使用torch算子的情况
- 即使测试用例中不包含某个功能或者分支对应的case，也要生成对应的ascend c kernel代码
- **🛑 同步机制强制门禁**: 涉及 MIX_AIC / CrossCore / WorkspaceQueue / 死锁 / 全零输出 时，必须先完成 **步骤 0-C** 的同步 checklist。详见下方步骤 0-C 章节。

## 目标任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 数据集原版测试用例，禁止修改、精简或备份
│   ├── design/           # 仅复杂算子路径需要
│   │   ├── design.md     # 可选设计记录
│   │   ├── block_level/  # TileLang block-level 设计（已由上一阶段完成）
│   │   └── tile_level/   # 仅复杂算子存在，作为转译输入
│   ├── kernel/           # AscendC kernel（op_host/ + op_kernel/ 分层）
│   │   ├── CMakeLists.txt
│   │   ├── setup.py      # whl 打包配置
│   │   ├── ops.h         # 算子声明
│   │   ├── register.cpp  # torch.ops.npu.* 注册（仅注册）
│   │   ├── op_host/
│   │   │   └── <op_name>.cpp  # Host 端: tiling + EXEC_KERNEL_CMD
│   │   ├── op_kernel/
│   │   │   └── <op_name>_kernel.cpp
│   │   └── utils/        # Polar prepare 预生成的固定工具
│   │       └── torch_kernel_helper.h
│   ├── model_new_tilelang.py # 仅复杂算子存在，可参考但不要修改
│   └── model_new_ascendc.py  # AscendC wrapper → 内部调用 torch.ops.npu.<op>()
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料：
- `.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md` — TileLang 转 AscendC 指南
- `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` — TileLang 与 AscendC API 映射表
- `.claude/skills/tilelang2ascend-translator/references/AscendCVerification.md` — AscendC 验证指南
- `.claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md` — Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention）
- `tools/ascendc_eval_pipeline.sh` — Polar RL 唯一允许执行的 AscendC 验证入口；禁止直接运行本 skill 的上游评测脚本
- `.claude/workflows/templates/archive_tasks/` — 历史成功任务，host/kernel 完整参考实现（**编译/运行时错误时优先查阅**）

### 🛑 官方文档目录（`$ASC_DEVKIT_DIR`，按需查阅）

`$ASC_DEVKIT_DIR` 是运行时只读挂载的 asc-devkit 根目录。开始前必须先确认
`test -d "$ASC_DEVKIT_DIR"`；不得假定文档仍保留旧版多层目录。当前 API 文档主要是
`$ASC_DEVKIT_DIR/docs/zh/api/` 下的扁平布局，必须按 API 名查找实际文件，并检查同名数字后缀变体：

```bash
find "$ASC_DEVKIT_DIR/docs/zh/api" -type f -name "${APIName}*.md"
```

首次实现时查阅实际使用 API 的文档；后续修复只查本轮错误、拟修改 API
或新引入 API 对应的条目。禁止凭记忆或猜测 API 签名、参数、dtype 支持矩阵。

| 查阅入口 | 内容 | 何时查阅 |
|----------|------|---------|
| `$ASC_DEVKIT_DIR/docs/zh/api/` | 用 `find` 按 API 名查找所有文档变体 | 首次实现或 API 不确定时 |
| `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/` | 官方 SIMD C++ 示例 | 标准范式或 API 用法不确定时 |
| `$ASC_DEVKIT_DIR/impl/` | API 实现与 tiling 参考 | 文档不足时 |
| `.claude/workflows/templates/archive_tasks/rms_norm/` | EXEC_KERNEL_CMD 正确传参模式 | 编写 op_host 时 |

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

---

### 🛑 步骤 0-A: Attention 算子模式路由（Attention / FlashAttention 类算子强制执行）

🛑 **无论算子类型如何，第一步必须读取 model.py**：
   Read `{output_dir}/model.py` 的 forward() 方法，逐行检查计算逻辑。
   转译阶段的输入固然是 tile_level 设计文件，但判断算子是否属于 Attention 类
   必须回到 model.py 的原始计算逻辑。**禁止**凭 tile_level 文件名或记忆跳过此步。

**触发条件**：读取 model.py 后，检查 forward() 是否包含以下任一特征：
- `softmax(Q @ K^T / sqrt(d)) @ V` 或等价 attention 计算模式（如 `F.softmax(matmul(Q, K^T) / sqrt(dk)) @ V`）
- `scaled_dot_product_attention` / `F.scaled_dot_product_attention`
- 类名包含 `Attention` / `SDPA` / `Flash`
- tile-level 设计中包含 Q/K/V 三输入 attention 结构

如果触发条件满足，必须逐个完成以下 checklist：

```
0-A.1 🛑 读取 AttentionPatternIndex.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md

0-A.2 🛑 逐条回答"生成前问题"中的 7 个诊断问题，记录命中的模式:
    1. 输入是标准 [B,H,S,D] 还是 (T,H,D) 拼接布局？
    2. K/V 是连续 tensor 还是 paged cache？
    3. Hq 和 Hkv 是否相等？
    4. Dqk 和 Dv 是否相等？
    5. 是否有 sink_k/sink_v？
    6. 是否有 indices/topk？
    7. 是否有 causal、padding、显式 mask？
    
    如果 7 项全否定 → 命中"标准 Attention" → 下一步 0-A.3 读 archive 模板
    如果任一命中 → 下一步 0-A.3 读对应的 pattern 文档（可组合）

0-A.3 🛑 只读取命中的文档（渐进式披露，只读需要的）:
    - 命中模式 → Read 对应文档顶部的"先读这个"

0-A.4 🛑 在思考中确认:
    - 已读的 pattern 文档列表及其关键语义边界
    - 组合顺序（多模式命中时按 TND → Head Sharing → MLA → Sink → Sparse → Paged → Mask 顺序理解）
    - 本算子的 AscendC 转译策略应与命中的 pattern 对齐
```

**门禁规则**：
- 如果触发条件满足但 0-A.1-0-A.4 未完成 → **禁止**进入步骤 1，**禁止**编写任何 kernel/ 代码
- 如果触发条件不满足 → 跳过步骤 0-A，直接进入步骤 0-B
- 禁止凭记忆或经验跳过模式文档直接转译

---

## 🛑 步骤 0-B: 错误驱动的官方文档查阅

**首次实现先确认标准范式和本实现实际使用的 API；修复迭代只确认本轮错误涉及、准备修改或新引入的 API。未变化的文档不重复读取。**

```
0.1 阅读标准范式:
    在 `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/` 中用 `find`/`rg` 定位与本题数据搬运、
    计算 API 和缓冲模式相符的官方示例
    → 确认 CopyIn→Compute→CopyOut 的完整流水线模式

0.2 阅读 EXEC_KERNEL_CMD 正确模式:
    .claude/workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp
    → 确认: 所有 tiling 参数必须是独立标量左值，禁止传 struct 指针
    → 确认: blockDim = usedCoreNum（多核统一分发），禁止 host 侧逐核循环
    → 确认: 核数来源于平台 API 动态获取（GetCoreNumAic/Aiv），非硬编码常量

0.3 逐个查阅要使用的 API 文档:
    根据算子计算逻辑，列出所有将使用的 AscendC API，然后**逐个**在
    `$ASC_DEVKIT_DIR/docs/zh/api/` 查找实际文档。对每个 API 都要执行：

        find "$ASC_DEVKIT_DIR/docs/zh/api" -type f -name "${APIName}*.md"

    如果返回多个同名变体，必须按实际调用形式逐个核对，不得凭数字后缀猜测版本。
    ⚠️ 每个 API 必须确认: ① 模板参数（类型/非类型）② 函数参数（个数/类型）③ dtype 支持矩阵 ④ work buffer 需求。
    **禁止凭记忆或猜测 API 签名**。

    ── 数据搬运 ──
    - DataCopyPad → 查找 `DataCopyPad*.md`
      ⚠️ 签名两态: GM→UB 4参(dst,src,cp,pp), UB→GM 3参(dst,src,cp)
    - DataCopy → 查找 `DataCopy*.md`

    ── 类型转换 ──
    - Cast → 查找 `Cast*.md`
      ⚠️ 确认 bfloat16→float32 和 float32→bfloat16 的 RoundMode 参数

    ── 矢量计算 (Memory) ──
    - Mul / Add / Sub / Rsqrt → 分别查找 `<APIName>*.md`

    ── 标量计算 (Reg) ──
    - Muls / Adds / Rsqrt (scalar) → 查找所有同名变体，以当前平台文档的真实签名为准

    ── 高阶 API ──
    - ReduceSum → 查找 `ReduceSum*.md`
      ⚠️ 模板: <T, pattern, isReuseSource>, 参数: (dst,src,workBuf,srcShape[],srcInnerPad)
      ⚠️ GetReduceSumMaxMinTmpSize 也要单独查找文档
    - Broadcast → 查找 `Broadcast*.md`
      ⚠️ 模板: <T, dim, axis, isReuseSource>, dim∈{1,2}, axis∈{0,1}
    - Cos / Sin → 查找同名文档，并分别查找对应的 `Get*MaxMinTmpSize*.md`

    ── 同步控制 ──
    - PipeBarrier → 查找 `PipeBarrier*.md`
      ⚠️ 确认 PIPE_MTE2/PIPE_MTE3/PIPE_V/PIPE_ALL 各 barrier 的放置位置规则
    - CrossCoreSetFlag/WaitFlag → .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md
      ⚠️ 确认 mode2 下 Set/Wait 两侧 PIPE 参数完整且配对
      ⚠️ AIC 侧: Set<0x2, PIPE_FIX> + Wait<0x2, PIPE_FIX>
      ⚠️ AIV 侧: Set<0x2, PIPE_MTE2> + Wait<0x2, PIPE_MTE2>（MTE3 写用 PIPE_MTE3）
      ⚠️ 封装泛型工具类（如 WorkspaceQueue）时 ProducerAcquire/ConsumerAcquire 必须将 PIPE 模板化传入

    📋 **查阅完成后，在思考中列出 API 签名清单**:
    对每个 API 记录:
    - 完整模板参数 (如 ReduceSum<float, AscendC::Pattern::Reduce::AR, false>)
    - 完整函数参数名和类型
    - work buffer 需求 (需要/不需要, 如需要则列出 GetXxxMaxMinTmpSize 的查阅结果)
    - dtype 约束

0.4 查阅 TBuf 用法:
    查找并阅读 `$ASC_DEVKIT_DIR/docs/zh/api/TBuf*.md`
    → 确认 UB 临时缓冲区的正确分配模式

0.5 🛑 验证所有 work buffer 尺寸（运行时正确性铁律）:
    对于每个使用 TBuf<uint8_t> 作为 work buffer 传入的 API，**必须在 host 端通过对应的
    GetXxxMaxMinTmpSize 计算正确尺寸，禁止在 kernel 中硬编码 work buffer 大小**。

    │ Work Buffer 使用者 │ 尺寸获取 API (host 端调用) │ API 文档 │
    │-------------------│---------------------------│---------│
    │ ReduceSum         │ GetReduceSumMaxMinTmpSize │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ ReduceMax         │ GetReduceMaxMaxMinTmpSize │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ ReduceMin         │ GetReduceMinMaxMinTmpSize │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ Cos               │ GetCosMaxMinTmpSize       │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ Sin               │ GetSinMaxMinTmpSize       │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ SinCos            │ GetSinCosMaxMinTmpSize    │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │
    │ Broadcast         │ GetBroadCastMaxMinTmpSize │ 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 按 API 名查找 │

    **验证步骤 (每次编写 kernel 前强制执行)**:
    a. 列出本算子所有使用 work buffer 的 API
    b. 逐一查阅上表中对应的 GetXxxMaxMinTmpSize 文档
    c. 确认每个 API 的 work buffer 最小/最大尺寸计算方法
    d. 在 host 端 tiling 函数中调用 GetXxxMaxMinTmpSize，将结果作为 tiling 参数传入 kernel
    e. kernel 中 InitBuffer 的 work buffer 尺寸必须来自 tiling 参数，**禁止**硬编码为固定值
    f. 在思考中记录: 每个 work buffer 的计算结果和对应的 API 名称

    ⚠️ 本步骤为**运行时正确性硬性要求**。硬编码 work buffer 尺寸 < 实际所需最小值
       会导致 vector core timeout (507034) / UB 内存违例等运行时错误。

⚠️ 未完成以上 0.1-0.5 全部步骤前，禁止进入步骤 1 编写代码。Attention 类算子还必须完成步骤 0-A。
   查阅完成后，记录与本轮改动直接相关的 API 约束；不要抄录无关文档或完整文档清单。
```

---

### 🛑 步骤 0-C: 同步机制门禁（涉及跨核同步 / MIX_AIC / 输出异常时强制执行）

**触发条件**（任一满足即触发）：
- kernel 使用了 `KERNEL_TYPE_MIX_AIC` 混合核模式
- 代码中出现 `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `WorkspaceQueue`
- 运行时出现**全零输出**、死锁、hang、vector core timeout (507034)
- 编译后功能验证 FAIL 但无编译错误

如果触发条件满足，必须逐个完成以下 checklist：

```
0-C.1 🛑 读取 ascendc-sync-guide.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md 全文

0-C.2 🛑 逐条在思考中确认以下 checkpoint:
    ① PIPE 配对:
       - AIC 侧所有 CrossCore Set/Wait → PIPE_FIX
       - AIV 侧所有 CrossCore Set/Wait → PIPE_MTE2（MTE3 写操作用 PIPE_MTE3）
       - WaitFlag 是否漏写 PIPE 模板参数
    ② CV1:2 模式信号计数:
       - AIC Set 1 次 → 两个 AIV 各 Wait 1 次
       - 两个 AIV 各 Set 1 次 → AIC Wait 2 次（等两个 AIV 都完成）
    ③ 封装泛型工具类:
       - WorkspaceQueue ProducerAcquire/ConsumerAcquire 是否通过模板参数将 PIPE 传入两侧
    ④ InitFreeSlots:
       - 是否仅 Consumer 侧调用一次（禁止 Producer/Consumer 两侧重复调用）
    ⑤ 条件分支:
       - 是否可能跳过 Set/Wait 导致对方死等（如提前 return）
    ⑥ TQue BUFFER_NUM:
       - 是否 ≥ 循环中同时持有的 queue tensor 数量 + 1

0-C.3 🛑 如果存在 AIC↔AIV 交叉依赖（如 AIC 等 AIV 的 SIG_P_READY，AIV 同时等 AIC 的 SIG_O_READY）:
    - 画出信号时序图，确认不存在循环等待（A 等 B 设 X，B 同时等 A 设 Y）
    - 确认 PRELAUNCH 延迟是否足够打破循环依赖
```

**门禁规则**：
- 触发条件满足但 0-C.1-0-C.3 未完成 → **禁止** Edit/Write 任何涉及同步的代码
- 禁止凭经验修改 CrossCore 参数而不查阅 sync-guide
- 仅当本轮涉及 MIX_AIC / CrossCore / WorkspaceQueue / 死锁 / 全零输出时重新检查

---

## 流程
首次实现前完成 **步骤 0-A（如触发）、步骤 0-B 和步骤 0-C（如触发）**；后续迭代按错误与改动范围只重读相关部分。

### 简单路径的内嵌设计与审查

当 `op_type == "simple"` 时，不调用额外 Agent，也不生成只为流程服务的 DESIGN / PLAN /
WALKTHROUGH / REVIEW 文档；在当前上下文中完成下面两次结构检查：

**实现前**：
- 从 `model.py` 确认完整接口、输入输出布局、dtype、shape、广播/索引维度和边界行为。
- 明确 Host 只负责校验、输出分配、tiling 和启动，所有 tensor 计算都在 kernel 内完成。
- 明确多核切分、UB 分配、32B 对齐和尾块处理；核数与 UB 容量必须动态获取。
- 对计划使用的 AscendC API 逐个查官方文档，不能凭记忆写签名。

**实现后、首次评测前**：
- 核对 `model_new_ascendc.py → register.cpp → ops.h → op_host → op_kernel` 的参数顺序、类型、返回值完全一致。
- 核对 loader、CMake、setup.py 和 utils 仍是预生成机制件，没有被重建或替换。
- 核对所有 GM/UB 偏移、Alloc/Free、EnQue/DeQue、对齐区和尾块均不越界。
- 核对没有 Host 侧 tensor 计算、plain-torch fallback、硬编码核数或硬编码 UB 容量。

这些检查只约束工程骨架与安全边界，不规定算子公式、API 组合或优化策略；具体数学实现由模型根据本题自行完成。

### 步骤 1: 在预生成骨架上完成 AscendC

先逐个读取现有骨架文件和 `model.py`。简单算子直接按 `model.py` 实现；复杂算子将
`{output_dir}/design/tile_level/` 的设计转译为 AscendC。以下文件已经存在，应在原路径补全，
不要删除后重建：
- `{output_dir}/kernel/op_host/<op_name>.cpp` — Host 端 (tiling 计算 + kernel launch)
- `{output_dir}/kernel/op_kernel/<op_name>_kernel.cpp` — Device 端 (CopyIn → Compute → CopyOut)
- `{output_dir}/kernel/ops.h` — 预生成算子声明，接口变化时才同步修改
- `{output_dir}/kernel/register.cpp` — 预生成注册，接口变化时才同步修改
- `{output_dir}/model_new_ascendc.py` — 预生成 wrapper，接口变化时才同步修改调用

下列机制件默认不得改写：
- `{output_dir}/kernel/setup.py`
- `{output_dir}/kernel/CMakeLists.txt`
- `{output_dir}/kernel/utils/`

参考文档：`.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md`
**🛑 首次实施前完成步骤 0-A（如触发）、步骤 0-B 和步骤 0-C（如触发）。复杂算子按实际用到的 TileLang 原语读取 `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` 对应映射；简单算子没有 TileLang 输入，不套用该映射。**

**op_host/<op_name>.cpp** 模式：
- include `torch_kernel_helper.h` + `tiling/platform/platform_ascendc.h`
 - 🛑 **核数获取（禁止硬编码）**：根据算子计算特征选择正确的 API，  **禁止 `constexpr int32_t = 20` 或 `min(20, ...)` 等硬编码**：
     ```cpp
     auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();

     // 纯 Vector 计算（Norm/激活函数/逐元素等）→ GetCoreNumAiv()
     int32_t totalCoreNum = platform->GetCoreNumAiv();

     // 纯 Cube 计算（MatMul/矩阵乘）→ GetCoreNumAic()
     int32_t totalCoreNum = platform->GetCoreNumAic();

     // Cube+Vector 融合（Attention/CV融合等）→ CalcTschBlockDim(), sliceNum为数据切分的份数
     int32_t totalCoreNum = platform->CalcTschBlockDim(
         sliceNum, platform->GetCoreNumAic(), platform->GetCoreNumAiv());
     ```
- 使用平台 API 获取 `GetCoreMemSize(UB)`
- Block 级 tiling: Cache Line 512B 对齐，formerNum/formerLength/tailNum/tailLength
- UB 级 tiling: bufferCoefficient 推导，32B 对齐 tileLength
- **🛑 EXEC_KERNEL_CMD 传参铁律**: 所有 tiling 参数必须是**独立标量左值**，**禁止传 struct 指针**。参照 `.claude/workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp` 的正确模式
- blockDim = usedCoreNum（多核统一分发），kernel 内部通过 `GetBlockIdx()` 计算工作范围
- `EXEC_KERNEL_CMD` 所有参数必须为**左值**（具名变量），禁止传入临时变量/右值/字面量。`double` 先转 `float` 局部变量，`bool` 用 `int64_t` 替代，表达式先赋给局部变量再传入

**op_kernel/<op_name>_kernel.cpp** 模式：
- template class `Kernel<OpName>` 含 Init/Process/CopyIn/Compute/CopyOut
- BUFFER_NUM = 2 (double buffer)；如算子需要在循环中同时持有多个 queue tensor，需相应增大 BUFFER_NUM
- DataCopyPad 用于 GM↔UB 搬运
- FP16/BF16 升精度到 FP32 计算
- 整核/尾核偏移和尾块对齐处理

   **ops.h** 模式：
   ```cpp
   namespace ascend_kernel {
   at::Tensor <op_name>(<参数列表>);
   }
   ```

   **register.cpp** 模式：
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

### 步骤 2: 核对接口接线 + 编译验证

检查预生成的 `{output_dir}/model_new_ascendc.py`，并保留其**双路径加载**模式：
- 优先 `import <op_name>_ext`（whl 安装后自动触发 TORCH_LIBRARY 注册）
- 失败回退 `torch.ops.load_library()` 直加载 `kernel/build/<op_name>_ext*.so`
- forward() 中调用 `torch.ops.npu.<op_name>(...)`

示例：
```python
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "<op_name>_ext*")

try:
    import <op_name>_ext  # noqa: F401 — whl path
except ImportError:
    # Fallback: direct .so loading
    if _LIB_PATTERN not in "".join(sys.path):
        import glob as _glob
        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])

class ModelNew(nn.Module):
    def forward(self, x, ...):
        ...
        return torch.ops.npu.<op_name>(x, ...)
```

**禁止**在 model_new_ascendc.py 中使用 `torch.*` / `F.*` 计算算子。只有真实接口发生变化时才同步修改 wrapper 的签名和 `torch.ops.npu.<op_name>(...)` 调用；不要重写 loader。

然后按工作目录 `CLAUDE.md` 的「固定入口」编译并验证，不要直接运行本 skill 自带的评测脚本。

---

### 步骤 3: 错误修复迭代

迭代上限只服从固定入口输出的 `attempt`、`limit`、`remaining` 和 `next_step`；不得在本 skill 内另建计数器。每次修复前执行以下步骤：

#### 🛑 3.0 修复前查阅（只读本轮错误相关资料）

**根据错误类型，查阅对应的 asc-devkit 文档或历史案例：**

🛑 **跨核同步专项门禁（涉及 CrossCore / WorkspaceQueue / MIX_AIC / 全零输出时强制执行）**:
- **必须先完成步骤 0-C 的全部 checklist**（含读取 ascendc-sync-guide.md 全文 + 逐条确认 6 项 checkpoint）
- 未完成步骤 0-C → **禁止** Edit/Write 任何 CrossCore/WorkspaceQueue 相关代码

| 错误类型 | 必须查阅 |
|---------|---------|
| **编译错误: API 签名不匹配** | 在 `$ASC_DEVKIT_DIR/docs/zh/api/` 查找该 API 的所有 `.md` 变体，确认当前平台签名 |
| **编译错误: 类型不匹配** | 查找 `$ASC_DEVKIT_DIR/docs/zh/api/Cast*.md` 及实际出错 API 的文档，确认 dtype 支持矩阵 |
| **编译错误: GlobalTensor/LocalTensor** | 查找 `$ASC_DEVKIT_DIR/docs/zh/api/{GlobalTensor,LocalTensor}*.md` |
| **运行时 vector core exception / UB 违例 / all-zero output** | ① 🛑 **优先执行步骤 0-C** 完成 sync checklist<br>② 查找 `$ASC_DEVKIT_DIR/docs/zh/api/TBuf*.md` 检查 buffer 大小<br>③ `.claude/workflows/templates/archive_tasks/rms_norm/` 对比 EXEC_KERNEL_CMD 传参模式<br>④ 检查是否有 struct 指针被传给 `EXEC_KERNEL_CMD`（常见根因） |
| **运行时 hang/死锁 / 跨核数据不流通** | 🛑 **必须先执行步骤 0-C**（含读取 ascendc-sync-guide.md 全文 + 6 项 checkpoint），再逐项排查 |
| **运行时 vector core timeout (507034)** | 🛑 这是硬件级别的 core 挂起错误。按顺序排查:<br>① **work buffer 尺寸**: 检查所有 API 的 work buffer (ReduceSum/Cos/Sin/Broadcast) 是否通过 GetXxxMaxMinTmpSize 正确计算 — 硬编码不足是最常见根因<br>② **Buffer 总溢出**: 计算所有 InitBuffer 分配的总 UB 字节数，确认不超过 GetCoreMemSize(UB)<br>③ **PipeBarrier 配对**: 每个 GM→UB (MTE2) 后必须有 PIPE_MTE2 barrier; 每个 V 计算块结束后必须有 PIPE_V barrier; 每个 UB→GM (MTE3) 前必须有 PIPE_V barrier<br>④ **循环边界**: 检查所有循环的边界类型一致性 (int32_t vs int64_t)，确认不会因类型不匹配导致死循环<br>⑤ **隔离法**: 将 kernel 逐步简化为 identity copy，每次恢复一个操作，定位触发 timeout 的具体 API<br>⑥ **参考历史**: 查阅 `.claude/workflows/templates/archive_tasks/` 中相似规模的融合算子，对比 work buffer 计算方式 |
| **精度不匹配 (MERE/MARE 超标)** | 调用 `ascendc-precision-debug` skill（见步骤 4） |

**⚠️ 在查阅完成并在思考中列出根因分析之前，禁止 Edit/Write 任何 kernel 代码。**

#### 3.1 分析错误输出，结合查阅结论确定根因
#### 3.2 修改 kernel/ 下的代码
#### 3.3 运行 CLAUDE.md 规定的固定入口
#### 3.4 如果 PASS → 完成，退出
#### 3.5 如果 FAIL 且固定入口仍给出修复预算 → 按 `next_step` 回到 3.0
#### 3.6 如果预算耗尽 → 停止调用工具，使用固定入口保存的 `.best`

---

### 步骤 4: 精度 Skill 深度诊断（固定入口分类为 D 类时）

只按固定入口的 D 类 `next_step` 调用精度 skill：

```
4.1 🛑 调用 Skill "ascendc-precision-debug"，传入 output_dir + 错误输出
    等待返回诊断结论和修复建议。此步骤不可跳过。

4.2 根据建议修改 kernel/ 代码，运行 CLAUDE.md 规定的固定入口

4.3 如果仍 FAIL 且 `next_step` 仍要求 precision-debug、预算尚有剩余 → 回到 4.1

4.4 如果 `next_step` 要求 precision-tuning →
    🛑 调用 Skill "tilelang2ascend-precision-tuning"，传入 output_dir + 错误输出
    等待返回取证→审计→修复分析。此步骤不可跳过。

4.5 根据建议修改 kernel/ 代码，运行 CLAUDE.md 规定的固定入口

4.6 如果仍 FAIL 且 `next_step` 仍要求 precision-tuning、预算尚有剩余 → 回到 4.4

4.7 如果预算耗尽仍 FAIL → 停止调用工具，报告当前状态并保留 `.best`
```

## 精度验证标准

**五类决策矩阵**（由 `verification_ascendc.py` 自动判定）：

| 类别 | 触发条件 | 判定标准 |
|---|---|---|
| 非计算类 | `ASCENDC_NON_COMPUTE=1` | view-as-int 二进制完全一致（含 NaN bit pattern） |
| bool 输出 | 输出 dtype 为 bool | `torch.equal` 严格相等 |
| 整数计算类 | 输入最高精度为 int 且输出为 int | `|actual − golden| == 0` |
| 量化计算类 | 输入最高精度为 float 且输出为 int | `|actual − golden| <= 1` |
| 浮点计算类 | 输出为 float 类型 | 三项 AND 判定（max_error_cap + matched_ratio ≥ 0.9 + MERE < rel_threshold） |

- **输入类型自动推断**：从实际输入 tensor 中取最高精度 dtype 分类为 float / int / no_tensor。
- **浮点三项判定**：① 100% 元素满足 `|diff| <= atol + rtol * |golden|`；② 分桶 matched_ratio ≥ 0.9（小值域绝对误差 / 正常域相对误差）；③ MERE（均值相对误差）< 阈值。三项全部通过才算通过。
