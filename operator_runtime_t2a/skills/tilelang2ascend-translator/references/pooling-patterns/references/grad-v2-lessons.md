# avg_pool3d_grad v2 补课：反向 pooling 落地的新经验（相对 v1 文档的增量）

> 使用范围：实现知识参考，按本题需要查阅，不新增执行阶段或验收门禁。原文中的架构、shape、性能数据与参数仅适用于所述场景；以当前 SDK、reference 和固定评测入口为准。不得据此修改 reference/case、容差、标杆、计时参数，或另起评测。


> **定位**：本文是 本地 `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md`（通用设计资料）/ [backward-implementation.md](backward-implementation.md) / [precision-patterns.md](precision-patterns.md) / [alignment-guards.md](alignment-guards.md) 的**增量补丁**，记录 avg_pool3d_grad v2（双路径 gather/scatter 分派 + fp16/bf16 补齐 + 定向优化 case0）在 v1 文档之上新增的踩坑。阅读前先看那三篇。
>
> **v2 核心结果（ops-nn 标杆，125-case 共同子集 NCDHW+NDHWC/C 对齐）**：NDHWC 原生快路径（§10）落地后 v2 geomean **0.99x~1.21x**（两次复跑），显著高于快路径前的 0.77x；同子集 v1 为 **0.59x**（v2 ≈ v1 的 1.7~2.0×）。之前 trace 里「2.00x」是对 torch.autograd 的虚高值。⚠️ 绝对加速比受小 shape launch 开销主导、run-to-run 方差约 ±20%（v1 自身也在 0.45~0.76x 间波动），结论以「同 session 同 case 相对 v1」为准。
>
> **延迟分桶泛化结果（§11，200-case，30/40/30 延迟分桶）**：v2 geomean **0.751x** vs v1 **0.337x**（v2 ≈ v1 的 **2.23×**）。分桶看：<500us **0.89x** / 500-1000us **0.58x** / >1000us **0.85x**——**medium（500-1000us）区段是 v2 的短板**（慢 ops-nn ~2x），小 shape 主导集上的 ~1.0x 是虚高。
>
> **medium 定向优化后（§12，阶段二.十）**：v2 geomean **0.96~1.00x**（独立复测 0.964x）。分桶看：<500us ~0.84~0.97x（launch-bound 噪声带）/ **500-1000us 0.88x（+52%）** / **>1000us 1.29x（+51%）**。三处 gather 省算：division-free（ow-outer 循环）、scatter→gather 分派重估、divisor 因式分解。

## 1. 纯 Vector 算子必须用 `GetCoreNumAiv()`，不是 `GetCoreNum()`（最高优先级）

**症状**：scatter 路径换了 `KERNEL_TYPE_MIX_AIC_1_2` + `GetCoreNum()` 后，case0 `matched_ratio≈0.0176`（sum ratio 1.97x）——shape 对、不崩、但数值完全错，且呈「部分工作单元被重复处理」的模式。

**根因**：`avg_pool3d_grad` 是**纯 Vector 算子**（pooling 反向 scatter/gather，无 Cube/MatMul）。Ascend 910B3 上：
- `GetCoreNum()` = **24**（AIC/Cube 核数）
- `GetCoreNumAiv()` = **48**（AIV/Vector 核数）

用 `GetCoreNum()` 做 round-robin 划片时，只覆盖了一半向量核，且 `KERNEL_TYPE_MIX_AIC_1_2` 的 kernel 声明与实际 vector-only 实现不匹配，导致部分工作单元被多核重复处理。这是**最隐蔽的一类精度 bug**：shape 全对、只差一个「核数」参数，matched_ratio 掉到 0.02 但极易被误判为算法错。

**规则**：
- 判定算子是否纯 Vector（Add/Mul/Reduce/Pooling scatter-gather，无 Cube）→ 是，则 kernel 声明 `KERNEL_TYPE_MIX_AIV_1_0`，host 侧 `PlatformAscendCManager::GetInstance()->GetCoreNumAiv()`。
- 只有涉及 MatMul/Cube 的算子才用 `GetCoreNum()`。
- 定位方法：单 case dump + 看输出是否有「重复累加」（sum 明显 > grads sum 的 divisor 倍）。这是核数用错的指纹，不是窗口公式错。

## 2. 累加用 `Axpy` 融合指令，避免 Muls+Add 的临时 buffer WAR

**症状**：case4（divisor_override，C=512）`matched_ratio=0.700983`。C 扫描（16/32/64/128/256/512）定位到 **C≥128 才出现**的向量累加竞争。

**根因**：gather 路径里 `Muls(tmpLocal, gradRow, inv)` 与 `Add(acc, acc, tmpLocal)` 复用同一个 `tmpLocal_`，C 大时 VECCALC 队列变长，两次相邻 kernel 调用之间的 `tmpLocal_` 读写在流水里重叠（WAR）。

**修复**：改用单条融合指令 `AscendC::Axpy(acc, gradRow, inv, C)`（`acc += inv * gradRow`，一步完成缩放+累加），消除临时缓冲竞争。`Axpy` 是 AscendC 提供的 `y = α·x + y` 融合指令，语义等价 Muls+Add 但无中间 buffer。

**规则**：需要「缩放再累加」时优先 `Axpy`（或 `Muls` 目标直接是最终累加器），**不要**引入会被下一次迭代复用的临时 buffer 做两步。

## 3. V→MTE3 的 WAR：累加循环结束后、CopyOut 前补 `PipeBarrier<PIPE_ALL>`

**症状**：把核数从 24 切到 48（§1）后，case4 出现精度退化（`max_abs_diff=0.037`，`matched_ratio=0.977`，误差恒为 `-1/divisor` 单位且只在 `iw=7` 处）。

**根因**：这是**第二种 WAR**，与 [backward-implementation.md](backward-implementation.md) §11.1（MTE2 `DataCopyPad` 读 `acc_` 后）不同——这里是**V 核写 `acc_`（Axpy 累加循环）与 MTE3 `CopyOut`（读 `acc_` 搬到 GM）竞争**。累加循环最后一条 `Axpy` 写 `accLocal_[W-1, :]`，紧接着的 `CopyOut`（MTE3）读它；C=512 时 VECCALC 队列最长、竞态窗口最大，最后一个 W 位置被读到旧值。v1 参考用显式 `SetFlag<V_MTE3>/WaitFlag<V_MTE3>` 同步，行为一致。

**规则**：gather 路径的 `CopyOut`（MTE3 读累加器）之前、累加循环之后，必须插 `PipeBarrier<PIPE_ALL>()`。两种 WAR 都要防：`DataCopyPad`(MTE2 读)→写 `acc_` 之间（§11.1），以及写 `acc_`(VEC)→`CopyOut`(MTE3 读) 之间（本条）。

## 4. fp16/bf16 下 Cast：`float→bfloat16_t` 不支持 `CAST_NONE`，须 `CAST_RINT`

**症状**：bf16 精度全失败（5/5，`max_abs_diff≈0.995`，输出为乱值，out sum 5263 vs ref sum 4078）。

**根因**：`precision-patterns.md` 的 `FinalizeOutputTensorHelper` 对 fp32→低精度下 Cast 一律用 `CAST_NONE`。但 AscendC 的 Cast dtype 支持矩阵：`float→half` 支持 `CAST_NONE`，**`float→bfloat16_t` 不支持 `CAST_NONE`**（仅 `CAST_RINT/FLOOR/CEIL/ROUND/TRUNC`）。对 bf16 用 `CAST_NONE` 产生未定义行为。

**修复**：
- 上 Cast（T→fp32）保持 `CAST_NONE`（精确）。
- 下 Cast（fp32→T）分类型：`half` 用 `CAST_NONE`，`bfloat16_t` 用 `CAST_RINT`（round-to-nearest-even）。

**规则**：**dtype 相关 Cast 必须按 `T` 分派 round mode**，不能无条件 `CAST_NONE`。尤其 bf16 下 Cast 只能 RINT 一族。用 `if constexpr` 按 `std::is_same_v<T, ...>` 分派即可。

## 5. C 对齐守卫是 scheme 相关的：v2 用 `C%8`(fp32) / `C%16`(fp16/bf16)，不是 `W*C%16`

**背景**：`alignment-guards.md` 给出 `C%8` + fp16 `W*C%16`/`OW*C%16`。但 v2 的 gather/scatter 用「`DataCopyPad` blockCount=C、blockLen=sizeof(T)」把每个 C 值 pad 到 32B 槽（`ubW_ = 32/sizeof(T)`），向量化槽是 **C 维**，因此守卫是：
- fp32：`C % 8 == 0`（32B / 4B = 8）
- fp16/bf16：`C % 16 == 0`（32B / 2B = 16）

**规则**：对齐守卫取决于**你的向量化槽在哪一维**，不是固定公式。槽在 C 维 → `C % (32/sizeof(T))`；槽在 W×C 行维 → `W*C % (32/sizeof(T))`。写 host `TORCH_CHECK` 前先明确自己 kernel 的 blockLen/blockCount 布局。注意 `C%16` 对 fp16/bf16 比 `W*C%16` 更严（C=8 会被 v2 拒绝）。

**放宽可行性（NDHWC round 评估结论）**：
- gather 路径的 `C%8`/`C%16` 是**结构性约束**——C 连续行上做向量化 `DataCopy`/`Cast`/`Axpy`（count=`ow*C`/`W*C`/`C`），AscendC 要求搬运/计算元素数 32B 对齐；半精度 `ow*C % 16` 依赖 ow 奇偶，不可靠，不能靠 c 本身凑。
- scatter 路径已用 `DataCopyPad`（blockCount=C, blockLen=sizeof(T)）**天然支持任意 C**（≤4095），唯一残留是 fp16/bf16+overlap 的 `ProcessCast` 末块（chunk=`C*ubW_`，末块 `n=total%chunk` 可能非 32B 对齐）。
非对齐输入需要合法尾块分支；这里的快路径条件不允许用于拒绝 reference 支持的 case。

## 6. 双路径分派：`kD*kH*kW >= 256` 是 gather→scatter 的实用阈值

> ⚠️ **已证伪（见 §12.2）**：本节的 `kD*kH*kW >= 256 → scatter` 阈值在本算子**实证失败**——v2 的 scatter 是 naive scatter（跨步读 + 每输出位置 vol 次原子写），实测 15 个 scatter case 中 14 个仅 0.04x~0.36x，任意 vol 都劣于 division-free gather。当前实现已把 `kScatterKernelVolume` 提到 `1<<30`（**全走 gather**），scatter 代码保留为兜底。
>
> **修正后的方向**：**默认 gather；naive scatter 几乎全劣，不要仅凭「窗口大」就切 scatter**；只有 output-driven transpose-scatter（[backward-implementation.md](backward-implementation.md) §12：块读 + Transpose + UB 内向量化 Add + SetAtomicAdd）才可能反超 gather，且属高复杂度重写。下文保留原阈值仅供理解历史决策。

v1 文档 §12.5 只给了定性判断（「kernel 大」→ scatter）。v2 落地的**具体可复用阈值**（后被 §12.2 证伪）：

```cpp
constexpr int64_t kScatterKernelVolume = 256;  // 8^3
const bool scatterMode = (kD * kH * kW >= kScatterKernelVolume);
```

- `kD*kH*kW < 256`（2³=8、3³=27 等小窗口）→ **input-driven gather**：NDHWC 内部布局 + C 维向量化 + `Axpy` 累加（§2），零原子、代码简单。
- `kD*kH*kW >= 256`（8³=512 大窗口）→ **output-driven scatter**：NCDHW 直通 + `SetAtomicAdd`，避免 gather 的大窗口整行重复读。

**注意（历史备注）**：这曾是一个工程阈值，来自「8³ 大窗口 gather 慢于 scatter、2³/3³ 小窗口 scatter 的原子开销不划算」的实测。**§12.2 实证该结论对 naive scatter 不成立**——方向修正为「默认 gather；仅 output-driven transpose-scatter 值得大 kernel 做」。

## 7. 不要丢 `data_format`：先丢再补的 rework 成本高于一开始就保留

v2 初版 host 去掉了 `data_format`，**恒按 NCDHW 处理**（gather 路径内部 `permute` 到 NDHWC 再 permute 回），导致：
- NDHWC 输入被 `grads.size(1)==C` 校验拒绝，或更糟，被当作 NCDHW 算出错误结果。
- ops-nn `avg_pool3_d_grad` 的 200-case 覆盖中，NDHWC 约占一半，v2 只能跑 65/200。

**NDHWC round 又补回**（schema 末尾加 `int data_format=0`，host 入口 `TORCH_CHECK(data_format ∈ {0,1})`，`data_format==1` 时 `grads.permute({0,4,1,2,3}).contiguous()` 归一化到 NCDHW 再走原逻辑）。补回实现本身很干净（几行 host permute + 一个 int 参数 + 默认值 0 向后兼容），但**整轮 rework（schema 变更 + host 改 + 重新编译 + 重新验证 5 类 smoke）的成本远高于一开始就保留**。

**规则**：反向 pooling 的 `data_format` 是 ops-nn 接口的**一等公民**（`.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md` §5）。不要为省一次 permute 而丢弃 NDHWC——host 侧 permute 归一化（NDHWC→NCDHW）是几行代码的代价，但丢弃后要补回来的代价是一整轮开发。除非算子 spec 明确 NCDHW-only，否则**从一开始就保留 `data_format` 分派**。

## 8. 性能数据口径

源案例采用的标杆与测量数据仅作结构分析参考；不改变当前固定 reference 和计时方式。

## 9. NDHWC 补齐的 perf 代价：host permute 归一化 ≠ 原生 NDHWC 快路径

「先丢再补」补回来的 NDHWC 是**正确但非原生优化**的：

- 实现是 host 入口 `grads.permute({0,4,1,2,3})` 把 NDHWC 归一化到 NCDHW，再复用原 NCDHW 逻辑。对 gather 路径（`kD*kH*kW < 256`），kernel 内部还会再做 NCDHW→NDHWC（内部布局）→ NCDHW（输出）两次 permute，等于 **NDHWC 输入在 gather 路径走了 3 次 permute**。
- 结果：NDHWC 补回来后，v2 相对 v1 的领先从「NCDHW-only 子集的 2.1×」收窄到「125-case 全子集的 1.31×」——因为 v1 的 NDHWC 是原生快路径，v2 的 NDHWC 是 permute 归一化，多出的 host/kernel permute 吃掉了部分优势。

**规则**：host permute 归一化是**保正确、不保性能**的最短路径，适合先让接口功能完整。要追平 ops-nn 的 NDHWC 变体，仍需像 [backward-implementation.md](backward-implementation.md) §12.3 那样按 layout 分派原生 kernel 变体（NDHWC 走 C 连续直接向量化，不做 NCDHW 往返）。功能补齐与性能补齐是两轮不同的工作，别把「补了 data_format」当成「NDHWC 也优化好了」。

## 10. NDHWC 原生快路径（已实现）：按 (data_format × scatterMode) 四象限分派 permute

§9 的「3 次 permute」在本轮被消除，且**只改 op_host、kernel 零改动**。

**关键洞察**：gather 路径内部本就使用 NDHWC 布局（C 维向量化），所以 `data_format==1` 的 grads **本来就是 NDHWC**，直接复用即可，无需 NDHWC→NCDHW→NDHWC 的往返。

**四象限 permute 分派**：

| data_format × scatterMode | 输入 permute | 输出 permute | 合计 |
|---|---|---|---|
| NCDHW + gather | NCDHW→NDHWC | NDHWC→NCDHW | 2 |
| NCDHW + scatter | 无 | 无 | 0 |
| **NDHWC + gather** | **无（原样传入）** | NDHWC→NCDHW | **1** |
| NDHWC + scatter | NDHWC→NCDHW | 无 | 1 |

**实现要点**：入口处**不要**无条件做 layout 归一化，先算 `scatterMode`，再按 `(data_format, scatterMode)` 决定 permute；Do/Ho/Wo 与 N/C 校验改按原生 layout 读（NDHWC 取 `C=size(4)`）。NDHWC-gather 从 3 次 permute 降到 1 次。

**结果**：独立计时（`perf_ndhwc_check.py`，shape (2,32,8,16,64) gather）显示 NDHWC-gather（1 permute）比 NCDHW-gather（2 permute）快约 **24%**。

**规则**：`data_format` 与内部工作布局要**一起规划**。若 kernel 某路径本就以 NDHWC 为工作布局，则 NDHWC 输入是「白送的」——按 `(data_format × 分派模式)` 四象限决定 permute，而不是入口无条件归一化。这样 NDHWC 输入非但不必付出 host 归一化代价，反而比 NCDHW 输入更省（少一次 permute）。「去掉入口归一化 + 四象限分派」是比「补回 data_format」更进一步的、纯 host 侧的零成本优化。

## 11. 计时分析

查看固定入口提供的逐 case 原始耗时与加速比，区分 launch 开销和大 tensor 访存瓶颈；不更改用例分布或计时参数。
