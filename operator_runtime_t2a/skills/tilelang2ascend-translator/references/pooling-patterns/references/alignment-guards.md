# Alignment Guards for Vectorized Pooling Kernels

> 使用范围：实现知识参考，按本题需要查阅，不新增执行阶段或验收门禁。原文中的架构、shape、性能数据与参数仅适用于所述场景；以当前 SDK、reference 和固定评测入口为准。不得据此修改 reference/case、容差、标杆、计时参数，或另起评测。


## 对齐约束总览

向量化 pooling kernel 有三层对齐约束：

| 约束 | 要求 | 原因 | 违规后果 |
|------|------|------|---------|
| C%8==0 | 本文整行向量化方案 | Add(C)/Muls(C) 操作 32B 对齐 | `UB address accessed by VEC instruction is not aligned` → 崩溃 |
| fp16 W*C%16==0 | 仅 fp16 | 输入 GM 行 DataCopy 32B 对齐 | 数据损坏（值变 512.0/nan） |
| fp16 OW*C%16==0 | 仅 fp16 | 输出 GM 行 DataCopy 32B 对齐 | 最后一个不完整输出行尾部损坏 |

> **⚠️ 方案相关（重要）**：上表假设「行粒度整行搬运」（`DataCopy` 以 `W*C` / `OW*C` 为单位）。若你的向量化槽在 **C 维**——即用 `DataCopyPad(blockCount=C, blockLen=sizeof(T))` 的 gather/scatter 方案（反向语义见 `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md`、实现见 [backward-implementation.md](backward-implementation.md)，另见 [grad-v2-lessons.md](grad-v2-lessons.md) §5）——fp16 守卫改为 **`C%16==0`（不是 `W*C%16`）**，且 **`C%16` 比 `W*C%16` 更严**（fp16 下 C=8 满足 `W*C%16` 但会被 `C%16` 拒绝）。写 host `TORCH_CHECK` 前先明确自己 kernel 的 blockLen/blockCount 布局（详见下文「方案相关的守卫」）。

## 为什么需要这些约束

### C%8==0: 向量计算对齐

AscendC V 核向量寄存器为 256-bit (32B)，一个 V 指令操作 256/32=8 个 fp32 元素或 256/16=16 个 fp16 元素。行粒度 kernel 使用 `Add(acc[ow*C], acc[ow*C], in[iw*C], C)` 这种 C-wide 向量操作，要求指针 32B 对齐。

C 非 8 倍数时，最后一个向量操作（C%8 元素）无法对齐，触发 UB 对齐 fault。

### fp16 W*C%16==0: 输入 DataCopy 对齐

`DataCopy(inLocal_, xGM_[rowBase], W*C)` 以 `W*C` 个 half 为单位搬运。GM→UB 的 DMA 要求源/目标地址 32B 对齐。W*C 个 half = W*C*2 bytes，需要 W*C*2 % 32 == 0，即 W*C%16==0。

不满足时，DMA 会损坏最后一个不对齐的 32B block 内的数据（非完整 block 的部分会被填充/截断）。

### fp16 OW*C%16==0: 输出 DataCopy 对齐

与输入对称：`DataCopy(yGM_[outRow*OW*C], outLocal_, OW*C)` 需要 OW*C 个 half 元素对齐。

不满足时，最后一个不完整 block 的尾部值损坏（实测值变为 512.0 或 nan）。

## 方案相关的守卫：向量化槽在 C 维时用 C%16，不是 W*C%16

上表三守卫对应「行粒度整行搬运」方案。但反向 gather/scatter（见 `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md`）和深度-only 快路径常用 `DataCopyPad` 把**每个 C 值 pad 到 32B 槽**（`blockCount=C, blockLen=sizeof(T)`），此时向量化槽在 **C 维**，守卫不同：

| 方案 | 向量化槽 | fp32 守卫 | fp16/bf16 守卫 |
|------|---------|----------|----------------|
| 行粒度整行搬运 | W×C 行 | C%8==0（计算对齐） | W*C%16==0 + OW*C%16==0（搬运对齐） |
| C 维向量化槽（gather/scatter / 深度快路径） | C | C%8==0 | **C%16==0** |

注意 **`C%16`（fp16）比 `W*C%16` 更严**：C=8 的 fp16 满足 `W*C%16`（W*8 为偶数即可）但会被 `C%16` 拒绝。规则：对齐守卫取决于**你的向量化槽在哪一维**，不是固定公式——槽在 C 维 → `C % (32/sizeof(T))`；槽在 W×C 行维 → `W*C % (32/sizeof(T))`。完整论证见 [grad-v2-lessons.md](grad-v2-lessons.md) §5（其中性能代价是设计权衡，不可用于拒绝本题合法输入）。

## 在当前题目中的使用

以上约束描述特定快路径的适用条件，不是输入合法性定义。不满足对齐时，应选择支持有效尾长的搬运/计算方案（如适用的 DataCopyPad 或尾块处理），保留全部合法元素，不能直接拒绝 reference 支持的输入。

出现对齐错误时核对本实现的有效长度、padding 长度、GM/UB 地址与 API 支持条件；诊断和复测仍走固定入口。
