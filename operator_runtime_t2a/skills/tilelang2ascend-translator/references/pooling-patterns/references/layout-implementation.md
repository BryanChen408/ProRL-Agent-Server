# NDHWC 布局的 AscendC 落地（Host Permute + Kernel 行地址）

> 使用范围：实现知识参考，按本题需要查阅，不新增执行阶段或验收门禁。原文中的架构、shape、性能数据与参数仅适用于所述场景；以当前 SDK、reference 和固定评测入口为准。不得据此修改 reference/case、容差、标杆、计时参数，或另起评测。


> **定位**：layout-strategy.md（设计语义：为何选 NDHWC + C 维向量化）的 **AscendC 实现篇**，即
> `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md` 的「实现模式」迁出内容。
> 先按 reference 确认布局与窗口语义；通用设计资料见 `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md`，无需寻找其他未安装技能。
> 反向 scatter-add 的布局分派（四象限 permute）另见 [grad-v2-lessons.md](grad-v2-lessons.md) §10。

## Host 侧：Permute + Dispatch

```cpp
// 1. NCDHW → NDHWC (contiguous)
at::Tensor xNdhwc = self.permute({0, 2, 3, 4, 1}).contiguous();

// 2. Kernel 操作 (NDHWC 内部)
// yFlat shape: [N*OD*OH, OW*C]

// 3. NDHWC → NCDHW (contiguous)
at::Tensor yNcdhw = yFlat.reshape({N, OD, OH, OW, C}).permute({0, 4, 1, 2, 3}).contiguous();
```

## Kernel 侧：行地址计算

```cpp
// NDHWC 下，元素总数为 N*D*H*W*C
// 偏移计算: (n*D + d)*H*W*C + h*W*C + w*C + c
// 简化: (n*D + d)*H*W*C + h*W*C  = 行基地址（W*C 元素）
//                          + w*C  = 列内偏移

const uint64_t rowBase =
    (static_cast<uint64_t>(nIdx) * D_ + static_cast<uint64_t>(id)) * H_ * W_ * C_ +
    static_cast<uint64_t>(ih) * W_ * C_;
// DataCopy 整行
AscendC::DataCopy(inLocal_, xGM_[rowBase], W_ * C_);
```

## 约束

- `permute().contiguous()` 在 host 侧产生额外的内存分配和拷贝
- 对于大 tensor，permute 开销可能显著，需要在性能分析中计入
- **仅在 kernel 计算收益 > permute 开销时使用**
- host 侧无条件 NCDHW↔NDHWC 归一化是「保正确、不保性能」的最短路径；要追平 ops-nn 需按 `(data_format × 分派模式)` 四象限决定 permute（见 grad-v2-lessons.md §9/§10）——若 kernel 某路径本就以 NDHWC 为工作布局，NDHWC 输入是「白送的」，非但不必付出 host 归一化代价，反而比 NCDHW 输入更省（少一次 permute）
