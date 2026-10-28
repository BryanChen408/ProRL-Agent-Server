# Pooling Pattern Index

> 使用范围：实现知识参考，按本题需要查阅，不新增执行阶段或验收门禁。原文中的架构、shape、性能数据与参数仅适用于所述场景；以当前 SDK、reference 和固定评测入口为准。不得据此修改 reference/case、容差、标杆、计时参数，或另起评测。


转译 pooling 类算子（AvgPool/MaxPool/AdaptivePool 及反向 Grad）前先读本文件，用它决定需要的模式文档。前向是「滑动窗口 reduce」，反向是「scatter-add/gather」。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的类别，再只读对应文档；只有实现细节不够时，再继续读文档后续章节。

> 设计语义按当前 reference 核对；通用设计参考 `.claude/skills/ascendc-design-doc-generator/references/pooling-tiling.md`。本目录提供实现细节，不需要调用额外设计工作流。

## 核心模式

| 如果看到 | 读这个文档 |
| --- | --- |
| TileLang 设计无法高效执行 / 需要逐语义转译映射（`T.copy→DataCopy`、AUTO_SYNC→`PipeBarrier`） | [tilelang-translation.md](references/tilelang-translation.md) |
| TQue 每窗口 Alloc/EnQue/DeQue/Free 开销大 / 需要直连 TBuf | [ub-management.md](references/ub-management.md) |
| NDHWC 布局落地：host permute + kernel 行地址 + 整行 DataCopy（布局语义以 reference 为准） | [layout-implementation.md](references/layout-implementation.md) |
| 前向 KH==1&&KW==1 深度-only 快路径落地：direct TBuf + 完整 kernel + barrier 决策（快路径条件以本题窗口参数为准） | [reduce-d-fastpath-implementation.md](references/reduce-d-fastpath-implementation.md) |
| `UB address not aligned` 崩溃 / 输出值 512.0/nan / 需要 host `TORCH_CHECK` 对齐守卫 | [alignment-guards.md](references/alignment-guards.md) |
| fp32 累加 / divisor 三级策略 / bf16 下 Cast round mode | [precision-patterns.md](references/precision-patterns.md) |
| 反向 Grad 算子（AvgPoolGrad/MaxPoolGrad/AdaptivePoolGrad）实现：WAR 陷阱、output-driven transpose-scatter、`SetAtomicAdd`、六路 tiling key | [backward-implementation.md](references/backward-implementation.md) |
| 反向落地增量踩坑：`GetCoreNumAiv` vs `GetCoreNum`、`Axpy`、Cast 分派、data_format 四象限、延迟分桶 | [grad-v2-lessons.md](references/grad-v2-lessons.md) |
| kernel 实现常见错误：TQue 过载、per-ow barrier、DeQue in-place、双重屏障缺失、固定核数 | [pooling-anti-patterns.md](references/pooling-anti-patterns.md) |

## 按需组合

前向实现按需查看转译、布局、buffer、精度与快路径文档；反向实现查看 backward-implementation.md，再按实际问题查 buffer、尾块和 grad-v2-lessons.md。设计仍以当前 reference 和既有设计结果为准。

## 生成前问题

1. 前向还是反向？（reduce 窗口 → 单值，或 grad 单值 → 窗口）
2. 前向：KH==KW==1 吗？（→ reduce_d 快路径）窗口固定还是 adaptive？
3. 反向：AvgPool（只依赖 shape）还是 MaxPool（依赖 argmax）？
4. 布局：NCDHW 还是 NDHWC？C 维是否可向量化（`C%8`/`C%16`）？
5. 反向实现策略：input-driven gather（默认）还是 output-driven transpose-scatter（大 kernel/重叠）？
6. 是否有 `data_format` 属性、ceil_mode、count_include_pad、divisor_override？

根据答案选择上面的文档。累加精度依 reference 的语义选择，FP32 策略见精度参考；对齐条件用于选择快路径，合法非对齐输入仍需实现（见对齐参考）。
