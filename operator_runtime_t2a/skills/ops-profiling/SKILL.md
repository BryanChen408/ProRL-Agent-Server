---
name: ops-profiling
description: 基于固定评测入口产出的真实逐 case 性能证据，定位 AscendC 瓶颈并实施一项通用优化；用于正确性通过但 speedup 尚未达到目标时。
---

# AscendC 性能诊断与优化

## 职责边界

本 skill 负责“读证据、改实现”，不负责另起一套计时流程。

- 唯一有效测速入口是工作目录 `CLAUDE.md` 规定的
  `/opt/workspace/agent_workdir/tools/ascendc_eval_pipeline.sh`。
- 不直接运行仓库底层 benchmark、msprof 或自建计时脚本。
- 不修改固定入口、评测脚本、参考实现和测试输入。
- 不用单个算子的特判换取分数；优化必须对同类 shape、dtype 和边界输入成立。
- 正确性没有通过时不得做性能优化，先按固定入口的 `next_step` 修复正确性。

## 调用条件

固定入口返回 `correctness_ok=true`，但 `perf_data.speedup_vs_torch < next_step.perf_target_speedup`（默认 `1.1`），且 `next_step.optimization_remaining > 0` 时必须直接 Read 本文件并按下述步骤执行。

调用时至少提供或读取：

1. `judge_out/metrics.json` 与 `judge_out/performance.json`；
2. `judge_out/perf.log` 中的逐 case 延迟；
3. 当前 `*_host.cpp`、`*_kernel.cpp` 及必要的头文件；
4. 固定入口给出的 `attempt`、`limit`、`remaining` 和 `next_step`。

缺少细粒度硬件计数器时，不得伪装成已经确认某一种 bound；只能把源码结构和逐 case 延迟支持的判断标为“优化假设”，并由下一次固定入口验证。

## 单轮优化流程

### 1. 找到主要损失

- 先确认当前正确实现和 `.best` 的状态。
- 比较全部有效 case，优先处理拖累整体 speedup 的慢 case，而不是只看最好 case。
- 结合 shape、dtype、尾块比例和源码，判断最可能的通用瓶颈类别。

### 2. 读取完整优化指导

每轮性能优化都读取
[optimization_quickref.md](references/optimization_quickref.md)，包括 Vector、MTE、Cube、
Scalar、负载均衡、Bank Conflict、DoubleBuffer、流水线、L2 Cache，以及 GroupedMatmul、
Matmul、FlashAttention 和 MC² 案例。当前算子看似简单也不跳过，避免后续融合算子或复杂
数据流被过早归类。

结合当前证据时优先使用最相关的章节：

- 多核切分、尾块或各核工作量差异：负载均衡；
- 搬运次数、非连续访问、UB 往返：MTE/内存；
- Vector 指令、Cast、循环或标量控制过多：Vector/Scalar；
- Cube 利用率或矩阵分块不合理：Cube；
- 流水重叠不足：双缓冲与流水。

如果工作区已经存在由受控评测流程产出的深度 profiling 结果，再按数据类型读取对应资料：

- 已有 `PROF_GROUP_*`、`op_summary_*.csv` 或 `task_time.csv`：读取
  [msprof-guide.md](references/msprof-guide.md) 的相关采集口径和 Bound 判定章节；
- 已有 `OPPROF_*` 或逐核 msprof-op 结果：读取
  [msprof-op-guide.md](references/msprof-op-guide.md) 的相关分析章节；
- 需要解释 `PipeUtilization.csv`、`ArithmeticUtilization.csv`、`Memory*.csv`、
  `L2Cache.csv` 或 `ResourceConflictRatio.csv` 字段时：读取
  [csv_fields_reference.md](references/csv_fields_reference.md) 的对应字段章节。

上述三份深度 profiling 手册仍以真实产物为前提：没有对应产物时不要为了“走流程”通读，
也不要绕过固定入口主动采集深度 profiling。完整优化速查表不受此限制。

### 3. 实施一项可验证改动

每轮只选收益预期最高、正确性风险最低的一项结构性改动，例如：

- 改善多核切分和尾块分配；
- 调整 tile/UB 大小，减少重复搬运；
- 合并相邻 CopyIn/CopyOut，保持连续访问；
- 在满足容量与同步约束时启用双缓冲；
- 减少不必要的 Cast、标量循环或重复中间结果。

改动必须保留预生成工程骨架、注册入口和公开函数签名。记录“证据 → 假设 → 改动”，不要堆叠多个无法归因的优化。

### 4. 有变化才复测

- 修改后确认 AscendC 源码内容确实变化，再运行一次固定入口。
- 若 `speedup >= next_step.perf_target_speedup`，停止优化并保留该正确实现。
- 若仍未达标且有优化预算，结合新结果继续按本文件指引优化。
- 若预算耗尽，停止调用工具并提交 `.best` 保存的最佳正确实现。

同一份未变化源码不得重复测速。固定入口是 attempt 计数和是否继续的唯一权威。

## 安全约束

- 禁止把计算搬回 CPU/host，或在 Python/host wrapper 中调用框架算子伪装 AscendC 实现。
- 禁止通过读取输出、硬编码 case 或放宽误差阈值获得 speedup。
- 新实现变慢或失去正确性时，立即以 `.best` 为基线继续，不要覆盖已知最佳正确版本。
