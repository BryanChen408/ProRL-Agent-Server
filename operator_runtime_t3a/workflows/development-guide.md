

## T3A 执行契约（优先于下文通用模板）

- 主 agent 只调度和检查；禁止主链初始化工程或写实现。只派发已注册的
  `tilelang2ascendc-kernel-generator`，其他设计/转译组件用 Skill 调用，不能当 Agent 类型。
- 每次派发/恢复必须原样传递 input 参考路径、用例策略、输出根目录、租约策略和取消末尾验证阶段的策略。
- 开发子 agent 按自身 Phase 1.2 创建 `{output_dir}/kernel/`，复用已安装 project-init
  模板中的固定文件；不要执行 standalone project-init 的 `ascend-kernel/csrc/ops/` 布局。
  model.py、model_new_tilelang.py、model_new_ascendc.py 和 design/ 均位于同一个 output_dir。
- simple 数据集保留输入的全部 5 条用例，不调用 case-simplifier；simple 不是算子分类。
- TileLang 验证是中间步骤，不能代替 Phase 4 AscendC 实现和 evaluate_ascendc.sh。
  子 agent 提前返回时，从未完成阶段恢复同一开发子 agent；禁止把 stub 或 TileLang 产物宣称为完成。
- 完成前读取真实 AscendC 评测结果和 trace.md，并检查 kernel/ 与 model_new_ascendc.py。
  失败按真实失败报告；独立 judge 负责最终验收，不能拿 judge 替代开发阶段。
- 本 RL 场景 pipeline/verify 不按次数终止，统一受外部 pipeline 时间预算约束。
  下文重试数字仅用于诊断分阶段建议，不构成次数上限；不得因达到次数提前转 Phase 7 或请求用户。
  同错反复出现应更换有证据的修复策略；D1/D2 的诊断和文档门禁继续保留。
- 禁止自行设置设备可见性；NPU 调用仅通过现有租约评测脚本。

# AscendC Ops Developer 开发指南

## 概述

本 Team 实现从 PyTorch Model 到 AscendC Kernel 的端到端自动开发流程。

## 工作流架构

```
用户输入 (npu, op_file, output_dir)
    │
    ▼
CANNBot Primary Agent (AGENTS.md)
    │  解析参数、调度 Subagent
    ▼
tilelang2ascendc-kernel-generator Subagent
    │
    ├── Phase 0: 参数确认 + 算子分类（简单/复杂）
    ├── Phase 1: 环境准备
    ├── Phase 2: Case 精简 (tilelang2ascend-case-simplifier)
    ├── Phase 3: 设计表达（分支）
    │     ├─ 简单: ops-direct-invoke 架构设计 + 设计串讲
    │     └─ 复杂: TileLang 设计 (tilelang2ascend-tilelang-designer) ← 迭代循环
    ├── Phase 4: AscendC 生成与验证（分支）
    │     ├─ 简单: ops-direct-invoke 开发实现 + 代码审查 + 修复循环（3轮上限）
    │     └─ 复杂: AscendC 转译 (tilelang2ascend-translator) ← 迭代循环（3轮上限）
    ├── Phase 5: 性能分析 (ops-profiling --compare 模式)
    └── Phase 7: Trace 记录 (tilelang2ascend-trace-recorder)
```

## Phase 3-4 迭代机制

### Phase 3: TileLang 设计迭代

```
generation → AST退化检测 → [通过] → 功能验证 → [通过] → 完成
                ↓ 失败                        ↓ 失败
            Conductor分析 → 生成修复建议 → 重新生成
                                     ↑
                        最多 5 轮迭代 ─┘
```

退化子类型：
- Type1: 无 TileLang kernel 导入（纯 PyTorch）
- Type2: 有导入但 forward() 未调用
- Type3: 调用 kernel 但部分计算仍用 PyTorch
- Type4: forward() 中存在逐元素 for 循环

### Phase 4: AscendC 转译迭代

```
generation → AST退化检测 → [通过] → 功能验证 → [通过] → 完成
                ↓ 失败                        ↓ 失败
            Conductor分析 → 生成修复建议 → 重新生成
                                     ↑
                        最多 3 轮迭代 ─┘
```

退化子类型：
- Type1: 无 AscendC 扩展导入（纯 PyTorch）
- Type2: 有导入但 forward() 未调用 kernel
- Type3: 调用 kernel 但部分计算仍用 PyTorch
- Type4: forward() 中存在逐元素 for 循环

## 错误分类

| 分类 | 含义 | 处理 |
|------|------|------|
| A 类 | 代码逻辑/算法错误（可修复） | 生成修复建议，继续迭代 |
| B 类 | 环境/基础设施错误（不可修复） | 立即终止 |
| C 类 | 同一 A 类子类型连续 ≥ 3 次 | 立即终止 |

## 产出物

```
{output_dir}/
├── model.py                  # 算子描述（只读）
├── <op_name>.json            # 原始测试用例文件（备份保留）
├── <op_name>.json.bak        # 原始 .json 备份
├── design/
│   ├── block_level/          # Block-level 设计
│   └── tile_level/           # Tile-level 设计
├── kernel/                   # AscendC kernel
├── model_new_tilelang.py     # TileLang 实现
├── model_new_ascendc.py      # AscendC 实现
├── preformance.json          # 性能数据
└── trace.md                  # 执行记录
```

## 关键约束

- TileLang 仅用于设计表达，不作为 correctness/performance gate
- model_new_*.py 中禁止使用 torch 算子
- 必须将核心计算融合成单个算子
- 文件操作范围限制在 {output_dir}/ 内
- 优先使用块级/向量化操作，避免标量逐元素写法

本 RL 流程取消 Phase 6：Phase 5 后直接进入 Phase 7 记录 trace，agent 不恢复用例或追加最终验证；独立 judge 使用 input/ 中的完整任务用例验收候选。
