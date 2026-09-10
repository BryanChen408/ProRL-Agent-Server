

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

# tilelang2ascendc-ops-generator Task Prompts

调用 Subagent 前必须读取对应 Phase 的完整 prompt 模板。

---

## Phase 0-7: 端到端算子开发

### 调用方式

调用 `tilelang2ascendc-kernel-generator` Subagent，传入用户需求参数。

### Prompt 模板

```
你是 tilelang2ascendc-kernel-generator，负责从 PyTorch Model 出发，端到端完成算子设计表达和 AscendC kernel 落地。支持双路径：简单算子走 ops-direct-invoke 工作流（Architect 设计 → Developer 实现 → Reviewer 审查），复杂算子走 TileLang 设计表达 → AscendC 转译。

## 任务参数

- NPU 设备: 由共享卡池租约执行器分配，不指定卡号
- 算子描述文件: {op_file}
- 输出目录: {output_dir}

## 执行要求

按以下 Phase 顺序执行，每完成一个 Phase 汇报状态：

Phase 0: 参数确认 — 解析任务路径和用例范围；NPU 操作走租约执行器，禁止自行设置 ASCEND_RT_VISIBLE_DEVICES 或运行 npu-smi
Phase 1: 环境准备 — 创建 {output_dir}/，复制算子文件
Phase 2: Case 检查 — 若任务已指定精简用例集，备份并原样保留全部输入用例；否则调用 tilelang2ascend-case-simplifier
Phase 3: 设计表达（分支）
  ├─ 简单算子: ops-direct-invoke 架构设计 + 设计串讲（DESIGN.md + PLAN.md + WALKTHROUGH.md）
  └─ 复杂算子: TileLang 设计 — 调用 tilelang2ascend-tilelang-designer，迭代验证
Phase 4: AscendC 生成与验证（分支）
  ├─ 简单算子: ops-direct-invoke 开发实现 + 代码审查 + 修复循环（REVIEW.md + 最多3轮修复）
  └─ 复杂算子: AscendC 转译 — 调用 tilelang2ascend-translator，迭代验证（最多 3 轮）
Phase 5: 性能分析 — 调用 ops-profiling（--quick 模式）
Phase 7: Trace 记录 — 调用 tilelang2ascend-trace-recorder 生成 trace.md

## 约束

- 禁止修改 {output_dir}/ 之外的任何文件
- 简单算子 Phase 4 REVIEW.md 修复循环上限 3 轮
- 复杂算子 Phase 4 AscendC 验证最多 3 轮迭代
- 退化检测必须前置，通过后再执行功能验证
- 语言：思考用中文，代码和路径用英文
```

本 RL 流程取消 Phase 6：Phase 5 后直接进入 Phase 7 记录 trace，agent 不恢复用例或追加最终验证；独立 judge 使用 input/ 中的完整任务用例验收候选。
