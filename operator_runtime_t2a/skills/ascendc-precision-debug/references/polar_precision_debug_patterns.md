# AscendC 精度与运行错误排查模式

本文用于候选 AscendC kernel 的定位和修复。正式结论仍以当前 Polar 固定评测入口为准。
辅助脚本只能帮助缩小范围，不能替换正式 reference、case、容差或对拍结论。

## 固定排查顺序

1. 先读取固定入口的 `metrics_error.log`、`verify.log`、`compile.log` 和错误分类。
2. 确认实际被评测的顶层任务目录、reference 快照、case JSON 和候选源码是同一轮产物。
3. 先检查输出契约：结构、数量、shape、dtype、device 和非有限值位置。
4. 再按失败形状归因：尾块/对齐、搬运、同步、未初始化、dtype/舍入、索引或数学公式。
5. 每轮只修一个有证据支持的根因，修改候选实现后重新走固定入口。

不要因为辅助复现更容易就替换正式 case；不要把辅助脚本的通过当作正式通过。

## 症状到根因

### 只在尾部或非对齐 shape 出错

- 检查 tile 长度、最后一块的有效元素数、GM/UB 行距和 DataCopyPad 的 padding。
- 确认 padding 值符合语义：sum 用 0，max/min 使用对应的单位值。
- 检查输出写回是否只写有效长度，不能把 padding 槽写进正式输出。
- 对 pooling、gather/scatter 等不同搬运布局分别计算对齐要求，不能套用另一条路径的守卫。

### 输出随机、跨 case 漂移或全零

- 优先检查 UB/GM 是否在读取前初始化，分配 buffer 不等于清零。
- 检查上一轮 tile 是否仍在被读取时被下一轮 CopyIn 覆盖。
- 检查 TQue 的 EnQue/DeQue、手动 SetFlag/WaitFlag 是否成对，event id 是否复用过早。
- 检查跨 pipe 数据依赖是否使用对应事件；单个 pipe 的 barrier 不能代替跨 pipe 可见性。
- 重复运行同一原始 case 只能用于确认确定性，不得因此修改 reference。

### FP16/BF16 误差随归约长度增长

- 先确认累加器是否升到 FP32，再按 reference dtype 转回。
- 检查归约顺序是否改变了误差量级；不能把正常的合法舍入差异误判成 bitwise 错误。
- 检查大均值、小方差场景的灾难性抵消；必要时采用两遍或稳定的统计方式。
- 检查 candidate 和 reference 的 NaN、Inf 符号及位置；非有限值不能用普通容差掩盖。

### Cast、量化或重复索引的值差一 ULP

- 先确认 reference 的 rounding、clamp、zero-point 和输出 dtype。
- 向更宽类型 Cast 时使用独立目标 buffer；源目标重叠可能覆盖尚未读取的数据。
- BF16/FP16 参与累加时，按 dtype 选择正确的 Cast round mode，避免逐次低精度舍入。
- 重复 index 的累加应保持 reference 的累加精度和写回语义；不因结果差异自行放宽阈值。

### 单次通过、重复运行失败

- 检查未初始化内存、双缓冲复用、跨核 atomic 顺序和事件残留。
- 单核与多核结果差异明显时，区分真实写冲突和浮点加法顺序差异。
- 若只是合法的多核舍入差异，仍以固定评测比较器和题目契约为准，不修改评测标准。

## 完整数值反例

- [BF16 固定偏置反例](/opt/workspace/agent_workdir/.claude/skills/ascendc-precision-debug/references/fixed_negative_bias_breaks_bf16_midpoint_rounding.md)
- [4097 拆分反例](/opt/workspace/agent_workdir/.claude/skills/ascendc-precision-debug/references/veltkamp_split_not_unconditional_rne11_quantizer.md)

## 证据边界

- 修复目标是候选 kernel、host tiling、buffer 生命周期、同步和 dtype 路径。
- 不修改 `model.py`、reference 快照、case JSON、prompt、比较阈值或固定评测脚本。
- 如果怀疑 reference 或 case 本身有问题，记录独立证据，停止在候选中“适配”它；该问题应由
  数据集维护流程单独审查和冻结。
- 资料中的硬件限制、阈值和 API 形式必须按当前 SoC、CANN 和 `ASC_DEVKIT_DIR` 核对，不能
  直接套用其他代际的实测数字。
