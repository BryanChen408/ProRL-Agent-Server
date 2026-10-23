# AscendC 算子实现知识补充

这是只读知识资料，不是新的 Skill、评测入口或工程初始化流程。实现任务仍按
`CLAUDE.md`、现有 Skill、预生成骨架和固定评测入口执行。

资料已经整理并固化在对应 Skill 的 `references/` 目录；下面的路径只是来源记录，训练或解题时不需要访问 cannbot
仓库，也不依赖其目录存在。原始来源是本机 cannbot 仓库 `97ee3d81` 的：

- `community/ops/tilelang2ascend-tilelang-designer/references/design-patterns/`
- `community/ops/tilelang2ascend-translator/references/ascendc_*_patterns.md`
- `ops/ascendc-tiling-design/references/index/`

## 使用边界

- `model.py`、JSON case、prompt、reference 快照、比较阈值和评测脚本是题目契约，不修改。
- 资料只用于选择计算结构、tiling、数据搬运、同步和 dtype 处理；不能用来改变题目语义。
- 不因为某条实现路径有对齐限制就拒绝原题合法 case；需要在 kernel 中处理尾块，或选择支持该
  case 的实现路径。
- 不修改 golden/reference 来迁就硬件输出，不自行放宽容差，不把候选实现接到 reference 中。
- 正式编译、对拍、状态检查和测速仍使用当前 Polar 固定入口。文档中的独立实验只能辅助定位，
 不能替代正式结论。
- 资料中的 SoC、API 重载、核数、UB 大小和阈值都要以当前 `npu-arch`、`ASC_DEVKIT_DIR`
 及实际工具链为准；不能把 950 或其他代际的数值直接套到 910B。

## 按算子选择资料

先读 `polar_operator_patterns.md` 的对应章节，再按需要查当前 Skill 中的 API 和调试资料。不要一次
通读整个资料包。
