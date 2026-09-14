# 本环境覆盖(优先级高于以上全部内容)

以上内容与本节冲突时,一律以本节为准。

## Skill 只作为本地知识文件

本 profile 禁用了 `Skill` 工具。任何“调用某个 skill”的上游表述，一律解释为先用 `Read`
打开 `.claude/skills/<skill-name>/SKILL.md`，再按其中明确引用的相对路径读取必要的
`references/` 文件。不要尝试调用 Skill，也不要等待 Skill 返回内容；读完后由当前会话直接
检查和修改工程。路径不确定时先列出 `.claude/skills/`，不要猜路径。

首次设计或找参考实现时，先 Read `.claude/workflows/cannbot-reference-index.md`。
它列出 cannbot 原版设计资料、host/kernel 模板及 archive 的真实文件名；只读本题相关条目，
已读且未变化的资料不重复读。旧资料仅提供实现参考，其中的调度、初始化与评测指令不适用。
Read 找不到时先 Glob 所在目录，再按真实文件名 Read；不要连续猜 `_kernel.cpp` 等后缀。

新增的两个目录是纯资料包，没有 `SKILL.md`，直接从以下文件读取：
- `.claude/skills/ascendc-design-doc-generator/templates/design-template.md`
- `.claude/skills/ascendc-code-gen/references/GUIDE.md`

以上 `.claude/...` 路径均相对会话工作目录 `/opt/workspace/agent_workdir/`，
不要使用宿主仓库的 `operator_runtime_t2a/skills/` 路径。两包的其他资料在各自
`references/`、`templates/` 下，具体文件名见上述索引；目录本身用 Glob，不用 Read。

## 开发源码只读范围

除工作区和 `$ASC_DEVKIT_DIR` 外，允许只读查阅**当前评测工具链**的 CANN 头文件、
实现源码、CMake 定义、官方 OPP 源码，以及当前 Python 安装的 TileLang 包源码和仓内示例。
只在文档不足、类型/签名不明或修复具体错误时读取，不重复探索已确认的目录。
先按 `.claude/workflows/cannbot-reference-index.md` 的「当前工具链源码」步骤定位，
复用 `tools/env.sh`，以实际存在的路径为准；不照搬历史版本目录、不搜索其他 SDK 版本。
这允许 Read/Grep/Glob 诊断，不允许改写 SDK/库文件、安装包、导入未知示例或执行其中的
构建/测试脚本。评测、设备选择、精度和预算仍由固定 pipeline 管理。

## 预生成骨架是唯一工程起点，不是算子语义契约

Polar prepare 在 Agent 启动前已经读取本题 `model.py` 的 `__init__` / `forward`，并在
`{output_dir}/` 中预生成当前算子的工程骨架。开始实现前先检查并复用这些现有文件：

- `model_new_ascendc.py`
- `kernel/CMakeLists.txt`、`kernel/setup.py`
- `kernel/ops.h`、`kernel/register.cpp`
- `kernel/op_host/{op_name}.cpp`
- `kernel/op_kernel/{op_name}_kernel.cpp`
- `kernel/utils/`

这是本环境唯一允许的工程初始化来源。不要调用或寻找额外的工程初始化/直调 skill，不要
复制其他任务或模板来重建工程，也不要把 device 文件另写成
`op_kernel/{op_name}.cpp`。

- 简单算子只跳过 TileLang，不跳过设计；由 `tilelang2ascend-translator` 读取 `model.py`，
  复用已有 cannbot 设计模板和知识文件，在主轨迹内完成设计与审查后再补全骨架。
- 复杂算子先形成 TileLang block/tile 设计，再由同一个 translator 在现有骨架上完成实现。
- `CMakeLists.txt`、`setup.py`、`kernel/utils/` 和 `model_new_ascendc.py` 的双路径 loader 是
  机制件，默认原样保留；不要为了“初始化”或统一风格而改写。
- prepare 生成的签名件通常已经彼此一致。如果真实 reference 接口要求变化，可以同步修改
  `model_new_ascendc.py` 调用、`register.cpp` schema、`ops.h` 声明和 `op_host` 实现；这属于
  必要接线，不属于重建工程。
- 必需文件确实缺失或损坏时，只在原路径结合 `model.py` 与相邻文件原位修复缺失部分，不要
  因一个文件缺失推倒整套骨架。

**唯一语义依据是原始 reference 和原版用例，不是骨架、算子名或历史实现。** 必须读取
`input/{op_name}.py` 的 `Model.__init__`、完整 `forward` 及其调用的辅助逻辑，核对
`get_inputs/get_input_groups/get_init_inputs` 中实际存在的入口与关联用例；不要把多步计算
缩减成文件名中的一个操作，也不要忽略子模块参数、返回结构或分支。

骨架中的 `empty_like(首个输入)`、fp16/fp32 限制、连续性检查、按字节数分发 dtype、
单输出接线、恒等拷贝与 elementwise tiling 都是**待改写占位**，不保证符合本题。
按 reference 修正输出 shape/dtype、全部输出、布局、累加精度、buffer 与尾块；BF16/整数
支持不能只删 host 检查，必须同步修正 kernel 类型与访存。静态签名提取不等于已实现
`__init__` 中的子模块与参数语义。模板或历史文档中的芯片示例以本环境 `SOC_VERSION` 为准，
其中的子代理、用例精简、独立构建/评测指令不适用，仍服从本节固定入口与预算。

## 固定入口

任何让你运行 `evaluate_ascendc.sh`、`validate_ascendc_impl.py`、
`msprof_profile_run.sh`、`msprof_perf_summary.py`、`verification_ascendc.py` 的地方,
一律改跑这一条 —— 包括本文档以上各 Phase、以及本地 SKILL.md 中读到的指示:

```bash
bash /opt/workspace/agent_workdir/tools/ascendc_eval_pipeline.sh --op_name {op_name} \
     --impl output/submission/{op_name}_impl.tar.gz --out_dir judge_out
```

它一次完成:退化检测 → 编译 → 对拍 → 测速 → 写 `judge_out/metrics.json`,
并自动把 `{op_name}/` 打包成提交物、保留历史最优版本。

- 每轮有效源码修改后跑一次；源码未变化时禁止重复运行。被中途截断时按历史最优版本判分。
- “有效修改”必须改变计算、访存、同步或必要接线；添加注释、时间戳、改文件名来触发 hash
  不算修复。收到“源码未变化”后先核对输出中的实际检查目录与本轮修改文件，再修改实现；
  不删除缓存状态来强迫复测。不要把 pipeline 输出管道接到 `head`（会提前关闭输出管道）。
- 迭代时可加 `--incremental` 复用上次解包目录,走增量编译。
- 它评的是 **AscendC 提交物**。Phase 3 的 TileLang 阶段还没有 AscendC kernel,
  那时跑它只会得到 `submission_missing` 并白白消耗一次评测配额。
- 不要另跑 `cmake` / `make` / `python setup.py`,也不要直接调 skill 里的
  AscendC 评测/对拍/测速脚本 —— 绕过它就没有基准复位、缓存检测和抢卡,结果不作数。
- 允许独立调试脚本做 CPU 语义、索引、dtype 对照和已有输出分析；候选 kernel 的
  编译与 NPU 运行仍走固定入口。不得修改、删减或替换正式评测用例、reference、
  输入生成逻辑和精度阈值，不得将自写测试接入正式评测或用其结果宣称评测通过。

## Phase 3 的 TileLang AST 门禁

- 复杂算子每次生成或修改 `model_new_tilelang.py` 后，都必须直接运行不占卡的
  `validate_tilelang_impl.py`：
  ```bash
  python3 .claude/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py \
      {output_dir}/model_new_tilelang.py
  ```
- 首次生成计为第 1 次候选，最多允许 3 份候选。AST 失败时只按脚本的
  `regression_type/suggestion` 修复；同一份未变化的候选禁止重复检查。
- 第 3 次仍失败时，丢弃未通过 AST 的 TileLang wrapper，Phase 4 直接使用 `model.py` 与
  Polar 预生成骨架继续求解，不能因为中间表示失败浪费整条 session。
- `verification_tilelang.py` 是占卡诊断工具，不属于自动 pipeline。只有后续错误明确需要验证
  TileLang DSL 时才允许经 NPU lease 手工调用；其结果不作为最终 correctness/performance gate。

## 本环境只有完成门禁 Hook

上游的 `skill_script_hook.py` 命令拦截机制仍然**未启用**。你的 Bash 调用一律直通执行，
不会被托管代跑；凡上游文中说“由 Hook 拦截/代为执行”的脚本，仍一律改跑上方固定入口。

本环境只安装一个 `Stop` 完成门禁。Polar 启用评测预算时，它向 gateway 只读查询完成状态；
gateway 根据宿主侧记录的固定入口调用结果与服务端预算判定，本地
`judge_out/task_state.json`、`metrics.json` 只作提示，修改它们不能改变完成判定。
gateway 查询失败时不按本地文件放行，agent 退出后服务端还会再次核验。
生成阶段尚未得到正确实现时，普通编译、精度或性能评测失败不能提前结束；
按 gateway 返回的 `generation_remaining` 继续修复，直到生成预算耗尽或明确 INFRA。
生成预算耗尽只表示允许结束失败任务，不表示已经解对；首次正确后转入独立优化预算。
独立运行且未配置 gateway 完成检查时才保留原本的本地状态检查。
门禁不执行评测、不替换 Bash、也不修改任何文件。当历史最佳实现已经
正确但 `task_complete=false`（性能目标未达且仍有优化预算）时，即使当前优化候选编译或运行
失败，它仍会拒绝结束；只有达到目标、预算耗尽，或当前属于 INFRA 时才放行。不要尝试绕过
或修改这个门禁。

`judge_out/`、`output/.selfcheck/`、`.best` 及其元数据均由固定入口维护，只读。
禁止修改或删除 `task_state.json`、`metrics.json`、budget/hash 状态来结束任务或重置次数；
也不要清空 `input/`、`output/`、整个 `judge_out/` 来“修编译缓存”。
`success=true`、已有 `.best`、自认为无法再优化，都不能替代 `task_complete=true`。
Stop feedback 要求继续时，下一轮执行一个新的诊断或实质修改，不重复提交最终总结。

- 保持计算在自定义 AscendC kernel。`at::add/sum/cumsum/repeat`、
  `at_npu::native::custom_ops::*` 等框架计算不能作为修复或性能优化的替代；
  保留一个未被调用的 kernel、实际返回框架结果，同样不符合要求。历史 PASS 不能豁免此约束。

本环境没有插件的 SessionStart 上下文注入。执行所需的路径、环境事实和流程约束已经写在
当前 `CLAUDE.md` 中，不要等待 hook 补充，也不要把“可能会被 hook 接管”作为跳过步骤的理由。

## 错误分类

固定入口输出末尾的 `错误分类:` 行是分类的权威来源:

| 分类 | 处理（全部直接 Read，不调用 Skill） |
|---|---|
| `通过` | 仅表示 `operator_valid=true`；严格按 `task_complete/next_step` 进入优化或结束 |
| `A类-提交物/AST/编译` | 读 `metrics_error.log` 与 `tilelang2ascend-translator/SKILL.md`；API 不明再读 `ascendc-docs-search/SKILL.md` |
| `A类-注册/加载` | 读 `ascendc-runtime-debug/SKILL.md` 和 `references/kernel_binary_debug.md` |
| `A类-崩溃/超时/启动失败` | 读 `ascendc-crash-debug/SKILL.md`；ACL 错误码再读 runtime-debug 的 `error_codes.md` |
| `A类-输出契约/状态化退化` | 严格按固定入口给出的 translator/precision-debug 文件路径修，不得冒充 D 类 |
| `D类-精度不匹配` | 仅当正常运行、输出契约正确且存在数值差异字段时进入；先读 precision-standard，再读 precision-debug |
| `A类-benchmark执行失败` | 读 `ops-profiling/SKILL.md`；若是 kernel/ACL 崩溃改走运行期路线 |
| `B类-INFRA` | **不要改 kernel、不要读取修复文档**,直接停止并说明 |

[A1] 照做:asc-devkit 就在 `$ASC_DEVKIT_DIR`。完整错误在 `judge_out/metrics_error.log`,先读它。

## 用例精简已移除(原 Phase 2 / Phase 6)

判分**恒用数据集原版用例**:`{op_name}/{op_name}.json` 在每次评测(含你的自检)时都会被
`input/` 里的数据集原件覆盖,改它、精简它、备份它都**无效**。上游的 Phase 2(用例精简)与
Phase 6(全量恢复)因此已从工作流移除。不要自行精简、修改或备份任何 `.json` 用例文件,
把轮次留给实现与修复迭代。

## 提交物

- 唯一提交物:`output/submission/{op_name}_impl.tar.gz`,由固定入口自动打包。
- 工程目录必须是工作目录顶层的 `{op_name}/`,不要另建别名或带时间戳的目录。
- `{op_name}/model.py` 与 `{op_name}/{op_name}.json` 判分时会被数据集原版覆盖,改它们无效。
- 不要删除或移动 `output/submission/` 下的任何文件。

## 性能目标

- 统一主指标：`judge_out/performance.json` 的 `geomean_speedup`，即各有效用例相对 PyTorch reference
  加速比的几何平均值。
- 加速比 **≥ 1.1x** PyTorch reference → 达标；低于 `1.1x` → 未达标。
- `operator_valid=true` 只表示实现正确；只有 `task_complete=true` 才表示任务允许结束。
  固定入口提示未达标且仍有 optimization 预算时，必须先 Read
  `.claude/skills/ops-profiling/SKILL.md`，再读取真实逐 case 结果和当前 kernel，实施一项有证据的
  通用性能改动。只有源码内容确实变化后才能重跑固定入口。
- `task_complete=false` 时 Stop 完成门禁会拒绝总结；按 `next_step.action` 继续即可。
- 达到 `1.1x` 后停止性能迭代；预算耗尽则停止调用工具，并使用固定入口保存的 `.best` 最佳
  正确实现。`1.1x` 只决定性能目标状态，不改变正确性判定，也不能成为不提交 tarball 的理由。
- attempt、limit、remaining、next_step 只服从固定入口输出。本文档或 skill 中残留的其他固定
  轮数均不作数，禁止另建一套 A/D/性能计数器。

## 环境事实

- `asc-devkit` 挂在 `$ASC_DEVKIT_DIR`(= `/opt/asc-devkit`),与本机 CANN 同版本。
  API 文档在 `$ASC_DEVKIT_DIR/docs/zh/api/`,示例在 `$ASC_DEVKIT_DIR/examples/`,
  实现代码在 `$ASC_DEVKIT_DIR/impl/`。写 kernel 前用 `ascendc-docs-search` 查 API 签名与
  命名空间,不要凭记忆写。
- 文档里凡写 `asc-devkit/...` 的地方,一律读作 `$ASC_DEVKIT_DIR/...`。
- 示例入口见 `.claude/skills/ascendc-docs-search/references/example-catalog.md`；
  目录随版本变化，先列实际目录再读取，不照搬历史绝对路径。
- 编译报 `no template named 'TQue'` / `did you mean 'AscendC::...'` 这类,是命名空间或签名
  记错,查 `$ASC_DEVKIT_DIR/docs/zh/api/` 核实,不要靠猜改。
- 检索文档一律从 `$ASC_DEVKIT_DIR` 根目录搜(Grep/Glob 的 path 填根目录),不要凭记忆
  猜子路径或假定存在 `docs/api/`；API 根目录搜索无结果时再扩大到整个资料仓；
  搜不到时换关键词(去后缀、换同义词)，再按「开发源码只读范围」查实际 SDK 定义，
  不要直接下「API 不存在」的结论。

## 禁止

- 自行设置 `ASCEND_RT_VISIBLE_DEVICES`。
- 运行 `npu-smi` 或任何探测 NPU 的命令。`SOC_VERSION` 已在环境变量里。
- 修改或删除 `tools/` 与 `.claude/skills/*/scripts/` 下的脚本。允许用 Read/Grep
  只读检查脚本，定位输入契约、参数和报错；不允许复制改写后执行或绕过固定入口判分。
- 修改评测参数(SOC_VERSION / warmup / repeats / 精度阈值)。
- 向用户提问。这是非交互运行,没有人会回答。
