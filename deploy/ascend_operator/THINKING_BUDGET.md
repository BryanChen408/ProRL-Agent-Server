# 单次回复的思考预算（Polar 插件 / vLLM Ascend 0.23）

思考上限插件和训练 mask 在 Polar，vime 无需修改源码。
引擎通过 vLLM 已有的 `--logits-processors` 接口加载插件。
精确区分自然结束与强制结束还需要下述 vllm-023 / vllm-ascend-023 元数据补丁；
没有补丁时仍能限制思考，但会保守地 mask 所有结束标记。
仅在 HTTP gateway 设置参数无法强制结束思考，下面两个步骤都需要。

## 启用

1. 每个推理引擎节点的 Python 环境必须能导入 Polar 源码。例如在引擎启动环境设置：

   ```bash
   export PYTHONPATH=/home/docker/polar_can/ProRL-Agent-Server/src:$PYTHONPATH
   ```

   vime 训练启动参数增加（vime 已支持透传，无需改代码）：

   ```bash
   --vllm-logits-processors polar.gateway.vllm_thinking_budget:ThinkingBudgetLogitsProcessor
   ```

   若直接 `vllm serve`，使用去掉 `vllm-` 前缀的原生参数：

   ```bash
   --logits-processors polar.gateway.vllm_thinking_budget:ThinkingBudgetLogitsProcessor
   ```

   Qwen 保留 `--reasoning-parser qwen3`（vime 参数为 `--vllm-reasoning-parser qwen3`）。
   重启所有推理引擎后才会加载插件。远程节点/容器也需部署同一份 Polar 源码和导入路径。

   没有 Polar 仓库的节点可只复制 `src/polar/gateway/vllm_thinking_budget.py` 为
   `/workspace/vime/polar_thinking_budget.py`，使用
   `--vllm-logits-processors polar_thinking_budget:ThinkingBudgetLogitsProcessor`。
   此时所有推理节点均需同名模块，`PYTHONPATH` 包含 `/workspace/vime`。

2. 在使用的 profile（如 `profile.t2a.yaml`）中设置，然后重启 Polar gateway：

   ```yaml
   operator:
     agent:
       max_output_tokens: 49152
       thinking_token_budget: 32768
   ```

   loader 会把 `thinking_token_budget` 导出为 gateway 的
   `POLAR_THINKING_TOKEN_BUDGET`，profile 显式值覆盖同名环境变量。
   未配置或设为 `null` 时，loader 不设置该变量，仍支持原来的环境变量方式：

   ```bash
   export POLAR_THINKING_TOKEN_BUDGET=32768
   # 然后使用原有的 Polar 启动命令
   ```

未配置预算时生成行为不变；当前 t2a profile 已显式配置 32768。
`0` 表示立即结束思考，不是关闭开关。
关闭预算需删除 profile 中该项（或设为 `null`），并
`unset POLAR_THINKING_TOKEN_BUDGET` 后重启 gateway；插件可保留，
没有预算的请求不受影响。请求显式设置更小预算时保留更小值。

## 行为与适用范围

预算按每次模型请求计数，不是整个 session 的总预算。达到预算后，插件复用
vLLM 原生状态机输出 `</think>`，随后继续正文/工具调用，不改 Cannbot 提示词。
插件复用服务端 Qwen3ReasoningParser 的边界判断：`</think>` 自然结束、强制结束，
或直接出现 `<tool_call>` 隐式结束思考后，本次回复的预算干预都会退出；
不会在已开始的工具参数或正文中途插入 `</think>`。
总 output 上限仍由 `max_tokens` 控制，prompt + output 仍受模型上下文上限约束。
output=49152、思考预算=32768 时，给结束标记及正文/工具参数留下约 16384 token；
上下文不足仍可能提前截断。预算是否影响复杂题正确率需通过实际任务对比验证。

当前针对 Qwen / Ascend V1 / **推理 PP=1**，支持 async scheduling。
不支持推测解码、推理 PP>1 或 `enable_reduce_sample=true`，配置冲突会明确报错。
这里的 PP 是推理引擎 PP，不是 Megatron 训练 PP。

每个请求单独持有上游状态机，批次换位由原生 AdapterLogitsProcessor 管理，
避开了之前需要修复的跨请求 holder 换位路径。
插件还在引擎进程内为 NPUInputBatch 的 sampling metadata 安装一个幂等小钩子：
Ascend 0.23 重建混合 KV 输入批次时会丢失输出 token 回填开关，需要为有预算的请求
保留异步回填。钩子只随插件加载生效，不写入或修改第三方源码文件。

训练保留原始 token IDs 和 logprobs。安装元数据补丁并更新插件后，引擎在
真正修改 logits 时记录强制 token，由 scheduler 核对实际采样结果，再按请求
输出位置返回 `choices[i].thinking_budget_forced_token_indices`。
位置从 0 计数，包括原始思考、正文和工具 token，流式响应在该 choice 的结束块返回累计位置。
Polar 只对这些强制位置设 loss_mask=0，自然生成的 `</think>` 恢复基线训练规则。
空数组 `[]` 明确表示未强制；字段缺失或 null 表示无记录能力，仍保守地 mask 所有 `</think>`。
无效位置（越界、重复、非整数、对应 token 不是结束标记）拒绝构建训练 trace。
其余 CoT、正文和工具 token 的训练口径不变，原有 reasoning 和整段截断 mask 仍生效。
训练 rollout 与评测/部署推理应使用相同的预算、插件和解析器配置。
此处保护 token、logprob 和上下文对齐，不保证模型本身始终生成合法工具参数、
充分完成推理，或在剩余 output/上下文预算内完成整个工具调用。
completion.response._polar_thinking_token_budget 记录实际发送预算；训练 trace 的
metadata.thinking_budget_loss_mask 记录预算及结束标记位置。

## 精确强制位置的引擎补丁

补丁只增加归因元数据，不改变预算、logits 的强制值、采样参数、token IDs 或 logprobs。
Ascend runner 在每次采样后按 request ID 快照归因，避免异步步进和 batch 换位污染记录；
跨进程输出及流式聚合保留位置。当前精确采集接在 Ascend V1 runner，仍限推理 PP=1、非推测解码。

本次开发工作区的两个引擎仓库已应用补丁；其他推理容器需部署同一版本。
把本目录的 `patches/` 复制到目标容器后，先检查再应用（下面路径替换为实际复制位置）：

```bash
git -C /workspace/vllm-023 apply --check /path/to/patches/vllm-023-thinking-budget-attribution.patch
git -C /workspace/vllm-ascend-023 apply --check /path/to/patches/vllm-ascend-023-thinking-budget-attribution.patch
git -C /workspace/vllm-023 apply /path/to/patches/vllm-023-thinking-budget-attribution.patch
git -C /workspace/vllm-ascend-023 apply /path/to/patches/vllm-ascend-023-thinking-budget-attribution.patch
```

已应用的节点不要重复应用；检查失败应先核对版本差异，不能强制覆盖其他改动。
同时更新所有节点的插件文件和 Polar 的训练 mask 代码；停止旧任务后，统一重启
所有推理引擎和 Polar gateway / rollout 服务。部署推理也使用同一套预算与插件。
仅复制插件不能启用精确归因；仅有部分补丁/旧插件的组合不要用于新一轮训练。
验收应在落盘 completion 中看到上述字段，并核对强制样本的位置及自然结束的空数组。

## 验证

从 Polar 仓库运行（无须加载模型；NPU 编号应选空闲卡）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 POLAR_TEST_NPU=1 \
PYTHONPATH=src:/workspace/vllm-023:/workspace/vllm-ascend-023 \
/workspace/vllm-023/.venv/bin/python -m pytest -q tests/gateway tests/trajectory
```

未设置 `POLAR_TEST_NPU=1` 时跳过真实 NPU 检查；未安装 vLLM 时跳过插件测试。
覆盖预算缺省/零预算、自然结束、批次换位/复用、异步回填、到限后继续输出，
以及预算落盘和训练 mask。NPU 检查验证真实设备上的 logits 处理，
不替代完整模型解题率和训练回归验证。

t2a profile 已配置 32768 思考预算；未修改训练启动脚本，也未重启现有服务。
