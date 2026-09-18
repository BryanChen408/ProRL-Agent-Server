# Vime + Polar session partial rollout（最多落后一轮）

实现位置：Polar `/home/l00830933/ProRL-Agent-Server`；Vime `/workspace/vime`。
**无需修改 vLLM 源码或增加 vLLM 启动参数。** 本次此前引入的 vLLM 协议补丁已撤回。
不需要复制历史 session 数据。

## 续接语义

以一个 group 的 8 条 session 为例：7 条已完成，第 8 条完成 43 轮模型调用，正在生成第 44 轮。

1. 保留 7 条完整 session 及其 reward，继续放在同一个 group 的缓冲区。
2. 保留第 8 条 session 的 agent 进程、工具状态、前 43 轮 prompt/response/真实 logprob。
3. 关闭 Gateway 新推理入口，取消第 44 轮尚未完成的上游 HTTP 请求。
   第 44 轮已经生成但尚未完整返回的内容全部丢弃，不进入历史、工具执行或训练数据。
4. Vime 调用 vLLM 原生 `/pause?mode=abort&clear_cache=false`，暂停并中止引擎请求；
   同时确认 Gateway 在飞请求已清理、重发请求已保存。两项确认都完成后才进入训练。
5. 权重同步并恢复引擎后，Gateway 优先原样重发第 44 轮 Chat Completions 请求：
   输入仍为前 43 轮完整历史（含 reasoning、tool calls、工具观测）及第 44 轮 prompt。
   不附加被丢弃的第 44 轮 token，不重跑前 43 轮工具。
6. 第 8 条 session 正常完成后，凑齐同组 8 条有效终态，再作为完整 group 进入后续训练。
   最多允许落后一轮；若到下一次更新边界仍未完成，整个过期 group 按原规则清退。

只有 session 最终正常结束才计算其 reward，已完成 session 的 reward 不因暂停重新计算。
前 43 轮旧策略 action 和恢复后的新策略 action 都参与训练，并保留各自实际生成时的 logprob
用于 TIS；不按版本 mask。prompt、工具观测仍使用正常的 loss mask。

重发的是一次全新完整的模型调用，max_tokens、stop、采样配置与原请求相同。
例如上限 32K，重发仍是 32K；此前丢弃的生成会产生额外计算开销，但不占最终回复额度。
**真实达到 32K 的 `finish_reason=length` 不会被视为计划暂停，不会因此自动重试。**

## 调度和训练边界

- `train.py` 交替执行 rollout、训练、权重同步；支持共卡和独立推理卡的同步训练。
- 使用持久化 `generate_rollout_polar_async` session_pool；async 指 agent 调度。
  不支持并发修改 serving 权重的 `train_async.py`。
- 默认每步 16 个完整 group，每组 8 条有效终态 session，共 128 条；最多 64 条并发，
  最多持有 24 个未消费 group。算子生成失败但正常得到 reward 的 session 仍是有效终态；
  基础设施失败不能用零 mask 占位轨迹补足 group。
- 优先重发被中止的请求，然后开放新任务 admission。未提交完的旧 group 先补齐，
  ready buffer 优先消费旧 group。已完成 session 不重新生成。
- 候选训练数据按所有保留 action 的最老实际 policy version 检查，最多落后 1。
  在飞 group 清退保守地使用创建版本，避免缺少部分 token 元数据时放过陈旧数据。
- 一次 policy epoch 必须恰好对应一次 optimizer step；wrapper 默认 global_batch_size=128。
- Gateway 取消 HTTP 和 vLLM 引擎原生 pause/abort 是两道屏障：前者保住 session 并结束
  等待响应头/半截响应的请求，后者确保训练前引擎停止生成。不能仅凭本地 HTTP 已关闭就训练。
- 已完整返回的请求保留；仅 Gateway 在这次计划边界明确取消的请求允许重发。
  传输异常、非计划 abort、缺失 logprob 仍按真实错误处理。

## 启用

所有 rollout/gateway 进程使用更新后的 Polar，trainer 使用更新后的 Vime。
vLLM 和 DP/PD proxy 沿用原有代码及启动配置。本次没有启动或重启运行中的服务。

Gateway 和 rollout server 使用原有启动命令，无需额外设置 partial 环境变量：

```bash
export PYTHONPATH=/home/l00830933/ProRL-Agent-Server/src:${PYTHONPATH:-}
polar serve_gateway -c "$POLAR_TOPOLOGY" --node-id "$POLAR_NODE_ID"
```

rollout server：`polar serve_rollout -c "$POLAR_TOPOLOGY"`。
沿用实际 topology/profile、模型、工具和算子数据路径。

Vime 入口：

```bash
bash /workspace/vime/scripts/start_partial_rollout.sh
```

wrapper 设置：

```text
TRAIN_ENTRY=train.py
POLAR_PARTIAL_ROLLOUT=1
POLAR_POLICY_TRANSITION_ENABLED=1
FEAT_OFFLOAD=1
POLAR_MAX_OFF_POLICY_STEPS=1
POLAR_MIN_COMPLETE_ACCEPT_FRACTION=1.0
ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 GLOBAL_BATCH_SIZE=128
POLAR_MAX_ACTIVE_SESSIONS=64 POLAR_MAX_OWNED_GROUPS=24
```

底层为 `--polar-partial-rollout --rollout-max-off-policy-steps 1 --get-mismatch-metrics`
以及 `--use-tis`。`--use-rollout-logprobs` 保持 false，禁用按版本 mask。
不依赖 Vime 原生单样本 `--partial-rollout` buffer。

Vime 在首次权重同步前，通过 bootstrap 显式发送 `partial_rollout` 和协议版本
（partial 为 2，普通模式为 0）。Polar 先关闭准入并暂停所有 Gateway，再配置运行模式；
只有 topology 中所有节点确认实际模式和 namespace 后才返回 bootstrap 成功。
旧版 Polar 没有这项协商能力时，新版 Vime 会明确报错，需要同步升级协调器和 Gateway。

Checkpoint 默认位于 `<rollout.save_dir>/partial_rollout/<namespace>/<node>`，路径分量使用
安全名称和身份哈希以避免碰撞。可选 `POLAR_PARTIAL_CHECKPOINT_DIR` 覆盖根目录，
自动配置仍会追加 namespace/node；若未配置 save_dir，必须提供这个根目录。
路径不可写、引擎不是 vLLM、节点不可达或仍有旧 session 时，准入保持关闭。
`POLAR_PARTIAL_ROLLOUT=1` 加 checkpoint 目录的旧式手动配置仍兼容旧客户端。

模式固定在一次 policy namespace 内，重复 bootstrap 幂等；不同 trainer 不可接管正在
serving 的运行。先完成现有 `/rollout/admin/policy/quiesce` 流程，再启动新 namespace，
即可从 partial 切换回普通模式（Vime 正常 dispose 会执行 quiesce）。模式和 checkpoint
路径随控制状态持久化，Gateway 重启后恢复配置；这不代表恢复崩溃前的 agent session。
新注册或重启节点只有确认当前运行配置和 policy epoch 后才可重新参与调度。

**只有 Polar 协调器和 Gateway 需要声明 `partial_rollout_protocol=2`**，用于区分 session
重发与此前的 token 续写协议，避免混用旧进程。vLLM 不需要协议标记、render 或 assemble 接口。
推理统一使用原有 `/v1/chat/completions`（Gateway 到引擎为 stream=false），
reasoning/content/tool_calls 由 vLLM 原生解析器正常生成和解析。

## 与 verl 对齐的概率指标

Polar 原有 VLLMEngine 已请求 `logprobs=True, top_logprobs=0, return_token_ids=True`。
只保留完整返回的回复及其实际 logprob。被中止的单轮输出不计入训练概率指标。
Vime `--get-mismatch-metrics` 要求冻结策略的训练侧 logprob 重算。

- `training/rollout_probs_diff_mean`
- `training/rollout_probs_diff_max`
- `training/rollout_probs_diff_std`
- `training/rollout_probs_diff_count`（有效 token 数）

定义为有效 response/action token 上的
`abs(exp(frozen_train_logprob) - exp(actual_rollout_logprob))`。
跨 microbatch、DP、CP 聚合 token 统计量，max 用 MAX，std 使用 correction=1。
全 mask 时为 NaN，单 token 的 std 为 NaN。保留原有 log 空间差异指标。
旧、新轮次都计入这些指标；差异同时包含数值失配和策略陈旧的影响。
该开关可能增加一次冻结策略的训练侧前向开销。

## 暂停预算和限制

- 仅等待模型调用恢复的计划暂停时间不计入 session 执行预算；工具执行正常计时。
  rollout 轮询同步暂停时间，SSE 心跳维持 agent 连接；Claude API 超时至少 24h。
  相同请求重连复用 Gateway 中的逻辑任务，完整结果保存一次、工具 ID 保持一致。
- 本版保留 Gateway 常驻 asyncio 任务 + HTTP/SSE；没有改造 agent SDK 为内部 RPC。
  其他 SDK 的绝对请求时限需足够覆盖训练窗口。
- checkpoint 在单轮请求被取消时保存原请求、版本及暂停次数，排空确认前落盘。
  完成后更新状态；不保存被中止回复的 token/logprob。完整轮次沿用正常 session 存储。
- 支持计划训练暂停后的原进程恢复，未实现 Gateway/agent 进程崩溃后的自动恢复。
  重启仍须清退旧任务，避免重放工具。
- 每次请求限 n=1。完整请求原样重发，不再因 token 前缀续写而额外限制 stop、
  frequency/presence penalty、repetition penalty 或 structured outputs；具体支持以原生引擎为准。

## 验证

CPU 测试覆盖 43 轮完整历史 + 第 44 轮半截响应丢弃/原样重发，8 samples 中 7 条保留后补齐，
原版本完整结果边界竞争、session 取消不复活、真实 length 不重试、落后两轮拒绝、恢复优先级、
完整回复重连去重、暂停预算、旧新 action 保留 loss、概率指标的 CP/microbatch/DP 聚合。

本轮相关测试：Polar 78 项、Vime 129 项，合计 207 项通过；新增文件 Ruff 检查和启动脚本
语法检查通过。Vime 概率指标包含双进程 Gloo 聚合验证，DP/PD proxy 使用撤回补丁后的代码测试。

尚未运行完整 NPU 训练或多节点断网测试，不能据 CPU 测试给出 rollout 加速比例。
实际验收需观察连续两个权重更新：同一 session 续接、未完成回复不入库、旧轮次有 loss、
最老版本差≤1、组内 8 条有效终态、工具不重复执行、三个概率差指标非空。
