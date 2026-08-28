# Qwen3.6 Polar RL 的 MTP 问题复盘

更新时间：2026-08-27  
主要证据日志：`/mnt/pipeline-data/train_log/train_qwen36_polar_20260826-130952.log`

## 1. 结论先行

这次所谓的“MTP 乱码问题”不能简单归结为“MTP 把 token 算错了”。现有证据实际指向三个彼此独立的问题：

1. **MTP 确实启动了，但 `enforce_eager` 没有启动。** 传入 speculative JSON 的 `enforce_eager=true` 没有落到 vLLM 的顶层引擎配置，最终日志明确显示 `enforce_eager=False`。
2. **Polar `operator_samples` 链路没有把 VIME 的停止 token 配置送进最终 vLLM 请求。** 实际 `SamplingParams` 只有 `stop_token_ids=[248044]`，缺少 Qwen 对话结束符 `<|im_end|>` 对应的 `248046`。这会让模型越过正常 assistant 回合边界继续生成，可能表现为角色串行、工具调用 XML 残缺或用户所说的“乱码/跑飞”。
3. **`507001`/`EngineDeadError` 是同步训推切换时的控制面故障，不是乱码证据。** `prepare_policy_update` 失败后 VIME 仍继续并恢复 Gateway，18 个在途请求随后撞上正在切换/休眠的共卡推理引擎，最终 NPU task scheduler 报错、引擎死亡。

因此：

- 目前没有证据证明 MTP 的 speculative verify 本身产生了错误 token。
- 异步模式也出现输出异常，并不能证明“MTP 有问题”；异步和同步共同经过同一条 Polar 请求转换链，同样会丢失 `248046`。
- 在修正停止 token 契约前，不能用当前现象评价 MTP 的文本正确性。
- 即使正确性修好，`num_speculative_tokens=3` 也未必稳定加速：本次运行后段的 draft acceptance rate 多次只有约 `2.7%~7.0%`，此时很可能是负收益，必须做关闭/开启 MTP 的同负载 A/B 测试。

## 2. 两种参数写法到底有什么区别

当前 VIME 启动链应使用：

```bash
--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

这是 **VIME 的包装参数**。VIME 解析后再把 JSON 作为 vLLM 的 `speculative_config` 传给各推理引擎。

此前使用过：

```bash
--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
```

这里的 `method` 和 `num_speculative_tokens` 属于 speculative config；`enforce_eager` 是 vLLM **顶层引擎参数**，不是 MTP 子配置。它写进这段 JSON 后虽然留在 VIME 打印的原始字典里，但没有控制最终引擎。

日志证据：

- 第 40、1122 行：VIME 的字典里能看到 `enforce_eager: True`。
- 第 1295 行：vLLM API server 收到的 `speculative_config` 字典里仍能看到它。
- 第 1371 行：最终 `EngineCore` 配置却是 `enforce_eager=False`。

如果确实需要 eager 模式，应作为 VIME/vLLM 的独立顶层参数传递，例如 VIME 当前参数体系中的：

```bash
--vllm-enforce-eager
```

但 eager 会关闭图模式带来的部分性能收益，只适合作为稳定性隔离实验，不应默认和“开启 MTP”绑定。

## 3. MTP 是否真的启用

答案是：**真的启用了，不是只被 argparse 解析。**

`130952` 日志中有完整运行证据：

- 第 1308 行：`Resolved architecture: Qwen3_5MoeMTP`。
- 第 1371 行：`SpeculativeConfig(method='mtp', ..., num_spec_tokens=3)` 已进入最终 `EngineCore`。
- 第 1528 行起：开始加载 draft model。
- 第 1540~1543 行：检测到 MTP 模型、共享 embedding，且 draft model 加载成功。
- 后续持续打印 `SpecDecoding metrics`，包含 drafted/accepted token 数和逐位置接收率。

所以本次不是“MTP 参数没有生效”，而是“生效后还同时存在停止条件丢失、参数层级写错和同步切换故障”。

## 4. “乱码”更准确地说是什么

目前日志里没有 UTF-8 解码失败、tokenizer decode exception 或非法字符证据。能直接看到的是：

- MTP 异步历史日志 `train_qwen36_polar_20260824-114318.log` 第 2870 行附近，Qwen3 Coder tool parser 报 `Error in extracting tool call from response`，最后是 `ValueError: substring not found`。
- 这说明生成内容中的工具调用结构不完整；它是“格式跑坏”，不是已证明的字符编码乱码。
- `130952` 崩溃时的最终 `SamplingParams` 明确只有 `stop_token_ids=[248044]`。如果模型先生成 `248046=<|im_end|>`，vLLM 不会把它当成本次请求的终止条件，生成就可能跨过 assistant 回合边界继续进行。

所以更通俗的描述是：

> 模型本来已经说完一句话并输出了“本轮结束”标记，但服务端没有把这个标记当刹车，仍让模型往后写；后面的角色标记、工具 XML 和正文混在一起，看起来像乱码。

这条机制与现象一致，而且停止 token 丢失已被真实请求证明；但在没有把一条具体异常响应的 token 序列逐 token 对齐前，仍不能断言所有格式异常都只由它造成。

## 5. 停止 token 的完整证据链

### 5.1 模型配置

实际模型目录 `/home/docker/Qwen3.6-35B-A3B` 中：

- `tokenizer_config.json`：`248044 = <|endoftext|>`。
- `tokenizer_config.json`：`248046 = <|im_end|>`。
- `generation_config.json`：`eos_token_id = [248046, 248044]`。

因此期望请求至少能在这两个 token 上结束。

### 5.2 该次 VIME 参数

`130952` 日志第 743 行实际是：

```text
rollout_stop_token_ids .......................... None
```

也就是说，这个具体 run 本身没有成功配置 `--rollout-stop-token-ids 248046 248044`。不能把后续另一次启动命令里的两个 token 倒推成这次 run 已经使用。

### 5.3 即使 VIME args 配了，当前 operator_samples payload 也没有对应字段

当前 VIME Polar 入口是：

```text
generate_rollout_polar_async
  -> _build_submission_payload
  -> /rollout/operator_samples/submit
```

`/mnt/pipeline-data/vime_56/vime_bridge/rollout.py:458` 的 `_build_submission_payload()` 在第 490~501 行构造的 thin payload 只包含：

- `task_id`
- `instruction`
- `num_samples`
- `sample`
- `metadata`

其中没有 `stop_token_ids`、`sampling_params` 或 `rollout_stop_token_ids`。

Polar 的 `src/polar/rollout/models.py:115` 中，`OperatorSampleRequest` schema 同样没有上述字段。因此即使 VIME argparse 里出现两个 token，它们也无法经这个接口进入 Polar。

### 5.4 Gateway 没有补回或覆盖 stop_token_ids

- `src/polar/gateway/transform/anthropic.py:368` 构造 OpenAI 请求。
- 第 381~382 行只把 Anthropic 的字符串 `stop_sequences` 转成 OpenAI 的 `stop`。
- `src/polar/gateway/engine.py:81` 的 `VLLMEngine.prepare_request()` 只补 `logprobs`、`return_token_ids`、`top_logprobs` 并整理 reasoning 字段，没有写入 `stop_token_ids`。

也就是说，`stop_sequences -> stop` 和 token ID 停止条件是两套东西；当前 Gateway 没有主动把 `[248046, 248044]` 写给 vLLM。

### 5.5 最终 vLLM 请求

核心验收证据在 `130952` 日志第 7709 行（第 8575 行还有同类请求）：

```text
SamplingParams(
  ...,
  stop=[],
  stop_token_ids=[248044],
  ...,
  max_tokens=24576,
  ...
)
```

结论非常明确：最终不是 `[248046, 248044]`，而是 `[248044]`。

此外，`max_tokens=24576` 与 Polar 的 `deploy/ascend_operator/profile.t2a.yaml:114` 中 `agent.max_output_tokens: 24576` 一致，说明这一项由 Polar agent/profile 控制，不是 VIME 的 rollout stop 参数控制。

## 6. 为什么异步也会出现

停止 token 缺失发生在：

```text
VIME generate_rollout_polar_async
  -> operator_samples payload
  -> Polar agent 的 Anthropic Messages 请求
  -> AnthropicToOpenAITransform
  -> VLLMEngine.prepare_request
  -> vLLM SamplingParams
```

同步和异步的差别主要在权重更新、pause/sleep/wake 以及训练卡是否与推理共卡；它们仍会经过上面的 operator_samples/Gateway 请求链。因此异步同样出现工具格式跑坏，与停止 token 契约缺失并不矛盾。

反过来说，异步复现也说明：**不能只用同步训推切换故障解释输出异常。**

## 7. `507001` 为什么不是同一个问题

`130952` 的同步切换时间线是：

1. 第 6843 行：Ray 无法反序列化 `HTTPStatusError`。
2. 第 6871 行：`prepare_policy_update failed ... Continuing`，即失败后仍 fail-open 继续。
3. 第 6909 行：Gateway 被主动恢复，返回 `paused=False, inflight=18`。
4. 训练完成后开始 offload/切换；与此同时，共卡 vLLM 仍收到新请求并执行。
5. 第 7445 行起：NPU `aclrtRecordEvent` 报 `507001`。
6. 第 7709 行：vLLM dump 出当时仍在调度的新请求。
7. 第 7824 行起：引擎进入 `EngineDeadError`。

这是“没有确认 Gateway 已暂停，就继续做共卡引擎状态切换”的控制顺序问题。它和 MTP 同时出现在这份日志中，但日志不足以证明 MTP 是 `507001` 的根因。

对应修复方向仍是：VIME 在 `prepare_policy_update` 失败时 fail-closed、异常可被 Ray 序列化、未确认 admission pause 时禁止 sleep/offload；Polar 返回明确的 `paused/drained/inflight` 契约。

## 8. MTP 实际有没有带来加速

现有日志只能说明“功能开启”，不能说明“端到端变快”。

本次运行前段曾出现较高接收率，例如：

- 第 3612 行：平均 draft acceptance rate `72.2%`。
- 第 3686 行：`77.4%`。
- 第 3711 行：`84.6%`。

但长轨迹后段明显下降，例如：

- 第 7353 行：`3.3%`。
- 第 7371 行：`2.7%`。
- 第 7383 行：`7.0%`。

vLLM 自身也在第 1310 行警告：单 MTP 层配置 `num_speculative_tokens > 1` 会对同一层做多次 forward，可能降低 acceptance rate。

对这类长上下文、工具调用式 RL 轨迹，接收率会随请求/阶段大幅波动。接收率只有几个百分点时，三次 draft forward 绝大多数被拒绝，无法形成稳定加速；这是根据计数作出的性能推断，最终仍需同一批请求做 wall-clock A/B 才能定量。

还需注意日志第 1481 行的兼容性提示：speculative decoding 下 `min_p` 和 `logit_bias` 不工作。本次实际请求 `min_p=0.0`，未看到有效 `logit_bias`，所以暂未构成主要差异，但今后改采样参数时要检查。

## 9. 推荐的最小启用方式与验收标准

如果目标只是开启 MTP，不同时改变图模式，参数只保留：

```bash
--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

但上线前必须满足以下验收条件：

1. 最终 `EngineCore` 打印 `SpeculativeConfig(method='mtp', ..., num_spec_tokens=3)`。
2. 从真实 vLLM scheduler dump 或临时请求日志看到：

   ```text
   SamplingParams.stop_token_ids=[248046, 248044]
   ```

3. 输出 token 在第一个 `248046` 或 `248044` 处终止，不跨 assistant 回合。
4. 同步模式下，任何 `prepare_policy_update`/pause 失败都必须停止本轮切换，不能继续 sleep 或主动 resume。
5. 对固定 workload 分别关闭/开启 MTP，比较：

   - 端到端 rollout wall time；
   - 总 generation tokens/s；
   - mean acceptance length；
   - draft acceptance rate；
   - tool parser error 数；
   - 完成轨迹数和有效训练 token 数。

只有最终 vLLM 请求和端到端计时满足要求，才算“MTP 正确生效并带来收益”；仅看到 argparse 字典或 MTP model loaded 不够。

## 10. 后续最小修复落点

停止 token 问题的主修复点应是接口契约，而不是改 vLLM：

1. VIME `vime_bridge/rollout.py::_build_submission_payload()`：把 `args.rollout_stop_token_ids` 放进 operator_samples payload。
2. Polar `OperatorSampleRequest`：增加明确的 `stop_token_ids: list[int] | None` schema，并在 profile 展开/Session dispatch 时保留。
3. Polar Gateway：把该字段写进最终 OpenAI/vLLM 请求；不得被 Anthropic `stop_sequences -> stop` 转换覆盖。
4. 增加端到端测试，断言最终 `VLLMEngine.prepare_request()` 的请求同时包含两个 token。

不建议只在 Gateway 中硬编码 Qwen token ID，因为这会把模型特定行为藏进通用网关，也无法表达未来不同模型的 EOS 配置。

本文只做问题整理，没有实施上述修改。

## 11. 复现快照说明

整理本文时：

- Polar checkout：`/home/docker/polar_can/ProRL-Agent-Server`，HEAD `94476486`。
- VIME checkout：`/mnt/pipeline-data/vime_56`，HEAD `4a6c5e51`。
- 两边工作树均有未提交修改，因此这两个 commit 只表示整理时基线，不等同于 `130952` 启动瞬间的完整代码状态。
