# dev_05 恢复 Skill 工具调用（替换 Read-SKILL.md 入口）方案

日期：2026-09-22。状态：待评审，未实施。

## 1. 背景：当初为什么禁

`ASCENDC_RL_HANDOFF.md`(2026-07-27 实测）记录的直接原因：

> claude-code 调 `Skill` 工具后，会用 qwen3.6 模板把下一轮历史上下文整个重渲染，
> 该动作丢掉历史里的 CoT,`<|im_start|>assistant\n<think>\n` 前缀消失 → 严格 token
> 前缀检查失败 → 断链。能对上轨迹的 3 个调过 Skill 的 ascendc session,3/3 全部断成 2 链。

当时的结论："决定断链率的是给不给 skill 文件路径"——于是 t2a 改为：禁 Skill 工具，
CLAUDE.md 里给全 14 个 skill 的文件级路径清单，入口统一为 Read。

## 2. 机制现在已经被修了

`preserve_thinking`(`src/polar/gateway/transform/anthropic.py`,2026-08-11 `ee8d37d2`)
让 chat template 对历史 assistant 轮全程渲染 `<think>`，其注释**点名** skill 注入：

> 任何纯 user 注入（截断 resume/skill 注入）都会把查询点前移、丢弃历史 thinking 的渲染
> → 老内容 token 化改变 → prefix merge 拆链（实测占拆链 ~88%)

即当年断链的机制根因（重渲染丢 CoT）已被消除。当前所有 run 默认 `POLAR_PRESERVE_THINKING=1`。

### 模拟验证（2026-09-22，真实 builder 代码)

用仓内 `PrefixMergingBuilder` 跑三种形态的合成 session:

| 形态 | 结果 |
|---|---|
| A: Read SKILL.md（现状） | 1 链完整，loss mask 正常 |
| B: Skill 调用 + preserve_thinking=1 | **1 链完整，与 A 逐位同构，mask 一致** |
| C': Skill 调用 + 旧模板（prompt 内部重 token 化） | 拆 2 链（复现当年故障形态） |

结论：训练侧链结构与 loss mask 上，Skill 与 Read 严格等价；工具结果段（skill 内容）
两种形态都被 mask 不进 loss。

### 附带：断链观测性的旧账

HANDOFF 还记录过"分组阶段断链不进 `break_reasons`"（只有 finalize 阶段计）。当前
`prefix_merging.py:547` 仍是 `break_reasons` 初始化为 `{}`、1180 行只在 finalize 记录。
恢复 Skill 后建议顺手补 `new_chain_reasons` 打点（分三类：真子 agent / 重渲染 / abort 空壳），
否则线上断链率回升也看不见。此条独立小改动，可一并做。

## 3. 训练一致性核对（恢复 Skill 是否引入训推不一致）

- 训练带 = gateway 实录的 `input_token_ids` + response logprobs，不重渲染；Skill 注入内容
  走同一 API 通道（下一轮请求 messages 内），同样实录。无二次渲染分叉。
- `record_filters.py` 的 completion filter 对工具名无感（只丢截断标记/空 choices/非 agent 侧）。
- builder 已把 Skill/Agent 分派识别为**独立子会话**（chain_role/agent_chain_id 身份），
  正常分链而非拆链；post-best masking 对独立子会话有专门处理。

风险不在一致性，在行为层：Skill 一次注入整份 SKILL.md（最大单份 ~52K 字符），而 Read 可
offset/limit 按需读。当前 76% session 本就撞 context limit 终止，恢复 Skill 后要盯
context-limit 占比和 session 时长。

## 4. 路径可达性核对（Skill 按名解析）

- Skill 工具从会话工作目录 `.claude/skills/<name>/` 解析；prepare 已把
  `operator_runtime_t2a/skills/`(14 个）落到每个 session 的该位置。
- 已核对：14 个 skill 的 SKILL.md 及 references 内部引用**无悬挂**（全部指向已安装的 14 个）;
  无残留 `skill:` 协议链接。
- **注意**:Skill 工具只注入 SKILL.md 本体；其引用的 `references/` 深链文件不进上下文，
  仍需 Read。所以文案不能写成"一律调 Skill"，要写"调用 Skill <name>;references 按
  SKILL.md 内引用用 Read 读绝对路径"。

## 5. cannbot 侧实现对照（每个改动点的参照系）

cannbot 插件（`plugins-community/tilelang2ascendc-ops-generator`,agents/
tilelang2ascendc-kernel-generator.md）的做法：

- 技能入口统一是"**调用 xxx skill**"（按名，不带路径）,Skill 工具全程可用；
- references 深链文件由 agent 顺 SKILL.md 指引 Read;
- 无"禁用 Skill"类禁令；钩子（doc_gate/skill_script_hook）在 PreToolUse 层托管，
  与 Polar 的预算/完成门禁职责不同，不冲突（Polar 侧 hook 不装这些）。

t2a 的 14 个 skill 即 cannbot `ops/` 的裁剪适配版（删了 `skill:` 交叉引用、路径改到
`/opt/asc-devkit`、加了 Polar 错误分类对接）,SKILL.md 的 `name:` 与 cannbot 一致，
按名调用兼容。

## 6. 改动清单（无残留版）

⚠️ 关键约束：`operator_runtime_t2a/CLAUDE.md` **是生成物**——由
`deploy/ascend_operator/build_claude_md.py`(upstream 副本 + DELTA 声明 + override）生成，
`--check` 会对不上就拒绝。直接手改 CLAUDE.md 会被 preflight 打回。
改动必须落在**生成器源文件**上，再 `--canonical` 重新生成。

### ① 配置层（解禁本体）

| 文件 | 现状 | 改法 |
|---|---|---|
| `deploy/ascend_operator/profile.t2a.yaml:127` | `disallowed_tools: "Skill AskUserQuestion ..."` | 删掉 `Skill ` 一个词 |
| `profile.t2a.yaml:132-133`(append_system_prompt) | "Skill 工具在本 profile 中禁用…禁止尝试调用 Skill" | 改为："知识查询调用 Skill 工具（按名）；其 references 深链文件用 Read 读绝对路径" |

### ② 生成器层（CLAUDE.md 的源头）

| 文件 | 位置 | 改法 |
|---|---|---|
| `deploy/ascend_operator/build_claude_md.py` | :189、:550、:611、:615 共 4 处 DELTA 里的"不要调用 Skill 工具/禁止调用 Skill 工具" | 反转措辞为"知识走 Skill 调用；references 用 Read" |
| `deploy/ascend_operator/gen_skill_reference_list.py` | :27、:31 生成文案 "Skill 工具已禁用；直接 Read…" | 改为"可按名调用 Skill；下列为各 skill 的 references 深链清单（Skill 不带出，需 Read)" |
| `deploy/ascend_operator/claude_override.md:7` | "本 profile 禁用了 Skill 工具…"整段 | 反转 |
| 重新生成 | `python3 build_claude_md.py --canonical operator_runtime_t2a` + `gen_skill_reference_list.py` | 再 `--check` 验证 |

### ③ 评测路由文案(`operator_runtime_t2a/tools/ascendc_eval_pipeline.sh`)

7 处路由串（`_AST_ROUTE`/`_COMPILE_ROUTE`/`_LOAD_ROUTE`/`_CRASH_ROUTE`/`_OUTPUT_ROUTE`/
`_PRECISION_ROUTE`/`_STATEFUL_ROUTE`/`_BENCHMARK_ROUTE`）现为 "参考资料：Read
.claude/skills/…"。改为 "参考资料：调用 Skill <name>；其 references 按文中引用 Read"。
:637 行预算耗尽的禁足令（"禁止再调用 Bash/Skill/…"）**保持不变**。

### ④ 测试层

| 文件 | 改动 |
|---|---|
| `tests/operator_runtime/test_eval_pipeline_budget.py:343-345` | 三条断言钉死了禁令，反转：allowed 含 Skill、disallowed 不以 Skill 开头 |
| 新增 `tests/trajectory/test_skill_call_chain.py` | 把 2026-09-22 的模拟固化：合成含 Skill 调用的 session 过 builder，断言单链 + mask 不变 |

### ⑤ 不动点（明确不改）

- 骨架生成：`prepare_operator_workdir.py` 预生成骨架维持不变（不恢复 cannbot 的
  init skill 做工程初始化——确定性、轮次成本、RL 跨 step 可比性三个理由）;
- `_non_project_skills`/`skillOverrides`（只关 CLI 内置 skill，与项目 skill 无关）;
- 预算、完成门禁、固定评测入口、loss mask 逻辑;
- 历史文档（`ASCENDC_RL_HANDOFF.md`、`SESSION_AUDIT_20260728.md` 等）是历史记录，不回改。

## 7. 观测指标与回退

上线后盯三个指标（首个含 Skill 的 run 跑 2~3 步后评估）:

1. `reconstruction_stats.chains_total` 与 break_reasons——含 Skill 的 session 不应比
   现状显著多链（若补了 new_chain_reasons 打点，可直接看"重渲染"类是否归零）;
2. `agent_context_limit_exceeded` 终止占比（当前 ~76%)与 session 时长——Skill 整份注入
   会推高 context 占用；
3. reward/成功率水位——不应有可归因于 Skill 恢复的变化。

回退：`profile.t2a.yaml` 把 `Skill ` 加回 disallowed_tools 即可（其余文案改动无害，
Read 入口措辞仍成立）。

## 8. 与 cannbot 回摆的关系

本次恢复 Skill 的直接收益是把 RL 学到的工具习惯从"Read 路径"换成"调 Skill"——
这是 cannbot 主流程的玩法，缓解"RL 模型回 cannbot 流程合规性退化"的风险项之一
（其余风险：固定入口漂移、预算节奏、输出纪律，不在本次范围）。
