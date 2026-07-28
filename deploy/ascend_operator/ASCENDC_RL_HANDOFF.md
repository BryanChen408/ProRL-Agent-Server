# AscendC 算子 RL — 交接文档

把 vime + polar 的算子 agentic RL 从"写 Triton 算子按 speedup 打分"改成"写 AscendC 算子",
用 `/home/docker/cannbot-skills` 的 ascendc-* skill 完成编写、用 polar judge 打分。

最后更新:2026-07-25。代码全部在本仓库分支 `feat/ascendc-rl`(10 个 commit,见文末)。

## 两条铁律(用户定)

1. **可随时切回 triton,绝不写死/覆盖** —— AscendC 全部并行新增,triton 侧一字不动。
   切换 = 启动时换 profile + 换数据集 env,零代码删改。
2. **参考成熟 triton 流程逐字派生**,保留其全部护栏,禁写丢护栏的"最小版"。
   > 今天的对抗审查证明这条不是洁癖:22 条确认缺陷里 10 条 critical,**6 条**源于"没逐字派生"
   > (judge 侧退化闸门退化成 `grep` 字符串、wheel 安装漏了、预算/best/哈希短路整块没派生)。

## 切换机制(2 个旋钮)

| | triton | ascendc |
|---|---|---|
| polar `POLAR_PROFILE` | `profile.sing52.yaml` ⚠️**本机 .52 用这份**,`profile.vime.yaml` 指向 80.48.5.88 另一台机 | `profile.ascendc.yaml` |
| vime `OPERATOR_DATA_ROOT` | cuda-llm filtered189 | `/home/docker/datasets/op_tasks/npukernelbench_level1_ascendc` |

启动:polar `POLAR_PROFILE=... POLAR_RUN_ID=... bash deploy/ascend_operator/restart_polar_host.sh`;
vime `bash /workspace/vime/scripts/start.sh`。

## 架构:一个固定入口,两侧自适应

**agent 自检与 judge 判分是同一个脚本**(对齐 triton 的 `triton_eval_pipeline.sh`),同一条命令:

```bash
bash tools/ascendc_eval_pipeline.sh --op_name {op} \
    --impl output/submission/{op}_impl.tar.gz --out_dir judge_out
```

判据是 workdir 顶层有没有 `{op}/` 源目录 —— agent 容器里有,judge 容器里没有。

| 阶段 | 占卡 | agent 侧 | judge 侧 |
|---|---|---|---|
| 自动打包(pack_submission.sh) | 否 | ✅ | ❌ |
| 预算计数 + LIMIT_EXHAUSTED / 内容哈希短路 / 更新 `.best` | 否 | ✅ | ❌(一次判定,无意义) |
| 解包 → 铲掉 `{op}/` 之外一切 → purge 二进制 → 注入数据集基准 | 否 | ✅ | ✅ |
| AST 退化检测(`validate_ascendc_impl.py`) | 否 | ✅ | ✅ |
| cmake + make + bdist_wheel + pip install | **否**(占卡编译会堵死 4 卡池) | ✅ | ✅ |
| 二次注入基准 + chmod 444 → 对拍 → 缓存探测 → 性能 | **是**(池内抢卡) | ✅ | ✅ |

**为什么必须合并**:曾经拆成两条路(judge 用 pipeline、agent 用 skill 的 `evaluate_ascendc.sh`),
结果持续漂移,并且 agent 侧的"自检通过"与 judge 判定基准不同(它用自己精简的用例和自己的 golden)
—— 18_Index 自检 `Result: pass`、judge `correctness_failed` 就是这么来的。

## 判分链的闸门(每一条都有对应的 reward hack)

reward 阶梯(`src/polar/trajectory/evaluator/operator_reward.py:67-85`):
`不成功: correctness_ok→0.4 | ast_check_ok→0.3 | else→0.2`;`成功: 0.75+0.25*tanh(ln speedup)`。

| 闸门 | 挡住什么 |
|---|---|
| 铲掉 `{op}/` 之外一切 | tarball 里塞 `.claude/skills/.../verification_ascendc.py`(内容 `print("Result: pass")`)接管判分。`cp -r A B` 在 B 已存在时**嵌套而非覆盖**,非恶意路径也可达(`tar czf x.tar.gz .`) |
| purge `*.so/*.a/*.whl/build/dist` | 夹带预编译二进制,与所交源码无对应关系 |
| 注入数据集 golden + 用例(**两次**,编译后再来一次 + chmod 444) | 改 golden 让参考实现迁就错 kernel;砍用例只留最简单一条。第二次是因为 Step2 会执行 agent 的 `setup.py`/import |
| AST 退化检测(不是 grep) | 纯 torch 实现 + 注释里写一句 `torch.ops.npu` → 编过 → 对拍必过 → speedup≈1 → **reward 0.75**。这是投入产出比最高的策略 |
| 缓存/常量输出探测(`detect_stateful_impl.py`) | `self._cache` 让 56 次测速只算 1 次 → speedup 上千 → ≈满分。对拍(只调一次)和退化检测都拦不住 |
| 完整 `metrics_error.log`(不 tail) | polar 整份下载做 infra 二次分类,`aclInit`/`InvalidDeviceId` 签名在栈**开头**,tail 会砍掉 → 环境故障被当算子错误算分 |

## NPU 卡:共享池 + 按需抢

`lease_at_start: false` → 容器只挂驱动、`ASCEND_RT_VISIBLE_DEVICES` 被清空,
polar 期待"容器里真正跑 NPU 活的子进程自己去抢"(`src/polar/runtime/docker.py:93-116`)。
agent 容器、judge 容器、polar 三方共用同一套 flock 锁 `/dev/shm/npu-locks/npu{N}.lock`,
抢卡器 `tools/npu_lease_exec.py` 是 triton 那份的逐字拷贝。池 = `0,1,2,3`(训练用 4-15)。

⚠️ **隔离靠的是环境变量,不是设备挂载**(容器是 `--privileged`)。所以 prompt/CLAUDE.md 里那条
"禁自写 NPU 探针、禁自设 `ASCEND_RT_VISIBLE_DEVICES`"是**护栏而非文风** —— 实测:triton
(有此禁令)自写探针 0%,ascendc(无禁令时)24% / 282 次。真要物理隔离只能 `lease_at_start: true`,
代价是并发从 24 掉到 ≤4。

## 本环境对 cannbot 原版工作流的覆盖

CLAUDE.md 的 Phase 0-7 是 cannbot `ascend-kernel-developer.md` 的**逐字拷贝**,覆盖声明写在文末契约区:

- **跳过 Phase 2(用例精简)与 Phase 6(全量恢复)** —— 数据集已定稿为每算子 5 个 case,
  judge 恒用数据集原版覆盖,精简只会让自检与判分不一致。不要调 `ascendc-case-simplifier`。
- **Phase 4.3 / 4-D / Phase 5 里 8 处"运行 evaluate_ascendc.sh / performance.py"** → 一律改跑固定入口。

## 当前状态与下一步

**代码**:全部提交在 `feat/ascendc-rl`。**没有任何一条在真机上端到端跑过。**

**上一次实跑(`polar_20260725_115157`,03:52 起)结论:该 run 无观察价值**,因为它跑在旧
prompt + 旧 topology 上:341 个 session,335 个 `submission_missing`(reward 0.2 地板,组内全等
→ GRPO 优势≈0),63% 被权重同步 abort;356 份 judge metrics 里 **350 份是 polar 在取件阶段
短路写的**(提交物不存在,judge 容器都没起),我们的 pipeline 只真正跑过 6 次。

**生效时机(重开 run 时必须清楚)**:

| 改动 | 何时生效 |
|---|---|
| 任务 prompt(`operator_tasks.jsonl`) | **run 启动时加载** |
| topology(judge_command / submission_candidates / prepare 动作 / env) | **polar 重启时烤死** |
| canonical(`CLAUDE.md`、`tools/*.sh`、skills) | **实时挂载,新 session 立刻生效** |

**重开步骤**:①`pkill` 清上一个 run 的 ray 残留;②`POLAR_PROFILE=profile.ascendc.yaml
restart_polar_host.sh`;③`bash /workspace/vime/scripts/start.sh`。
**建议先缩成小冒烟**(降 `NUM_ROLLOUT` / `POLAR_MAX_ACTIVE_SESSIONS`),只验
"agent 调固定入口 → 自动打包 → judge 出真 metrics"这条链,再放大。

## 序列长度(2026-07-25 确认)

**Qwen3.6-35B-A3B 原生支持 262144(256K)**:`config.json → text_config.max_position_embeddings
= 262144`(顶层没有,VLM 嵌套),`tokenizer_config.model_max_length` 互相印证;
`rope_type: "default"` + `theta=1e7`,**不需要 rope scaling**。
40 层里只有 **10 层 full_attention**(`full_attention_interval=4`),其余 30 层 GDN 线性注意力
状态大小与序列长度无关 → 长上下文比同规模纯注意力模型便宜得多。

要提到 256K 需**四件套同时抬**,最后一条最容易被忘:

```
SEQ_LENGTH / ROLLOUT_MAX_CONTEXT_LEN / VLLM_MAX_MODEL_LEN = 262144
MAX_TOKENS_PER_GPU × CP ≥ 262144      # 65536 × 4;vime_bridge/rollout.py:779 _resolve_max_tokens
                                       # 超过它的 trace 在 adapter.py 被**丢弃**(不是截断)
```

⚠️ `max_position_embeddings` 没在启动脚本里显式给,Megatron 拿 `seq_length` 填
(`arguments.py:886` 断言 ≥ seq_length),**起来后核 arg dump**,没跟上要补 `--max-position-embeddings`。
⚠️ 记忆 `gdn-train-infer-logprob-mismatch`:vime 的 `qwen3_5` spec 用标准 RoPE,而模型是
interleaved M-RoPE(`mrope_section: [11,11,10]`),误差随位置增长 —— 上长序列前该先修。

## 未做清单

- **端到端实跑**(最要紧;上面每一条都只过了本地单测)
- 对抗审查 22 条里未处理的:预算可绕过(直调 skill / `bash -c` / Skill 工具不被 watcher 计数)
- vime 侧缺"系统性全 ERROR 就早停":438 个 session 全挂时它仍一组组提交、一组组丢,白跑 8 分钟
- CLAUDE.md 内联 27,976 字符占 prompt 注入的 79%,要瘦身得把 Phase 3/4 迭代伪代码挪进按需加载的 skill
  ⚠️ 瘦身本身不影响断链率;**真正决定断链率的是 CLAUDE.md 给不给 skill 的文件路径**,
  见下面「prefix merge 断链」一条 —— 动第 171-180 行那份路径清单前先读那条
- B7:用 `profile.sing52.yaml` 起一次 triton run,确认可切回

## 遗留:prefix merge / 轨迹重建

### 1. 断链不可观测(优先级最高的一条,因为它挡住其它判断)

**现象**:`PrefixMergingBuilder` 在**分组阶段**接不上任何开链时会直接新起一条链
(`prefix_merging.py:_find_extendable_chain` 返回 `None`),这条路径**不写 `break_reasons`**
—— 只有 finalize 阶段的断因才计。所以线上看到的 `break_reasons` 恒为 `{}`,
唯一信号是 `reconstruction_stats.chains_total > 1`,而它把三类完全不同的东西混在一起:

| 类别 | 特征 | 是否问题 |
|---|---|---|
| 真 subagent(Agent 工具 / Explore) | 第二链 prompt 小(≈6k)、**system prompt 不同** | 否,本来就该独立成样本 |
| 重渲染丢 CoT 断链 | 第二链 prompt 大(35k~46k)、**system prompt 相同**、只是同一会话在更深处重启 | 是,同一 session 被计成 N 个样本 |
| abort 空壳 | `prompt_ids=[]`,空 tip 永不匹配前缀 → 每条自成一链 | 见下条 |

**建议**:`_find_extendable_chain` 返回 `None` 时打点,按上表三分类记进
`reconstruction_stats.new_chain_reasons`。**这不是为了好看** —— 任何"按 session 归一化 trace 权重"
的处理都必须先能把「同会话碎片」和「真子 agent」分开,不做这条就无从下手。

**已知触发条件(2026-07-27 实测)**:claude-code 调 `Skill` 工具后,会用 qwen3.6 模板把下一轮的
历史上下文整个重渲染一遍,该动作**丢掉历史里的 CoT**,`<|im_start|>assistant\n<think>\n` 这段
前缀消失 → 严格 token 前缀检查失败 → 断链。ascendc 侧目前几乎不断链**不是因为修好了,是因为
agent 基本不调 Skill 工具**;能对上轨迹的 3 个调过 Skill 的 ascendc session,3/3 全部断成 2 链。

**为什么不调(核实过,别照抄直觉结论)**:不是"CLAUDE.md 里已经有 skill 的内容"——
11 个 skill 合计 ≈912 KB(`ascendc-precision-tuning/SKILL.md` 单个 51,863 字符就比整份
CLAUDE.md 大),CLAUDE.md 只有编排骨架(分支、迭代计数器、传参、产物清单),
模板 / API 手册 / TileLang 映射表 / 精度知识库一个字都没有。
真实区别在**入口**——两边同样重度消耗 skill 内容,只是进门方式不同:

| | ascendc(64 session) | triton(86 session) |
|---|---|---|
| Read/Grep 命中 `skills/` 路径 | 486 / 792 = 61% | 476 / 727 = 66% |
| 有调 `Skill` 工具的 session | **5(8%)** | **72(84%)** |
| 首次读 skill 文件早于首次调 Skill(或从不调) | **57(89%)** | 3(3.5%) |
| 直接 `Read` 到 `SKILL.md` 本身 | **47 次** | **0 次** |

- triton:入口只能是 Skill 工具(`SKILL.md` 从没被 Read 过,是工具注入的),之后才顺着
  SKILL.md 指的路径 Read `references/`。CLAUDE.md 里 `skills/` 只出现 1 次,还是**禁令**。
- ascendc:入口是 Read。CLAUDE.md 有 16 处 `skills/` 路径,**给到文件名级别**——
  第 68-71 行脚本路径表、第 171-180 行整份「Skill 参考资料」清单、第 234 行还有一条
  直接从 skill 模板目录 `cp` 的命令。路径给了,Read 就是最短路径,Skill 工具沦为可选。

⇒ **决定断链率的是"给不给 skill 文件路径",不是 CLAUDE.md 的长度或伪代码内联。**
只搬伪代码、留着第 171-180 行的路径清单 → agent 照样 Read,断链率不变;
删掉路径清单强制走 Skill 工具 → 断链率立刻回到 triton 水平。
反过来,想永久规避断链最省事的办法就是继续暴露路径,代价是 skill 的
progressive disclosure 失效:该按需加载的 900 KB 参考变成 agent 自己翻,
吃上下文且不保证翻对(实测它连 `ascendc-code-gen/SKILL.md` 都要自己 Read 27 次)。

### 2. abort 空壳 trace 污染统计

`polar_20260725_115157` 的 805 条 trace 里 **244 条(30%)** 是
`prompt_ids=[] / response_ids=[] / finish_reason="abort"` 的空壳(权重同步 abort)。
session 已被 dev_09 判 `status="ERROR"` 整体丢弃、不进训练,**但**这些空壳仍会各自成链
(空 tip 永不匹配任何前缀),把 `chains_total` 顶高,让上面第 1 条的信号更没法用。

**建议**:在 `record_filters.exclude_completion_reason` 里加一条
`finish_reason == "abort"` → `"aborted_completion"`,在 filter 阶段就剔掉,
不要让它进分组。session 级的 ERROR 判定(`build()` 里的 `session_had_abort`)读的是
`session.completions` 原始列表,不受 filter 影响,所以加这条不会削弱现有的丢弃逻辑。

### 3. reward 广播到每条 trace(与第 1 条联动)

`gateway/node.py:1002-1006`:`outcome_reward` 会被**广播**到 trajectory 的每一条 trace
(实测每条 trace 的 `reward` 与 `outcome_reward` 完全相等)。所以一个 session 断成 N 链
= N 个带同一 reward 的训练样本。碎片本身是 on-policy 的(prompt 是当时真发出去的、
response 是真采样的),**不是正确性 bug**;但 N 与"这条轨迹调没调 skill"相关,
会系统性地给某一类轨迹加权。vime 侧把 traces 展开成几条样本、如何加权,不在本仓库,
**未核实** —— 要动加权前先确认 bridge 行为。PPO/critic 路线(sao-adapt)上,
把终局 reward 贴到每个碎片对 value target 是错的,影响比 GRPO 大。

## 关键路径速查

- canonical:`operator_runtime_ascendc/`(judge 挂 `/opt/canonical:ro`)
- 数据:`/home/docker/datasets/op_tasks/npukernelbench_level1_ascendc/`(31 算子 × 5 case)
- 镜像:`ascendc-sandbox:v1` = `sandbox:v1` + `pip install "cmake<4"`(base 未动);容器内 claude **2.1.168**
- 真实工作流参照(非 RL):`cannbot-skills/plugins-community/ascendc-ops-lab-developer/`
- 评测脚本真身:`cannbot-skills/ops-lab/tilelang-to-ascendc/skills/`(canonical 里是拷贝)
- **诊断口诀**:session ERROR 的真因永远在 `polar_sessions/<sid>/logs/prepare.NN.stderr.log`,
  不在 gateway.log,也不在 vime 日志;judge 结果在 `polar_sessions/<sid>/artifacts/`
- 截 prompt 的实测装置:`/workspace/claude`(与镜像同为 2.1.168)+ 本地假端点记录 POST body

## 提交记录(`feat/ascendc-rl`)

```
19e84490  fix(budget): watcher 认 ascendc 的固定入口
f6f28aff  feat(ascendc): load_polar_profile 接线
c2e09820  feat(ascendc): prepare_operator_workdir 加 --backend ascendc 分支
5fecb93c  feat(ascendc): profile.ascendc.yaml
749b3de4  feat(ascendc): 数据集生成 gen_ascendc_tasks.py
8389e4d8  feat(ascendc): 反 reward-hacking(缓存探测 + 测速输入轮换)
056a18fc  feat(ascendc): 提交物打包与档位留存
51fc0f47  feat(ascendc): 统一固定评测入口
da10c4c5  feat(ascendc): CLAUDE.md 追加 polar RL 判分契约
a8f802ed  feat(ascendc): canonical 运行时骨架(逐字拷贝 cannbot)
```

共用文件只改了两个(都是加分支,triton 默认不变):`operator_runtime/runtime/prepare_operator_workdir.py`、
`deploy/ascend_operator/tools/load_polar_profile.py`(+ watcher 加一个 marker)。
`profile.vime.yaml` / `operator_runtime/` / `triton_eval_pipeline.sh` / triton 数据集 = 零改动。
