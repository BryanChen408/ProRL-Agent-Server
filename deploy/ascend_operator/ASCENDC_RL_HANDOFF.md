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
- B7:用 `profile.sing52.yaml` 起一次 triton run,确认可切回

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
