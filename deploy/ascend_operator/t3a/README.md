# T3A rollout wiring

## Migration gate (2026-09-08)

The committed runtime is a rollback snapshot, **not** a verified reproduction
of the historical successful workflow. Keep the T2A service, scheduling, leases,
judge criteria and training boundary unchanged during migration. Do not restore
T2A solving rules inside the cannbot agent or enable training on empty main chains.

`baseline_audit.json` records the current evidence: 748 saved developer bodies,
129 body variants, zero exact matches against the pinned current agent. These
are file counts (exports can overlap), not independent success-rate measurements.
An exact text mismatch alone does not prove a semantic difference; representative
outlines additionally show older design-document/template-generation skills in
the successful corpus versus the current design/development/review route.

The older complete skill/script source is still needed to pin the workflow.
Do not assemble missing historical skills from similarly named modern ones.
The existing git history's initial `ascendc-ops-lab-developer` already documents
the newer route; the historical benchmark runner was not located there.

Reproduce the read-only audit (exit 2 means unverified, not a crash):

```bash
python3 deploy/ascend_operator/t3a/audit_workflow_baseline.py \
  --samples /home/docker/01_passed --repo /home/docker/cannbot-skills \
  --ref 13b2ae5652c75fe83a3e4114552a6477d3a01f3d \
  --agent-path plugins-community/tilelang2ascendc-ops-generator/agents/tilelang2ascendc-kernel-generator.md
```

Before the next stage, provide the old source/installation archive or explicitly
select a different documented baseline. Then validate fixed-checkpoint generation
and one full operator session before changing deployment defaults. Port 8011 was
serving a newer training run during this work; no replay requests or restarts
were sent to that live model.

Independent adapter fix: direct verification compound commands now execute as
`lease -- bash -c <quoted original command>`; a configured but missing executor
fails instead of silently running without a lease. CPU build/validation commands
remain unwrapped, and the self-wrapped AscendC pipeline is not double wrapped.

## Existing snapshot usage

Keep the original T2A dataset unchanged. Derive T3A instructions once:

```bash
python3 deploy/ascend_operator/t3a/convert_task_prompts.py \
  /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/operator_tasks.npukernelbench.jsonl \
  /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/operator_tasks.npukernelbench.t3a.jsonl \
  --case-mode simple
```

The converter refuses to overwrite an existing dataset. All non-prompt fields remain unchanged.
`--case-mode simple` preserves all five curated cases through Phase 2; the independent
judge validates all five. Agent Phase 6 is disabled: Phase 5 proceeds directly to
Phase 7 without restoring cases or repeating final validation. Use `NPUKernelBench/op_tasks` for both VIME's
`OPERATOR_TASKS_DIR` and Polar's `operator_runtime.task_assets_dir`. This does not
force the operator onto the simple development route. For `NPUKernelBench/src`
with the original case set, use `--case-mode full` (the converter default).
Use `profile.t3a.yaml` for Polar, and the derived JSONL as VIME's `OPERATOR_TASK_JSONL`.
`/workspace/vime/scripts/start_sync_hybrid_t3a.sh` wraps the existing local hybrid launcher
with those dataset settings and main-chain filtering; it does not change the resource layout.
For another launcher, set the dataset path there explicitly. Starting services is a separate action.

The snapshot's installed developer agent and all dispatch/reentry references use
`tilelang2ascendc-kernel-generator`. The replica build applies this installation adaptation;
the workflow follows that vendored source with the explicit Phase 6 removal above,
not yet proven equal to the historical success corpus.

Hook snapshots remain in agent session artifacts after the agent stops. The gateway recognizes
the candidate index without requiring a legacy submission tarball. The evaluator verifies each
snapshot hash, uploads only candidate files and the attempt stream, and uploads a relocated index
whose paths refer to the fresh judge container. `process_reward.json` is downloaded alongside
metrics; a failed download must not reuse a previous judge attempt's reward.

`profile.t3a.yaml` enables `gateway.t3a_attempt_spans`, which the profile loader
exports as `POLAR_T3A_ATTEMPT_SPANS=1` for the gateway. Keep the shared
`POLAR_ATTEMPT_CREDIT` switch enabled (its default) in Polar and VIME.
Main-chain filtering remains enabled in the T3A VIME wrapper.
Polar joins Claude's native transcript (`isSidechain`, `agentId`) and stream events
(`parent_tool_use_id`) to captured completions by message ID. Native identities
also keep agents with identical prompts in separate chains. VIME accepts only
verified subagent traces; missing/conflicting identities are logged and excluded,
without changing the terminal Polar result or relaxing the group acceptance gate.
Update/restart both Polar and VIME together: old system-based role labels are not
accepted as verified identities. An exclusively main-agent solution has no
subagent tokens to train and still produces a masked placeholder.

Attempt recording, credit and candidate selection do not impose a call limit.
`operator_runtime.budget.enabled: false` stops the old count watcher and skips
starting it; generation/optimization caps are not injected into the agent.
`operator.agent.max_turns: null` omits the CLI turn cap. `operator.timeout_seconds`
sets the initialization/solving budget. Once solving ends,
`operator.evaluator.postrun_timeout_seconds: 5400` gives postprocessing, trajectory
building and final judging a separate budget (including queueing, runtime setup
and infrastructure retries). `judge_timeout: 5400` also caps each judge command.
Rollout callback waiting includes both budgets plus its existing cleanup grace.
Solving timeout preserves valid traces and records `agent_time_budget_exceeded`;
successful final scoring remains trainable. Policy cutoffs and builder/evaluator
errors still reject the session. Tool/request timeouts remain in effect.
T2A retains its existing workflow and call caps.

CPU regression:

```bash
PYTHONPATH=src python3 -m pytest tests/examples/test_t3a_task_prompts.py \
  tests/examples/test_t3a_prepare_tools.py tests/trajectory/test_operator_judge.py \
  tests/gateway/test_lazy_eval_runtime.py
```

Before long training, verify one real session: registered developer dispatch succeeds, original
skills initialize and evaluate the project, judge consumes the same candidate hash, and VIME has
nonzero subagent loss tokens. CPU tests do not establish NPU correctness or resolve inference NaNs.

### 2026-09-09 session 回归修复（仅 T3A）

- R-paths 同时修复首次安装和已经漏插函数的安装；上游锚点不匹配立即构建失败。
  TileLang verifier 使用任务目录导入 design，并在成功/异常退出时恢复 Python 搜索路径。
- 工程初始化直接复用开发子 agent 的 Phase 1.2；project-init 保留模板，但其 T3A
  Skill 入口改为同一段初始化说明。主链只派发、检查，输出根目录保持不变。
- TileLang/helper 继续进入 attempt stream，仅 AscendC 评测且具备提交布局的工程进入
  candidate index。时间戳不作为进步；零通过失败候选可以保留供 judge 诊断，但不记 promote 奖励。
- Stop 复用现有候选记录，只读检查 AscendC 候选和 trace.md；缺失则要求恢复开发子 agent。
  不额外评测，不恢复 Phase 6，不参与权重同步。通过此检查不代表 judge 通过，真实失败仍可如实报告。
- T3A 的 doc_gate 去掉累计 12 次终止，保留 D1/D2 诊断与编辑门禁；入口说明明确
  重试数字只作策略切换建议，pipeline/verify 由外部时间预算截止。
- 租约下的评测命令若自行赋值 ASCEND_RT_VISIBLE_DEVICES，会在执行前被拒绝并提示删除赋值。

回归检查：

```bash
pytest -q tests/examples/test_t3a_session_regressions.py tests/examples/test_t3a_task_prompts.py \
  tests/examples/test_t3a_prepare_tools.py tests/examples/test_t3a_lease_commands.py \
  tests/examples/test_t3a_workflow_baseline.py
```

其中 fresh-replica 检查从本地 cannbot 源码重新构建，核对全部运行时文件。
已准备的 session 使用各自的 skill 副本，不会被回溯修改；新的 prepare 才读取修复件。
训练进程保留原始已载入 prompt；新 prepare 通过下述执行约定使用当前模板，无需重启训练。


### 剩余闭环修复

- `workflow_hook.py` 复用 cannbot 的原生 `agent_id` / PreToolUse 协议检查角色。
  主链只读与派发；仅允许已注册的开发 Agent，开发子链使用 Skill、不继续嵌套 Agent。
  Bash 先完成角色检查，再调用原生 skill_script_hook，避免并行 hook 提前执行被拒绝的命令。
- prepare 从同源 `convert_task_prompts.py` 和原生 task-prompts 生成任务约定，UserPromptSubmit
  补入当前主链说明，Agent updatedInput 传入当前开发模板并保留 resume 与错误上下文。
  NPUKernelBench 已提供 5 条 JSONL 用例时全部保留；其他任务保持 full 策略。
- 新增依赖逐项检查，并补入原生开发 agent 已声明的 `ascendc-crash-debug`，原件复制。
- 每个 judge 候选使用独立输出目录。接受者优先，否则保留排名靠前的真实实现错误；
  最终 metrics 包含实际候选 SHA256 和 judge_candidates 汇总，metrics_error.log 与选中结果一致。
  新一轮 judge 不复用旧 metrics，子进程无有效指标也留下明确失败结果。
- 过程分只允许有正确性进展的 AscendC attempt 触发一次 promote 奖励；没有候选被接受时
  正的过程总分归零，逐次 attempt 记录与负反馈保留，T2A 分数逻辑不变。
- 执行器及租约内 shell 开启 pipefail，尾部 tail 不再掩盖评测失败；非零退出不能判 PASS。
  T3A profile 明确允许原生 AscendC、TileLang 的 evaluate/verification 四种入口。

新增检查：`tests/examples/test_t3a_workflow_and_judge.py`。
已启动/prepare 的 session 保持原副本；上述 runtime 改动由后续新 prepare 加载。
profile 中入口说明的调整在下一次正常加载 profile 时生效。本次未重启运行中的服务。
