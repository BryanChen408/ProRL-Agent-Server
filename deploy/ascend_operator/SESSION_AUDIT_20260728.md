# AscendC RL — 落盘 session 审计(2026-07-28)

对 `output/ascend_operator/runs/` 下已落盘的 rollout session 做的一次系统扫描,目的是找出
**"agent 本应成功、却因为环境/文档/框架问题失败"** 的动作,以及轨迹重建(prefix merge)侧的问题。

## 数据来源与方法

| 侧 | run | session 数 | 说明 |
|---|---|---|---|
| ascendC | `polar_20260727_151247` | 8 | 最新 run,新 prompt + 新 topology |
| ascendC | `polar_20260725_171524` | 48 | |
| ascendC | `polar_20260725_175829` | 8 | |
| ascendC | `polar_20260725_115157` | 388 | 旧 prompt,仅用于 abort/断链统计(HANDOFF 已判该 run 无观察价值) |
| triton(对照) | `polar_20260715_144815` | 50 | |
| triton(对照) | `polar_20260724_111704` | 36 | |

两份落盘互为印证:

- `polar_sessions/<run>/session-<sid>/.claude/projects/*/<uuid>.jsonl` —— claude-code 的**消息级**原始对话
- `rollout_results/<run>/task_*/ses_<uuid>.json` —— polar 的 **token 级**轨迹(prefix merge 产物)+ reward

统计口径:ascendC 侧 64 份 transcript、**2799 次 tool_result**、792 次 Read/Glob/Grep;
triton 侧 86 份 transcript、727 次 Read/Glob/Grep。两侧 session 通过
`evaluation.metrics_path` 里的 `session-sk-polar-*` 目录名互相对齐。

---

## 问题清单(按归属分类)

上游 pin = `https://gitcode.com/chenshushu2020/cannbot-skills.git` @ `br_asc_dev` @ `6cf50e29`
(本地 `/home/docker/cannbot-skills` 就是该 commit,已核)。**归属划分是决定改法的关键**:
适配问题改我们自己的代码,上游债改 canonical 副本或走覆盖声明。

### 类别一:适配问题 —— 我们移植时引入的

| # | 问题 | 证据 | 改法 | 状态 |
|---|---|---|---|---|
| 适-1a | CLAUDE.md Phase 0 教 agent `设置 ASCEND_RT_VISIBLE_DEVICES=${npu}`,与 task prompt 的 `Never set … yourself` **正面矛盾** | Phase 0「参数校验」第 3 条 | 删掉该条(卡由 `npu_lease_exec.py` 统一管) | 待改 |
| 适-1b | `{output_dir}` 在 CLAUDE.md 出现 **50 次**,契约却要顶层 `{op}/`;实测 agent 写工程 顶层 338 vs `output/<op>/` 60,摇摆导致 `AGENT_SIDE=0`、自动打包整段跳过 | 见「适-1 调研结论」 | 统一钉成 `{op_name}/` | 待改 |
| ~~适-1~~ | ~~用错 agent 定义(subagent vs primary)~~ | — | **调研后撤回**:primary 是纯调度壳,RL 下职责全不适用;用 subagent 当 CLAUDE.md 是合理简化 | 已撤回 |
| 适-2 | `archive_tasks` 不存在(只搬 skills,没搬 plugin 的 `workflows/`) | 实测 5 次找该目录;实体在 `workflows/templates/archive_tasks` | prepare 一并铺进 workdir,或覆盖声明宣告没有 | 待改 |
| 适-3 | `ops-lab/` 路径失效(详见下文 A) | ≥20 次 Bash 失败 + 3 次 `ops-lab not found` | prepare 建 symlink + 对账断言 | **已改,已验** |
| 适-4 | judge pipeline `OP_DIR_NAME` 从 tarball 内部布局反推,不与 `--op_name` 对账;同一退化让隔离闸静默失效(详见下文 G) | `tar -C 3_Add .` → `op 'work'` → verification 崩 | 以 `--op_name` 为准;`TASK_DIR == $WORK` 时规范化或明确报错 | 待改 |
| 适-5 | `rm -rf` 被拒 8 次。**不是 flag 失效**(实测 `permissionMode=bypassPermissions`),是 agent `cd` 进 `kernel/build` 后在里面删它自己,命中「删除当前工作目录或其父目录」这条独立于 permission 体系的硬规则 | 见「适-5 更正」 | 给 pipeline 加 `--clean-kernel-build`(或 `cd ..` 再删) | 待改 |
| 适-6a | **引擎 OOM 被终止 → 4 个 session 收到 502 → 以 `COMPLETED` + `reward=0.2` 进训练** | gateway 8 条 502 集中在 15:54 的 3 秒内;OOM 已确认 | ① 控引擎内存 ② agent 侧识别 `API Error: 5xx` → 判 infra,retry 不计分 | 待改 **P0** |
| 适-6b | 取件循环 `except Exception: continue`(`operator_judge.py:209-215`)把「文件不存在」与「源容器已销毁/传输失败」塌缩成同一条 `submission_missing`,而两者 reward 处理应相反 | 1 个 session 打包 10 次仍 missing、真因不可知 | 分别捕获,真实异常写进 error;infra 类走 retry | 待改 |
| 适-6c | 提交物目录对 agent 完全可写:agent `rm -rf …/output/submission/*.tar.gz` 把 `.best` 一起删掉,破坏 prompt「被截断也按 best 判分」的承诺 | 1/64 | best 存到 agent 够不到的位置,或 pipeline 结束即取件 | 待改 |
| 适-7 | prefix merge 三条(断链不可观测 / abort 空壳 / reward 广播) | 见 HANDOFF「遗留」章 | 已记录 | 遗留 |
| 适-8 | judge 把 tarball 存成 `submission_impl.py`(triton 单文件时代命名) | `artifacts/operator_judge/` | 改名 | 低优先 |
| 适-9 | 固定入口编译失败只回一句「完整错误在 metrics_error.log」,agent 只能反推 → 自编译 `cmake` 163 次 / 24 session(38%);**triton 侧 0 次** | 见「skill-1 降级」 | 编译失败时把 `compile.log` 关键段直接打到 stdout | 待改 |

### 类别二:skills 自身的问题 —— pin 版上游债,装不装都在

已实测验证:在隔离沙箱里按上游 `init.sh project claude` 装一遍,悬空引用 **51 种 → 50 种**,
即这一类**与我们怎么移植无关**,换基点解决不了(详见下文「安装对照实验」)。

| # | 问题 | 实体位置 | 改法 |
|---|---|---|---|
| skill-1<br>**(已降级为可选)** | hook 注册缺失 → CLAUDE.md 56-88 行整章描述的拦截机制不生效 | pin 版 `hooks.json` 只注册 `SessionStart`,`skill_script_hook.py` 及其 9 条 `INTERCEPTED_PATTERNS` 都在但永不触发 | **实测该绕过在我们这儿不发生**(2/64,且只有最无害的 `validate_ascendc_impl.py`);真实绕过是自编译(适-9)。上游 hook 的行为是「代为执行」,照搬 = 把绕过合法化。详见「skill-1 降级」与「固定入口 vs hook」两节 |
| skill-2 | skill 名笔误:`ascendc-operator-code-gen`(真名 `ascendc-code-gen`,7 处);agent 定义 70/71/93 行 `tilelang-designer`/`performance-analyzer` 漏 `ascendc-` 前缀 | — | 直接改 canonical 副本 |
| skill-3 | 目录单复数:`@scripts/performance.py` vs 真实 `script/`;CLAUDE.md:428 `script/evaluate_tilelang.sh` vs 真实 `scripts/` | — | 同上 |
| skill-4 | 引用不存在的资源:`references/AscendC_knowledge/`、`ascendc-trace-recorder/scripts/*.sh`、precision-debug 的 `./env_setup.sh` | 全仓不存在 | 删引用或覆盖声明宣告 |
| skill-5 | 跨 plugin 断裂:precision-tuning 引 `ascendc-evaluation`/`dsl-lowering` 共 10 处 | 在 `master` 的 `plugins-community/collaborative-agent-kernel-evolution/skills/` | 见下「skill-5 调研结论」 |
| skill-6 | asc-devkit 引用路径过时:`docs/api/…` vs 实际 `docs/zh/api/SIMD-API/…` | devkit 当前 HEAD | 引入时同步改引用;不引入就删引用 |
| skill-7 | 路径形式不完整:清单只给文件名不给子目录;`templates/ascend-kernel/` 没给树(详见下文 B/C) | — | 脚本从目录树生成清单 + 进 preflight |

### 原则:允许修上游笔误(2026-07-28 定)

skill-2/3/4/6/7 要动上游文件,与 HANDOFF 铁律 2「逐字派生」表面冲突。**认定:铁律 2 的目的是
不丢护栏,不是连笔误一起保留。**把一个指向不存在文件的路径改成正确路径不改变任何语义 ——
上游那条路径本来就走不通,改它不构成行为分叉。本仓库本来也是 fork + 覆盖声明模式。

⇒ 这几条**直接在 canonical 副本里修**,不走覆盖声明(实测 agent 是照正文直接执行的,
靠文末一句话推翻正文,可靠性低)。

### 不建议 rebase(2026-07-28 定)

- **rebase 到同 commit 的安装产物**:悬空引用 51→50,净收益 −1;唯一真收获是暴露适-1。
- **rebase 到新 HEAD(`2ff3d2d`,+78 commit)**:上游把整套 skill 改名成 `tilelang2ascend-*`,
  **删除了 `ascendc-code-gen` 和 `ascendc-design-doc-generator`**(全仓 0 命中),简单算子那一路
  从 `design.md → code-gen` 换成了 `ops-direct-invoke`。我们的数据集分类路由、CLAUDE.md
  Phase 3-S/4-S、固定入口 pipeline、judge 对 `{op}/kernel/{op_host,op_kernel}` 的结构假设
  全部长在被删掉的那条路线上 —— **那是重做,不是 rebase**。
- 可单独摘的三样(都不需要动基点):新 HEAD 的 `hooks.json` PreToolUse 段、
  `ascendc-evaluation`/`dsl-lowering`、asc-devkit。

---

## 调研结论(2026-07-28,清单里标「待查/需调研」的项)

### 适-1:primary vs subagent —— **结论:我们的选择是对的,但 Phase 0 有真冲突**

读完上游 `AGENTS.md`(primary,157 行)后确认:**primary 是纯调度壳**,它明确"禁止直接参与设计、
开发或代码修改",职责只有需求接收 / 算子分类 / 调 subagent / 进度监控 / 断点恢复 / 争议仲裁。
**干活的逻辑 100% 在 `ascend-kernel-developer.md`(subagent)里**,也就是我们用的那份。

在 RL 场景下 primary 那层的职责基本都不适用:需求接收→由 task prompt 替代;调度 subagent→
我们只起一个 claude session;进度监控/断点恢复→单 session 一次性;仲裁→无人可仲裁。
⇒ **用 subagent 定义当 CLAUDE.md 是合理简化,不需要改。**

**但 Phase 0 有两条真冲突,必须处理:**

1. **`设置环境变量 ASCEND_RT_VISIBLE_DEVICES=${npu}`(CLAUDE.md Phase 0「参数校验」第 3 条)**
   与 task prompt 的 `Never set ASCEND_RT_VISIBLE_DEVICES yourself` **正面矛盾**。
   HANDOFF 已记「ascendc(无禁令时)自写探针 24% / 282 次」—— 这条正指文档在教它这么做。
   → **改法:删掉 Phase 0 这一条**(NPU 由 `npu_lease_exec.py` 抢卡器统一管)。
2. **`{output_dir}` 占位符在 CLAUDE.md 里出现 50 次**,而 task prompt 要的是顶层 `{op}/`,
   `pack_submission.sh` 也是 `cd $WORKDIR && tar czf ... "$OP_NAME"`(即顶层)。
   实测 agent 写工程的位置:**顶层 338 次 vs `output/<op>/` 60 次** —— 在两套约定间摇摆。
   → **改法:把 CLAUDE.md 里的 `{output_dir}` 统一钉成 `{op_name}/`**,与固定入口契约对齐。

### 适-8:`submission_missing` 7/8 —— **结论:五个不同原因,主因是 502**

逐 session 拆开(`polar_20260727_151247`,8 个):

| session | 原因 | 归属 |
|---|---|---|
| a2ridi1g / f9y93kcn / hpblxt06 / xchaxt29 | **`API Error: 502 Upstream request failed: All connection attempts failed` 掐断**,`stop_reason=stop_sequence`,从没跑成固定入口 | **infra(引擎侧)4/8** |
| 3cqr3f4w | 打包成功后 **agent 自己删掉了提交物**:`rm -rf .../judge_out .../output/submission/*.tar.gz`(通配符把 `.best.tar.gz` 一起带走) | agent 行为 + 缺护栏 |
| z5mgj8vb | 跑了 6 次入口但 `[pack]` 输出为 0 —— `AGENT_SIDE=0`(`$PWD/{op}` 不存在,工程写在 `output/<op>/`),自动打包整段跳过 | 适-1 第 2 条的直接后果 |
| 7c95jgx8 | **打包成功 10 次仍报 missing**,真因不可知(见下) | 取件层,**未闭合** |
| jn5za7tz | 唯一判到分(`correctness_failed`,见下文 G) | — |

**⚠️ 502 那 4 个 session 全部以 `status=COMPLETED` + `reward=0.2` 进了训练** —— 引擎故障被当成
算子失败计分,直接毒化 GRPO。polar 的 infra 二次分类(`classify_infra_error_text`)只看 judge 侧
的 `metrics_error.log`,而 502 发生在 agent 侧、根本没进 judge → 这道闸完全拦不住。

**7c95jgx8 为什么不可知** —— `operator_judge.py:209-215` 的取件循环:

```python
for cand in self.submission_candidates:
    try:
        await source.download_file(self._abs(cand), str(local_impl)); picked = cand; break
    except Exception:      # ← 吞掉一切
        continue
```

"文件不存在"和"源容器已销毁 / 传输失败"塌缩成同一条 `submission_missing`。
这两者的 reward 处理**应该完全相反**(前者 agent 的锅记 0.2,后者是 infra 该 retry 不计分)。
→ **改法:分别捕获并把真实异常写进 error,infra 类走 retry 而不是 0.2。**

### skill-1:hook —— **结论:挂载点可借,行为不能借**

- pin 版 `hooks.json` 只注册 `SessionStart`,`skill_script_hook.py` 及其 9 条 `INTERCEPTED_PATTERNS`
  都在但**永不触发**(上游债);新 HEAD 补上了 `PreToolUse: Bash → skill_script_hook.py`。
- **但上游 hook 的行为是"代为执行"被拦截的脚本**(`evaluate_ascendc.sh` / `performance.py` /
  `verification_ascendc.py` …),这恰恰是我们覆盖声明**禁止**的 —— 直接执行会绕过预算计数、
  NPU 抢卡、判分基准注入、退化检测。**照搬 = 把绕过合法化。**
- ⇒ **正确做法:自己写 PreToolUse hook,复用上游的 `should_intercept` 匹配逻辑,
  但 decision 改成 `deny` + 提示改跑固定入口。** 上游那套 `_LEADING_PREFIX` 已经能识别
  `ENV=VAL cd dir && python3 x.py` 这类前缀绕过形态,正是我们需要的,可直接复用。
  这同时关掉 HANDOFF 未做清单里的「预算可绕过」。

### skill-5:precision-tuning 的跨 plugin 依赖 —— **结论:该路径当前不可用**

`ascendc-precision-tuning/SKILL.md` 对 `dsl-lowering` 有 **3 条「必须读取」**(第 334-336 行),
并执行 `skills/ascendc-evaluation/scripts/{generate_pybind,evaluate}.py`(第 122/688/698 行)。
两个 skill 在 pin 版**全仓不存在**(实体在 `master` 的 `collaborative-agent-kernel-evolution`)。
而 CLAUDE.md:623 又写「D 类错误必须先调用 Skill 工具 ascendc-precision-tuning」。
→ **改法二选一**:①从 master 取这两个 skill 进来;②在覆盖声明宣告深度审计不可用,
D 类只走 `ascendc-precision-debug`。**建议 ②** —— 它执行的 `evaluate.py` 本来也会绕过固定入口。

### 适-5 更正:不是 flag 没生效,是「站在要删的目录里删它」

`logs/agent/claude-code.txt` 的 init 消息实测 **`"permissionMode":"bypassPermissions"`** ——
`--dangerously-skip-permissions` 完全生效。完整拒绝原文:

```
Dangerous rm operation detected: '/opt/workspace/agent_workdir/output/3_Add/kernel/build'
This command would remove the current working directory or one of its parent directories.
This requires explicit approval and cannot be auto-allowed by permission rules.
```

触发条件是第二句 —— agent 先 `cd` 进了 `kernel/build`(同条命令里跟着 `cmake ..` 为证),
再在里面 `rm -rf` 它自己。这条规则独立于 permission 体系("cannot be auto-allowed by
permission rules"),bypassPermissions 覆盖不到。

⇒ **不是硬阻塞,是可绕开的用法问题**:`cd ..` 再删、或 `rm -rf build/*` 都能过。
我们侧仍建议给 pipeline 加 `--clean-kernel-build`,让 agent 根本不必自己 rm。
(更正:先前写的「claude-code 一律禁 rm -rf、改配置没用」——方向对,原因说错了。)

### 适-6 补充:502 的直接原因是引擎 OOM 被终止

gateway 侧证据:整个 run **只有 8 条** upstream error,全部落在 `15:54:55–15:54:58`
这 3 秒内,先 `Server disconnected without sending a response` 再 `All connection attempts failed`;
8 条打在 4 个还活着的 session 上(group 1 的 sp000-sp003),15:55:00 这 4 个 task 被标 completed;
502 之后 gateway 自身 health/sessions 持续 200 —— 挂的是 upstream 不是 gateway。
upstream = `sglang_router_url: http://80.48.5.52:8001`。

**根因(2026-07-28 由用户确认):推理引擎 OOM 被终止。**进程被杀的信号与上述特征完全吻合
(瞬时、先断连后拒连、之后 gateway 正常)。

⇒ 结论不变且更强:这 4 个 session 以 `COMPLETED` + `reward=0.2` 进训练,**引擎 OOM 被当成
算子失败计分**。polar 的 infra 二次分类只看 judge 侧 `metrics_error.log`,而 502 发生在 agent 侧、
根本没进 judge → 这道闸拦不住(适-6a)。

### skill-1 降级:上游 hook 防的绕过在我们这儿不发生

先前把 skill-1 排在「收益最大」,依据是 HANDOFF 的「预算可绕过(直调 skill 脚本)」。
**没有验证它实际有没有发生。实测后降级:**

```
64 个 session 中,只有 2 个直调过上游拦截清单里的 9 个脚本
   2 次  validate_ascendc_impl.py   ← 最无害的那个(纯 AST 检查,不上卡、不烧预算)
   0 次  evaluate_ascendc.sh / verification_ascendc.py / performance.py / build_ascendc.py …
```

**真正发生的绕过是自编译,而上游那 9 条模式一条都匹配不到:**

| 动作 | ascendC(64 session) | triton(86 session) |
|---|---|---|
| `cmake` | **163 次 / 24 session(38%)** | **0** |
| `make` | 38 次 / 9 session | **0** |
| `g++ / gcc / bisheng` | 19 次 / 2 session | **0** |
| `npu-smi` | 0(prompt 禁令在这条上有效) | 0 |

**triton 侧一次都没有** —— 因为 triton 的交付物是单个 `.py`,没有编译步骤,agent 没有"自己动手
看看"的动机。AscendC 要编 C++ kernel,而编译错误的细节**只有编译器给得出来**,固定入口却把编译
包在里面、失败只回一个 `metrics_error.log`。**这个动机是任务性质带来的,不是模型不听话。**

而且它并不算严重违规:`cmake`/`make` 不占卡、不烧预算、不污染判分,prompt 禁的是
"custom tests / probes",编译算不算 probe 有解释空间。

⇒ **skill-1 从优先级 4 降为「可选」**,且内容要换:不是移植上游 hook,而是(如果要做)自写一个
拦自编译的 PreToolUse hook。上游那套唯一值得复用的是 `_LEADING_PREFIX` 前缀识别
(能识破 `ENV=VAL cd dir && cmd`,而 24 个 session 的 cmake 调用大多正是 `cd .../build && cmake ..`)。

### 固定入口 vs hook 的本质区别(架构结论)

- **固定入口**把护栏放在「被调用的东西」里面 → **正向保证:走对了路一定有护栏**。
  但它**不能保证 agent 只走这条路** —— 唯一性必须由入口之外的机制保证,
  而现在保证它的只是 prompt 里那句 "the only executable validation path"(建议,非强制)。
- **hook** 是在「调用动作」之前设卡 → **反向封堵:走错了路被拦回来**。

两者互补而非替代。triton 的「统一入口」设计没有问题,缺的那一半从来没补上,只是 triton
的任务性质没把它暴露出来。

⇒ **建议:先治根因而非症状。**那 38% 的自编译没有造成实际损害,它反映的是**固定入口的
编译错误反馈不够好**(见适-9)。改反馈比加 hook 便宜,而且 hook 会让 agent 拿不到编译细节、
调试能力下降,可能反而拉低 reward。

### 适-9(新增):固定入口的编译失败反馈不足

实测 agent 在编译失败后的行为:去 `cat judge_out/compile.log`(3 次,含读在文件尚未生成的时机)、
`find /usr/local -name kernel_macros.h`、手写 `g++ -E` 探针、给 `kernel/` 建 include symlink……
即**它在费力反推固定入口没告诉它的东西**。

⇒ 改法:`ascendc_eval_pipeline.sh` 编译失败时,把 `compile.log` 的关键错误段直接打到 stdout
(而不是只留一句"完整错误在 metrics_error.log")。这条同时削弱自编译的动机,
比封堵便宜、也不损失调试能力。

---

## 迁移到官方新版的前置调研(2026-07-28)

目标版本:`gitcode.com/cann/cannbot-skills` @ `master` @ `f76269e`(2026-07-28),
plugin = `plugins-community/tilelang2ascendc-ops-generator`。注意与我们 pin 的
`chenshushu2020/cannbot-skills` @ `br_asc_profiling` 是**两个仓库**。

### Q1 交付物形态 —— **完全没变(推翻先前判断)**

新版 Phase 1 定义的目录结构与 pin 版逐字相同:

```
{output_dir}/kernel/{CMakeLists.txt, setup.py, ops.h, register.cpp,
                     op_host/<op>.cpp, op_kernel/<op>.cpp, utils/torch_kernel_helper.h}
{output_dir}/model_new_ascendc.py
```

简单路径(`ops-direct-invoke`)与复杂路径(`translator`)**产出同一套东西**,
简单路径只多一个 `docs/REVIEW.md`。Phase 0-7 骨架也一样,只换了两个分支的内部做法。

⚠️ **收回先前的判断**:早前写的「简单算子那一路整个换了,我们的分类路由 / pipeline /
judge 结构假设全长在被删掉的路线上」是**错的**。换的是"怎么写出 kernel"
(模板生成 → 渐进式开发 + 代码审查),不是"写出什么"。
⇒ **数据集、判分链、结构假设全部直接可用**,🔴 重新对齐的工作量从 ~200 行降到 ~90 行。

### Q2 性能测量 —— 决定:**对齐 msprof 版本**

| | pin 版 | 新版 |
|---|---|---|
| 工具 | `ascendc-performance-analyzer/script/performance.py` | `ops-profiling/scripts/msprof_perf_summary.py --quick` |
| 测什么 | wall-clock,56 次取平均 | **msprof kernel 时间**,warmup 3 + timed 1 |
| 输出 | `preformance.json` | `performance.json` |
| 加速比字段 | `overall_speedup` | `mean_speedup` / `geomean_speedup` / `per_case[].speedup` |

**迁移动作**:改 2 处读取点(`ascendc_eval_pipeline.sh:428`、`pack_submission.sh:82`),
并决定用 mean 还是 geomean。
⚠️ **msprof 测纯 kernel 时间、排除 host 开销,同一实现的 speedup 通常高于 wall-clock**,
reward 阶梯 `0.75+0.25*tanh(ln speedup)` 的分布会整体上移 —— 是否跟着调阶梯要跑一批数据再定。

### Q3 I-6:测速缓存漏洞 —— **新版更严重,补丁必须重做**

**先前遗漏**:`skills/ascendc-performance-analyzer/script/performance.py` 是我们**唯一改过的
上游 skill 文件**(17 行,`ASCENDC_PERF_INPUT_VARIANTS`,默认关闭 → 上游行为一字不变)。
作用:预生成 K 份同 shape 不同数值的输入并轮换,堵掉"按输入 id/值缓存"造成的虚假加速。

新版 msprof wrapper(`msprof_perf_summary.py:_WRAPPER_SCRIPT_TEMPLATE`)的跑法:

```python
{inputs_code}                 # 构造一次 inputs
for _ in range({warmup}):     # 默认 3 次
    _ = model(*inputs)        # 同一个 inputs
_ = model(*inputs)            # 正式计时,还是同一个 inputs
```

| | pin 版 | 新版 msprof |
|---|---|---|
| 正式计时 | 56 次取平均 | **仅 1 次** |
| 缓存命中率 | 55/56 ≈ 98%,平均值里仍掺一份真实耗时 | **100%** —— warmup 3 次已填满缓存,timed 那次必然命中 |

`--compare` 的 8 轮采集**不解决问题** —— 每轮新起 msprof 进程跑同一个 wrapper,
进程内仍是"构造一次输入 → warmup → timed"。

**这个洞是条件触发的,不是恒定偏高。**测量方法的前提是"每次调用都真算";人工流程里
该前提自动成立(没人给自己的 kernel 写缓存,那只会让自己看不到真实性能),
**所以上游的数字在实现诚实时是准确的**。RL 里前提不成立:reward 直接奖励高 speedup,
任何压低测得时间的行为都被梯度强化 —— 模型不需要"想作弊",只需某次采样碰巧写出
带缓存的实现拿到高分,这个方向就被强化(HANDOFF 记的 `self._cache` → speedup 上千即实测)。

**注意区分两种效应**:warmup 带来的硬件 cache / DVFS 稳定效应是**故意的**,
且 `reference` 与 `ascendc` 走同一套流程,比值公平;软件缓存作弊只有作弊方有,直接摧毁比值。

**缓存命中后的两种结局(已核 `msprof_perf_summary.py:1464-1470`)**:

- ① 缓存在 Python 层但 kernel 仍被启动 → `task_time.csv` 有记录、时间极短 →
  `asc_sum > 0` → **speedup 虚高、静默通过**。这是真实风险,且对拍与 AST 退化检测都拦不住。
- ② 缓存写在 `forward` 顶层、完全不启动 kernel → 采不到记录 →
  `"no task_time or api_statistic csv found"` → 记 `asc_error`、`total_speedup = None` →
  **自曝报错,不会虚高**。
  (⚠️ 更正:先前把这条写成"更严重的证据"是错的,它其实是安全的那一侧。)

**结论:补丁必须在新工具上重做**,针对的是结局 ①。
新版反而更好打:正式计时只有 1 次,只需让**那一次用一份 warmup 没碰过的输入**,约 5 行:

```python
_variants = [inputs] + [_clone_value(inputs) for _ in range(K)]   # 预生成在计时区外
for i in range({warmup}): _ = model(*_variants[i % {warmup}])
_ = model(*_variants[-1])                                          # timed 用全新的一份
```

⚠️ `tools/detect_stateful_impl.py` **仍然必需** —— 它堵的是 `self._cache = result` 这种
不看输入的缓存,输入轮换对它无效。两者射程互补,缺一不可。

### Q4 hooks —— 决定:**v1 不接**(分析留档)

**`skill_script_hook.py`(挂 `PreToolUse: Bash`)**:命中 9 个脚本名就
`subprocess` 自己跑一遍 → `deny` + 把 exit_code/stdout/stderr 回注上下文。
即**"代为执行"**。直接跑 `evaluate_ascendc.sh` / `verification_ascendc.py` / `performance.py`
会绕过我们的抢卡、预算、判分基准注入、退化检测 —— **照搬等于把绕过合法化**。
可复用的只有它的 `should_intercept` / `_LEADING_PREFIX`(能识破
`ENV=VAL cd dir && python3 x.py` 这类前缀绕过)。

**`doc_gate.py`(挂 `PreToolUse: Edit|Write` + `PostToolUse: Read|Skill`)**:
把"改 kernel 代码"变成配额资源,配额用"读了多少 asc-devkit 文档"兑换。

```python
KERNEL_SEGMENTS = ("/kernel/", "/op_kernel/", "/op_host/")   # 受管辖
KERNEL_EXTENSIONS = (".cpp", ".h", ".hpp", ".cc")
DOC_SEGMENTS = ("/asc-devkit/docs/", "/asc-devkit/examples/")
BUDGET_BY_COVERAGE = {0: 0, 1: 10, 2: 20, 3: 35, 4: 50}      # 覆盖类数 → 允许改几次
```

4 类基础文档(矢量编程流水线 / TBuf-TQue 管理 / API 参考 / 官方示例代码),
每类读过任意一篇即算覆盖;覆盖 0 类 → `permissionDecision="deny"`,一行 kernel 都改不了。
另有 `precision_gate`:调用精度类 skill 后额外发放一笔编辑额度。

**不接的理由**(不是"它不好"):

1. **不接成本为零** —— 不装这个 hook,行为与现在完全一致。
2. **接的成本高且不可逆** —— 必须先引进 `asc-devkit`(196 MB);没有 devkit 就永远覆盖 0 类
   → 配额 0 → **所有 kernel 编辑被全拒,agent 直接瘫痪**。
3. **它与 RL 的塑造方式相反** —— 它为"真人开发别不看文档瞎写"设计,用**硬 deny** 塑造行为;
   RL 里策略由 reward 塑造,硬 deny 只让模型学到"被拒了",学不到"为什么该先读文档",
   同时直接压缩探索空间。而且它的编辑配额会与我们的评测预算**两套叠加**。

**值得偷师的一点**:`--mode post` 追踪 Read、按"读了什么"打状态标签这套记账机制,
可以拿来做**观测**(而不是拦截)—— 看模型有没有读该读的文档、读文档与 reward 的相关性。
零风险,建议单独做。

---

### F(agent 自行 cmake/make 撞 include)—— **结论:agent 违规,不是模板问题**

本机 CANN 9.0.0(与 session 日志里的路径同版本)实测,agent 报"找不到"的头文件**都在**:

```
kernel_operator.h → /usr/local/Ascend/cann-9.0.0/x86_64-linux/tikcpp/tikcfw/
kernel_tpipe.h    → /usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include/basic_api/
kernel_macros.h   → /usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/impl/basic_api/
```

它们由 AscendC 的 cmake 工具链(`ascendc_kernel_cmake`)负责注入,裸 `g++` 指错目录就找不到。

⚠️ **更正(2026-07-28,`ascendc-sandbox:v1` 容器内实测)**:先前写的"`-I…/include/ascendc`
那个目录根本不存在"是**错的**。sandbox 里 `kernel_operator.h` 就在
`x86_64-linux/include/ascendc/basic_api/kernel_operator.h` —— 目录存在,头文件在**下一层**
`basic_api/`。agent 差的是一级目录,不是路径瞎写。
(先前的查证是在另一个容器里做的,那里 `find | head -1` 命中的是 `tikcpp/tikcfw/` 那份副本。)
结论不变,理由要改。

**反证**:jn5za7tz 走固定入口时
`Step2 compile + install (no NPU)` 顺利通过并进到 Step2b —— **官方路径编得过**。
⇒ 不是模板缺陷,是 agent 绕过固定入口自己 `make` 造成的。护栏层面归入 skill-1 的 hook。

---

## 问题清单(按现象,含证据细节)

| # | 问题 | 类别 | 状态 | 影响面 |
|---|---|---|---|---|
| A | CLAUDE.md 指的 `ops-lab/` 路径在容器里不存在 | canonical 文档 | 确证 | **每个 session** |
| B | 参考资料清单只给文件名、不给子目录 → agent 猜路径 | canonical 文档 | 确证 | 高频 |
| C | `ascendc-operator-project-init` 模板层级被反复猜错 | canonical 文档 | 确证 | 高频 |
| D | `archive_tasks` 目录不存在但 CLAUDE.md 说它有 | canonical 文档 | 确证 | 中 |
| E | `rm -rf` 被 claude-code 安全策略挡掉,非交互环境无人可批 | infra 阻塞 | 确证 | 中 |
| F | agent 在固定入口之外自己 cmake/make,撞 include 路径 | 违规 + 可能的模板问题 | **待验** | 中 |
| G | judge 判分链把工作目录名当算子名传进 `_resolve_task_dir` | judge bug | 确证 | 污染唯一跑通的样本 |
| H | prefix merge 三条(断链不可观测 / abort 空壳 / reward 广播) | 轨迹重建 | 已记录 | 见 HANDOFF |

---

### A. CLAUDE.md 指的 `ops-lab/` 路径在容器里根本不存在

**证据**

```
operator_runtime_ascendc/CLAUDE.md:171
  **Skill 参考资料**(各 skill 独立维护,位于 `ops-lab/tilelang-to-ascendc/skills/<skill-name>/`):

operator_runtime_ascendc/CLAUDE.md:234
  cp ops-lab/tilelang-to-ascendc/skills/ascendc-operator-project-init/
     templates/ascend-kernel/csrc/utils/torch_kernel_helper.h {output_dir}/kernel/utils/
```

agent workdir 里没有 `ops-lab/`,canonical 目录 `operator_runtime_ascendc/` 里也没有。
真身在 `.claude/skills/`。

实测:**≥20 次** Bash 失败的对象是 `/opt/workspace/agent_workdir/ops-lab/tilelang-to-ascendc/skills/...`,
另有 3 次直接输出 `ops-lab not found`。

**影响**:第 234 行不是说明文字,是 **Phase 1.2 的固定动作**;`torch_kernel_helper.h` 是 kernel
编译必需的头文件。等于每个 session 在第一步就撞墙,要自己摸索到 `.claude/skills/` 才能往下走
(该文件后来被 Read 了 25 次,说明 agent 最终都绕到了正确位置,代价是白烧若干轮)。

**根因**:那个路径是 cannbot 的**宿主机布局**,已确认真实存在于
`/home/docker/cannbot-skills/ops-lab/tilelang-to-ascendc/skills/`;而 polar 起 session 时把
skill 平铺到了别处:

```python
src/polar/agent/presets/claude_code.py:24     self._config_dir = f"{RUNTIME_SESSION_DIR}/.claude"
src/polar/agent/presets/claude_code.py:50-53  cp -r {skills_path}/* {config_dir}/skills/
```

⇒ **HANDOFF 铁律 2「逐字派生」只保证内容不丢,没有配套的「路径重写」步骤。**
文末覆盖声明覆盖的是*流程*差异(Phase 2/6、8 处 evaluate 脚本),漏了**环境布局差异**这一整类
(D 是同一缺口的另一个实例)。triton 侧不受影响只是因为它的 CLAUDE.md 压根不写 skill 路径。

**修**:171 / 234 两处的 `ops-lab/tilelang-to-ascendc/skills/` → `.claude/skills/`。
canonical 实时挂载,改完新 session 立刻生效。

### B. 参考资料清单只给文件名、不给子目录

第 171-180 行写的是 "`ascendc-code-gen`:elementwise_op_host.cpp、…、GUIDE.md、data-copy-api.md…",
但这些文件分属 `templates/` 和 `references/` 两个子目录,清单里看不出来。实测猜错:

| agent 猜的路径 | 次数 | 真身 |
|---|---|---|
| `ascendc-code-gen/GUIDE.md` | 7 | `ascendc-code-gen/references/GUIDE.md`(之后成功 9 次) |
| `ascendc-code-gen/references/elementwise_op_host.cpp` | 1 | `templates/` 下 |
| `ascendc-code-gen/references/index-tiling.md`、`elementwise-tiling.md` | 2 | 属于 `ascendc-design-doc-generator` |
| `ascendc-code-gen/templates/reduction_op_host.cpp` | 1 | **不存在**;只有 elementwise / index / index_per_elem / pool_ndhwc / row / sort 六套 |

**根因**:第 171-180 行是**人手写的摘要,不是从目录树生成的**;而且仓库里没有任何东西把文档里的
路径与真实文件对账 —— `check_render_contract.py` 检的是 DockerRuntime render 契约,
`preflight.sh` 不检文档。写漏子目录、写上不存在的 `reduction_op_host.cpp`,没有闸门会报警。
更深一层:一旦决定「把 skill 路径暴露给 agent」(见下文入口机制),CLAUDE.md 就从文档变成了
**接口**,而这个接口没有测试。

**修**:清单补成完整相对路径;顺手删掉清单里不存在的模板名。最好由脚本从目录树生成并进 preflight。

### C. `ascendc-operator-project-init` 模板层级被反复猜错(≥10 次)

agent 试过 `templates/ascend-kernel/setup.py`、`ascend-kernel/register.cpp`、`ascend-kernel/ops.h`、
`csrc/setup.py`,全部失败。真实结构:

```
templates/ascend-kernel/
├── CMakeLists.txt   build.sh
├── csrc/            CMakeLists.txt  ops.h  register.cpp
│   ├── utils/       torch_kernel_helper.h
│   └── ops/helloworld/{op_host,op_kernel}/
├── python/ascend_kernel/setup.py
├── tests/  third_party/
```

CLAUDE.md 只写了 "templates/ascend-kernel/(完整项目模板)"。

**根因**:与 B 同源(手写摘要 + 无对账),再叠加 A —— Phase 1.2 给的 `cp` 命令源路径本身就是错的,
agent 连一个可用的起点都没有,只能自己摸层级。

**修**:补一棵 5 行目录树。

### D. `archive_tasks` 目录不存在但 CLAUDE.md 说它有

```
operator_runtime_ascendc/CLAUDE.md:130
  - archive_tasks 目录是历史成功任务,可作为参考实现
```

cannbot 原版逐字拷贝带过来的,本环境没有。实测 5 次去找
`/opt/workspace/agent_workdir/archive_tasks`。

**根因**:同 A —— cannbot 原版对**环境设施**的假设(宿主机上有历史任务归档目录)被逐字拷进来,
覆盖声明没列这一条。A 和 D 是同一个系统性缺口的两个实例,建议按「环境布局差异」整类过一遍
CLAUDE.md,而不是逐条打补丁。

**修**:删掉该行,或按 HANDOFF「本环境对 cannbot 原版工作流的覆盖」的惯例在文末契约区写明本环境无此目录。

### E. `rm -rf` 被 claude-code 安全策略挡掉(8 次)

```
Dangerous rm operation detected: '<WD>'
This Bash command contains multiple operations. The following part requires approval:
  rm -rf <WD>/output/3_Add/kernel/build && mkdir -p ...
```

被挡的多数是合理动作(清 `{op}/kernel/build` 重编)。**非交互环境没有人能批准**,所以这条路径
永远失败。固定入口的 `--incremental` 管的是 `judge_out/work`,补不上这个缺口。
(注:在 build 目录里执行的 `rm -rf *` 被挡是挡对了,不要一起放开。)

**根因:不是 polar 配错。**polar 已经传了 flag、也设了沙箱标记:

```python
src/polar/agent/presets/claude_code.py:62   "--dangerously-skip-permissions"
src/polar/agent/presets/claude_code.py:84   "IS_SANDBOX": "1"
```

flag 已传而 `rm -rf` 仍被拒 ⇒ **claude-code 对破坏性命令的守卫独立于 permission 体系,
该 flag 覆盖不到**(由这两条证据推出,属推断);复合命令 `rm -rf X && mkdir -p X` 还会被拆成
多段分别要批准,非交互模式(`--output-format=stream-json`,无 TTY)无人应答 → 自动拒。
所以**只能绕过,改配置没用**。

**修**:二选一 —— ① 在 session 的 settings.json 里把 workdir 内的 `rm -rf` 加进 allow;
② 给 `ascendc_eval_pipeline.sh` 加 `--clean-kernel-build`,让 agent 不必自己 rm。

### F. agent 在固定入口之外自己 cmake/make(待验)

实测 `make -j` / 手写 `cmake ..` 失败,报缺头文件:

| 头文件 | 次数 |
|---|---|
| `kernel_operator.h` | 10 |
| `kernel_tpipe.h` | 7 |
| `kernel_macros.h` / `ATen/ATen.h` | 5 / 5 |
| `torch_npu/csrc/core/npu/NPUStream.h` | 3 |
| `Python.h` / `torch/library.h` / `graph/types.h` | 2 / 1 / 2 |

还有一次是 prompt 明令禁止的 API 探针:

```bash
g++ -I/usr/local/Ascend/cann-9.0.0/x86_64-linux/include/ascendc -E -x c++ - <<'EOF'
#include "kernel_operator.h"
```

**两种可能没区分开**:(a) 纯 agent 违规(该走固定入口);(b) 模板 CMakeLists 的 include 路径在
`ascendc-sandbox:v1` 这个镜像里对不上,agent 自己 cmake 只是把问题提前暴露。
**验法**:直接用 `templates/ascend-kernel/` 编一次 helloworld,五分钟能分清是谁的锅。

### G. judge 判分链把工作目录名当算子名传进 `_resolve_task_dir`

`polar_20260727_151247` 里唯一真正走到对拍的样本(session `sk-polar-jn5za7tz` / `ses_35a70334`,
算子 `3_Add`):

```
verification_ascendc.py:408 _resolve_task_dir
  FileNotFoundError: Cannot find task directory for op 'work'
```

传进去的是工作目录名 `judge_out/work`,不是算子名 `3_Add`。结果被记成 `correctness_failed`
拿 0.3 分。**这不是模型写错算子,是判分链自身的参数传递 bug**,而它污染的恰好是目前唯一一条
跑到对拍的样本。

**触发命令(从该 session 的 transcript 里找到)**:agent 手工打了包 ——

```bash
cd output && tar -czf submission/3_Add_impl.tar.gz -C 3_Add .
```

`-C 3_Add .` 把 **`3_Add/` 的内容**放到 tarball 根,于是:

```
解包 → model_new_ascendc.py 落在 $WORK/ 而非 $WORK/3_Add/
ascendc_eval_pipeline.sh:251  TASK_DIR="$(dirname "$MNA")"           → $WORK
ascendc_eval_pipeline.sh:256  OP_DIR_NAME="$(basename "$TASK_DIR")"  → "work"
ascendc_eval_pipeline.sh:396  run_npu_phase verify ... "$OP_DIR_NAME"
verification_ascendc.py:408   _resolve_task_dir("work") → 两条候选路径都不存在 → FileNotFoundError
```

**根因在 pipeline 侧**:`OP_DIR_NAME` 是从 tarball 的**内部布局反推**的,而 `--op_name`
(`3_Add`)明明已经从命令行传进来了,**两者从不对账**。任何 `-C {op} .` 形状的包都会退化成 `work`。
(`pack_submission.sh:90-94` 自己打的包是对的 —— `cd "$WORKDIR" && tar czf ... "$OP_NAME"`;
出问题的是 agent 绕过固定入口手工打的那份。)

**连带一处:同一形状让隔离闸静默失效**

```bash
ascendc_eval_pipeline.sh:265  _TOP_REL=$(realpath --relative-to="$WORK" "$TASK_DIR" | cut -d/ -f1)   # → "."
ascendc_eval_pipeline.sh:266  if [[ ... "$_TOP_REL" != "." ... ]]; then    # 条件不成立,整段跳过
```

HANDOFF 里列为头号 reward-hack 闸门的「铲掉 `{op}/` 之外一切」在这种包形下**根本没执行**。
skills 那条被第 321 行 `rm -rf "$SK"; mkdir -p "$SK"`(先铲后铺)兜住了,所以这次没造成实际
危害,但闸门语义是失效的 —— 修 `OP_DIR_NAME` 时要一并处理 `TASK_DIR == $WORK` 这个退化分支。

**调试陷阱**:polar 把 tarball 存成 `artifacts/operator_judge/submission_impl.py` —— triton
单文件时代的命名遗留,文件内容其实是 gzip 二进制。

同 run 另外 7 个 session 全是 `submission_missing`(0.2 地板),3 个
`termination_reason=pipeline_budget_exceeded` —— 这批的根因尚未定位,待查。

### H. prefix merge / 轨迹重建

三条已写入 `ASCENDC_RL_HANDOFF.md` 的「遗留:prefix merge / 轨迹重建」章节,此处只留指针,
避免两份文档漂移:

1. **断链不可观测** —— 分组阶段起新链不写 `break_reasons`(线上恒为 `{}`),`chains_total` 把
   「真 subagent / 重渲染断链 / abort 空壳」三类混在一起。
   **根因**:`break_reasons` 只在 finalize 阶段记录(`_finalize_chain` 里那 5 处 break);
   分组阶段起新链走的是 `_find_extendable_chain` 返回 `None` 的**正常路径** —— 真 subagent
   也走这条,设计上把「新链」当正常事件所以没打点。缺的不是打点位置,是**区分「正常新会话」
   与「异常断裂」的判据**(system prompt 是否相同)。

2. **abort 空壳 trace** —— `polar_20260725_115157` 的 805 条 trace 里 244 条(30%)是
   `prompt_ids=[] / finish_reason="abort"` 的空壳,各自成链,顶高 `chains_total`。
   **根因**:`record_filters._has_empty_choices` 只检**形状**不检内容 ——
   `not (isinstance(choices, list) and choices and isinstance(choices[0], dict))`;
   abort 记录的 `choices[0]` 是个合法 dict(`finish_reason:"abort"`,内容空),形状检查放行 →
   进分组 → `prompt_ids=[]` 永不匹配任何 tip(`_find_extendable_chain` 有 `0 < n` 守卫)→ 各自成链。

3. **reward 广播** —— `gateway/node.py:1002-1006` 把 `outcome_reward` 广播到每条 trace,
   一个 session 断成 N 链 = N 个同 reward 样本。
   **根因是结构性不匹配**:judge 是 **session 级**判分(一个 tarball 一个分),
   `EvalResult.trace_rewards` 恒为 `None`,`_merge_eval_result` 只能广播 `outcome_reward`;
   而 prefix merge 会把一个 session 拆成 N 条 trace。**「一个 reward」与「N 条样本」之间
   没有归一化层。**

---

## 关键背景结论:Skill 入口 vs Read 入口

解释「为什么 triton 断链、ascendC 不断链」的机制,同时也是上面 A/B/C 三条的成因。

已知触发条件:claude-code 调 `Skill` 工具后会用 qwen3.6 模板重渲染下一轮的历史上下文,
该动作**丢掉历史里的 CoT**,`<|im_start|>assistant\n<think>\n` 前缀消失 → prefix merge 的严格
token 前缀检查失败 → 断链。实测两侧对 skill 内容的消耗量几乎一样,**区别只在入口**:

| | ascendC(64 session) | triton(86 session) |
|---|---|---|
| Read/Grep 命中 `skills/` 路径 | 486 / 792 = 61% | 476 / 727 = 66% |
| 有调 `Skill` 工具的 session | **5(8%)** | **72(84%)** |
| 首次读 skill 文件早于首次调 Skill(或从不调) | **57(89%)** | 3(3.5%) |
| 直接 `Read` 到 `SKILL.md` 本身 | **47 次** | **0 次** |
| 能对上轨迹的 Skill session 断链率 | **3/3** | — |

- **triton**:入口只能是 Skill 工具(`SKILL.md` 从没被 Read 过,是工具注入的),之后才顺着
  SKILL.md 指的路径去 Read `references/`。CLAUDE.md 里 `skills/` 只出现 1 次,还是禁令。
- **ascendC**:入口是 Read。CLAUDE.md 有 16 处 `skills/` 路径且给到文件名级别,
  Read 成了最短路径,Skill 工具沦为可选。

⇒ **决定断链率的是「给不给 skill 文件路径」,不是 CLAUDE.md 的长度或伪代码内联。**
A/B/C 三条正是这条路线的账单:路径给得不准,agent 就一路猜。修好它们会让 Read 路线更稳固;
反过来若把路径清单删掉,断链率会立刻回到 triton 水平。

⚠️ 这个行为**不是硬保证**:已有 8% 的 session 调了 Skill,而且这是未经 RL 训练的初始策略,
reward 里没有任何一项惩罚调 Skill,训起来之后会不会漂移无人担保。

---

## 已核实推翻的假设(别再重新推导一遍)

| 曾经的说法 | 核实结果 |
|---|---|
| 「ascendC agent 不调 Skill,是因为 CLAUDE.md 已内联了 skill 的全部内容」 | **错**。11 个 skill 合计 ≈912 KB(`ascendc-precision-tuning/SKILL.md` 单个 51,863 字符就比整份 CLAUDE.md 的 45,774 大);CLAUDE.md 只有编排骨架,模板 / API 手册 / TileLang 映射表 / 精度知识库一个字都没有。真实原因是入口不同,见上一节 |
| 「Skill 工具是不是没在 ascendC 的 frontmatter 里注册」 | **不是**。两边 skill 都正常出现在 session 的 `skill_listing` attachment 里,且确有 5 个 session 成功调用。frontmatter 格式确实不同(triton 是 `tools.skill: true`,ascendC 是 cannbot 的 `mode: subagent` + `permission:`),但不影响可用性 |
| 「`judge_out/metrics_error.log` 读不到 10 次 = infra bug」 | **不是**。`write_metrics` 在所有失败路径都会写该文件,缺失说明 agent 在跑 pipeline **之前**就去读了(prompt 明写 validation 失败后才读)。`compile.log` / `verify.log` 同理,确实是 pipeline 产物,只是当时还没生成 |
| 「`preformance.json` 是拼写 bug」 | **不是**。上游 `ascendc-performance-analyzer/script/performance.py:696` 就这么写,`ascendc_eval_pipeline.sh` 和 `pack_submission.sh` 两边都同时匹配两种拼法,是刻意兼容 |

---

## 建议的处理顺序

1. **A/B/C/D** —— 都在 canonical(`operator_runtime_ascendc/CLAUDE.md`),实时挂载、新 session
   立刻生效,改动量小、收益直接。
2. **G** —— 定位 `_resolve_task_dir` 拿到 `'work'` 的参数传递路径。它污染的是目前唯一一条走到
   对拍的样本,不修就没有可信的判分数据。
3. **F** —— 先花五分钟用模板编一次 helloworld,把"agent 违规"和"模板 include 路径不对"分开。
4. **E** —— 权限放开或给 pipeline 加 `--clean-kernel-build`。
5. **H** —— 见 HANDOFF,按那边记的优先级走。
