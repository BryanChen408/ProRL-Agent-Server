# 迁移到官方新版 skills(tilelang2ascendc-ops-generator)—— 节奏规划

分支:`feat/ascendc-rl-t2a`(t2a = tilelang2ascendc),从 `feat/ascendc-rl` @ `47e31856` 拉出。

## 目标基座

```
仓库    https://gitcode.com/cann/cannbot-skills   (官方,非 chenshushu2020 那个 fork)
分支    master
commit  f76269e   2026-07-28 16:40
plugin  plugins-community/tilelang2ascendc-ops-generator
```

与现基座(`chenshushu2020/cannbot-skills` @ `br_asc_profiling` @ `6cf50e29`,2026-05-28)
是**两个仓库**,且我们 pin 的那个 commit 已因 `br_asc_dev` 易主而不在该分支上。

## 为什么迁(已核实的收益)

| | 现基座 | 新基座 |
|---|---|---|
| skills 位置 | `ops-lab/tilelang-to-ascendc/skills/`(跨目录相对路径,agent 找不到) | **plugin 内部 `skills/`** —— 适-3 天然消失 |
| `hooks.json` | 只注册 `SessionStart`,拦截机制空挂 | 注册 `SessionStart` + `PreToolUse` + `PostToolUse` |
| skill 名笔误 `ascendc-operator-code-gen`(7 处) | 有 | 无(该 skill 被删) |
| precision-tuning 跨 plugin 断裂(10 处) | 有 | 无 |
| `performance-analyzer` 的 `script/` vs `@scripts/` | 有 | 无(换成 `ops-profiling`) |

**⚠️ 但仍有 6 条上游债照样带过来**:skill-3b(`script/` 单数)、skill-4a(`AscendC_knowledge/`)、
skill-4b(trace-recorder 引用不存在的 scripts)、skill-6(asc-devkit 路径)、skill-7(参考清单)、
适-1a(Phase 0 教设 `ASCENDC_RT_VISIBLE_DEVICES`)、适-2(`archive_tasks`)。见 `SESSION_AUDIT_20260728.md`。

## 关键前提(已核实,决定工作量)

- **交付物形态完全没变**:`{op}/kernel/{CMakeLists.txt,setup.py,ops.h,register.cpp,
  op_host/<op>.cpp,op_kernel/<op>.cpp,utils/torch_kernel_helper.h}` + `model_new_ascendc.py`。
  简单路径(`ops-direct-invoke`)与复杂路径(`translator`)产出同一套,简单路径只多 `docs/REVIEW.md`。
  ⇒ **数据集、判分链、judge 结构假设全部直接可用。**
- Phase 0-7 骨架不变,只换了两个分支的内部做法。
- 我们 pipeline 依赖的 4 个脚本全在(`tilelang2ascend-translator/scripts/`),
  `build_ascendc.py` **零差异**,`verification_ascendc.py` 接口兼容(首个位置参数仍是算子名/目录)。

## 工作量分布

```
🟢 原样搬  ~700 行   抢卡 / 解包 / metrics.json / 隔离 / purge / 完整 error log / polar 侧全部 / preflight
🟡 改路径  ~360 行   pipeline 的 SK 路径与 skill 名、pack、prepare、profile/topology、perf 字段
🔴 重对齐   ~90 行   CLAUDE.md 覆盖声明按新 Phase 重写 + msprof 输入轮换补丁重做
```

---

# 阶段划分

**总原则:每阶段末有一个可验证闸门,不过不进下一阶段。**
**并行策略:新建 `operator_runtime_t2a/`,`operator_runtime_ascendc/` 一字不动** ——
符合铁律 1「可随时切回」,切换 = 换 profile 的 `paths.operator_runtime_dir`。

## 阶段 A — 基座就位

**做**
- 把 plugin 的 `skills/` + `agents/tilelang2ascendc-kernel-generator.md` 拷进 `operator_runtime_t2a/`
- `tools/` 从 `operator_runtime_ascendc/` 整体搬(🟢 那 700 行)
- `runtime/prepare_operator_workdir.py`:删 `_link_cannbot_skills_path()`(新基座无 `ops-lab/`),
  skills 源改成 plugin 的 `skills/`

**闸门 A**:`prepare_operator_workdir.py` **干跑通过**(不起 polar)——
产出的 workdir 里 `input/{op}.py+.json`、`tools/`、`.claude/skills/`、`CLAUDE.md` 齐全,
`skillOverrides` 正确写入。用仿真 workdir 跑一遍路径扫描,悬空引用数记录为基线。

## 阶段 B — 判分链接通(**最关键**)

**做**
- `ascendc_eval_pipeline.sh`:`SK` 路径 + skill 名(`ascendc-translator` → `tilelang2ascend-translator`)
- Step3 换成 `ops-profiling/scripts/msprof_perf_summary.py --quick`
- perf 字段:`overall_speedup` → `mean_speedup` / `geomean_speedup`(2 处:pipeline:428、pack:82;
  **决定用 mean 还是 geomean** —— 新版日志里 geomean 标注为"主指标")
- `pack_submission.sh`:skill 名

**闸门 B**:**手工造一个已知正确的 tarball**,只在 judge 侧跑固定入口,
`metrics.json` 出 `ast_check_ok=true / correctness_ok=true / perf_data.speedup_vs_torch` 有值。
⚠️ 这是全流程最关键的闸门 —— 不通过,后面所有阶段都是空的。

**风险点**:msprof 需要什么权限/环境(`msprof` 可执行是否在 `ascendc-sandbox:v1` 里、
是否需要额外 device 权限)**尚未验证**,是本阶段的头号未知。

## 阶段 C — 与基座无关的 P0

四条全在我们自己代码里,新旧基座通用。详见 `SESSION_AUDIT_20260728.md`。

| # | 改什么 | 位置 |
|---|---|---|
| 适-4 | `OP_DIR_NAME` 以 `--op_name` 为准 + 处理 `TASK_DIR == $WORK` 退化分支(隔离闸同时失效) | `ascendc_eval_pipeline.sh:251-267` |
| 适-6a | session 以 upstream 5xx 收尾 → 判 infra、retry 不计分(**建议在 gateway 侧打标**,比解析 transcript 干净;插入点待定位) | polar gateway / evaluator |
| 适-6b | 取件循环分别捕获异常,真实原因写进 error | `operator_judge.py:209-215` |
| 适-6c | `.best.tar.gz` 存到 agent 够不到的位置,或 pipeline 结束即取件 | pack + judge 取件 |

**闸门 C**:构造性验证 ——
① `tar -C {op} .` 形状的包不再退化成 `work`,且隔离闸不再被跳过;
② 模拟 upstream 5xx,session 被判 infra 而不是 `reward=0.2`。

## 阶段 D — 护栏与补丁

**做**
- **I-6 重做**:msprof wrapper 输入轮换(`_WRAPPER_SCRIPT_TEMPLATE`,约 5 行)——
  让 timed 那唯一一次用 warmup 没碰过的输入。新版命中率 100%,比 pin 版更需要
- 上游债(D1 原则已授权直接改 canonical 副本):skill-3b / 4a / 4b / 6 / 7
- 适-1a(删 Phase 0 设 `ASCEND_RT_VISIBLE_DEVICES`)、适-1b(`{output_dir}` → `{op_name}/`)、适-2
- **CLAUDE.md 覆盖声明重写** —— 对着新版 Phase 逐条数(`evaluate_ascendc.sh` 出现处、
  Phase 2/6 跳过、判分契约)。骨架和每条理由可从旧版直接复用

**闸门 D**:各造一个反例,验证两道防 hack 闸门在新基座上仍然有效 ——
① 纯 torch 实现 + 注释写 `torch.ops.npu` → AST 退化检测拦住;
② 带 `self._cache` 的实现 → `detect_stateful_impl.py` 拦住;
③ 按输入缓存的实现 → 输入轮换后 speedup 不再虚高。

## 阶段 E — 单算子冒烟

**做**:用刚提交的 `gen_ascendc_tasks.py --ops 3_Add` 生成单算子任务集,起 4-8 session。

**闸门 E**:四件事同时成立 ——
1. 上游 5xx 被判 infra,不进 0.2 地板
2. `output/submission/{op}_impl.tar.gz` 真的产出并被 polar 取到(不是 `submission_missing` 短路)
3. judge 出真 metrics(`judge.stdout.log` 存在 = judge 容器真起过)
4. `speedup` 正常落进 `metrics.json`

## 阶段 F — 放大与观察

**做**:扩到 31 算子,长跑。

**观察项**(不是闸门,是要盯的指标):
- **reward 分布** —— msprof 测纯 kernel 时间、排除 host 开销,同一实现的 speedup 通常高于
  wall-clock,`0.75+0.25*tanh(ln speedup)` 的分布会整体上移。**是否要调阶梯,跑完这批再定**
- **断链率** —— 新基座下 agent 调 Skill 工具的频率会不会变(决定 prefix merge 断链;
  参考清单仍给到文件名级别,预期仍以 Read 为主)
- **自编译率** —— 现在 38%(163 次 `cmake` / 24 session)。若居高不下,做适-9(编译反馈)

---

## 遗留:自写脚本的注释要极致精简(**兼有信息泄露风险**)

**范围**:`tools/ascendc_eval_pipeline.sh`、`pack_submission.sh`、`detect_stateful_impl.py`、
`npu_lease_exec.py`、`env.sh` —— 凡是**挂进 agent workdir、agent 能读到**的自写脚本。

**问题有两层:**

1. **信息泄露(主要)** —— 实测 agent 会 `cat tools/ascendc_eval_pipeline.sh`
   (0725/0727 的 session 里有多次)。而现在的注释把**每一道反作弊闸门防的是什么**
   写得清清楚楚,等于给模型一份攻击面清单:

   ```
   # 挡住 tarball 里塞 .claude/skills/.../verification_ascendc.py 接管判分
   # 缓存实现能拿虚高 speedup(→满分),而对拍(只调一次)和退化检测都拦不住
   # 纯 torch 实现 + 注释里写一句 torch.ops.npu → 编过 → 对拍必过 → reward 0.75
   ```

   在 RL 里这不是文档,是**提示词**。梯度只要撞上一次就会强化。

2. **prompt/turn 预算** —— 该文件 470 行、注释占 116 行;agent `cat` 一次全进上下文。

**改法**:把"为什么这么写"的长篇论证**移出脚本**,留到 `SESSION_AUDIT` / `HANDOFF` /
commit message 里(那些 agent 读不到);脚本里只留必要的机械说明
(参数含义、单位、调用契约),不写"这条防的是什么攻击"。

**注意别矫枉过正**:`--repeats 1` / `--device` 那类**改了会破坏正确性**的约束,
必须在脚本里留一行"别改 + 后果",否则后人无声改回。折中:脚本里写
「别改,理由见 MIGRATION_T2A.md」,把理由本身放到 agent 读不到的地方。

**时机**:阶段 D 之后、阶段 E 冒烟之前 —— 那时闸门都已验证,精简不会丢掉验证依据。

## 明确不做(v1)

| | 理由 |
|---|---|
| 接 `skill_script_hook.py` | 它是"代为执行"被拦脚本,会绕过抢卡/预算/基准注入/退化检测 —— 照搬等于把绕过合法化。且该绕过实测只有 2/64 |
| 接 `doc_gate.py` | 依赖 `asc-devkit`(196MB);没有 devkit 就永远覆盖 0 类 → 配额 0 → **所有 kernel 编辑被全拒**。且它用硬 deny 塑造行为,与 RL 用 reward 塑造相反,会直接压缩探索空间 |
| 引进 `asc-devkit` | 阶段 F 之后按需评估。注意 skill 里写的 `docs/api/…` 与 devkit 实际的 `docs/zh/api/SIMD-API/…` 对不上 |
| rebase 到 fork 的 `br_asc_dev` | 那是另一拨人的线,且我们 pin 的 commit 已不在其上 |

## 待定位/待验证清单

| # | 待办 | 卡住哪个阶段 |
|---|---|---|
| ~~1~~ | ~~`msprof` 在 `ascendc-sandbox:v1` 里可用性 + 权限要求~~ | **已解除,见下** |
| 2 | 适-6a 在 gateway 侧的具体打标位置(`SessionResult` 有无承载位) | 阶段 C |
| 3 | `mean_speedup` vs `geomean_speedup` 选哪个 | 阶段 B |
| 4 | `validate_ascendc_impl.py` 新旧 108 行 diff —— 退化判据是否仍等价 | 阶段 D |

---

## msprof 可用性探测结果(2026-07-28,`ascendc-sandbox:v1` 容器内实测)

探测脚本:`/home/docker/t2a_probe_msprof.sh`(v2,自动 `docker run` 进容器,
容器参数逐字取自 `src/polar/runtime/ascend.py:_DRIVER_MOUNTS` + `ascend_mount_create_args`)。

**13 项全 PASS —— msprof 路线可行,阶段 B 按计划做。**

```
msprof     /usr/local/Ascend/cann-9.0.0/bin/msprof   (source set_env.sh 后在 PATH)
容器 CANN  只有 9.0.0 一套
torch      2.8.0+cpu   torch_npu 2.8.0.post4.dev20260309   npu 可用
msprof 采集 rc=0,op_summary(5行) / task_time(8行) / api_statistic(26行) 全部生成
cmake      3.31.10(符合 ascendc-sandbox:v1 = sandbox:v1 + cmake<4)
权限       /dev/davinci* 17 节点,perf_event_paranoid=4,日志无权限类报错
```

**顺带确认的三件事:**

1. **镜像烤了默认 `ASCEND_RT_VISIBLE_DEVICES`** —— `/dev/davinci*` 有 17 个节点但
   `torch.npu.device_count()==1`。正是 `docker.py:112-114` 注释所述
   "Docker image ENV survives unless explicitly overridden, clear any baked default here"。
   代码与现实对上。
2. **pipeline 能拿到 msprof,但靠的是巧合** —— `tools/env.sh` 把
   `BISHENGIR_BIN=/usr/local/Ascend/cann-9.0.0/bin` 加进 PATH,而 msprof 恰好在同一目录。
   那个变量本意是给 bisheng 编译器用的。⇒ **阶段 B 加 msprof 步骤时必须显式解析路径**
   (优先 `$ASCEND_HOME_PATH/bin/msprof`,回落 PATH),不依赖这个巧合。
3. **更正 F 的一处细节** —— sandbox 里 `kernel_operator.h` 在
   `x86_64-linux/include/ascendc/basic_api/`(不是我先前在别的容器查到的 `tikcpp/tikcfw/`)。
   所以 agent 那条 `g++ -I.../x86_64-linux/include/ascendc` 探针,**目录是存在的**,
   只是头文件在下一层 `basic_api/` —— 它差一级目录,不是路径瞎写。
   F 的结论(agent 违规、非模板缺陷)不变,理由要改。

**探测脚本 v1 → v2 修的两个 bug**(值得记,同类脚本别再犯):

- **CANN 版本挑错**:宿主机有 8.5.1 与 9.0.0 两套,`find | head -1` 按字典序挑到 8.5.1。
  改为优先 `ASCEND_HOME_PATH`、否则 `sort -V | tail -1`。
- **`msprof --version` 不存在**(报 `unrecognized option`),改用 `--help` 探活。
- v1 还犯了个方法错误:**在宿主机上跑**(宿主机没 torch),结论无效。
  v2 检测到不在容器就自动 `exec docker run` 进去。
