# 设计：math 作为 polar 第三套 workflow（平行 legacy / cannbot，零算子改动）

状态：**待 review**。作者：Claude（复盘自己前一版错误实现后重写）。
隔离位置：worktree `/home/docker/math_polar_wt`（分支 `feature/math-judge`）+ vime worktree
`/home/docker/math_val_wt`（分支 `feature/math-pipeline-validation`）。**不碰主 checkout、不碰算子任何文件。**

---

## 0. 目标（不变）

验证 polar 接入整条 vime RL 管线是否正确：把任务从算子生成换成 **DAPO-17k 数学题**，
仍走 claudecode 后端 + 轨迹捕获/rebuild，看 reward 能否训上去。判据：reward 上升 + `train/tis≈1`。
**不是训模型，是给管线冒烟。**

---

## 1. 机制真相（我上一版没读懂，这次读透了）

polar 的运行拓扑**不是手写 `topology.yaml`，而是代码生成的**：

```
POLAR_PROFILE=profile.X.yaml
   └─> deploy/ascend_operator/tools/load_polar_profile.py
         读 profile 字段 → 生成 effective_topology.yaml（写到 run_artifact_dir/）
         └─> polar 用这个 generated topology 启动
```

- **`deploy/ascend_operator/topology.yaml` 根本不参与运行**（每次启动被 profile 重新生成覆盖）。
  → 我上一版手改 topology.yaml 加 `math_npu` 是**纯无用功**，且污染算子 checkout。这就是你问
  "为什么会改 topology"的答案：**永远不该改它。**
- 隔离缝 = `operator_runtime.workflow` 字段。当前只支持 **`legacy` / `cannbot`**
  （`load_polar_profile.py:145` 硬校验）。每套 workflow 在生成器里各有自己的分支：
  - `_operator_runtime_dir(workflow)` —— 用哪套 runtime 目录
  - `_operator_prepare(workflow)` —— agent 工作目录怎么搭
  - `_evaluator_config(workflow)` —— 判题怎么配
  - evaluator strategy（`:261` 现硬编码 `"operator_judge"`）

→ **正确姿势：math 加成第三套 workflow `math`，和 legacy/cannbot 完全平行，各自隔离。**

---

## 2. 上一版失败根因复盘（存档，避免重犯）

1. 手改 `topology.yaml` → 不参与运行，空忙 + 污染。
2. 把算子的 `prepare` / `.agents` / `tools` 当"算子内容"删了 → 那其实是**跑 claude 的基础设施**，
   删了 agent 起不来。
3. `math_runtime` 只放了个 skill（残缺），没有 CLAUDE.md 等 → workdir 搭不起来。
4. 实跑用的是 `profile.vime.yaml`（= operator_npu / workflow=legacy），它的 prepare **硬要上传
   `op_tasks/{op_name}.py`**；math 的 `op_name=math_000123` 没这文件 → prepare 失败 →
   **agent 不产出 → 0 trace**。（这才是空 trace 真根因，与端口无关。）

---

## 3. 数据流（已逐行代码核实，是设计命门）

### 3.1 ground-truth 答案怎么到 judge（leak-safe，不进 agent prompt）
```
vime prep：sample.metadata = { op_name, answer }        # scripts/prep_dapo_math.py
  └─ operator_profile.py:194  request.metadata["sample_metadata"] = deepcopy(sample.metadata)
       └─ pipeline.py:502      trajectory.metadata["task_metadata"] = dict(request.metadata)
```
→ **答案最终在 `trajectory.metadata.task_metadata.sample_metadata.answer`**。
→ **改动点：math_judge 读这个路径**（当前读 `task_metadata.answer`，错，要改）。

### 3.2 math 不需要任务文件（关键，解释为何不碰 upload_file）
`operator_profile.py:64` `_attach_task_source`：`if task_source is None: return rendered_profile`。
→ OperatorSampleRequest 的 `sample.task_source` 保持 `None`（默认）→ 完全不碰任务文件缓存/上传。
→ 前提：**math workflow 的 prepare 里不能有 `upload_file {op_name}.py`**（否则去传不存在的文件）。

---

## 4. 逐文件改动（全部在隔离 worktree，主 checkout / 算子零改动）

### A. 新增 `deploy/ascend_operator/profile.math.yaml`
以 `profile.vime.yaml` 的 service 块为基线（推理端点跟 vime 一致），关键差异：
```yaml
service:                          # 与 vime 一致：vllm 后端 + model_served + 推理端点
  inference_engine: vllm
  model_served: /home/docker/Qwen3.6-35B-A3B
  sglang_router_url: http://80.48.5.88:8001   # 单引擎则 :15000；开 LB proxy 则 :8001（沿用 vime 现状）
paths:
  operator_runtime_dir: math_runtime          # ← 指向 math 自己的完整 runtime
operator_runtime:
  workflow: math                              # ← 第三套 workflow（新）
  budget: { generation_max: 5, optimization_max: 2, interval_seconds: 2 }
  npu_lease:
    enabled: false                            # ← 数学题纯 CPU 解，agent 不租 NPU（与算子最大区别）
operator:
  profile: math_npu                           # generated topology 的 default_operator_profile
  timeout_seconds: 1200.0
  agent:
    model_name: claude-opus-4-5
    max_turns: 20
    append_system_prompt: "解一道整数答案的竞赛数学题。可用 Bash 跑 Python 计算/验证。
      逐步推理后，最后单独一行给出 'Answer: <整数>'。非交互，勿向用户提问。"
  evaluator: {}                               # math_judge 纯 in-process，无 judge_command
```

### B. `deploy/ascend_operator/tools/load_polar_profile.py` —— 加 `math` 分支（4 处）
1. `:145` 允许集合加 `"math"`：`if workflow not in {"legacy", "cannbot", "math"}`。
2. `_operator_runtime_dir`：math 走默认分支即可（profile 的 `operator_runtime_dir: math_runtime`），
   无需改（或显式加 `if workflow=="math": return repo/"math_runtime"`）。
3. `_operator_prepare`：加 `if workflow == "math":` → **只搭 workdir，无 upload_file**：
   ```python
   return [{"type": "exec", "cwd": workdir,
            "command": "mkdir -p output && cp /opt/canonical/CLAUDE.md CLAUDE.md "
                       "&& ln -sfn /polar/session/.claude .claude"}]
   ```
4. evaluator 生成块（`:260-269`）：加 math 分支 → **in-process，无 docker runtime**：
   ```python
   if workflow == "math":
       evaluator_block = {"strategy": "math_judge",
                          "config": {"format_bonus": 0.0}}   # 无 refresh_runtime / runtime
   else:
       evaluator_block = { ...operator_judge 原样... }
   ```
   （strategy `:261` 由此参数化，不再硬编码。）

### C. 新增 `math_runtime/`（完整、隔离的一套 runtime，不复用 operator_runtime）
```
math_runtime/
  CLAUDE.md                     # 数学 agent 的工作说明（简短：解题→Answer: <int>）
  skills/
    math-solver/SKILL.md        # 已有（feature/math-judge）
```
无需 `.agents/` / `tools/` / NPU 相关（算子才需要）。skills 经 `skills_path=/opt/canonical/skills` 挂载。

### D. `src/polar/trajectory/` —— 注册 math_judge（已在 worktree，仅需一处修正）
- `evaluator/math_judge.py`（已有）：**改答案读取路径** → `task_metadata.sample_metadata.answer`。
- `registry.py` + `evaluator/__init__.py`（已在 worktree）：`register("math_judge", MathJudgeEvaluator)`。

### E. vime 侧（已在 `feature/math-pipeline-validation`，无新增逻辑）
- `scripts/prep_dapo_math.py`：DAPO → operator_samples；prompt 保持 list；补
  `metadata.op_name`（安全文件名）+ `metadata.answer`（judge-only）。
- 提交走 **operator_samples**；vime **不传 profile**（polar 用 `default_operator_profile`）。
- 启动：`start_math.sh`（一键）。**vime 只需要一个 polar url**，profile 全在 polar 侧。

---

## 5. 隔离保证（对照你的"完全隔离、不走捷径"）

- 生成器新增的是**独立 `math` 分支**，legacy/cannbot 代码路径逐字不变。
- math 用**独立 profile 文件 + 独立 runtime 目录 + 独立 evaluator strategy**，与算子零共享。
- 全部改动在 worktree；主算子 checkout 已复核 **8/8 核心文件逐字未动、HEAD 未移动**。
- 跑 math：`POLAR_PROFILE=profile.math.yaml`；跑算子：`POLAR_PROFILE=profile.vime.yaml`。**换 profile 即换整套**，正是你要的机制。

---

## 6. 验证计划

1. 干跑生成器：`load_polar_profile.py --profile profile.math.yaml` → 检查生成的
   effective_topology：default_operator_profile=math_npu、evaluator.strategy=math_judge、
   prepare 无 upload_file、无 NPU 租借。
2. 单条 math 任务：确认 session 建目录、agent 有 `/v1/messages` 调用、trace≥1。
3. 查 `EvalResult.metadata`：`gt_found=True`（答案确实经 sample_metadata 到了 judge）、
   `pred/gt/correct` 合理。
4. 小规模（bs8×N8=64）：reward 有组内方差(非 zero-std)、`train/tis≈1`、reward 趋势上升。

---

## 7. 决策 / 开放问题

已定（2026-07-17，你拍板）：
- **Q1 端口 → 沿用 :8001**（单引擎也走 LB proxy，与当前 vime 一致）。
- **Q4 → 选"生成器加 math 分支"**（在隔离 `math` 分支上改 `load_polar_profile.py`，算子线不受影响；
  最贴合"换 profile 即切"机制，比静态 topology 绕开更少发散）。

待 review 时确认：
- **Q2 CLAUDE.md 来源**：prepare 里 `cp /opt/canonical/CLAUDE.md`——需 math_runtime 自带一份；
  内容我拟最简版（解题纪律），非算子那份。
- **Q3 evaluator in-process**：需确认 polar 对"无 runtime 的 evaluator"支持（math_judge 纯解析）。
  若 BaseTrajectoryEvaluator 强制 runtime，则退化为给一个空 docker runtime（成本极低）。
