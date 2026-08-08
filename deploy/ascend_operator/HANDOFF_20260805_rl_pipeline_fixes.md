# AscendC 算子 RL（t2a）排查与修复交接 — 20260805

> 一次长对话的完整交接。目标:让 RL 训练的 reward 有区分度、提交物不再出打包/判分问题。
> 主仓: `/home/docker/polar_can/ProRL-Agent-Server`(branch `feat/ascendc-rl-t2a`)。

---

## 0. 系统一句话

agent（claude-opus-4-5）在容器里写 AscendC C++ 算子 → 固定评测入口
`operator_runtime_t2a/tools/ascendc_eval_pipeline.sh` 判分
（AST退化检测→编译→注册冒烟→对拍→状态检测→测速）→
`reward_from_metrics` 阶梯给分 → GRPO。observer（`deploy/ascend_operator/tools/polar_rollout_observer.py`）是只读监控屏。

reward 阶梯（`src/polar/trajectory/evaluator/operator_reward.py`）:
`0.2 没调真op / 0.3 调了但没对拍过 / 0.4 对拍过没测速 / 0.75+0.25·tanh(ln speedup) 成功`。

---

## 1. 本轮已根治并已提交的(commit,全在 `feat/ascendc-rl-t2a`)

| commit | 修的根因 | 文件 |
|---|---|---|
| `bdc59111` | submission_missing(跑错目录):pipeline 从脚本位置反推 WORK_ROOT,不再依赖 $PWD;readlink 解软链接 | ascendc_eval_pipeline.sh |
| `f1a31060` | benchmark wrapper `model(*tensor)` 崩:get_inputs 单 case 包一层(对齐 triton resolve_inputs) | msprof_perf_summary.py |
| `02784964` | observer 识别 cached verdict / 后台启动壳 / submission_missing,不再落 unknown/command_error | polar_rollout_observer.py + 测试 |
| `9f08c3ab` | Bash 超时默认/上限提到 600s(env),prompt 英文引导前台跑评测 | claude_code.py + profile.t2a.yaml |
| `2edd5caa` | benchmark 空表(枚举 0 case):`_load_compare_cases` 加 model fallback(没 case 文件时从 model.py 数) | msprof_perf_summary.py |
| `f8fd6e68`(你们提的) | Step2a 注册冒烟(check_op_registered.py)、fail_hint 首个异常、write_metrics force_type、**set -e 修复**(`if !( set -e;..)` 里 `!` 禁用 errexit,build 失败被 setup.py 的 `|| echo` 兜底成 0 → 编译失败被错标 op_not_registered;改 `|| exit $?`)、改动3(compile/verify 错误打 stderr) | ascendc_eval_pipeline.sh 等 |
| `46d23d21`(你们提的) | kernel_skeleton 模板 + check_op_registered | 多文件 |
| 远程拉进来的 | `aada4a84` honor_cancel=False:budget 取消时 eval_prepare 照铺 golden(input_load_failed 根因修复) | node.py |

**judge 级重试(方向1)+ golden 预检补传(方向2)** 在 `src/polar/gateway/node.py`,已提交(在 f8fd6e68/46d23d21 一带)。

---

## 2. 已修但**还没提交**(在工作区,需你确认后提交)

| 文件 | 修了什么 |
|---|---|
| `operator_runtime_t2a/skills/ops-profiling/scripts/msprof_perf_summary.py` | **wrapper 缩进 bug**:我上个 wrapper 修复(f1a31060)把 get_inputs fallback 写成 4 空格基础缩进,但那段代码嵌进 wrapper 模板**模块顶层**(0 缩进)→ 生成脚本 `IndentationError` → benchmark 崩 → 空表。已改成 0 基础缩进,生成脚本 py_compile 通过。**这是 205907 里 benchmark_failed 的主因。** |
| `operator_runtime_t2a/tools/ascendc_eval_pipeline.sh` | **classify 错标**:`get_input` 裸匹配把 benchmark_failed(日志里有 get_inputs 函数名)错标成 input_load_failed(本该得 0.4 被判成 infra 重试)。收紧:get_input 需搭配「无法加载/未就位/缺失」等失败上下文。 |
| `operator_runtime_t2a/runtime/prepare_operator_workdir.py` | **骨架预生成(根治打包)**:`_instantiate_kernel_skeleton` 把 kernel_skeleton 按 op_name 实例化进 `workdir/{op}/`(占位符+文件名都替换),只在 agent 侧(`require_claude`)铺,judge 侧不铺。agent 只写 kernel 数学。 |
| `operator_runtime_t2a/CLAUDE.md` | 指引改成「骨架已预生成,只写 Compute 数学 + host tiling + forward 调用」(原来是「cp -r 复制模板」)。 |
| `deploy/ascend_operator/profile.t2a.yaml` | ⚠️ **这是你的本地配置**(model_served→-4t 变体、pool 8-11→0-7),**不是我的修复,别当我的提交**。 |

验证:4 个修复文件的测试/编译全过;prepare 9 个测试过;全量 232 过(1 个 gateway inflight 失败是 HEAD 预存在,与本次无关)。

---

## 3. 只暴露、根因还在(下一步该根治的,按 reward 影响排)

1. **abort session 记 0.2(最该根治)**:weight-update cutoff 把没跑完的 session 掐了,没打包 → judge 报 submission_missing → 记 0.2 地板分。这是把 RL 机制的损失算到 agent 头上,假信号。要在 `operator_judge.py` 把「abort 且无提交物」从「记 0.2」改成「不计入训练」。约 6/164,量不大但毒化 reward。
2. **数据集有 ~39% 先天无解题**:189 道里大量 int64 / 极小 shape(如 4 元素的 torch_subtract),AIV 不支持逐元素 int64、凑不满 32B 对齐,怎么 rollout 都同 reward、零方差纯烧预算。要体检剔除。
3. **T1 档太粗(0.30 混编不过+算错)**:GRPO 组内方差=0 的 dead group 来源。要拆:编不过 0.25 / 算错 0.35。改 `operator_reward.py`。
4. **observer 聚合窗口**:subagent completion 把主 agent pipeline 历史挤出 `files[-2:]` → precision 假 0/0。定位了没修(有界回溯)。
5. **skill 文档阅读 265K**(实测最大上下文冗余,比错误日志大):没减。要调研按需加载。
6. **judge infra(NPU 争抢/超时/传输)**:方向1/2 是兜底不是根治。

---

## 4. 验证状态(重要,别当已验)

- **能离线验的都验了**:模板加载链(真编 .so + load_library + torch.ops.npu 解析 + ModelNew 调用)、枚举修复、classify 修复、骨架实例化、gating、observer 识别。
- **没验的(需真 NPU/judge 跑一次)**:预生成骨架端到端(铺给 agent→只填数学→编过+注册+判分过)、benchmark 修好后真出 speedup(T3 可达)。**先跑一个 sign(输出 ±1,零精度风险)的小 run 验证。**

---

## 5. 关键路径

- 主仓:`/home/docker/polar_can/ProRL-Agent-Server`(t2a RL)
- triton 参照(高完成度):`/home/docker/polar_debug/ProRL-Agent-Server`
- 67 个跑通算子(另一套 harness,纯写算子非 RL):`/home/docker/output1`
- 数据集:`/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/`(189 题,`operator_tasks.simple4.jsonl` 是我筛的 4 道干净题)
- judge pipeline:`operator_runtime_t2a/tools/ascendc_eval_pipeline.sh`
- benchmark:`operator_runtime_t2a/skills/ops-profiling/scripts/msprof_perf_summary.py`
- 骨架模板:`operator_runtime_t2a/workflows/templates/kernel_skeleton/`
- reward:`src/polar/trajectory/evaluator/operator_reward.py`
- judge/gateway:`src/polar/gateway/node.py`、`src/polar/trajectory/evaluator/operator_judge.py`
- run 落盘:`output/ascend_operator/runs/<run>/{polar_sessions,rollout_results,logs}`

---

## 6. 方法论坑(别再踩)

- **扫描口径**:数 session 里的报错要用 `error_type=xxx`(真实运行)且按 run 去重,别用宽松字符串扫(会把「agent 读脚本源码」算进去,数字吹大几十倍 —— 我把 submission_missing 误判成「很常见」就是因为这个,真实只有 8 session/11 次)。
- **pid 文件不可信**:run 的 pid 是宿主命名空间的号,容器内 `kill -0` 一律 DEAD。判断 run 活着看**文件是否在增长**,别看 pid。
- **代理劫持**:本机 `http_proxy` 会劫持对 80.48.5.64 的探测(no_proxy 不含它)。直连要 `env -u http_proxy -u https_proxy ...`。
- **profile 主机值是分歧点**:本地 80.48.5.64 vs 远程 80.5.25.119,rebase 后核对 `profile.t2a.yaml` 的 service/image 段没被远程盖掉(这次是你的「双机」commit 保住了本地值)。
- **bash `if ! ( set -e; ... )` 的 `!` 会禁用 set -e** —— 传播失败要用 `|| exit $?`。

## 7. 下一步建议(一句话)

先跑一个 sign 小 run 验证「预生成骨架 + benchmark 修复」端到端通(能看到 benchmark_success / speedup),再回来做「abort 记 0.2 排除」「T1 拆档」「数据集体检」。
