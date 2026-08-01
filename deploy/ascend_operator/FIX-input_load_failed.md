# 修复方案:input_load_failed 假 INFRA + 丢弃已产出算子

状态:**待实施**(根因已双实锤,代码未改)
适用:算子场景 profile.t2a / profile.ascendc(operator_judge + ascendc_eval_pipeline)
日期锚点:2026-08-01 定位于 run 20260731-180255-795ae6 + 20260801-193002-58a116

---

## 1. 现象
- 单 run 中 ~34%(旧 run 62/180)的 session 判 `error_type=input_load_failed`。
- pipeline 自诊断把它归为 **INFRA-环境故障 → retry**(不计分、重跑一整个 ~2h session)。
- 判分侧 `judge.stdout.log` 报 `[ascendc-eval] golden missing`,即 judge 容器里 `input/{op}.py`(golden 参考输入)不存在。

## 2. 根因(双实锤:探针 + 统计 + 代码路径)

**触发链(全部有证据):**
```
agent 反复调验证 pipeline 自测 → 第 7 次调用 > 预算 generation_max=6
  → pipeline_budget_watcher 取消该 session:
       "cancel <sid>: generation pipeline calls 7>6"   （pipeline_budget_watcher.log 实证,6/6）
       gateway.log cancel reason = pipeline_budget_exceeded（6/6）
  → managed.cancel_requested = True                    （dispatcher.py:129 / :157）
  → 该 session 仍进入 lazy eval（postrun 判分）
  → _prepare_eval_runtime → _run_runtime_prepare
       第一行 `if managed.cancel_requested: return`     （node.py:296-298）
       在 step 0/3 直接 return → upload input/{op}.py、upload {op}.json、exec 三步全跳过
  → 但 return 无异常 → _prepare_eval_runtime 视作成功、返回可用 runtime（node.py:454-482）
  → operator_judge 在这个「什么都没铺」的 fresh 容器里跑 judge_command
       → input/{op}.py 不在 → ascendc_eval_pipeline.sh:255-258 报 golden missing
       → metrics.json error_type=input_load_failed
  → operator_judge._scored（operator_judge.py:557-562）判 INFRA → raise → session ERROR → retry
```

**探针实证(20260801 run,新加的 [EVALPREP-PROBE]):**
```
submission_missing=False                       # 有提交物(best-so-far 已提取)
eval_prepare_is_none=False n_actions=3 types=[upload_file,upload_file,exec]   # 配置正确
run_prepare n_steps=3 cancel_at_entry=True      # ★ 进 prepare 时已是取消态
CANCEL 在第 0/3 步中途返回 → 剩余步(含铺 input/)被跳过   # ★ 三步全跳过
_run_runtime_prepare 正常返回(eval_prepare 已跑完)       # ★ 被当成功
fresh_eval_runtime=True → judge 将用 fresh 运行时
```

**统计实证(前 10 个到达 eval 的 session):**
| cancel_at_entry | 数量 | → 跳过铺 input | → input_load_failed |
|---|---|---|---|
| True  | 6 | 6 | **6** |
| False | 4 | 0 | **0** |

→ `input_load_failed ⟺ cancel_at_entry=True`,6/6 对应、0 假阳性。

## 3. 为什么是「浪费」
- 探针 `submission_missing=False` → 这些 session 的**提交物(含 best-so-far 最优 kernel)是存在、已被 operator_judge 提取的**。
- 但 eval_prepare 被跳过 → judge 无法验证 → 判 input_load_failed → **已产出的算子没被评分,整个 session 作废 + 重试(再花 ~2h)**。
- 即:agent 可能已写出能用的算子,只因多验证 1 次(7>6)被取消,连产出一起被扔。**双重浪费。**

## 4. 对照:撞顶 262k 上下文的 session 是「对」的(不丢)
- 旧 run 17 个撞 `maximum context length 262144`(引擎 http_400)。
- 结局:status=ERROR,但 **reward=0.2/0.3 都在** —— best-so-far 被评了分。
- 原因:上下文超限**不置 cancel_requested** → cancel_at_entry=False → eval_prepare 正常跑 → 正常评分。
- 结论:**丢产出的只有 budget 取消(7>6)这一条路**,因为只有它置 cancel_requested、才命中跳过 bug。目标就是让 budget 取消也走撞顶那样的「照评分」路径。

## 5. 修复设计

### 主修(推荐):取消时 eval_prepare 照铺,把 best-so-far 评掉
`cancel_requested` 的语义是「停 agent」,不该连「给已产出物打分的判分准备」也停。

- **改动点**:`_run_runtime_prepare`(node.py:283)的 `if managed.cancel_requested: return` 只对 **agent prepare** 生效;**eval_prepare 路径不理会取消**。
- **实现**:给 `_run_runtime_prepare` 加参数 `honor_cancel: bool = True`;`_prepare_eval_runtime`(node.py:474)调用时传 `honor_cancel=False`。循环里改为 `if honor_cancel and managed.cancel_requested: return`。
- 效果:budget 取消的 session,eval_prepare 仍铺 input/ → judge 评 best-so-far → 拿到真实 reward(correctness/speedup 或如实的低分),**不再 input_load_failed、不再无谓 retry**。

### 防御修(配合):判分前断言 golden 就位
- `ascendc_eval_pipeline.sh` 在 golden missing 时,当前直接判 input_load_failed。可加一层:若检测到运行环境是「取消态但要求评分」,而 input/ 仍缺,则明确区分是 prepare 漏铺(基础设施)还是数据缺(真 infra),避免二义。（可选,主修生效后此路不再触发。）

### 不采用:取消就跳过判分
- 「取消的 session 干脆不评分」会把 best-so-far 一起扔,仍浪费,**否决**。

## 6. 上游(单独议题,非本修复)
- `generation_max: 6` 是否偏紧:实测有 agent 到第 7 次。是否放宽是调参决策,与本正确性修复解耦。放宽能减少「取消」发生,但**即使不放宽,主修也保证取消不再丢产出**。

## 7. 验证方案(改完重启后)
1. `input_load_failed` 判分数 → 应趋近 0(`grep 'infra failure (input_load_failed)' gateway.log`)。
2. budget 取消仍会发生(`grep 'calls 7>6' pipeline_budget_watcher.log` 照旧),但这些 session 的 metrics.json 应有**真实 verdict**(correctness_ok/success/speedup),不再是 input_load_failed。
3. [EVALPREP-PROBE] 应显示:cancel_at_entry=True 时**不再**「第0步跳过」,而是 upload OK + exec 跑完。
4. 对照撞顶 session:两类取消/超限的 session 都应能拿到 reward,不再有「有提交却 input_load_failed」的。

## 8. 关联/清理
- 调试探针 [EVALPREP-PROBE](node.py `_run_lazy_eval`/`_prepare_eval_runtime`/`_run_runtime_prepare`)为临时,**验证通过后回滚**。
- 之前误判并已回滚:①「golden 指向宿主机路径」(容器内不可达);②「agent 删 input」(refresh_runtime=true,agent 碰不到 judge 容器);③「cwd 偏移」(judge cwd 写死 workdir)。均见 git 历史。
- 采集侧 T1–T8 已全链路跑通(见 telemetry/ + [[telemetry-t1-t2-file-collection]]),与本修复无关但同期。
