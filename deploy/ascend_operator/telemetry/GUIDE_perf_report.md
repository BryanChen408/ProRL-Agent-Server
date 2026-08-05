# 性能分析报告生成指南(给零上下文的新 session)

> 目的:让**没有任何上下文**的 session 也能:①找到采集到的性能数据 ②看懂每个字段是什么 ③基于现有 HTML 报告继续优化。
> 你要做的是**在现有 `perf_report_html.py` 生成的 HTML 上优化**,不是从零写。先读本文件,再读 `perf_report_html.py` 和 `perf_charts_html.py`。

---

## 0. 一句话背景(这是什么系统)

这是 **ProRL agentic-RL 训练**的**推理侧性能采集**。系统在跑:一个 RL rollout 里,agent(claude_code)反复调推理引擎(vllm)生成 AscendC 算子代码,再送去 NPU 验证。我们采集了整条链路的负载,分成 **T1–T8** 八张表。

**硬件拓扑**:16 个 NPU chip = **12 卡推理**(3 个 vllm engine,每个 TP4,记作 `infer-0/1/2`,端口 15000/15002/15004)+ **4 卡验证池**(算子编译+对拍时才用)。

**一轮 rollout** = `train_batch_size(8) × n_samples_per_prompt(8) = 64 个 session`(配置在 `/workspace/vime/scripts/start-ascendc.sh`)。

---

## 1. 数据在哪(路径)

### 全局实时采集(T1–T4),跨 run 累积,重启会自动归档
根目录:**`/mnt/share/polar_engine_metrics/`**(环境变量 `POLAR_ENGINE_METRICS_DIR`)
```
npu_state/npu_card.jsonl        # T1  NPU 卡负载(16 chip 每 5s 一行)
vllm_state/infer-0.jsonl        # T2  引擎 0 态(每 5s;infer-1/infer-2 同理)
vllm_state/infer-1.jsonl
vllm_state/infer-2.jsonl
host_state/host_proc.jsonl      # T3  主机 + 进程(每 5s)
engine-15000.jsonl              # T4  引擎 0 逐请求(每请求一行;15002/15004 同理)
engine-15002.jsonl
engine-15004.jsonl
archive/<时间戳>/               # 历史 run 的 T1-T4(每次重启 polar 自动归档到这)
```

### 每 run 独立的派生表(T5–T8)
在 **`<run_dir>/telemetry_derived/`**,`<run_dir>` 形如 `output/ascend_operator/runs/20260802-123724-d3b14e/`(取最新的:`ls -dt .../runs/*/ | head -1`)。
```
rollout_span.jsonl    # T5  每 session 时间拆解(推理/验证/工具/等待)
verify_job.jsonl      # T6  每次算子验证的结果
rollout.jsonl         # T7  每 session 汇总(status/reward/timing)
rollout_step.jsonl    # T8  每 rollout step 聚合(goodput/engine 均衡)
```
> T5-T8 由 `telemetry_aggregator.py` 常驻 daemon 每 30s 从 run 目录的 `rollout_results/ses_*.json` + `polar_sessions/` 转录重建。若某 run 没有 telemetry_derived,说明 aggregator 没扫到它,可手动:`python3 telemetry_aggregator.py --run-dir <run_dir> --once`。

### 原始 run 数据(派生表的来源,一般不用直接读)
- `<run_dir>/rollout_results/**/ses_*.json` — 每 session 的完整结果(status/timing/trajectory/evaluation)
- `<run_dir>/polar_sessions/**/session-*/` — 每 session 的 agent 转录、artifacts(metrics.json/judge.stdout.log)
- `<run_dir>/logs/gateway.log` — polar gateway 日志(session 请求/错误)

---

## 2. 每张表是什么、关键字段(看懂数据)

**所有表都是 JSONL(一行一个 JSON)。时间戳字段 `recorded_at_unix`(float 秒)。**

### T1 `npu_card.jsonl` — NPU 卡负载
每 5s,每 chip 一行(16 行/轮)。
| 字段 | 含义 |
|---|---|
| `recorded_at_unix` | 采样时刻 |
| `card_id` | 全局 Phy-ID 0-15 |
| `pool` | `inference` 或 `verify` |
| `engine_id` | `infer-0/1/2` 或 `verify-pool` |
| `aicore_util_pct` | **AI Core 算力利用率 %**(核心指标) |
| `hbm_used_mb` / `hbm_util_pct` | 显存已用 MB / 占比(单卡满 65536MB) |
| `power_w` | 功耗瓦(temp_c 这台采不到,别用) |

### T2 `infer-{0,1,2}.jsonl` — 引擎态(vllm 原生 /metrics 抓取)
每 5s 每引擎一行。**大多是累计 counter**,要**相邻两行差分**才是区间值。
| 字段 | 含义 |
|---|---|
| `scrape_ok` | **false=这次抓取失败,该行不可用**,分析时必须过滤掉(见下方陷阱) |
| `vllm:num_requests_running` / `_waiting` | 瞬时:正在跑/排队的请求数 |
| `vllm:kv_cache_usage_perc` | 瞬时:KV cache 占比(0-1,乘 100 得 %) |
| `vllm:generation_tokens_total` | 累计生成 token(差分/Δt = 吞吐 tok/s) |
| `vllm:prompt_tokens_total` | 累计 prompt token |
| `vllm:time_to_first_token_seconds_sum` / `_count` | TTFT 累计和/次数(Δsum/Δcount = 平均 TTFT 秒) |
| `vllm:inter_token_latency_seconds_sum` / `_count` | ITL(token 间延迟) |
| `vllm:e2e_request_latency_seconds_sum` / `_count` | 端到端延迟 |
| `vllm:num_preemptions_total` | 累计抢占次数(差分) |
| `vllm:prefix_cache_hits_total` / `_queries_total` | prefix cache 命中/查询(差分求命中率) |
| `..._bucket@le=<x>` | histogram 桶,相邻两行差分 + 桶插值可求 TTFT 的 p50/p90/p99 |

> **prefill/decode 拆分**:引擎侧近似 —— prefill 占比 ≈ ΔTTFT_sum / Δe2e_sum,decode ≈ 1−prefill。V1 非流式拿不到墙钟级 token 计时,只能这样近似,报告里要标注口径。

### T3 `host_proc.jsonl` — 主机 + 进程
每 5s。`kind=host` 是整机一行;`kind=proc` 是各角色进程(gateway/rollout/proxy/vllm_engine/ray/agent)。
| 字段(kind=host) | 含义 |
|---|---|
| `mem_avail_mb` / `mem_used_mb` / `page_cache_mb` | 内存(MB) |
| `cpu_pct` / `load1` | CPU% / 1分钟负载 |
| `nfs` / `net` | NFS(/mnt/share)读写字节 / 各网卡收发 |
| (kind=proc) `role` `rss_mb` `cpu_pct` `num_threads` `num_fds` | 进程级;ray 那种海量进程滚成 1 行 `rollup:true n_procs=N` |

### T4 `engine-<port>.jsonl` — 逐请求(vllm 内埋点)
每个推理请求一行。**注意 `recorded_at` 是字符串(ISO),不是 `recorded_at_unix`**。
| 字段 | 含义 |
|---|---|
| `num_prompt_tokens` | 该请求 prompt 长度(=prefill 量) |
| `num_generation_tokens` | 生成 token 数(=decode 长) |
| `num_cached_tokens` / `prefix_cache_hit_pct` | 命中缓存的 prompt token / 命中率 % |
| `trace_id` / `session_id` | 关联键(=`session_id:turn_seq`) |
| `finish_reason` / `aborted` | 结束原因 / 是否被中止 |
> V1 非流式下 `ttft_ms/decode_ms` 不可信(拆不出),别用 T4 的延迟字段,延迟看 T2。

### T5 `rollout_span.jsonl` — 时间拆解(块④)
每 session 一行,占墙钟比(0-1):`infer_frac` `verify_frac` `tool_frac` `wait_cpu_frac`,加 `inference_ms/verify_ms/tool_ms` 绝对值、`n_turns`。

### T6 `verify_job.jsonl` — 算子验证结果
每次验证一行:`op_name` `success` `error_type`(`ascendc_compile_failed`/`correctness_failed`/`input_load_failed`…) `speedup` `lease_wait_s`(验证池排队,需带补丁运行时才有)。

### T7 `rollout.jsonl` — 每 session 汇总
`session_id` `status`(COMPLETED/ERROR/TIMEOUT) `reward` `rollout_step` `op_name` `run_ms` `init_ms` `num_turns`。**reward 在这来自 ses.json 的 `trajectory.metadata.evaluation`**。

### T8 `rollout_step.jsonl` — 每 step 聚合
`rollout_step` `num_sessions` `goodput`(=COMPLETED/total) `reward_dist` `engine_balance`(含 `request_cv` 引擎均衡、各 engine 请求/token 份额)。

---

## 3. 怎么生成报告(现成脚本,先跑起来看)

目录:`deploy/ascend_operator/telemetry/`。依赖:`pip install matplotlib plotly`(已装)。

```bash
cd deploy/ascend_operator/telemetry
# HTML 版(推荐,交互/矢量/中文/离线自包含)——你要优化的就是这个
python3 perf_report_html.py                          # 默认最新 run 全部 session
python3 perf_report_html.py --run <run_dir> --window-sessions 64   # 指定 run + 只取最近 64(一轮)
# 产物: <run_dir>/perf_report_html/report.html  (双击浏览器打开) + source_data/(切片)
```
- `perf_report_html.py` = 报告主体(数据加载 + 分节 + 文字结论)。
- `perf_charts_html.py` = plotly 图函数(饼/sunburst/时序折线/直方/条形),各图返回 `<div>`,plotly.js 在 HTML 头部内联一次。
- 还有 `perf_report_detailed.py`(ASCII/PNG 版,备用)、`perf_report.py`(早期简版)。**优化就改 html 那两个。**

报告现有 7 节:1 引擎/NPU/显存性能 · 2 长度 · 3 时间构成(含 prefill/decode sunburst + 一轮墙钟绝对耗时) · 4 Agent 结局/验证 · 5 多粒度(step/op) · 6 主机 · 7 瓶颈总结。

---

## 4. 必须知道的数据陷阱(踩过的坑,别重犯)

1. **T2 `scrape_ok=false` 的行要过滤**。vllm /metrics 在负载下常超时,失败行字段为空/0。不过滤会把 counter 差分击穿(吞吐算出天文数字)。代码里用 `_vt2()` 过滤。
2. **T2 是累计 counter,必须相邻两行差分**,不能直接取值当区间量。
3. **T4 时间用 `recorded_at`(字符串),T1/T2/T3 用 `recorded_at_unix`(float)**,别混。
4. **"一轮时长"= 墙钟跨度(并发),不是单 session 累加**。取最近 64 个完成 session:墙钟=max(end)−min(start),累加=Σrun_ms,并发度=累加/墙钟(实测~12,约等于 12 推理卡)。用户明确要墙钟口径。
5. **temp_c 采不到**(这台 npu-smi 不出),别画温度图。
6. **prefill/decode 是引擎侧近似**(§T2),不是墙钟,标注口径。
7. **图内标签用英文**(这台没中文字体,中文会变豆腐块),正文/表格用中文。HTML 里浏览器有字体,可用中文——若装了 Noto CJK 字体则图内也能中文。
8. **session id 两套命名**:polar_sessions 目录用短 slug(`sk-polar-xxxx`),ses.json 用长 UUID,**离线无法按 id join**,所以 T5 时间拆解只能给全局分布,不能 per-session 关联 reward。

---

## 5. 用户想要的优化方向(基于现有 HTML)

现有 HTML 已实现:PNG→plotly、时间占比饼图(推理拆 prefill/decode 的 sunburst)、显存时序、一轮 64-session 墙钟+并发度、绝对耗时(分钟)、KPI 卡、三段式溯源(【来源】→【计算】→【依据】)。

用户反复强调的品味:
- **要图不要纯数字**:任何列了一堆数字的地方,配一张图。
- **绝对耗时**很重要:不只百分比,给分钟数(单 session 各段 min、一轮墙钟 min)。
- **美观**:用户嫌 matplotlib/ASCII 丑,选了 **plotly HTML**。要更美可考虑 ECharts(需内联 echarts.min.js,之前 CDN 下载超时,可让用户提供或换源)。
- **每个结论要能溯源**:数据来自哪张表哪个字段、怎么算的、为什么能下这个结论(三段式)。
- **多粒度**:per-request / per-session / per-step / per-op / per-engine 都要能切。

优化时:先 `python3 perf_report_html.py` 跑一版看现状 → 读 `perf_report_html.py` 对应 `_sec_*` 函数 → 改图/加图(在 `perf_charts_html.py` 加函数)→ 重跑对比。**改完一定重跑确认 HTML 能开、图能显示、数据没算错(拿 source_data 复核)。**

---

## 6. 快速自检(新 session 上手先跑这些)

```bash
cd /mnt/share/c00937190/src/cannbot_debug/ProRL-Agent-Server/deploy/ascend_operator/telemetry
ls /mnt/share/polar_engine_metrics/{npu_state,vllm_state,host_state}/   # T1-T3 在不在
NEW=$(ls -dt ../../../output/ascend_operator/runs/*/ | head -1); echo "最新 run: $NEW"
ls "$NEW/telemetry_derived/"                                            # T5-T8 在不在
python3 perf_report_html.py                                            # 生成,看报告路径
```
生成的 `report.html` 双击打开,就是你要优化的基线。source_data/ 里是切好的数据,可 pandas/DuckDB 复核任何一个数字。

