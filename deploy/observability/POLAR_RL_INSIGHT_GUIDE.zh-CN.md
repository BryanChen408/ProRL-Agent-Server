# Polar × RL-Insight 可观测性使用指南

本文说明 Polar 与 RL-Insight 集成后的能力边界、指标口径、Polar Dashboard，以及注册和部署方法。

## 1. 整体架构

```text
                                      ┌──────────────────────────┐
Polar Gateway /metrics ──────────────>│ RL-Insight Prometheus    │──> Grafana :3000
                                      │ :9090                    │
VIME PD Proxy /metrics ──────────────>│                          │
  └─ 聚合 Prefill/Decode vLLM 指标     └──────────────────────────┘

Polar Session OTLP spans ────────────> RL-Insight Tempo :4318 ──> Trace 查询

rollout_results + persisted traces ──> Polar Dashboard :8090
                                      └─ Task/Session/Trace/Artifact 明细
```

两个 UI 的用途不同：

| UI | 默认端口 | 主要用途 |
| --- | ---: | --- |
| RL-Insight Grafana | `3000` | 跨 Session 的吞吐、TTFT、TPOT、队列、缓存、健康状态等时序聚合 |
| Polar Dashboard | `8090` | 单个 Task/Session 的执行过程、每轮 LLM 调用、工具调用、轨迹和产物 |

## 2. 术语解释

### 2.1 可观测性与协议

| 术语 | 全称或定义 | 在本文中的作用 |
| --- | --- | --- |
| Observability | 可观测性 | 通过指标、Trace 和日志理解系统状态与性能 |
| OpenTelemetry / OTel | OpenTelemetry | 统一生成、传输和关联 Metrics、Traces、Logs 的标准与工具集合 |
| OTLP | OpenTelemetry Protocol | OpenTelemetry 的遥测数据传输协议；Polar 使用 OTLP/HTTP 把 Session Trace 发送到 RL-Insight |
| Metric | 指标 | 随时间变化的数值，例如吞吐、队列长度和延迟分布 |
| Trace | 追踪 | 一次完整操作的调用链，例如一个 Session 从排队到结束的全过程 |
| Span | 跨度/执行片段 | Trace 内的一段操作，包含开始时间、结束时间、名称、属性和父子关系 |
| Correlation | 关联 | 使用 `session_id`、`task_id`、`trace_id` 将 Gateway、Agent 与 Engine 数据对应起来 |

OTLP 是协议，Tempo 是接收和存储 Trace 的后端，Grafana 是查询和展示界面。三者关系为：

```text
Polar spans → OTLP/HTTP → Tempo → Grafana
```

### 2.2 Prometheus 与 Grafana

| 术语 | 含义 |
| --- | --- |
| Prometheus | 按固定周期从 `/metrics` 拉取并存储时序指标的系统 |
| Scrape | Prometheus 主动访问一个 `/metrics` 端点并采集数据 |
| Target | 被 Prometheus 抓取的 `host:port` |
| Job | 一组同类 Target；本集成使用 `polar-gateway` 和 `polar-inference-engine` |
| Label | 指标维度，例如 `node_id`、`pd_role`、`pd_backend` 和 `status` |
| PromQL | Prometheus 查询语言，用于计算速率、聚合和延迟分位数 |
| Recording rule | 周期执行 PromQL 并把结果保存为新指标，用于统一指标名和降低查询开销 |
| Alert rule | 当 PromQL 条件持续成立时产生告警状态的规则 |
| Grafana | 查询 Prometheus/Tempo 并组织 Dashboard、Panel 和告警的可视化系统 |
| Dashboard | 一组相关的 Panel |
| Panel | 一张图或一个状态卡片，每个 Panel 包含查询、标题、单位和展示方式 |

### 2.3 Prometheus 指标类型

| 类型 | 含义 | 正确使用方式 |
| --- | --- | --- |
| Counter | 只增不减的累计值，进程重启时可归零 | 使用 `rate()` 看每秒速率，使用 `increase()` 看区间增量 |
| Gauge | 可升可降的瞬时值 | 直接查询或使用 `sum/avg/max` 聚合 |
| Histogram | 将样本累计到多个上界 bucket 中，同时输出 `_bucket/_sum/_count` | 用 `histogram_quantile()` 算 P95/P99，或用 `_sum/_count` 算平均值 |
| Bucket | Histogram 的累计区间 | `le="1.0"` 表示“小于等于 1 秒”的累计样本，不代表延迟等于 1 秒 |

`rate(counter[5m])` 表示根据最近 5 分钟的 Counter 增量计算平均每秒速率。窗口越短越实时、波动越大；窗口越长越平滑、响应越慢。

### 2.4 推理性能术语

| 术语 | 全称 | 含义 |
| --- | --- | --- |
| TTFT | Time To First Token | 从请求进入推理服务到产生第一个输出 token 的时间，通常受排队和 Prefill 影响 |
| TPOT | Time Per Output Token | 首 token 之后平均生成一个输出 token 的耗时，主要反映 Decode 性能 |
| ITL | Inter-Token Latency | 相邻两个输出 token 之间的时间间隔 |
| Prefill | 提示词预填充 | 对输入 Prompt 做前向计算并构建 KV Cache 的阶段 |
| Decode | 自回归解码 | 基于已有 KV Cache 逐个生成输出 token 的阶段 |
| PD 分离 | Prefill/Decode Disaggregation | 将 Prefill 和 Decode 部署到不同推理实例或资源池 |
| KV Cache | Key-Value Cache | 保存注意力历史状态，避免 Decode 时重复计算历史 token |
| Prefix Cache | 前缀缓存 | 在不同请求间复用相同 Prompt 前缀对应的 KV Cache |
| Prefix Cache Hit Ratio | 前缀缓存命中率 | 输入 token 中复用了缓存的比例；高命中时逻辑 Prompt 吞吐可能远高于实际 Prefill 计算吞吐 |
| Throughput | 吞吐 | 单位时间完成的请求数或 token 数，常见单位为 req/s 或 token/s |
| P95/P99 | 95/99 分位数 | 95%/99% 的样本不超过该值，用于观察长尾延迟 |

TTFT、TPOT 和 ITL 都是延迟指标，但口径不同，不能互相替代。对于 PD 分离：TTFT 主要关注 Prefill 路径，TPOT/ITL 主要关注 Decode 路径。

## 3. RL-Insight 提供的能力

RL-Insight 在本集成中提供集中式可观测性基础设施。

### 3.1 服务栈

| 服务 | 默认端口 | 能力 |
| --- | ---: | --- |
| RL-Insight Server | `18080` | 控制 API、Prometheus target 动态注册 |
| Prometheus | `9090` | 拉取和存储 Polar/vLLM 指标，执行 recording/alert rules |
| Grafana | `3000` | 指标查询、Dashboard 和告警可视化 |
| OTLP HTTP Receiver | `4318` | 接收 Polar 输出的 Session spans |
| Tempo Query | `3200` | 存储和查询分布式 trace |

### 3.2 动态发现与持久化

Polar Gateway 启动时调用 RL-Insight：

```text
POST /api/v1/prometheus/targets
```

自动注册两个 job：

| job | target | 标签 |
| --- | --- | --- |
| `polar-gateway` | Gateway 的 `host:port` | `node_id` |
| `polar-inference-engine` | 推理入口的 `host:port` | `node_id`、`engine_type` |

注册是幂等的，并按 `registration_refresh_seconds` 周期刷新，默认 30 秒。动态 targets 默认持久化到：

```text
~/.rl-insight/data/targets/prometheus-targets.yml
```

RL-Insight/Prometheus 重启不会清空该文件。

### 3.3 指标与 Trace

- Prometheus 拉取 Gateway 和推理入口的 `/metrics`。
- Recording rules 将不同推理后端的原生指标归一到 `polar_inference_*`。
- Polar 在 Session 结束时通过 OTLP 输出 Gateway、Engine、Agent/Tool 等关联 spans。
- 已落盘但未发送的 Perfetto trace 可以使用 `polar backfill_traces` 补传到 Tempo。
- Grafana 可导入本仓提供的 Polar Dashboard JSON。

RL-Insight 默认没有配置告警通知接收人。Prometheus alert rules 只负责产生告警状态；发送邮件、Webhook 等仍需配置 Alertmanager 或 Grafana Alerting。

## 4. Polar 提供的指标

Gateway 指标端点：

```text
http://<gateway-host>:<gateway-port>/metrics
```

### 4.1 Session 与调度指标

| 指标 | 类型 | 主要标签 | 含义 |
| --- | --- | --- | --- |
| `polar_sessions_total` | Counter | `node_id,status` | 按终态统计完成的 Session 数 |
| `polar_session_duration_seconds` | Histogram | `node_id,le` | Session 端到端执行耗时 |
| `polar_gateway_sessions` | Gauge | `node_id,stage` | 各调度阶段的实时 Session 数 |
| `polar_llm_calls_total` | Counter | `node_id` | 已完成的 LLM 调用次数 |
| `polar_llm_tokens_total` | Counter | `node_id,direction` | Gateway 记录的 prompt/response token 数 |

`polar_gateway_sessions.stage` 包含：

- `init_queue_depth`
- `init_inflight`
- `ready_depth`
- `run_inflight`
- `postrun_queue_depth`
- `postrun_inflight`

常用查询：

```promql
sum by (stage) (polar_gateway_sessions)
```

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(polar_session_duration_seconds_bucket[5m]))
)
```

### 4.2 Gateway 记录的推理指标

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `polar_inference_requests_total` | Counter | 已完成的推理请求数 |
| `polar_inference_tokens_total` | Counter | 按 `direction=prompt/response` 统计推理 token |
| `polar_inference_cached_prompt_tokens_total` | Counter | 推理引擎返回的缓存 prompt token 数 |
| `polar_inference_request_duration_seconds` | Histogram | Gateway 看到的完整推理请求耗时 |
| `polar_inference_queue_seconds` | Histogram | 引擎返回的请求排队耗时 |
| `polar_inference_ttft_seconds` | Histogram | 首 Token 延迟 |
| `polar_inference_prefill_seconds` | Histogram | Prefill 耗时 |
| `polar_inference_decode_seconds` | Histogram | Decode 耗时 |
| `polar_inference_prefix_cache_hit_ratio` | Histogram | 单请求 Prefix Cache 命中比例 |
| `polar_observability_export_failures_total` | Counter | OTLP trace 输出失败次数 |

除 `request_duration` 外的细分耗时依赖推理引擎在响应中返回 `metrics` 或 `timings`。引擎不返回某字段时，Gateway 不会伪造该样本。

实时请求与 Token 吞吐：

```promql
sum(rate(polar_inference_requests_total[1m]))
```

```promql
sum by (direction) (rate(polar_inference_tokens_total[1m]))
```

### 4.3 vLLM 原生指标与 PD 聚合

Polar 会把配置的推理入口注册成 `polar-inference-engine`。在 VIME PD 模式下，该入口是 Mooncake PD Proxy；VIME 的 `/metrics` 会并发抓取全部 Prefill/Decode 后端，保留原生 histogram，并增加：

- `pd_role="prefill"` 或 `pd_role="decode"`
- `pd_backend="<role-index>@<host:port>"`
- `polar_pd_backend_up`：每个后端的指标抓取健康状态

值得关注的原生指标：

| 指标 | 含义 |
| --- | --- |
| `vllm:prompt_tokens_total` | 逻辑 Prompt token；可能包含 Prefix Cache 命中的 token |
| `vllm:generation_tokens_total` | 实际生成的输出 token |
| `vllm:prompt_tokens_by_source_total` | 按来源拆分的 Prompt token；是否存在取决于 vLLM 版本 |
| `vllm:num_requests_running` | 正在运行的请求数 |
| `vllm:num_requests_waiting` | 等待调度的请求数 |
| `vllm:gpu_cache_usage_perc` / `vllm:kv_cache_usage_perc` | KV Cache 占用比例，名称取决于版本 |
| `vllm:num_preemptions_total` | 请求抢占累计次数 |
| `vllm:request_time_per_output_token_seconds` | 请求级 TPOT histogram |
| `vllm:inter_token_latency_seconds` | 相邻 token 延迟 histogram |
| `vllm:prefix_cache_hits_total` | Prefix Cache 命中 token 累计值 |
| `vllm:prefix_cache_queries_total` | Prefix Cache 查询 token 累计值 |

实际 Decode 输出吞吐：

```promql
sum by (pd_role) (rate(vllm:generation_tokens_total[1m]))
```

逻辑 Prompt 吞吐：

```promql
sum by (pd_role) (rate(vllm:prompt_tokens_total[1m]))
```

当 Prefix Cache 命中率很高时，逻辑 Prompt 吞吐可能达到很大的数值。这表示单位时间内处理的逻辑上下文长度，不等于实际执行 Prefill 计算的 token/s。若引擎提供来源标签，实际 Prefill 计算吞吐应查询：

```promql
sum by (pd_role) (
  rate(vllm:prompt_tokens_by_source_total{source="local_compute"}[1m])
)
```

平均 TPOT：

```promql
sum(rate(vllm:request_time_per_output_token_seconds_sum{pd_role="decode"}[5m]))
/
sum(rate(vllm:request_time_per_output_token_seconds_count{pd_role="decode"}[5m]))
```

### 4.4 归一化推理指标

文件 `prometheus/polar_engine_recording_rules.yaml` 将 vLLM 原生指标归一为：

- `polar_inference_queue_seconds_{bucket,sum,count}`
- `polar_inference_ttft_seconds_{bucket,sum,count}`
- `polar_inference_prefill_seconds_{bucket,sum,count}`
- `polar_inference_decode_seconds_{bucket,sum,count}`
- `polar_inference_prefix_cache_hit_ratio`

这些序列带有 `engine="vllm-native"`、`node_id`、`pd_role` 和 `pd_backend`，适合比较 Prefill/Decode 后端。

TTFT P95：

```promql
histogram_quantile(
  0.95,
  sum by (le, pd_role, pd_backend) (
    rate(polar_inference_ttft_seconds_bucket{engine="vllm-native"}[5m])
  )
)
```

只需要一条全局 Prefill TTFT 曲线时：

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(polar_inference_ttft_seconds_bucket{
      engine="vllm-native",
      pd_role="prefill"
    }[5m])
  )
)
```

Histogram 的 `le="160.0"` 表示“耗时小于等于 160 秒”的累计 bucket，不是 160 秒的延迟曲线。必须使用 `histogram_quantile()` 才能得到以秒为单位的 P95/P99。

### 4.5 告警规则

`prometheus/polar_alerts.yaml` 提供：

- Gateway target 缺失或不可抓取
- 推理 target 不可抓取
- 单个 PD backend 指标不可抓取
- OTLP trace 输出失败
- Session 调度积压
- Session 失败率过高
- Session P95 延迟过高
- 推理请求 P95 延迟过高

阈值是部署默认值，接入生产通知前应根据实际 workload 调整。

## 5. Polar Dashboard 实现

Polar Dashboard 是 Polar 自带的 Session 明细 UI，不依赖 Grafana Panel。它通过两类数据构建视图：

1. 从 Rollout/Gateway API 获取正在运行的 Task 和 Session。
2. 从 `--save-dir` 扫描已经落盘的 `task_*/ses_*.json`；同时支持 `run_*/task_*/ses_*.json` 布局。

### 5.1 构建前端

从源码运行且 `web/dist` 尚未生成，或前端代码发生变化时执行：

```bash
cd /home/docker/polar_can/ProRL-Agent-Server/web
npm install
npm run build
cd ..
```

依赖未变化且已有 `node_modules` 时，可只执行 `npm run build`。构建完成后需要重启 Dashboard 进程，但不需要因为 UI 变化重启训练。

### 5.2 启动

```bash
source /root/polar-venv/bin/activate
cd /home/docker/polar_can/ProRL-Agent-Server

polar dashboard \
  -c output/ascend_operator/runs/<run-id>/run_artifacts/effective_topology.yaml \
  --save-dir output/ascend_operator/runs/<run-id>/rollout_results/run_<polar-run-id> \
  --host 0.0.0.0 \
  --port 8090
```

若 `effective_topology.yaml` 中的 `rollout.save_dir` 已准确指向结果目录，可以省略 `--save-dir`；否则必须显式传入。浏览器访问：

```text
http://<polar-host>:8090
```

`--host 0.0.0.0` 会监听所有网卡；Dashboard 本身没有在本文范围内增加公网认证，应通过可信网络或防火墙限制访问。

### 5.3 页面与数据

| 页面 | 能力 |
| --- | --- |
| Dashboard | Rollout/Gateway 拓扑、运行中 Task、近期 Task |
| Tasks | 状态/Harness/Task ID 筛选、Reward 横向分布、任务列表 |
| Task Detail | 同一 Task 下的所有 Session，比较状态、节点、Reward、Init/Run/Postrun |
| Compare | 比较两个 Task 的成功数、平均 Reward 和阶段平均耗时 |
| Session Detail | 单 Session 的 Timeline、Trace、Artifacts、Completions、Trajectory、Evaluation、Raw JSON |

Tasks 总览中的 `session_time`：

- 对单 Session Task，优先显示 `timing.total_ms`。
- 旧数据没有 `total_ms` 时，回退为 `queue + init + run + postrun`。
- 未完成、没有持久化结果或一个 Task 包含多个 Session 时显示 `—`，避免把聚合值误标为单 Session 耗时。

### 5.4 Session Detail

入口：

```text
Tasks → task_id → session_id
```

页签说明：

| 页签 | 内容 |
| --- | --- |
| Timeline | Queue、Init、Run、Postrun 和 Total 阶段耗时 |
| Trace | Gateway、Engine、Agent 三条时间线及瀑布图 |
| Artifacts | 日志、metrics、profiling 等有界持久化产物 |
| Completions | 每轮原始请求、转换请求、响应及 TTFT/Prefill/Decode/Queue/Cache |
| Trajectory | Prompt/Response 消息、reasoning、tool calls、reward、finish reason |
| Evaluation | outcome reward、评测策略和原始报告 |
| Raw JSON | Session 原始落盘数据 |

Trace 中的重要 span：

- `llm_call_N/inference`：第 N 次调用 vLLM 的完整区间。
- `llm_call_N/engine/queue`：vLLM 排队。
- `llm_call_N/engine/prefill`：Prefill。
- `llm_call_N/engine/decode`：Decode。
- `llm_call_N/agent/think`：模型调用之间的 Agent 思考区间。
- `llm_call_N/tool/...`：Bash、读写文件、评测等工具动作。

点击 span 可查看 duration、trace ID、engine URL、token 数和缓存信息；也可跳到对应 Completion/Trajectory。`Download Perfetto JSON` 可下载后拖入 `https://ui.perfetto.dev` 做更细的缩放分析。

## 6. Polar 如何注册到 RL-Insight

### 6.1 前置条件

- RL-Insight 和 Polar 应运行在能够互相访问的宿主机网络中。
- Polar Gateway 的 `/metrics` 必须可被 RL-Insight Prometheus 访问。
- PD 模式下，配置的推理入口必须提供 `/metrics`；VIME Mooncake PD Proxy 的指标聚合能力负责汇总各 Prefill/Decode 后端。
- `127.0.0.1` 仅适用于 Polar 与 RL-Insight 位于同一宿主机网络命名空间。若任一服务在容器内，需填写对端可达的宿主机 IP。

### 6.2 启动 RL-Insight

推荐在独立 venv 中启动：

```bash
cd /home/docker/insight/rl-insight
python3 -m venv /root/rl-insight-venv
source /root/rl-insight-venv/bin/activate
pip install -e .

rl-insight server start --detach
```

Python 报 `externally-managed-environment` 时不要向系统 Python 强制安装，使用 venv 即可。

### 6.3 配置 Prometheus rules

RL-Insight 使用的源配置路径不固定；editable 安装时通常位于 clone 目录：

```text
/home/docker/insight/rl-insight/rl_insight/config/services/prometheus/prometheus.yml
```

可通过以下命令确认：

```bash
python3 - <<'PY'
from rl_insight.utils.monitor_config_loader import load_server_config_file
print(load_server_config_file().prometheus.config_file)
PY
```

在源配置加入：

```yaml
global:
  scrape_interval: 10s

rule_files:
  - /home/docker/polar_can/ProRL-Agent-Server/deploy/observability/prometheus/polar_alerts.yaml
  - /home/docker/polar_can/ProRL-Agent-Server/deploy/observability/prometheus/polar_engine_recording_rules.yaml

scrape_configs: []
```

不要直接修改自动生成的：

```text
~/.rl-insight/runtime/prometheus.yml
```

源配置修改后重启 RL-Insight：

```bash
rl-insight server stop
rl-insight server start --detach
```

长期部署建议通过 `rl-insight server start --config <server.yaml>` 指向仓库外的独立 Prometheus 源配置，避免 `git pull` 覆盖。

### 6.4 配置 Polar

在 Polar profile 或最终 topology 的 Gateway 下配置：

```yaml
gateway:
  observability:
    prometheus_enabled: true
    rl_insight_url: http://127.0.0.1:18080
    otlp_endpoint: http://127.0.0.1:4318/v1/traces
    registration_refresh_seconds: 30
    export_timeout_seconds: 3
    service_name: polar-gateway
```

可选项：

```yaml
    otlp_headers: {}
    otlp_include_action_content: false
```

`otlp_include_action_content=false` 默认不把可能较大或敏感的 Agent action 内容写入 OTLP 属性；需要内容级调试时再显式开启。

`deploy/ascend_operator/profile.t2a.yaml` 已提供同宿主机默认配置。运行启动脚本后，应核对生成的：

```text
output/ascend_operator/runs/<run-id>/run_artifacts/effective_topology.yaml
```

修改 observability 配置后需要重启 Polar Gateway 才能加载；不需要仅为了重启 RL-Insight 而重启训练。若 RL-Insight 晚于 Polar 启动，Gateway 的周期刷新会在默认 30 秒内重新尝试注册。

### 6.5 验证注册

1. 检查 Gateway 指标：

```bash
curl --noproxy '*' http://127.0.0.1:<gateway-port>/metrics | head
```

2. 检查持久化 targets：

```bash
cat ~/.rl-insight/data/targets/prometheus-targets.yml
```

应看到 `polar-gateway` 和 `polar-inference-engine` 对应 target/labels。

3. 查询 Prometheus：

```bash
curl --noproxy '*' -G http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job=~"polar-gateway|polar-inference-engine"}'
```

结果应为 `1`。PD 后端健康：

```bash
curl --noproxy '*' -G http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=polar_pd_backend_up'
```

4. 验证 recording rules：

```bash
curl --noproxy '*' -G http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=polar_inference_ttft_seconds_count'
```

### 6.6 导入 Grafana Dashboard

Grafana 打开：

```text
http://<rl-insight-host>:3000
```

依次点击：

```text
Dashboards → New → Import → Upload dashboard JSON file
```

上传：

```text
deploy/observability/grafana/polar_gateway.json
```

选择 RL-Insight 的 `Prometheus` 数据源后点击 Import。Dashboard 包含：

- Gateway/Inference/PD Backend 健康状态
- Session Queue/Inflight 与终态
- Session Duration P95
- 推理 Token 吞吐与请求吞吐
- Request/TTFT P95
- Queue/Prefill/Decode P95
- Prefix Cache Hit Ratio
- OTLP Export Failures

## 7. 历史 Trace 补传

对启用 OTLP 之前已经持久化的 Perfetto trace，可以执行：

```bash
polar backfill_traces \
  output/ascend_operator/runs/<run-id>/perfetto_traces \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces \
  --service-name polar-gateway
```

建议先验证：

```bash
polar backfill_traces \
  output/ascend_operator/runs/<run-id>/perfetto_traces \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run
```

旧版 trace 没有真实 epoch 时间戳时，补传器会根据文件修改时间推断锚点，并标记 `timing.inferred=true`。

## 8. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| `18080` 无法访问 | RL-Insight Server 是否已启动、是否监听正确网卡；`rl_insight_url` 不是一个由 Polar 启动的服务 |
| Grafana 要求登录 | 使用 RL-Insight/Grafana 配置的认证方式；GitHub OAuth 用户应使用对应授权入口 |
| Grafana 没有 Polar 数据 | `up{job="polar-gateway"}`、targets 文件、Gateway `/metrics` 网络连通性 |
| `polar_inference_ttft_seconds_count` 不存在 | rules 是否加在 RL-Insight 的源配置而非 runtime 文件、RL-Insight 是否重启、原生 vLLM 指标是否存在 |
| `le="160.0"` 出现很多曲线 | 当前查询的是 histogram bucket；增加 Rate、按 `le` 聚合和 Histogram quantile |
| Prompt 吞吐异常高 | `vllm:prompt_tokens_total` 是逻辑 token；结合 Cache 命中率与 `source="local_compute"` 判断真实 Prefill 计算量 |
| Polar Dashboard Tasks 为空 | `--save-dir` 是否指向实际 `rollout_results/run_*`，而不是其父目录或另一个 run |
| Session `Trace (0)` | Session 是否由新版 Polar 产生、`session_trace_json` 和 `persist_traces_dir` 是否启用 |
| OTLP trace 没数据 | `polar_observability_export_failures_total`、`otlp_endpoint` 和 Tempo `4318` 连通性 |
