# Agentic-RL 推理侧负载采集(P0 + P1)

覆盖:块①显存/内存、块②rollout-step 耗时/长度、块③batch(engine 连续批 + rollout group)长度。
块④(rollout 非推理占比)= P2,埋 `rollout_span`/`verify_job`,尚未在此实现。

两个物理隔离卡池:**12 卡推理(3 engine×4,TP4)+ 4 卡验证**。见 `card_topology.yaml`(按你实际卡号/端口改)。

## 关联键(P0 已接)
gateway 在发往 engine 的请求头注入 **`x-polar-trace-id = {session_id}:{turn_seq}`**(`server.py` 两处 completion 调用 + `proxy.py` 转发)。
- gateway 事件:`completion_metrics.jsonl`,schema v2,新增 `trace_id / policy_version / engine_url / group_id / rollout_step / op_name`。
- engine 事件:`engine_request_logger` 写 `engine_metrics/<engine_id>.jsonl`,同 `trace_id`。
- 两者 **join on trace_id** = 每 request 既有 gateway 口径(latency/tokens)又有 engine 真值(TTFT/prefill/decode/queue/prefix-cache/抢占)。

## 组件与部署

### 1. npu-smi exporter(块① + 利用率,两池分标签)— 零侵入
```bash
python3 npu_smi_exporter.py --topology card_topology.yaml --port 9800 --interval 5
# 校准:--once 看每卡采样;--raw --card N 看 npu-smi 原始字段名(不同 CANN 版本对齐 _ALIASES)
```
Prometheus scrape `:9800/metrics`;指标 `npu_aicore_util_pct / npu_hbm_used_mb / npu_hbm_util_pct / npu_power_watts …`,标签 `card_id,pool,engine_id,tp_rank`。
→ Grafana 看**池间、engine 间、engine 内 4 卡**不均。

### 2. engine per-request logger(块②/③ engine 真值)— **已接入 vllm,无需再改代码**
已直接接进你的 vllm(`/workspace/vllm`,editable 安装):
- 新增 `vllm/entrypoints/openai/polar_telemetry.py`(自包含 extract+写盘)。
- `chat_completion/serving.py` 两处 guarded hook:`_create_chat_completion` 存 trace-id(contextvar),
  `chat_completion_full_generator` 拿到 final RequestOutput 后 `log_final()`。
- 落 `$POLAR_ENGINE_METRICS_DIR/<engine_id>.jsonl`。

**每 engine 只需两个环境变量**(在各 engine 启动处设):
```bash
export POLAR_ENGINE_ID=infer-0                 # 各 engine 改 id,与 card_topology 对齐;不设则回落 $VLLM_PORT
export POLAR_ENGINE_METRICS_DIR=<run_dir>/engine_metrics
# 应急关闭:export POLAR_ENGINE_METRICS_DISABLE=1
```
> `deploy/ascend_operator/telemetry/engine_request_logger.py` 是同逻辑的独立参考实现,已被 vllm 内的
> `polar_telemetry.py` 取代(留作 schema 文档)。
> **部署核查**:确认运行时加载的 vllm 就是 `/workspace/vllm`(`python -c "import vllm,os;print(os.path.dirname(vllm.__file__))"`);
> 若 vllm 被烤进镜像而非挂载/editable,需重建镜像或把改动同步进镜像。

### 3. gateway 补丁(P0 trace-id + v2 事件)— 已改,**下次重启 polar 生效**
改动文件(均**加性、向后兼容**,失败不影响推理热路径):
`proxy.py`(转发 trace 头)·`server.py`(两处注入)·`storage.py`(`peek_sequence` + 透传)·`completion_metrics.py`(schema v2)。

### 4. vllm /metrics(块① KV/queue + 引擎内部态)— 配置项
每 engine 开原生 Prometheus 端点,Prometheus 按 `engine_id` 抓 `card_topology.yaml:engine_endpoints`。
拿 `gpu_cache_usage_perc / num_requests_running|waiting / num_preemptions_total / prefix_cache_*`。

### 5. 块④ rollout 时间拆解(inference/verify/tool/wait)— 无需改 agent
从 agent 转录(每事件带 timestamp)重建,`verify_job` 由 `npu_lease_exec.py`(已加 `wait_seconds`/`exec_seconds`)补 lease_wait:
```bash
python3 build_spans.py <run_dir>     # → <run_dir>/telemetry_spans/{spans,rollout_spans}.jsonl
python3 to_perfetto.py <run_dir>/telemetry_spans/spans.jsonl --session <sid> -o trace.json  # ui.perfetto.dev
```

## 分析(一条龙)
```bash
python3 build_spans.py <run_dir>     # 块④ span 重建(可选,先跑)
python3 analyze.py     <run_dir>     # 块②/③分布 + 每 engine 均衡 + 块④占比 + T8 goodput
```

## 落地形态
- 时序(块①+引擎态)→ **Prometheus + Grafana**(`npu_smi_exporter` + vllm /metrics + `grafana_dashboard.json`)。
- 事件(per-request/per-span)→ **JSONL**(`completion_metrics` + `engine_metrics` + `telemetry_spans`),`analyze.py` 离线出报表。
- 单 rollout 瀑布(块④)→ `to_perfetto.py` 生成 Chrome/Perfetto trace。

## 状态(P0–P3 全部落地,无遗留代码步骤)
- ✅ P0 trace-id 贯通(gateway 补丁,重启生效)。
- ✅ P1 `npu_smi_exporter`(块①)+ **vllm `polar_telemetry` 已接入**(块②/③ engine 真值)+ gateway v2 事件。
- ✅ P2 `build_spans.py`(块④,转录重建)+ `npu_lease_exec.py` lease_wait/exec(验证池饱和)。
- ✅ P3 `analyze.py` T8 goodput/rollout_step 聚合 + `grafana_dashboard.json` + `to_perfetto.py`。

## 起 running 前的三步(全是配置,非代码)
1. 各 engine 启动加环境变量:`POLAR_ENGINE_ID=infer-{0,1,2}` + `POLAR_ENGINE_METRICS_DIR=<run_dir>/engine_metrics`。
2. 起 exporter:`python3 npu_smi_exporter.py --port 9800`(填好 `card_topology.yaml` 的卡号/端口)。
3. 重启 polar(gateway/lease/pipeline 补丁随之加载)。
之后 `build_spans.py` + `analyze.py` 一跑即完整基线(含 engine 真值 + 块④ + goodput)。

## 归属
gateway 补丁属 **main-npu(核心)**;telemetry 部署件 + `npu_lease_exec`/pipeline 属 **feat/operator**。当前落 feat/swe-tasks 工作区以便即用,分支重整时归位。
