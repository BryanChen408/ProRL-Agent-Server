# Agentic-RL 推理侧负载采集(P0 + P1)

## 主机内存耗尽探针

在 **64 宿主机 root 终端**运行，无需重启 Polar 或训练：

```bash
cd /home/docker/polar_can/ProRL-Agent-Server
bash deploy/ascend_operator/telemetry/start_memory_probe.sh
```

需要 py-spy。本工作区已单独安装到 `output/ascend_operator/memory_probe_tools/bin/py-spy`；
其他机器可通过 `POLAR_PY_SPY=/absolute/path/to/py-spy` 指定已有二进制。
后台使用 transient systemd service，退出终端仍运行；**宿主机重启后需要重新启动**。
服务本身限 512 MiB、20% 单核 CPU，并设置 OOMScoreAdjust=-900，尽量保留取证进程。
`systemctl status polar-memory-probe` 查看状态，`journalctl -u polar-memory-probe` 查看错误，
`systemctl stop polar-memory-probe` 停止。

每 2 秒按 `(MemTotal-MemAvailable)/MemTotal` 记录整机内存及所有可见进程中 RSS 最大、增长最快的各 30 个。
首次达到 85%、90%、95%、98% 时抓快照；持续高水位每 60 秒重抓。启动时已高水位也立即抓取。
阈值可以追加参数覆盖，例如 `--thresholds 80 85 90 95 --interval 2`。
取证写本地共享工作区 `output/ascend_operator/memory_probe/`，不要放在 NFS 上。

- `memory.jsonl`：时间戳、内存分类、RSS/增长排行榜、PID/父 PID/cgroup。RSS 求和含共享页，不能当整机唯一占用。
- `incident-*/processes.json`：全部可见进程；cgroup 路径含 Docker ID，可用于映射评测容器。
- `incident-*/cgroups.jsonl`：cgroup v2 的 memory.current/stat/events/max，含匿名、文件缓存等分类。
- `incident-*/<pid>/`：最大和增长最快的各 5 个进程的 smaps_rollup、状态、最多 64 个线程内核栈及 Python 栈。
- `incident-*/meminfo`、`slabinfo`、`vmstat`、压力和内核日志：辅助定位不反映在进程 RSS 上的内存。

Python 栈通过 `py-spy dump --nonblocking` 读取，不暂停训练，不抓局部变量/张量，也不发信号。
单次调用限 3 秒，最多选 10 个进程；取栈期间采样会延后。非 Python 进程不提供用户态栈，仍保存内核栈和占用；
权限不足、进程退出、解释器不受支持会记录到对应文件，不能保证每个目标都有 Python 栈。
这是**当时的调用栈，不是历史内存分配栈**。没有故障前的分配追踪，不能仅凭一张栈判定泄漏来源。
日志轮转约 32 MiB×2；保留第一次和最近 31 次事故快照。输出目录默认仅当前用户可读。
默认拒绝在 PID 隔离容器运行，避免把局部进程榜误当整机全貌；`--allow-container --once` 仅用于测试。

覆盖:块①显存/内存、块②rollout-step 耗时/长度、块③batch(engine 连续批 + rollout group)长度。
块④(rollout 非推理占比)= P2,埋 `rollout_span`/`verify_job`,尚未在此实现。

两个物理隔离卡池:**12 卡推理(3 engine×4,TP4)+ 4 卡验证**。见 `card_topology.yaml`(按你实际卡号/端口改)。

## 关联键(P0 已接)
gateway 在发往 engine 的请求头注入 **`x-polar-trace-id = {session_id}:{turn_seq}`**(`server.py` 两处 completion 调用 + `proxy.py` 转发)。
- gateway 事件:`completion_metrics.jsonl`,schema v2,新增 `trace_id / policy_version / engine_url / group_id / rollout_step / op_name`。
- engine 事件:`engine_request_logger` 写 `engine_metrics/<engine_id>.jsonl`,同 `trace_id`。
- 两者 **join on trace_id** = 每 request 既有 gateway 口径(latency/tokens)又有 engine 真值(TTFT/prefill/decode/queue/prefix-cache/抢占)。

## 组件与部署

### 统一 NPU 信息查询

所有受管 session 和本仓库 telemetry exporter 的 `npu-smi` 只读查询由宿主
`npu_smi_snapshot.py` 执行。容器读取普通 `info` 快照；参数查询通过同目录
`query.sock` 请求宿主，保留真实 stdout、stderr 和返回码，不回退到容器直接探测。
目录按 `POLAR_NPU_SMI_CACHE_DIR` 配置，默认 `/dev/shm/npu-locks/npu-smi-snapshot`。

支持 `info` 整个只读命名空间，包括 `-l`、`-m`、`-t TYPE -i ID -c CHIP`、
`info proc`、帮助和版本。命令与参数按 argv 传递，不经过 shell；
`set/reset/clear/upgrade` 等写操作拒绝执行。硬件实际支持哪些查询类型由宿主
安装的 npu-smi 决定，未知参数原样返回其错误，不伪造字段。

宿主只有一个 owner，定时采集与参数查询串行，不同时启动驱动探测；相同 argv
共用 30 秒结果缓存，失败缓存 300 秒。`--interval`、`--failure-interval`
可调。缓存最多 256 种查询。原生探测默认限 5 秒，客户端等待限 15 秒；
持续 `info watch` 也受该上限约束，超时返回已有输出和 124，不能持续独占接口。
驱动子进程即使无法退出，也保留 singleton 锁，阻止新 owner 重叠探测；
读者快速返回超时或使用已有普通 info 快照，不绕过统一接口。
这些是硬件信息查询，算子验证、测速继续走评测入口和 NPU 租约。

部署在每台需要读取本机 NPU 的**宿主机**执行：

```bash
python3 -u deploy/ascend_operator/telemetry/npu_smi_snapshot.py \
  --directory /dev/shm/npu-locks/npu-smi-snapshot
# 另一个终端测试；不会在当前进程直接访问驱动
python3 src/polar/runtime/npu_smi_cached.py info -t board -i 0 -c 0
```

Polar 启动脚本和 `start_all_telemetry.sh` 均会启动 owner。
从旧版升级时，先确认旧 collector 的 PID/命令行并停止**仅该采集器**，再启动新版；
不要因此重启训练、Gateway 或评测。旧 owner 没有 query.sock，新版检测后报错，
不能把“already running”误当作参数查询可用。读取快照的裸 info 在切换期间仍可用。
若另有不属于本仓库的 exporter 或硬件工具，必须接入相同 client；本接口无法拦截
任意用户直接调用 DCMI、torch_npu 或其他未接入工具。

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
