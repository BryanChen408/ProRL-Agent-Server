# A2 双机 Polar + VIME 稳定训练

本文只描述一套边界清楚的部署：VIME 训练容器负责 Ray、Megatron、vLLM 和数据集；Polar 在宿主机负责 rollout、gateway、observer、watcher，并通过 Docker 启动 agent runtime。VIME 只需要 Polar rollout URL、数据集路径和 resource layout。

## 1. 目录和前提

Polar 仓可以放在任意宿主机目录，下面用 `${POLAR_ROOT}` 表示仓根目录：

```text
${POLAR_ROOT}/deploy/ascend_operator/
${POLAR_ROOT}/operator_runtime_t2a/
${POLAR_ROOT}/output/ascend_operator/

/workspace/vime/                         # vime 容器内
```

Profile 中的相对路径都相对于 `${POLAR_ROOT}` 解析；不要把 `${POLAR_ROOT}` 写死进 profile。两台机器都准备好相同的数据集路径，或把数据集放在 vime 稳定可读的数据卷中。

需要满足：

- A2/A3 驱动、CANN 和 `npu-smi` 在训练容器与 runtime 容器可用。
- Polar 宿主机有 Python 3.11+、可用的 Docker CLI 和 Docker daemon。
- Polar 宿主机可拉取 runtime 镜像 `ascendc-tilelang:v1`。
- 训练容器能访问 Polar 宿主机的 rollout URL；双机防火墙放通 rollout、gateway、router 和 observer 端口。

## 2. 启动 VIME 容器

每台参与 Ray 的机器启动一个 VIME 容器。挂载项按本机 Ascend 安装位置调整：

```bash
docker load -i <vime-image>.tar
docker run -it --name vime --network host --ipc host --shm-size=500g \
  --privileged \
  -e http_proxy= -e https_proxy= \
  -v /dev:/dev -v /home:/home \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /mnt/pipeline-data:/mnt/pipeline-data \
  <vime-image> bash
```

Polar 不要在这个容器内启动；容器内通常没有启动 agent runtime 所需的 Docker daemon。

## 3. Polar 宿主机

```bash
cd "${POLAR_ROOT}"
python3 -m venv /root/polar-venv       # 仅首次执行
source /root/polar-venv/bin/activate
pip install -e .
docker image inspect ascendc-tilelang:v1 >/dev/null
```

已有 venv 时只需 `source /root/polar-venv/bin/activate`。runtime 镜像必须在 Polar 宿主机可见；它不是 VIME 镜像。

## 4. Polar profile

复制 `deploy/ascend_operator/profile.t2a.yaml`，并按双机卡位改环境相关值。仓内基线 profile 的租约池是整机示例值；在本文拓扑中必须把它改成 Polar 可用卡（例如 `[0, 1, 2, 3]`），否则 agent/eval 会和 VIME 抢卡。profile 的关键结构如下（端口和路径均为示例）：

```yaml
service:
  bind_host: 0.0.0.0
  rollout_url: http://<polar-host>:8180
  gateway_url: http://<polar-host>:8200
  sglang_router_url: http://<vllm-router-host>:8011
  inference_engine: vllm
  model_served: <vllm-served-model-name>

paths:
  output_dir: output/ascend_operator
  operator_runtime_dir: operator_runtime_t2a

operator_runtime:
  workflow: legacy
  backend: ascendc
  task_assets_dir: <polar-or-shared-dataset>/op_tasks
  asc_devkit_dir: <prepared-asc-devkit>
  budget:
    generation_max: 30
    optimization_max: 6
    interval_seconds: 2
  npu_lease:
    enabled: true
    pool: [0, 1, 2, 3]
    lock_dir: /dev/shm/npu-locks

observer:
  host: 0.0.0.0
  port: 18189

gateway:
  node_id: ascend-node-01
  max_init_workers: 8
  max_run_workers: 64
  max_postrun_workers: 32

operator:
  profile: operator_npu
  timeout_seconds: 21600
  runtime:
    image: ascendc-tilelang:v1
    network: host
    workdir: /opt/workspace/agent_workdir
    env:
      SOC_VERSION: ascend910b1
      ASCENDC_SOC_VERSION: ascend910b1
      ASC_DEVKIT_DIR: /opt/asc-devkit
```

不要把 `pipeline_budget` 写成顶层字段；当前 loader 读取的是 `operator_runtime.budget`。不要使用 `npu_pool`；租约池字段是 `operator_runtime.npu_lease.pool`。`task_assets_dir` 必须是 Polar 宿主机能读到的目录，并且包含算子同名 `.py`/`.json` 资产。`operator_runtime_dir`、`output_dir` 使用相对路径即可。

若使用其他 A2 机器，只替换 `service` 地址、模型名、数据集、`asc_devkit_dir` 和租约卡号；不要复制另一台机器的旧端口或旧 profile。租约池必须与 `polar_reserved` 完全一致。

## 5. 启动 Polar

在 Polar 宿主机执行：

```bash
cd "${POLAR_ROOT}"
source /root/polar-venv/bin/activate
POLAR_PROFILE=deploy/ascend_operator/profile.t2a.yaml \
POLAR_RUN_ID=polar_$(date +%Y%m%d_%H%M%S) \
NO_PROXY=127.0.0.1,localhost,<polar-host>,<vllm-router-host> \
no_proxy=127.0.0.1,localhost,<polar-host>,<vllm-router-host> \
bash deploy/ascend_operator/restart_polar_host.sh
```

确认服务和节点：

```bash
curl -fsS http://<polar-host>:8180/health
curl -fsS http://<polar-host>:8200/health
curl -fsS http://<polar-host>:8180/nodes
```

## 6. VIME 资源布局和数据集

VIME 侧只配置 rollout URL、数据集和资源布局；不要把 Polar runtime、skills、judge 或 output 路径配置到 VIME。示例（141 训练，140 rollout，140 的 0-3 给 Polar）：

```yaml
roles:
  actor:
    - {node: <train-host>, devices: "0-15"}
  rollout:
    - {node: <polar-host>, devices: "4-15"}
  polar_reserved:
    - {node: <polar-host>, devices: "0-3"}
rollout:
  num_gpus_per_engine: 2
  vllm_dp_size: 2
```

`polar_reserved` 是规划标记；VIME placement 只消费 `actor` 和 `rollout`。`num_gpus_per_engine` 是 TP 卡数（PP=1），`vllm_dp_size` 是 P+D engine 总数。PD 配置通过 VIME 的 `VLLM_PD_CONFIG` 指定，不能用它替代 resource layout。

数据集至少需要：

```text
<DATA_ROOT>/operator_tasks.jsonl
<DATA_ROOT>/op_tasks/
```

启动 VIME 时设置：

```bash
export OPERATOR_DATA_ROOT=<DATA_ROOT>
export OPERATOR_TASK_JSONL=<DATA_ROOT>/operator_tasks.jsonl
export POLAR_ROLLOUT_URL=http://<polar-host>:8180
```

## 7. 双机启动顺序

先在训练机启动 Ray head，再在 Polar/vLLM 机启动 Ray worker，最后由 head 提交训练。使用 VIME 镜像中实际存在的 launcher；启动前先执行 `test -x`，避免把旧脚本名写进自动化：

```bash
# 训练机
cd /workspace/vime
test -x scripts/start_pd.sh
bash scripts/start_pd.sh

# Polar/vLLM 机的 VIME 容器
cd /workspace/vime
test -x scripts/start_pd_worker.sh
bash scripts/start_pd_worker.sh
```

若环境使用 `start_sync_hybrid.sh` 或其他 launcher，以该 launcher 的参数为准；不要同时启动两套 Ray head。先用小规模 `NUM_ROLLOUT` 和 `POLAR_MAX_ACTIVE_SESSIONS` 验证链路，再放大到目标并发。

## 8. 验证和日志

```bash
ray status
curl -fsS http://<vllm-router-host>:8011/health
tail -f /workspace/slime-ascend/output/polar_bridge/train_<RUN_ID>.log
tail -f ${POLAR_ROOT}/output/ascend_operator/runs/<POLAR_RUN_ID>/logs/rollout.log
tail -f ${POLAR_ROOT}/output/ascend_operator/runs/<POLAR_RUN_ID>/logs/gateway.log
```

observer 地址为 `http://<polar-host>:18189`。所有 Polar 运行产物都在 `${POLAR_ROOT}/output/ascend_operator/`；不要从 VIME 容器写 Polar output。短 debug run 结束时 session pool 预取的尾部任务可能在 observer 中显示异常收尾，先以训练消费数据和 session 结果为准；长稳训练不应持续出现该现象。
