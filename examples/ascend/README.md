# Ascend 真实推理 smoke(Polar × vllm-ascend × Claude Code)

目标:在你的 NPU 节点上**真跑一次** —— Claude Code 在 Polar 的 runtime 容器里执行,它的 Anthropic 调用经
Polar gateway 翻译后打到 **vllm-ascend(qwen35)**,Polar 捕获一条 token-faithful 轨迹。这是第一次让**真实推理引擎**
被调起,并验证整条 rollout 链。**不改 Polar 核心代码**(用独立 `submit_claude_code.py`,不碰 `run.py`)。

## URL 是怎么流的(重要:vllm 地址不经过 rllm/这个脚本)
```
submit_claude_code.py ──(rollout :8080)──▶ Polar rollout server
                                              └─dispatch─▶ Polar gateway(:8100)
                                                  ├─ 给 agent 注入 ANTHROPIC_BASE_URL = 自己(:8100/sessions/<id>)
                                                  └─(topology.yaml 的 inference.base_url)─▶ vllm-ascend(:8003)
```
- **vllm 的 URL 只写在 `topology.qwen35.yaml` 一行**(`inference.base_url: http://127.0.0.1:8003`);本脚本/agent 都不碰它。
- `model_served: qwen35` 必须 == 你 vllm-ascend 的 `--served-model-name`。

## 前置 + 一个关键决策:Polar 在哪跑、用什么起容器
Polar 的 rollout/gateway 通过 `docker create/start/exec` 起 **agent 容器**,所以**必须能访问容器运行时**。你现在在「训练主容器」里——通常它没有 docker daemon,三选一:
- **A. 宿主机跑 Polar**(宿主有 docker)——最简单。
- **B. 训练容器挂 docker.sock**(容器启动时 `-v /var/run/docker.sock:/var/run/docker.sock`),Polar 在容器内起宿主的兄弟容器。
- **C. apptainer**(很多 NPU/HPC 节点用这个,无需 docker daemon):下面所有 `--backend docker` 换 `--backend apptainer`。

> agent 容器**不需要 NPU/vllm**,只要能网络到 Polar gateway(单机用 `network: host` 即可)。

vllm-ascend 起服务时建议带 parser(harness 要用 tool/thinking;P0b 不需要):
```bash
# 用你原来的 vllm-ascend 命令,补上:
--served-model-name qwen35 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

## 步骤(在能访问 docker/apptainer 的地方执行)

**0. 装 Polar(纯 Python,普通 venv;不需要 torch/uv):**
```bash
cd <ProRL-Agent-Server>
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**1. build agent 镜像(claude-code 预装;受限网络用代理):**
```bash
# 基础镜像走 docker daemon 代理(配 ~/.docker/config.json 或 systemd http-proxy);apt/npm 走 build-arg:
docker build -f examples/ascend/Dockerfile \
  --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy \
  -t polar-ascend-agent:latest examples/ascend
# build 期 `claude --version` 跑过 = 镜像里 CLI 没问题(装/代理错会当场 build 失败)
```

**2. 确认 topology**(已填好 `:8003`/`qwen35`,核对一眼):`examples/ascend/topology.qwen35.yaml`

**3. 起 Polar(rollout + 1 个 gateway 节点):**
```bash
polar serve_rollout -c examples/ascend/topology.qwen35.yaml &
polar serve_gateway -c examples/ascend/topology.qwen35.yaml --node-id ascend-node-01 &
polar status      -c examples/ascend/topology.qwen35.yaml      # 健康检查,等节点 READY
```

**4. 真跑一次(真实推理在这步发生):**
```bash
python examples/ascend/submit_claude_code.py \
  --rollout-url http://127.0.0.1:8080 --image polar-ascend-agent:latest --backend docker
# apptainer: --backend apptainer
```
它会提交 → 轮询 → dump `ascend_session.json` → 打印每条 trace 的 prompt_ids/response_ids/logprobs 长度。
**看到 trace 有非空 token_ids/logprobs = 真实推理被调起 + token-faithful 捕获成功。**

**5. 全契约校验(同时也证明 adapter 的合成 fixture == 真实形状):**
```bash
python examples/ascend/verify_trajectory.py ascend_session.json --expect-per-request
```
绿 = 每条 Trace 的 `len(loss_mask)==len(response_logprobs)==len(response_ids)`、mask 全 1 —— 正是 rllm `integrations/polar/adapter` 消费的契约。

## 排错
- **gateway 打不通 vllm**:核对 `topology.inference.base_url=http://127.0.0.1:8003`(不带 `/v1`)、vllm 在服务、`polar status` 绿。
- **agent 找不到 claude / 容器拉不起**:确认镜像 build 过;受限网络确认 daemon 代理(拉基础镜像)和 build-arg 代理(apt/npm)都配了;apptainer 记得 `--backend apptainer`(镜像名自动加 `docker-daemon:`)。
- **401 / 模型名报错**:gateway 用 `model_served` 覆盖请求里的 model;确认 `topology.model_served` == vllm `--served-model-name`(qwen35)。
- **tool_call/thinking 解析失败**:vllm-ascend 加上面那几个 `--*-parser` 旗标。
- **容器到 gateway 不通**:单机 `network: host` + `127.0.0.1`;跨网络命名空间时把 topology 的 `public_url`/`host` 换成可达 IP。
- **拿到真实 `ascend_session.json` 后**:把它发回去,可直接喂 rllm 的 `polar_episode_builder` / `pytest tests/integration/test_polar_engine.py` 用真数据再验一遍(L1)。
