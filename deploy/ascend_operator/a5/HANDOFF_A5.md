# A5 Agent 交接：补齐 Ascend950DT 算子执行环境

更新：2026-11-17。本文可单独交给目标机器上的 coding agent，按顺序执行并记录结果。

## 最新进度：torch_npu 单卡矩阵乘法对拍已通过

用户已回传：Python 为 `/usr/local/python3.12.13/bin/python3`，
`torch=2.12.0+cpu`、`torch_npu=2.12.0`，两者导入成功且 `torch.npu.is_available()` 为 True。
用户随后回传单卡测试结果：`PASS: Ascend950DT_9582 矩阵乘法对拍通过`。
该测试完成了 64×64 FP16 矩阵的 NPU 分配、矩阵乘法、结果回传、同步及与 CPU FP32 参考对拍
（rtol=1e-2，atol=1e-2）。这不等于所有算子或八卡通信已经通过。
复用此解释器，不重复安装，不降级到此前候选 2.10，
也不要创建并切换到空 venv。需要复查版本时执行：

```bash
python3 - <<'PY'
import sys
from importlib.metadata import version
print('python:', sys.executable)
for name in ('torch', 'torch-npu'):
    print(name, version(name))
PY
```

若查不到包，先检查是否用了错误的解释器。
下文的 Python 命令及 `OPERATOR_PYTHON` / `AST_CHECK_PYTHON` / `PY_BIN`
统一指向 `/usr/local/python3.12.13/bin/python3`。
第 4.2 节 torch 安装命令仅作为重建配方；当前补齐环境脚本、版本约束和构建依赖后，
从第 4.4 节继续 TileLang 构建与 SIMT/SIMD 对拍。第 4.3 节保留为复现步骤，未改变环境时无需重跑。
后续安装 TileLang 时约束 torch 版本，避免依赖解析替换已安装的组合。

## 1. 接手任务

用户正在把原 RL 中 `ascendc-tilelang:v2` 的 **Polar 算子 agent / 评测环境**迁移到 A5。
当前优先在已经启动的 A5 容器内补齐依赖、验证实际算子执行，再把有效步骤固化到 Dockerfile。
原始镜像名暂时未知，不阻塞容器内安装和测试。当前容器没有 Docker，无需为此安装嵌套 Docker。

本阶段交付：

1. 可重复加载的 A5 环境、明确的 Python 解释器和软件版本。
2. torch_npu 上卡、TileLang SIMT/SIMD 对拍、一个真实 AscendC pipeline 的分层结果。
3. 安装日志、构建 wheel/源码提交、失败原因，以及更新后的构建配方。

本阶段不包含 VIME/Megatron/vLLM 训练推理栈迁移，不自动启动完整 RL。
保持已有宿主驱动、固件、CANN 和运行中的服务；复用当前已验证可导入 torch 的 Python 环境。
先只在一张当前空闲的卡上做小规模验证。不要执行设备 reset 或沿用旧 Box16 恢复脚本。
在此范围内推进安装、编译和验证，遇到缺文件/网络/权限时给出具体缺项，不把未执行步骤标为通过。

## 2. 已确认的目标环境

以下来自安装前用户执行 `inspect_a5.sh` 和 `npu-smi info -t board -i 0` 的输出；
torch / torch_npu 的后续安装进度以上方“最新进度”为准。

| 项目 | 已知值 |
|---|---|
| 节点 / 用户当前目录 | `C04-B03-OS4` / `/home/z00893068`，root shell |
| 运行位置 | **容器内部**，根文件系统 overlay，Docker 命令不存在 |
| 容器发行版 | Ubuntu 22.04.5 LTS |
| 宿主内核 | `6.6.0-145.3.34.165.oe2403sp3.aarch64` |
| CPU | aarch64，Kunpeng 950 7592C；可见 192 CPU、4 NUMA 节点 |
| 内存 | 可见约 1.5 TiB；容器的实际 cgroup 配额尚未核实 |
| NPU | 8 张 `Ascend950DT`，ID 0–7，每张 98304 MiB = 96 GiB |
| 精确编译 SOC | **`Ascend950DT_9582`**；Chip Version V120 |
| 驱动 / 固件 | `25.1.rc3.1` / `9.2.0.102.200`；board Compatibility=OK |
| CANN toolkit | **9.1.0**，`V100R001C11SPC001B243` |
| CANN 路径 | `/usr/local/Ascend/cann -> /usr/local/Ascend/cann-9.1.0` |
| 兼容路径 | `/usr/local/Ascend/ascend-toolkit/latest` 同样指向 CANN 9.1.0 |
| Python | `/usr/local/python3.12.13/bin/python3`，3.12.13，Python.h 存在 |
| 编译器 | GCC 11.4.0；Bisheng 15.0.5，位于 CANN `bin` |
| CMake | 3.22.1，需为 TileLang 升级 |
| 已有工具 | `npu-smi`、`msprof`；msprof 不支持 `--version` |
| 当前 Python 缺失 | torch、torch_npu、tilelang、triton-ascend、pybind11 |
| 当前 PATH 缺失 | Claude、Node、npm、Docker |

补充：

- 固件 `9.2.0.102.200` 不代表安装了 CANN 9.2，不能据此选择 CANN 9.2 配套包。
- 用户此前下载了 `node-v24.19.0-linux-x64`，因 CPU 架构不匹配出现 Exec format error。
  如需 Node 应用 `linux-arm64`；本任务安装原生 Claude，无需 Node/npm。
- 设备节点 `/dev/davinci0..7`、`/dev/davinci_manager`、`/dev/hisi_hdc` 可见；
  `/dev/devmm_svm` 不存在。不要凭单一旧设备节点缺失判断驱动坏了，也不要手工 mknod。
- 历史采样时 NPU 健康且未列出进程，但每卡约占用 4.7 GiB；不能把整段占用归为用户进程，
  也不能把该历史快照当作接手时卡仍空闲的证明。

## 3. 软件选型与来源

- 当前实际组合：`torch==2.12.0+cpu`、`torch-npu==2.12.0`。
  官方[配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
  将 torch 2.12.0 / torch_npu 2.12.0 对应到 CANN 9.1.0，并支持 Python 3.12。
  本机已验证导入、NPU 可用性和一次 64×64 FP16 矩阵乘法对拍，其他算子仍需分别验证。
- 重新准备 wheels 时使用 cp312/aarch64，保存实际包来源和 SHA256；
  不复用旧机器的 cp311/x86_64 wheels。
- TileLang 使用 `https://github.com/tile-ai/tilelang.git`，固定提交
  **`562cbc7c8188d4521920e76b708d00168e778f3b`**。
  [950 后端文档](https://github.com/tile-ai/tilelang/blob/562cbc7c8188d4521920e76b708d00168e778f3b/tilelang/ascend/README.md)
  使用 `USE_ASCEND=ON USE_CUDA=OFF`，架构 `dav-3510`，语言模块 `tilelang.ascend.language`。
- 旧 `tile-ai/tilelang-ascend` 并非可直接迁移的 A5 后端，参见
  [维护者说明](https://github.com/tile-ai/tilelang-ascend/issues/1830)。
- Claude 可沿用原镜像记录中的版本 `2.1.318`，但文件必须改为 `linux-arm64`。
  不要安装旧 x64 文件，不把任何登录凭据写进构建上下文。

## 4. 当前容器中的执行步骤

下面的示例在同一个 Bash 会话中执行，逐块查看结果；不要把整篇 Markdown 当作脚本运行。
如果接手时环境已经补过依赖，先读取实际状态，复用匹配的软件，不盲目重装。

### 4.1 记录环境并配置已有 Python

当前已安装 torch 栈，不创建新 venv。记录环境并补必要的系统依赖；
已有 env.sh 时先读取并按以下已确认路径调整。

```bash
set -eo pipefail
test "$(uname -m)" = aarch64
mkdir -p /opt/polar-a5/{logs,src,wheels,bin}
date -u | tee /opt/polar-a5/logs/start.txt
npu-smi info | tee /opt/polar-a5/logs/npu-before.txt
npu-smi info -t board -i 0 | tee /opt/polar-a5/logs/board.txt
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null || true

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates file util-linux \
  libnuma1 libgomp1 libedit-dev zlib1g-dev

/usr/local/python3.12.13/bin/python3 -V
```

创建 `/opt/polar-a5/env.sh`：

```bash
cat > /opt/polar-a5/env.sh <<'ENV'
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export ASCEND_SETENV=/usr/local/Ascend/cann/set_env.sh
source "$ASCEND_SETENV"
export PATH="/usr/local/python3.12.13/bin:/opt/polar-a5/bin:${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin:$PATH"
export SOC_VERSION=Ascend950DT_9582
export ASCENDC_SOC_VERSION=Ascend950DT_9582
export ASCEND_NPU_ARCH=dav-3510
export TILELANG_DEFAULT_TARGET=ascend
export ACL_OP_INIT_MODE=1
export OPERATOR_PYTHON=/usr/local/python3.12.13/bin/python3
export AST_CHECK_PYTHON=/usr/local/python3.12.13/bin/python3
export PY_BIN=/usr/local/python3.12.13/bin/python3
ENV
source /opt/polar-a5/env.sh
python3 -V
command -v python3 bisheng ld.lld
```

这里显式设置解释器，是因为现有 Polar `tools/env.sh` 会探测 `/usr/local/bin/python` 等位置，
只改 PATH 不一定能阻止 pipeline 选回另一个 Python。不要全局替换系统 Python 软链。

### 4.2 安装 torch 栈与构建依赖

当前 torch 栈已经安装，跳过下面第一段安装命令；它仅用于重建环境。

```bash
python3 -m pip install --upgrade pip
python3 -m pip install 'torch==2.12.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu \
  2>&1 | tee /opt/polar-a5/logs/install-torch.log
python3 -m pip install 'torch-npu==2.12.0' \
  --index-url https://pypi.org/simple \
  2>&1 | tee /opt/polar-a5/logs/install-torch-npu.log
```

当前从这里开始，记录约束并补构建依赖：

```bash
python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path
assert version('torch') == '2.12.0+cpu'
assert version('torch-npu') == '2.12.0'
Path('/opt/polar-a5/torch-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torch-npu'))
)
PY
python3 -m pip install -c /opt/polar-a5/torch-constraints.txt \
  'cmake>=3.26.1,<4' ninja pybind11 packaging pytest
cmake --version
```

包源不可达时使用可信离线 wheels，检查 cp312/aarch64 和来源；pip 缺包不能直接解释为硬件不支持。
不要为绕过依赖冲突无约束升级 torch，也不要换成旧 v2 的 torch_npu post7。

### 4.3 先验证 torch_npu 上卡

重新确认卡是否空闲；下面示例选择物理卡 0，必要时换成当前允许使用的卡。
`ASCEND_RT_VISIBLE_DEVICES` 只对这条测试生效，不写进通用 env.sh。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout -k 10s 120s python3 - <<'PY' 2>&1 | tee /opt/polar-a5/logs/torch-smoke.log
import torch
import torch_npu
assert torch.npu.is_available(), 'NPU unavailable'
name = torch.npu.get_device_name(0)
assert '950' in name, name
print('device:', name, 'torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
torch.manual_seed(0)
a = torch.randn(64, 64, dtype=torch.float16)
b = torch.randn(64, 64, dtype=torch.float16)
actual = (a.npu() @ b.npu()).float().cpu()
torch.npu.synchronize()
torch.testing.assert_close(actual, a.float() @ b.float(), rtol=1e-2, atol=1e-2)
print('PASS: torch_npu allocation, matmul, synchronize, correctness')
PY
```

导入、分配或执行失败时，先定位实际加载的 CANN/驱动库、950 ops 和包版本。
不要先继续构建整个 TileLang，也不要用驱动 stub 库替代真实运行时。

### 4.4 构建并验证 TileLang

下面目录应为新的源码目录；若已经存在，核对 HEAD 和未提交修改后复用，不执行强制清理。

```bash
git clone --no-checkout https://github.com/tile-ai/tilelang.git /opt/polar-a5/src/tilelang
git -C /opt/polar-a5/src/tilelang checkout --detach 562cbc7c8188d4521920e76b708d00168e778f3b
git -C /opt/polar-a5/src/tilelang submodule update --init --recursive
cd /opt/polar-a5/src/tilelang
git rev-parse HEAD | tee /opt/polar-a5/logs/tilelang-commit.txt
git submodule status --recursive > /opt/polar-a5/logs/tilelang-submodules.txt

CMAKE_BUILD_PARALLEL_LEVEL=2 USE_ASCEND=ON USE_CUDA=OFF \
  python3 -m pip wheel --no-deps --wheel-dir /opt/polar-a5/wheels . \
  2>&1 | tee /opt/polar-a5/logs/tilelang-build.log
cd /opt/polar-a5
python3 -m pip install -c /opt/polar-a5/torch-constraints.txt /opt/polar-a5/wheels/tilelang-*.whl
python3 -c 'import torch_npu, tilelang, tilelang.ascend.language; print(tilelang.__version__, tilelang.__file__)'

ASCEND_RT_VISIBLE_DEVICES=0 timeout -k 10s 600s python3 - <<'PY' 2>&1 | tee /opt/polar-a5/logs/tilelang-smoke.log
import sys
import runpy
import torch_npu
example = '/opt/polar-a5/src/tilelang/examples/ascend/example_vecadd.py'
sys.argv = [example, '--no-bench']
runpy.run_path(example, run_name='__main__')
PY
```

预期 SIMT 和 SIMD 两种 vector-add 都报告 correctness passed。
从源码目录外导入，避免源码树遮蔽已安装 wheel。默认并发 2，核实 cgroup 配额后才调整。
若工具链无法编译该固定提交，保留错误证据，核对 CANN 编译器要求；不要把它描述为已保证兼容。

### 4.5 安装 ARM64 原生 Claude（已存在时先核对）

```bash
curl -fL --retry 3 \
  https://downloads.claude.ai/claude-code-releases/2.1.318/linux-arm64/claude \
  -o /opt/polar-a5/bin/claude.download
printf '%s  %s\n' \
  '53a0d7e219d7b0804ecba5dbc002745e89f786cc946d2de85a6470d5b2d1fb06' \
  '/opt/polar-a5/bin/claude.download' | sha256sum -c -
chmod 755 /opt/polar-a5/bin/claude.download
mv /opt/polar-a5/bin/claude.download /opt/polar-a5/bin/claude
file /opt/polar-a5/bin/claude
/opt/polar-a5/bin/claude --version
```

该 SHA256 来自 2.1.318 官方 manifest 的 linux-arm64 条目。下载或校验失败时核对
`https://downloads.claude.ai/claude-code-releases/2.1.318/manifest.json`，不要跳过哈希校验。
Claude `--version` 通过只证明 CLI 能运行；Polar gateway 的模型调用接线还需单独验证。

## 5. 接入现有 Polar runtime 时必须检查的地方

目标机的 Polar 仓库路径尚未知。先定位实际仓库和它的 AGENTS.md；有 `.codegraph/` 时按其规则先查图。
来源机器上的仓库路径是 `/home/docker/polar_can/ProRL-Agent-Server`，不假定 A5 也有这个路径。

现有相关文件：

- `deploy/ascend_operator/profile.t2a.yaml`、`profile.t3a.yaml`
- `operator_runtime_t2a/tools/env.sh`、`tools/ascendc_eval_pipeline.sh`
- `operator_runtime_t2a/skills/tilelang2ascend-translator/scripts/build_ascendc.py`
- `operator_runtime_t2a/skills/tilelang2ascend-tilelang-designer/`
- 同结构的 `operator_runtime_t3a/`；实际使用哪一套，以选定 profile 为准。

复制一个 A5 profile 后逐项调整，不覆盖正在使用的 profile：

1. `SOC_VERSION`、`ASCENDC_SOC_VERSION` 都为 `Ascend950DT_9582`。
2. `ASCEND_SETENV=/usr/local/Ascend/cann/set_env.sh`；检查 pipeline 是否又 source 了其他旧 CANN 路径。
3. 本文直接配置容器时，`OPERATOR_PYTHON`、`AST_CHECK_PYTHON` 和必要的 `PY_BIN`
   指向 `/usr/local/python3.12.13/bin/python3`。将来若采用随附 Dockerfile，
   应核对最终镜像内实际解释器路径，不沿用未经确认的路径。
4. `operator_runtime.asc_devkit_dir` 指向 **A5 / CANN 9.1 匹配**的 SDK/API 文档。
   原配置的 `asc-devkit-9.0.0-a2a3-readable` 曾将 A5 专用页面裁剪为不可用提示，不能照搬。
5. 当前工作区 T2A profile 的 `operator_runtime.npu_lease.pool` 为 **0–15**，
   而这台机只有 **0–7**。先用单卡池 `[0]` 做验证，再按实际资源扩展。
   同时审查 `npu_recovery` 的 Box16 维护假设，未适配前不要启用自动整机恢复。
6. 旧网络地址 `80.48.5.56/.64`、gateway/rollout URL、模型路径、数据路径、挂载路径
   都是原环境配置，必须按 A5 的真实拓扑核实，不能直接启动旧 profile。
7. 950 TileLang 使用 `tilelang.ascend.language`、`target="ascend"`；
   旧模板的 `tilelang.language`、`target="ascendc"`、旧内存作用域/同步 API 需逐例核对。
   不做盲目的全仓替换。Triton/PTO 的依赖按实际使用路径另行适配，不因名称相近默认兼容。

在准备好的独立评测工作区中，沿现有固定入口验证一个真实 AscendC 提交物。
已有入口接受 `--op_name`、`--impl`、`--task`、`--out_dir`；使用绝对路径避免工作目录歧义。
示意（必须替换为真实任务和提交物）：

```bash
source /opt/polar-a5/env.sh
POLAR_NPU_LEASE_POOL=0 bash /path/to/prepared-runtime/tools/ascendc_eval_pipeline.sh \
  --op_name '<真实算子名>' \
  --impl /absolute/path/to/operator_impl.tar.gz \
  --task /absolute/path/to/reference.py \
  --out_dir /opt/polar-a5/logs/ascendc-smoke
```

检查编译参数确为目标 SOC、正确性、性能结果和 metrics 文件。上游 TileLang vector-add
通过不能替代这一步。缺任务数据或 runtime 时明确报告缺失路径，先完成其余环境验收。

## 6. 镜像固化与现有交付物

交接包内有 `Dockerfile`、`README.md`、`inspect_a5.sh` 和两项静态/模拟测试。
Dockerfile 已默认 `SOC_VERSION=Ascend950DT_9582`，固定 TileLang 提交，支持通过 `wheels/`
装 torch 栈，保留 Polar 所需的 `ENTRYPOINT []`，带 `polar-a5-smoke` 上卡检查命令。

它尚未被 Docker 构建验证。当前容器的原始镜像名要从启动平台或宿主运行时取得，
不能由容器主机名推断。也要确认 CANN/Python 是镜像自带还是挂载进来的；挂载内容不会自动进入新镜像。
取得基底后，在支持 ARM64 的构建环境上按 README 构建 `ascendc-tilelang:a5`。
不要覆盖老 `:v2` 标签，不要把包含凭据、模型和训练数据的当前容器直接作为发布成品。

若只收到本文，没有交接包，仍可完成第 4 节容器内安装；取得配方后再做镜像构建。

## 7. 验收与回传

至少留下以下信息，避免只报告“安装成功”：

- `python3 -m pip freeze`、`python3 -m pip check`、实际解释器和模块导入路径。
- torch / torch_npu / TileLang / Claude / CMake / Bisheng 版本与 TileLang commit。
- wheel SHA256、安装日志、torch-smoke、TileLang 两种模式对拍结果。
- AscendC pipeline 是否执行、使用哪个任务、正确性和性能结果；没执行则说明具体缺项。
- 实际改动文件、A5 profile 和后续复现命令；镜像构建是否完成及镜像 ID。

收尾时可执行：

```bash
source /opt/polar-a5/env.sh
python3 -m pip freeze > /opt/polar-a5/logs/pip-freeze.txt
python3 -m pip check 2>&1 | tee /opt/polar-a5/logs/pip-check.txt
sha256sum /opt/polar-a5/wheels/*.whl > /opt/polar-a5/logs/wheel-sha256.txt
npu-smi info | tee /opt/polar-a5/logs/npu-after.txt
```

交接时的验证边界：来源环境只做过 Shell/Python 语法检查、wheel 安装分支和只读采集的模拟测试。
**已确认 torch 2.12.0+cpu / torch_npu 2.12.0 的导入、NPU 可用性及单卡矩阵乘法对拍；TileLang、完整 AscendC pipeline、Docker 构建及完整 RL 尚未得到验证。接手 agent 需要实际完成并记录后续分层结果。**
