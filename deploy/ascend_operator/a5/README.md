# A5 / Ascend 950 算子执行镜像

这是当前 `ascendc-tilelang:v2` 的 **Polar agent / 算子评测** 镜像迁移配方，
不包含 VIME、Megatron 或 vLLM 训练推理服务。
参考本机 2026-11-04 的 v2 构建记录和原 sandbox 分阶段构建流程。
使用单独标签 `ascendc-tilelang:a5`，原 T2A/T3A profile 不自动切换。

在目标 A5 容器中直接接手安装与验证，请先阅读 [Agent 交接文档](HANDOFF_A5.md)。

## 先采集目标机器配置

在 A5 宿主机执行（脚本也可以单独复制过去，不依赖仓库其他文件）：

```bash
bash deploy/ascend_operator/a5/inspect_a5.sh 2>&1 | tee a5-config.txt
# 若已有候选基础镜像，可额外查看其 CPU 架构等元数据：
bash deploy/ascend_operator/a5/inspect_a5.sh '<本地镜像:标签>' 2>&1 | tee a5-config.txt
```

只读采集 CPU/NPU、驱动、CANN 安装路径、当前 Python 包版本、Node 文件架构和
本地 Docker 镜像列表。不安装依赖、不启动容器、不加载 torch 或执行 NPU kernel。
宿主机没有 torch_npu 属正常情况，不能据此推断镜像里也没有。

2026-11-17 用户回传的 **C04-B03-OS4 当前容器** 配置：

| 项目 | 已确认值 |
|---|---|
| CPU / 容器 OS | aarch64，Kunpeng 950，192 CPU；Ubuntu 22.04.5 |
| NPU | 8 × Ascend950DT，每卡 98304 MiB（96 GiB），Health OK |
| 编译 SOC | `Ascend950DT_9582`（Chip Name=Ascend950DT，NPU Name=9582；Chip Version=V120） |
| 驱动 | 25.1.rc3.1 |
| 固件 | 9.2.0.102.200；board 查询显示 Compatibility=OK |
| CANN | 9.1.0，V100R001C11SPC001B243；`/usr/local/Ascend/cann` 指向 `cann-9.1.0` |
| Python | 3.12.13，`/usr/local/python3.12.13/bin/python3`，含 Python.h |
| 最新安装进度 | 上述 Python 已安装 torch 2.12.0+cpu / torch_npu 2.12.0；导入、NPU 可用性及单卡 64×64 FP16 矩阵乘法对拍通过 |
| 其他软件 | 初次采样未安装 TileLang，PATH 无 Claude、Node、Docker；后续状态未回传 |

构建平台应为 `linux/arm64`，Claude 用 `linux-arm64`，PyTorch wheels 用
CPython 3.12 / aarch64。按官方[版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)，
实际已安装的 `torch 2.12.0` + `torch_npu 2.12.0` 对应 CANN 9.1.0；
目前已验证导入、NPU 可用性和一次单卡矩阵乘法对拍；TileLang 和完整评测链路仍待验证。
不要直接复用旧 v2 的 cp311/x86_64 wheels 或 CANN 9.2 配套的 post7。
精确编译 SOC 已根据 `npu-smi info -t board -i 0` 的 Chip Name / NPU Name
确认为 `Ascend950DT_9582`，作为 Dockerfile 默认值。
固件版本号不等于 CANN toolkit 版本；此处 CANN 仍为 9.1.0。
原始容器镜像名需要从启动平台或宿主运行时确认。

## 与 v2 的必要差异

- 原 v2：x86_64、Python 3.11.9、CANN/910B ops 9.2.0、
  torch 2.10.0+cpu / torch_npu 2.10.0.post7、TileLang-Ascend `1065aea14ec9`。
- A5：必须提供 **950 对应** 的 CANN toolkit、ops 基底，以及匹配的 torch_npu；
  不能把原 v2 的 910B 算子包当作 950 算子包，也不默认沿用原 torch 组合。
  Dockerfile 可从 `wheels/` 安装配套包；目录为空时沿用基底的 torch/torch_npu。
  安装 TileLang 前锁定 torch/torch_npu 版本，并在安装后检查未改变。
- TileLang 使用主仓库 `tile-ai/tilelang` 的固定提交 `562cbc7c8188d4521920e76b708d00168e778f3b`，
  开启 `USE_ASCEND=ON USE_CUDA=OFF`，目标架构 `dav-3510`。
  [旧分支维护者说明（2026-10-20）](https://github.com/tile-ai/tilelang-ascend/issues/1830)
  和 [950 后端安装文档](https://github.com/tile-ai/tilelang/blob/562cbc7c8188d4521920e76b708d00168e778f3b/tilelang/ascend/README.md)。
- 原 CLI 使用原生可执行文件，继续复用这一方式；不需要安装 Node。
  原镜像的 x64 文件不能用于 aarch64 主机。

## 构建

在有 Docker/BuildKit 的目标架构机器上准备：

1. Debian/Ubuntu 系的 A5 开发基底，包含 Python >= 3.10、与该 Python 匹配的
   `Python.h`、pip、A5 CANN toolkit/ops。
   需要 `bisheng`、CCE `ld.lld`、AscendC 开发头文件/CMake 模板和 `msprof`。
   驱动/固件由宿主机提供，不在 Dockerfile 里安装。
2. 本机 SOC 使用 `Ascend950DT_9582`；换其他机器时根据 board 查询结果覆盖构建参数。
3. 对应 CPU 架构的可信 Claude 原生二进制，放到临时构建目录。
4. 基底缺少 PyTorch 时，准备配套的 torch 和 torch_npu wheels；
   可在目标 Python/CPU 架构环境下载，或使用厂商提供的配套包。

从 Polar 仓库根目录运行，替换基底镜像名和本地文件路径：

```bash
build_context=$(mktemp -d)
cp deploy/ascend_operator/a5/Dockerfile "$build_context/Dockerfile"
cp /path/to/claude-linux-arm64 "$build_context/claude"
mkdir -p "$build_context/wheels"
# 当前 CANN-only 容器需要这一步；已有配套 torch/torch_npu 的基底可留空目录。
cp /path/to/matching-wheels/*.whl "$build_context/wheels/"

docker build --platform linux/arm64 --progress=plain \
  --build-arg BASE_IMAGE='<你的 A5 CANN 开发镜像>' \
  --build-arg CANN_HOME=/usr/local/Ascend/cann \
  --build-arg SOC_VERSION=Ascend950DT_9582 \
  --build-arg BUILD_JOBS=2 \
  -t ascendc-tilelang:a5 "$build_context"
```

`CANN_HOME` 下必须存在 `set_env.sh`；若基底使用
`/usr/local/Ascend/ascend-toolkit/latest`，相应修改参数。
构建需要访问系统包源、PyPI 和 GitHub；代理可通过 Docker 标准构建参数传递。
仅复制 Dockerfile、CLI 和 wheels，避免把工作区、模型、数据或凭据打包进构建上下文。
安装本地 wheels 时仍可从包源解析依赖；这不是完整离线构建。
`BUILD_JOBS` 只限制编译并发，并不限制整个 Docker daemon 的资源占用。

## 上卡验证

复用目标机器已能运行 torch_npu 的容器启动命令及设备/驱动挂载，将镜像替换为
`ascendc-tilelang:a5`，命令替换为 `polar-a5-smoke`。命令会验证设备确为 950，
然后编译、执行并对拍固定版本上游的 SIMT/SIMD vector-add，不做性能跑分。
若容器已按该方式启动，也可以执行：

```bash
docker exec <A5测试容器> polar-a5-smoke
docker exec <A5测试容器> bash -lc 'claude --version; cmake --version; command -v bisheng msprof'
```

镜像构建时只检查软件包、版本、头文件和工具；上卡 smoke 才会导入/执行 NPU 栈。
通过 vector-add 也不代表完整 AscendC pipeline、模型或 RL 流程已通过验证。

## 接入 Polar

复制现有 profile 为单独的 A5 profile，再修改这些字段：

```yaml
operator_runtime:
  asc_devkit_dir: /path/to/A5-matching-readable-asc-devkit
operator:
  runtime:
    image: ascendc-tilelang:a5
    env:
      SOC_VERSION: "Ascend950DT_9582"
      ASCENDC_SOC_VERSION: "Ascend950DT_9582"
      ASCEND_SETENV: "/usr/local/Ascend/cann/set_env.sh"
```

profile 中原有的 `ascend910b1` 会覆盖镜像环境，必须同时修改。
`ASCEND_SETENV` 要与构建参数一致；现有 `tools/env.sh` 会读取它。
原来经过 A2/A3 裁剪的 `/opt/asc-devkit` 文档不能继续作为 A5 API 依据。
按新节点实际情况核对 NPU device/driver 挂载和 lease 卡池，再运行一个真实算子 pipeline。

**TileLang API 需要迁移**：950 后端使用 `import tilelang.ascend.language as T`
及 `target="ascend"`；现有 skills/模板中的 `tilelang.language`、
`target="ascendc"` 和旧 A2/A3 DSL 不能保证直接运行。不要只替换 import 或
SOC 名称后批量启动 RL。若任务还依赖 Triton，基底也必须提供已验证适配 A5 的
triton-ascend；此配方不猜测或覆盖其版本。

当前 Dockerfile 尚未实际构建或运行；A5 当前容器的 torch 单卡测试已通过。
镜像固化仍需原始基底镜像名，并根据基底是否自带 torch 栈准备配套 wheels。
