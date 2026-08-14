# 设计：受限容器内的进程级 runtime（LocalRuntime）

证据分级沿用 `MIGRATION_T2A.md`：
`[实测]` 跑过命令看到输出 / `[读码]` 读源码确认 / `[未定论]` 有推断无证据 / `[未验证]` 没查过。

---

## 1. 约束变更

原拓扑要求 polar 跑在**宿主机**上，唯一理由是它要通过 docker/apptainer 起
per-session 沙箱。现在环境只给容器内权限：

| 项 | 状态 |
|---|---|
| 宿主机权限 | **拿不到** |
| DinD / DooD | 都试过，各自阻塞（DinD 卡容器权限）。暂不考虑 |
| 工具链 | 受限容器内**可以装** AscendC 工具链 |
| NPU 卡 | 可以给 polar **独占卡** → 与 vime 抢卡问题不存在 |
| 容器形态 | 与 vime **同镜像、独立容器** |
| 共享盘 | `/mnt` 可挂 |
| 平台侧额外配置 | **要尽量少**，加 docker 参数可能有困难 |

最后一条是最强的约束：三处只读挂载和 workdir 都必须由 **polar 进程自己**在容器内
解决，不能要求平台传任何 docker 参数、不能要求 privileged。

同事的 apptainer 方案**未经验证**（我们没跑过），不作为推荐依据。

---

## 2. 为什么进程级可行

`profile.t2a.yaml` 的 `operator.runtime` 只设了 `image` 和 `network: host`，
**没有** `cpus` / `memory_mb` / `storage_mb` `[读码]`。即网络隔离和资源配额本来就没在用。
沙箱实际只提供四件事：

1. AscendC 工具链环境（`ascendc-tilelang:v1`）→ 受限容器内可装，等价
2. 每 session 独立工作目录 → `base.py` 的 bind-mount 快路径已是 backend 无关
3. NPU 卡隔离 → 见下，**不是 runtime 做的**
4. 进程清理（拆容器即全杀）→ **唯一真实退化项**，见 §6

### 2.1 关键发现：租卡与 runtime 无关

`load_polar_profile.py:295` 构建的是
`kwargs.ascend = {pool, lock_dir, lease_at_start: False}` `[读码]`。

`lease_at_start=False` 意味着 DockerRuntime **只做 driver 挂载、不租卡**
（`docker.py:110-116`）；真正的租卡是**沙箱内** `tools/npu_lease_exec.py` 逐命令
flock，只靠 `POLAR_NPU_LEASE_POOL` / `POLAR_NPU_LOCK_DIR` 两个 env 驱动
（`load_polar_profile.py:221-234`）。

推论：**LocalRuntime 完全不需要 import `ascend.py`**。`--privileged -v /dev:/dev`
那整套 recipe 的唯一目的是「把宿主 NPU 环境搬进一个新容器」（`ascend.py:1-31` 明确
写了这点），而进程级运行时直接继承 gateway 容器已有的 `/dev/davinci*` 和 CANN。

`ascend.py:21` 另有一句要紧：现有方案本来就是**软隔离**（所有卡在容器内都可见，
靠 `ASCEND_RT_VISIBLE_DEVICES` 限定），不是设备文件隔离。所以进程级在卡隔离强度上
**没有退化**。

---

## 3. 核心机制：`/polar/session` 前缀重写

这是本设计的关键，比「改 workdir」更根本。**先前一版设计只打算改 workdir，
那是不够的，会在 judge 阶段静默出错。**

`gateway/node.py` 每 session 调 `create_runtime` **两次** `[读码]`：

| 用途 | session_id | session_dir |
|---|---|---|
| agent | `<sid>` | `<managed.session_dir>` |
| fresh eval judge | `<sid>-eval` | `<managed.session_dir>/eval_runtime`（`node.py:554`）|

Docker 下两者各自把自己的 session_dir bind 到**同一个** `/polar/session`，互不可见。
**进程级没有 mount namespace，做不到这件事。** 若照抄绝对路径，agent 和 judge 会
共用一个目录 —— judge 本该在「fresh 容器」上判分（`node.py:575` 注释），
共用后这个语义直接破掉。

### 3.1 一条规则解决

LocalRuntime 在 `exec()` 里把 `/polar/session` 前缀替换成**本实例的** `session_dir`，
作用于三处：命令字符串、env 的 value、cwd。

`/polar/session` 的引用面已全量确认，只有 5 处 `[实测 grep]`：

| 位置 | 性质 |
|---|---|
| `runtime/base.py:14` | 常量定义本身 |
| `agent/presets/claude_code.py:24` | `_config_dir` = `/polar/session/.claude` |
| `agent/presets/claude_code.py:118` | agent 日志 tee 目标 |
| `trajectory/evaluator/_patch_utils.py:268`、`operator_judge.py:138` | 注释 |
| `load_polar_profile.py:82` | cannbot 分支的 `.claude` 软链（**我们走 legacy，不涉及**）|

全部以 `/polar/session` 为前缀，无例外。一条前缀替换即可覆盖，约 15 行。

### 3.2 workdir 顺带 session 化

`workdir` 现为 `/opt/workspace/agent_workdir`（`load_polar_profile.py:266`），
在 `/polar/session` **之外**。改成 `/polar/session/agent_workdir` 后由 §3.1 的规则
自动落到各自 session 目录，agent / judge 隔离随之成立。

顺带收益：`operator_judge.py:135-143` 的 docstring 指出 bind-mount 快路径只覆盖
`/polar/session`，workdir 在外面时 judge 取 submission 得走 `docker cp`。挪进来之后
这条路径变成纯本地文件访问。

改 workdir 的影响面已查实，**agent 侧文档零处写死** `[实测 grep]`：
`operator_runtime_t2a/` 内 `/opt/workspace` 仅 3 处命中，全是可覆盖默认值
（`prepare_operator_workdir.py:262` 的 argparse default、`tools/env.sh:14` 的
`${WORKSPACE_BASE:=...}`、及其 triton 参考副本）。`ASC_DEVKIT_DIR` 有 81 处引用但
都是**环境变量名**而非路径，软链即可。

代码侧两处硬编码必须一起改，否则静默走错目录：

- `load_polar_profile.py:87` —— prepare 命令里写死 `--workdir /opt/workspace/agent_workdir`，
  **没走 `workdir` 变量**
- `_runtime_volumes:54` —— `{tools_dir}:/opt/workspace/agent_workdir/tools:ro`
`check_render_contract.py:123` 有同一个字面量，但**不需要改** `[实测]`：它校验的是
mainline 路径（`polar_config.yaml` + `expand_operator_sample_request`）、自建
expected volumes，不经过 `load_polar_profile.py`。改完本文件后未动它即通过
（`task_request=ok / topology=ok`）。本文档上一版把它列进必改清单，是照字面量搜索
就下结论、没核对调用链。

---

## 4. 改动清单

| # | 文件 | 改动 | 量 | 状态 |
|---|---|---|---|---|
| 1 | `src/polar/runtime/local.py` | **新增** LocalRuntime | ~180 行 | 待做 |
| 2 | `src/polar/runtime/models.py` | `backend` Literal 加 `"local"` | 1 行 | 待做 |
| 3 | `src/polar/runtime/factory.py` | `_BUILTIN_BACKENDS` 加一项 | 2 行 | 待做 |
| 4 | `profiles/profile.t2a.yaml` | `runtime.backend: local` | 1 行 | 待做 |
| 5 | `tools/load_polar_profile.py:297` | `"backend": "docker"` 改读 profile | ~2 行 | 待做 |
| 6 | `tools/load_polar_profile.py` | workdir 迁移 + 去两处硬编码 | ~12 行 | **已落 `8265c230`** |
| 7 | `launch_polar_and_guide.sh` | 共享盘根改自动探测 | ~25 行 | **已落 `d87f22aa`** |

`check_render_contract.py` 不在清单里 —— 见 §3.2 末尾。

#7 不属原设计，是实施中发现的：沙箱容器的共享盘挂在 `/mnt/host-model`，而 launcher
四处默认值写死 `/mnt/model/cbx/...`，不改则 venv、数据集、asc-devkit 全落空。

分支 `feat/local-runtime`（基线 `ac1a5673`）。`ascendc-szai` 不受影响。

### 4.1 LocalRuntime 要点

- `supports_gpus` / `supports_ascend` = **True**。`supports_ascend` 的语义是
  「Ascend 环境已由宿主容器提供」而非「本 backend 挂载 driver」——
  必须在 docstring 写清，否则 `factory.py:55` 硬拒 `kwargs.ascend`
- `supports_cpu_limits` / `memory` / `storage` = False。profile 未设这三项，
  `factory.py:49-54` 不会触发
- `can_disable_internet` = False。profile 是 `allow_internet: true`，门禁不触发
- `start()`：建目录 + 按 §5 处理 volumes；`stop()`：杀进程组 + 清理
- `exec()`：前缀重写后交 `bash -lc`，复用 `base.py` 的 `_run_local_command`
- 四个 upload/download：`/polar/session` 内走 `base.py` 现成快路径
  （`_copy_to_bind_mount` / `_copy_from_bind_mount`），外部就是本地 `shutil` ——
  没有容器边界要跨，是本方案里最省的一块

## 5. volumes：软链实现，契约不改

`_runtime_volumes()` 产出三项 `src:dst:ro`（`load_polar_profile.py:51-59`）：

| dst | src | 大小 | 性质 |
|---|---|---|---|
| `/opt/canonical` | `operator_runtime_t2a/` | 21M | 全局共享 |
| `<workdir>/tools` | `operator_runtime_t2a/tools/` | — | **session 内** |
| `/opt/asc-devkit` | `asc-devkit-9.0.0` | 317M | 全局共享 |

LocalRuntime 把 `kwargs.volumes` 解释成软链，profile 侧契约一字不改。dst 经 §3.1
重写后判断：落在 session 目录内的按 session 建，全局的建一次共享。

先澄清一个容易搞错的前提：`prepare_readonly_tools.py` 看名字像是已有的只读拷贝层，
**它不服务运行期**。`preflight.sh:48` 的 `READONLY_TOOLS_DIR` 默认是 `mktemp -d`，
那份拷贝只用于 preflight 自检；运行期 `_runtime_volumes` 挂的是**仓库树本体**
`[实测]`。所以三处暴露都是活仓库，没有既存的拷贝层兜着。

### 5.1 软链保不住 `:ro`，靠降权来保

Docker 的 `:ro` 是内核级的；软链没有这个语义。agent 以 uid 0 跑就能写穿软链改坏
共享树。后果不是崩，是**静默漂移**：asc-devkit 被改坏 → docs-search 返回错内容 →
agent 照着写出编不过的 kernel，且**一份坏了污染后续所有 session**。
这正是本文档反复警惕的那类失败，不能只靠事后检测。

**解法：agent 侧全程非 root，共享树 root 持有 0555。**

**2026-08-13 在沙箱容器实测成立** `[实测]`：`runuser -u polar` 下
`torch.npu` 建 tensor 成功（`nonroot tensor ok 16.0`）。**不需要任何 capability** ——
capability 恰恰是受限环境最难拿的（同一次实测里 CAP_SYS_ADMIN 就是没有的，见 §5.3）。

沙箱的设备权限比训练容器更宽：`crw-rw-rw-`（0666）对训练容器的 `crw-rw----`
（0660），属主同为 uid/gid 1000 而非 root。**0666 意味着连 gid 1000 都不必加**，
但脚本里仍加上 —— 权限位随镜像变，加了不会错。

一处要处理：实测出现 `can not create directory: /home/polar/ascend/log`。
`useradd -M` 不建 home，而 CANN 要往 home 写日志。除了 §5.1 表格里 `HOME` 指进
session 目录，还要显式给 `ASCEND_PROCESS_LOG_PATH`，否则每次 exec 都刷这行警告。

| 项 | 处理 |
|---|---|
| 降权点 | LocalRuntime.`exec()` 统一包 `runuser -u polar --` |
| 专用用户 | `polar`，附加组 **gid 1000** |
| 共享树 | `root:root` `0555` —— 内核拒写，非事后检测 |
| session 目录 | `chown` 给 `polar`，可写 |
| `lock_dir` | `/dev/shm/npu-locks` 必须 `polar` 可写，否则 `npu_lease_exec.py` 拿不到 flock |
| `HOME` / `TMPDIR` | 指进 session 目录（`CLAUDE_CONFIG_DIR` 已在 session 内）|

**降权点必须选 `exec()`：** 这一层同时覆盖 prepare、agent、judge 三个阶段，不用改
agent preset、profile、prepare 动作列表。包在 agent preset 里则 prepare 和 judge 漏掉。

**已实测通过** `[实测]`（`RUN_AS=polar` 的 e2e + `test_local_runtime_run_as.py`）：
进程以 `polar` 身份跑、`HOME` 指进 session、agent 写的 tarball 与 `judge_out` 属主都是
`polar`、往共享树写被内核拒（`Permission denied`）、canonical 与 asc-devkit 事后
`git status` 干净。

**踩到一个必修的顺序问题**：`workdir` 必须在 `chown` **之前**建好。原先它由后来的
`exec()` 以 root 身份 `mkdir` 出来，agent 于是拿到一个 `root:root 0755` 的工作目录，
第一次写就 `Permission denied` —— 而 `start()` 的 `chown -R` 早已跑完，救不回来。
`exec()` 里新建的 cwd 也要跟着 `chown`（只对刚建的、且在 session 内的）。

另一处易错：只读拷贝的**重定向失败由 shell 本身报到它自己的 stderr**，
命令里写 `2>&1` 拦不到（那只作用于 `echo`，而 `echo` 根本没跑起来）。
写断言时要查 stderr + 内容未变，否则会误判成「写保护没生效」。

`killpg` 不受影响：root 杀 `polar` 的进程组照样有效（§6）。CLI 的
`--dangerously-skip-permissions` 在 root 下靠 `IS_SANDBOX=1`（`claude_code.py:82`）
放行 `[读码]`，降权不影响它。

### 5.2 残留取舍

- **agent 装不了包**（`pip install` / `npm i -g` 失败）。沙箱镜像本就自带整套 AscendC
  依赖，这是选它当 polar 容器的理由。实跑若发现依赖临时装包，补镜像而非放权限
- **0555 下 polar 运行期也改不了共享树**。asc-devkit 自举（浅克隆 + 软链修复）要在
  降权之前跑完，属 run 间隙动作，不冲突
- `git status --porcelain` 校验保留为兜底（两者都是 git 仓 `[实测]`：asc-devkit 在
  `9.0.0` 分支，operator_runtime_t2a 是 polar 仓内 624 个跟踪文件）
- 容器内 root 对**容器本身**从来不是安全边界；这里只需防共享盘那两棵树被写坏

### 5.3 已否决的替代方案

- **per-session 拷贝共享树**：canonical 21M 可行，asc-devkit 317M 不行（按
  `polar-session-scale` 的实测规模 37 session / 2.5 小时 = 十几 GB 无谓 IO）。
  硬链更糟，写会穿透原文件
- **`chattr +i`**：能挡 root，但要 CAP_LINUX_IMMUTABLE —— 正是受限环境不保证给的
- **per-session mount namespace**（`unshare --mount` + `--bind -o ro`）：能拿回真
  `:ro` 和 `/polar/session` 隔离、省掉 §3 的前缀重写。**已实测排除** `[实测]`：
  沙箱容器 `CapEff=00000000a80425fb`（Docker 默认无特权集，无 CAP_SYS_ADMIN），
  `unshare --mount --fork true` 报 `Operation not permitted`。与 DinD 同类阻塞
- **只做事后 `git status` 校验**（本文档上一版写法）：检测到时坏内容已进过若干
  session 的 prompt，那些 session 的分数无法追溯甄别

---

## 6. 唯一实质退化：进程清理

Docker 下「拆容器即全杀」是免费的。进程级必须自己做，而 `base.py:78-87` 的
`cancel()` **只 kill `_active_process`** —— agent 会派生 claude CLI、编译器、
eval 子进程，全都漏网。

做法：`_run_local_command` 起进程时带 `start_new_session=True`（新进程组），
`stop()` / `cancel()` 用 `os.killpg(pgid, SIGKILL)`。

这是整个方案里唯一需要真写逻辑而非委托的部分，也是最可能漏的地方 —— 漏进程会
攒着占卡（NPU 租约是 flock，进程不死就不释放），几个 session 之后卡池空转。
§8 的测试专门盯这条。

---

## 7. 实施顺序：先做什么以减少返工

按「一旦错了返工代价」排序，不是按代码依赖排。

**Step 0（写代码前，零成本，必做）**
在**沙箱容器**内跑 §10 的三段脚本。工具链检查已可略去（沙箱镜像自带 AscendC 依赖，
这正是选它当 polar 容器的理由），但下列三条的结论会改变方案本身：

**2026-08-13 全部跑完，三条结论如下** `[实测]`：

| 判据 | 结果 | 影响 |
|---|---|---|
| 非 root 能用 NPU | **是**（`nonroot tensor ok 16.0`）| §5.1 降权方案成立 |
| CAP_SYS_ADMIN | **无**（`CapEff=...a80425fb`）| §5.3 mount ns 排除，本方案维持主线 |
| claude 可同容器并发 | **是**（solo 与并发 8 全 `rc=0`）| §9 通过，吞吐无忧 |
| `SOC_VERSION` | `Ascend910_9382` | 与既有记录一致 |

Step 0 已完成，可直接进 Step 1。

**Step 1：改 workdir 与去硬编码（#6/#7）—— 代码已落，等 docker 环境验一次。**
仍用 `backend: docker`，这样把「workdir 迁移」和「新 runtime」两个变量分开，
任一出问题都能立刻归因。已落 `8265c230` + `d87f22aa`。

在能跑 docker 的环境上验三条：

1. prepare 不报 `exit 1`（失败在 agent 起来之前，日志只有一行，最难归因）
2. `<session>/agent_workdir/input/{op}.py` 在位
3. **judge 出分非 0.2** —— 0.2 是 `submission_missing` 地板分

第 3 条同时会顺带验证 §8.1 第五个用例要覆盖的隐患：judge 跨实例取 submission 到底
靠不靠 gateway 显式传的 `submission_host_path`（`node.py:878`）。docker 上正常出分 =
那条转移路径存在且有效，LocalRuntime 只要不破坏它即可；docker 上就出 0.2 = 问题在
workdir 迁移本身，与 runtime 无关。两种结果都有用。

**Step 2：写 LocalRuntime（#1/#2/#3），配 §8.1 的单测。**
不接 gateway，纯 runtime 契约。

**Step 3/4：单 session 实跑到 judge 出分 —— 已在训练容器跑通** `[实测]`。

我先前判断这两步需要沙箱环境，那是错的：依赖齐（fastapi/uvicorn/httpx/pydantic/pyyaml）、
不需要 docker（这就是本方案的意义）、不需要真 LLM（shell harness）、不需要真 NPU
（假 tarball 到不了编译阶段）。脚本 `verify_local_runtime_e2e.sh`。

跑通的链路：起真 rollout + gateway（`backend: local`）→ prepare → agent（shell
harness）→ judge 出分。取到的证据：

| 判据 | 结果 |
|---|---|
| agent 与 eval 各有独立 workdir | `<session>/agent_workdir` 与 `<session>/eval_runtime/agent_workdir` 两个不同宿主目录 —— §3 前缀重写的直接证据 |
| tarball 落在 session 内 | 两处各一份；宿主根上没有 `/polar/session` |
| judge 取到并解开了 agent 写的 tarball | 见 §8.3 |
| 无残留进程、共享树未被写坏 | `git status` 干净 |

`probe_gateway_runtime.py`（原 Step 3）没用上：全量 e2e 本身只要 ~80s，probe 省下的
时间不值得多维护一条路径。它仍可用（已有 `--backend` 参数）。

### 7.1 这一步撞出的两个坑，都不是静态检查能看出的

**session 内的 volume 不能用软链**（已修，`633ec98a`）。`judge_out/` 落进了共享树
`operator_runtime_t2a/` 而非 session。根因在 `tools/ascendc_eval_pipeline.sh:11` ——
它**故意**对 `$BASH_SOURCE` 做 `readlink -f`（注释写明：被软链调用时 `dirname` 会拿到
软链的目录），再取 `WORK_ROOT="$_SCRIPT_DIR/.."`。docker 下 `<workdir>/tools` 是 bind
挂载、是真目录，`WORK_ROOT` 正是 `<workdir>`；做成软链后 `readlink -f` 穿过去，
`WORK_ROOT` 变成共享树。**那句为「安全」而加的 `readlink -f`，恰好是打破软链方案的
东西。** 改法见 §5。

**测试操作本身会制造假象。** e2e 脚本被 timeout 掐断后仍在后台轮询，它退出时 trap 会
`cleanup` 掉**新一轮**的进程；旧 gateway 占着端口时新 gateway `bind` 失败直接退出，而
rollout 会连上旧的 —— session 照样跑完、状态照样 `completed`，判据却全部落空，看起来
像 LocalRuntime 的 bug。排查了三轮才发现 gateway 日志只有 6 行、末尾是
`address already in use`。脚本开头因此加了清理 + 端口释放等待。

## 8. 测试：够用即止

不追求商用级覆盖。原则是**只测那些「错了会静默」的路径** —— 会明确报错的靠实跑
自然暴露，不值得写测试。

### 8.1 单测（`tests/runtime/test_local_runtime_contract.py`）

照 `tests/runtime/test_docker_runtime_contract.py` 的形态改（monkeypatch
`_run_local_command`，`asyncio.run` 包一层）。LocalRuntime 比 DockerRuntime 好测：
它本来就是跑本地子进程，多数用例可以直接**真跑** `bash`，不用 mock。

五个用例，都对着「静默失败」写：

| 用例 | 防的问题 |
|---|---|
| agent 与 eval 两个实例的 `/polar/session` 重写到不同宿主目录 | §3 的 judge 共目录 |
| `exec` 的 cwd / env value / 命令字符串三处前缀都被重写 | 漏一处就走错目录 |
| `stop()` 后子进程的**孙进程**也没了 | §6 漏进程占卡 |
| `factory.create_runtime` 对 `kwargs.ascend` 不再抛 | `supports_ascend` 忘了置 True |
| **agent 实例写的文件，judge 实例能取到** | 见下 |

第三个用例是重点：起 `bash -c 'sleep 300 & sleep 300'` 再 `stop()`，断言两个
`sleep` 都消失。这条最容易写漏，也是最贵的 bug。

第五个用例是前缀重写最刁的一处，实施中才想到：agent 的 workdir 重写到
`<session>/agent_workdir`，judge 的重写到 `<session>/eval_runtime/agent_workdir`
—— 两个不同的宿主目录。而 `operator_judge._abs()` 拼出的绝对路径会按 **judge 自己的
实例**重写，落到 `eval_runtime/` 下面，可 submission 其实是 agent 写在另一处的。
Docker 下不成问题（judge 在 fresh 容器里，gateway 显式传 `submission_host_path`
做转移，`node.py:878`），但那条转移路径在 LocalRuntime 下必须同样有效。
判据 = judge 出分非 0.2。Step 1 在 docker 上跑过之后，这条的性质就清楚了。

### 8.2 实跑判据（Step 4，一个 session）

按「先暴露的先看」排序：

1. prepare 阶段无 `exit 1`。这里失败在 agent 起来之前，日志只有一行
   （`load_polar_profile.py:112-118` 记录过这个坑）
2. `<session>/agent_workdir/` 和 `<session>/eval_runtime/agent_workdir/`
   **都存在且内容不同** ← §3 的直接验证
3. agent 日志落在 `<session>/logs/agent/claude-code.txt`
4. judge 出分**非 0.2** —— 0.2 是 `submission_missing` 的地板分
   （`operator_judge.py:135-143`），出 0.2 说明路径没通而不是 kernel 差
5. session 结束后 `pgrep -f <session_id>` 为空、`/dev/shm/npu-locks` 无残留持有
6. `git status --porcelain` 在 asc-devkit 与 operator_runtime_t2a 上都干净（§5）

第 4 条要当心：judge 出 0.2 时**看起来是模型不行**，实际是路径断裂。这是本方案
最像「跑通了」的失败模式。

### 8.3 不测什么

- 网络隔离 / 资源配额 —— profile 本来没用
- 多 session 并发 —— 先跑通单 session；并发只会放大 §6，判据一样
- apptainer 路径 —— 不在本方案内

---

## 9. 同容器内并发多个 claude CLI：已实测通过

Docker 下 N 个 session = N 个容器，每个容器里一个 `claude`。LocalRuntime 下变成
**同一容器内 N 个 `claude` 子进程**。这曾是本方案最大的未知
（`polar-session-scale` 那次 37 session / 2.5 小时是每 session 独立容器）。

**2026-08-13 在沙箱容器实测通过** `[实测]`：solo 与并发 8 全部 `rc=0`、8 个
输出一致、同一秒完成、无 claude 残留。脚本见
`/mnt/model/cbx/local_runtime_probe/probe_concurrent_claude.sh`（对着死端口测不出
东西 —— CLI 会卡满 timeout；必须用 stub 返回合法响应让它真跑完一轮，并且要有
solo 对照组）。

顺带一条教训：那次实测第一版**看起来是并发挂住**，实际是探测脚本自己的 bug ——
裸 `wait` 连永不退出的 stub server 一起等了。8 个 claude 早已完成。
判断「挂住」时先看 `rc.txt` 和日志，别只看进程还在。

### 9.1 已隔离的部分 `[读码]`

- `CLAUDE_CONFIG_DIR` = `/polar/session/.claude`（`claude_code.py:24`），经 §3.1
  重写后每 session 各一份。配置、`.claude.json`、skills 拷贝都不共享
- 调用的是 `claude -p`（非交互，`claude_code.py:117`），没有 TUI、没有终端状态、
  没有 stdin 争用。日志各自 `tee` 到 session 内
- API key 是 session id，gateway 靠它区分请求来源 —— 本来就是 per-session 设计
- `HOME` 按 §5.1 指进 session 目录，落在 `~` 下的东西随之隔离

### 9.2 仍未覆盖的部分

实测的是 8 个并发、每个只跑一轮、工具调用为零。真实负载是 8+ 个 session 各跑几小时、
每个几十次 Bash 工具调用（`polar-session-scale`：单 session 中位 168 条 assistant
消息 / 85 次工具调用）。以下仍无证据：

- 长时程下是否有资源泄漏（fd、`/tmp` 堆积）
- 大量并发 Bash 工具调用之间是否互相干扰
- 内存：8 个 node 进程 × 数小时。沙箱容器上限 2TB、空载 1.5G `[实测]`，
  余量充足，但没测过满载

这些等 §7 Step 4 的单 session 实跑之后自然覆盖，不单独测。

---

## 10. 沙箱容器内的验证脚本

沙箱镜像（`ascendc-tilelang:v1`）自带整套 AscendC 依赖，将直接用作 polar 容器。
**以下检测必须在那个容器里跑**，训练容器的结果不能替代 —— 权限位、用户组、CLI 行为
都可能不同。

三段独立，可分开跑。全部只读或在 `/tmp` 内操作，不动共享盘。

### 10.1 权限与降权可行性

```bash
set +e
echo "=== 身份与 capability ==="
id; grep CapEff /proc/self/status
echo "=== NPU 设备属主（关键：是否 root）==="
ls -l /dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>&1 | head
echo "=== 驱动目录 ==="
ls -ld /usr/local/Ascend/driver /usr/local/Ascend/ascend-toolkit 2>&1
echo "=== 降权工具 ==="
for b in runuser setpriv su unshare; do printf '%-9s %s\n' "$b" "$(command -v $b || echo -)"; done
echo "=== 设备 gid 是否有名 ==="
DEVGID=$(stat -c %g /dev/davinci0 2>/dev/null); echo "davinci0 gid=${DEVGID}"
getent group "${DEVGID}" || echo "gid ${DEVGID} 无名（直接用数字即可）"
echo "=== CAP_SYS_ADMIN 实测（决定 mount ns 方案是否反超）==="
unshare --mount --fork true 2>&1 && echo "mount ns: OK" || echo "mount ns: 不可用"
echo "=== SOC_VERSION（填错会静默编到另一块芯片）==="
python3 -c "import acl; print(acl.get_soc_name())" 2>&1 | tail -1
echo "=== 非 root 能否真用 NPU（核心判据，只需 1 张卡）==="
CARD="${CARD:-0}"
useradd -M -s /bin/bash polar 2>/dev/null; usermod -aG "${DEVGID}" polar 2>/dev/null
install -d -o polar -g "${DEVGID}" -m 0775 /dev/shm/npu-locks /tmp/polar-home
runuser -u polar -- env HOME=/tmp/polar-home ASCEND_RT_VISIBLE_DEVICES="${CARD}" \
  python3 -c "
import torch, torch_npu
print('device_count', torch.npu.device_count())
torch.npu.set_device(0)
x = torch.ones(8, device='npu:0')
print('nonroot tensor ok', (x + 1).sum().item())
" 2>&1 | tail -4
```

**两个实测踩出来的坑，别用旧写法** `[实测]`：

- `acl.get_soc_name()` **不能**当「能否用卡」的判据 —— 它不需要 `acl.init()`、
  不打开设备，在卡完全不可用的容器里照样返回 `Ascend910_9382`。它只够测 SOC_VERSION
- **`ASCEND_RT_VISIBLE_DEVICES` 若是空字符串，`device_count` 直接为 0**。训练容器里
  就是空的（`docker.py:110` 的 mount-only 路径会显式清空它）。脚本必须显式赋值，
  否则会因为这个而非权限问题失败，结论完全误导

判据：最后一行出现 `nonroot tensor ok 16` → 降权方案成立（§5.1）。
`mount ns: OK` → 按 §5.3 该方案反超成为首选。

**卡数要求：这段只用 1 张**（`CARD=<卡号>` 可指定），§10.2 用 0 张。少于 16 张
完全够验证设计；卡数只影响生产期 `npu_lease.pool` 的并发验证数，不影响任何结论。

### 10.2 并发 claude CLI（回答 §9）

用**故意写错的 base URL** 让请求在网络层失败 —— 此前 CLI 已完成配置初始化、
snapshot 等全部本地动作，正好能暴露并发争用，且不需要真的起 gateway、不烧 token。

```bash
set +e
N=8; BASE=/tmp/cc-concur; rm -rf $BASE; mkdir -p $BASE
for i in $(seq 1 $N); do
  ( export CLAUDE_CONFIG_DIR=$BASE/s$i/.claude HOME=$BASE/s$i \
           ANTHROPIC_BASE_URL=http://127.0.0.1:59999 ANTHROPIC_API_KEY=sk-test-$i \
           IS_SANDBOX=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1
    mkdir -p $CLAUDE_CONFIG_DIR
    timeout 120 claude --dangerously-skip-permissions -p "say hi" \
      > $BASE/out.$i.log 2>&1
    echo "$i rc=$?" >> $BASE/rc.txt ) &
done
wait
echo "=== 退出码分布（要求：N 个一致）==="
sort $BASE/rc.txt | awk '{print $2}' | sort | uniq -c
echo "=== 错误类型（要求：全是连接失败，无锁/权限/EEXIST）==="
cat $BASE/out.*.log | grep -oiE "connect|refused|econnrefused|lock|permission|denied|EEXIST|EADDRINUSE|already running" \
  | sort | uniq -c
echo "=== 是否写了 CLAUDE_CONFIG_DIR/HOME 之外的路径 ==="
find / -xdev -newer $BASE -path /proc -prune -o -newermt '-3 minutes' -type f -print 2>/dev/null \
  | grep -vE "^($BASE|/tmp/cc-concur|/proc|/sys|/dev|/run|/var/log)" | head -20
```

判据：退出码 **N 个一致**、错误只有连接类、最后一段不出现 `/root`、`/tmp` 下的
固定路径或任何全局 CLI 状态文件。出现 `lock` / `already running` / `EADDRINUSE`
即说明 CLI 有单实例约束，§9.2 的坏情况成立。

最后一段的 `find` 可能有噪声（日志、时区文件），看的是**有没有 CLI 自己的固定路径**，
不是要求一条不出。

### 10.3 npu-smi 并发影响（对应 §11 遗留项）

`CLAUDE.md:443-453,1293` 禁 agent 调 `npu-smi`，因为它无视
`ASCEND_RT_VISIBLE_DEVICES` 枚举全部卡、抢全局 DCMI 锁。独占卡缓解了与 vime 的
冲突，但同容器 session 之间是否互撞未定论。

```bash
set +e
for i in 0 1 2 3; do ( timeout 30 npu-smi info >/tmp/smi.$i.log 2>&1; echo "$i rc=$?" ) & done
wait
grep -oiE "\-8005|resource.?busy|dcmi|fail" /tmp/smi.*.log | sort | uniq -c
```

出现 `-8005` 或 `Resource_Busy` 就证实并发探测会互撞，届时应把
`skills/ascendc-env-check` 从启用列表移除（它整个 skill 就是调 `npu-smi`）。

---

## 11. 遗留项

- **`/tmp/npu`** 在 eval pipeline 出现一处（`ascendc_eval_pipeline.sh:33`）
  `[实测 grep]`。workdir 已 session 化，但这个是绝对路径、全 session 共享。
  处理：给每 session 一个 `TMPDIR`，随 §4 的 diff 一起落
- **`backend` 这个 key 在两个层级是两回事**，改 #5 时极易搞混：
  `operator_runtime.backend` = `triton`/`ascendc`（决定 prepare 上传布局，
  `load_polar_profile.py:194` 已有同名局部变量占位）；
  `operator.runtime.backend` = `docker`/`local`。**新变量务必换名**，
  否则会静默改掉 ascendc 的 prepare 逻辑
- **`npu-smi` 禁令仍然有效。** `CLAUDE.md:443-453,1293` 已明令 agent 不得调用
  （它无视 `ASCEND_RT_VISIBLE_DEVICES` 枚举全部卡、抢全局 DCMI 锁 → -8005）。
  但 `skills/ascendc-env-check/` 整个 skill 就是干这事的，且在启用列表里。
  独占卡缓解了与 vime 的冲突，**但同容器内 session 之间仍会互撞** `[未定论]` ——
  实跑时若见 `-8005` 或 `aclInit 507899` 先查这里
- **`operator_judge.py:135` 的 docstring 提到 workdir**，改完顺手更新，
  否则下一个 session 会照着过时描述下结论（handoff §9 的原话）




