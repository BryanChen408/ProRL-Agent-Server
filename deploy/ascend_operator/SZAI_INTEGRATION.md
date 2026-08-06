# szai 拓扑集成契约(分支 `ascendc-szai`)

分支:`ascendc-szai`,从 `feat/ascendc-rl-t2a` @ `2edd5caa` 拉出。
对端:vime 仓 `feature/swe-tasks`(5 个提交未推)。

这份文档说明这套拓扑下 vime 与 polar 之间的接口、为什么需要运行期渲染、
站点值放哪里,以及为什么没有合并同事在 `luoss-lzc` 上的那套方案。

结论一律标注证据等级:

- `[实测]` 跑过/量过 —— 可直接用
- `[对比]` 逐字 diff / 读代码 / 读配置 —— 可直接用
- `[推断]` 从若干事实推的因果 —— **未验证,别据此决策**

> ⚠️ 全链路**从未跑通过一次真实训练**。`scratch/logs/`、`checkpoints/` 自 2026-08-01
> 起为空 `[实测]`。以下所有机制只做过静态验证(YAML 解析、渲染模拟、分支行为),
> 没有任何一条被真实 run 证明过。

---

## 一、拓扑

```
┌─ polar 宿主(固定机器,IP 已知,能跑 Docker 沙箱)
│    polar rollout   :ROLLOUT_PORT
│    polar gateway   :GATEWAY_PORT
│    Docker 沙箱     编译/评测 AscendC kernel,占本机 npu_lease.pool 的卡
│
└─ 训练节点(调度平台临时分配,IP 到起任务才知道)
     vime actor(Megatron)+ rollout(vLLM)
     vLLM router      :VLLM_ROUTER_PORT
```

两侧的卡是**两台机器上的两组卡**,互不相干:`npu_lease.pool` 是 polar 宿主自己的卡池,
和 vime 的 resource layout 切分没有冲突关系 `[对比]`。

## 二、集成契约

依赖是**非对称的,不是循环的**。polar 的 IP 固定且已知;只有训练侧是动态的;
训练任务在启动时同时知道两侧地址。所有动态绑定都是单向的 训练 → polar。

| 方向 | 传什么 | 怎么传 |
|---|---|---|
| 训练 → polar | vLLM router 的 IP | 共享盘交会文件 `vime_router_ip.txt` |
| polar → 训练 | 无(polar IP 是固定值,写在训练侧配置里) | —— |

交会文件的**新鲜度判据是"存在性",不是时间戳**:master 起跑时先 `rm -f`,
就绪后才写;文件不存在就等于"本次 run 的 master 还没就绪"。

不用时间戳的原因是实测的:共享盘 mtime 比本机时钟**快约 30 秒** `[实测]`。
跨机比绝对时间戳的偏斜暴露和比 mtime 完全一样,而且偏斜方向是**把陈旧读成新鲜**
—— 正好是危险的那一侧。两侧写方都是我们自己的脚本,判存在性不依赖时钟,比容差带更硬。

### 要防的失败模式

**"看起来成功"**:交会文件残留上一次 run 的 router IP → polar 正常启动,
`/health` 返回 200,vime 那 600 秒等待通过,训练开始 —— 但生成请求打到一台
已被回收的机器,rollout 静默饿死,哪一侧都不报错 `[推断]`。
`rm -f` + 判存在性就是为了堵掉这条路径。

---

## 三、为什么渲染步骤是必需的

polar 的配置加载是裸 `yaml.safe_load`,`src/polar/config/` 里**没有** `os.environ`
也没有 `expandvars` `[对比]`。上游 32 个提交之后重新核过,仍然如此 `[对比]`。

所以 profile 里的动态值**不可能**靠环境变量注入,sed 渲染是硬要求,不是风格选择。

```
profile.t2a.yaml (跟踪,含 __TOKEN__)
        │  launch_polar_and_guide.sh 渲染
        ▼
/tmp/polar_profile_runtime.yaml (运行期产物,原文件不动)
```

渲染后必须校验残留 token:未替换的 `__X__` 会作为字面主机名送进 polar,
那时报的错离根因很远,所以在渲染处就 `exit 1`。

## 四、站点值一律 token 化,不提交字面值

**规则:跟踪的模板里不出现任何站点值。** 站点值以 token 出现在 profile,
由 launcher 给默认值并允许环境变量覆盖。

理由:把站点值提交成字面值,每次 rebase 上游都要手工解一遍同样的冲突。
反正渲染步骤本来就必需,扩展 token 覆盖面是零额外代价。

| profile 里的 token | 性质 | 默认值来源 |
|---|---|---|
| `__POLAR_HOST__` | 本机 IP | launcher 探测 |
| `__VIME_ROUTER_HOST__` | 训练侧动态 IP | 交会文件 |
| `__ROLLOUT_PORT__` / `__GATEWAY_PORT__` | 本地约定 | launcher,与 vime 侧同源 |
| `__SOC_VERSION__` | 宿主是 A3 还是 A2 | launcher(A3=`ascend910_9391`,A2=`ascend910b1`) |
| `__NPU_POOL__` | polar 宿主自己的卡 | launcher |
| `__MODEL_SERVED__` / `__TASK_ASSETS_DIR__` / `__ASC_DEVKIT_DIR__` | 本地挂载路径 | launcher |

端口只在 profile 声明一份,launcher 显示的端口从渲染产物里读回来,不重复写死。

### 已知的两处不一致(待随 token 化一并修掉)

1. vime 的 `POLAR_ROLLOUT_URL` 等 `:8080/health`,而 profile 声明 `8180` `[对比]`
   —— 不对齐会一直挂在那 600 秒。两侧端口要同源。
2. `profile.t2a.yaml` 头部注释写"npu_lease(0-3)全部照 vime 不变",实际值是
   `[8, 9, 10, 11]` `[对比]` —— 注释和值自相矛盾,token 化时这条注释要跟着改。

---

## 五、同事的 `luoss-lzc` 方案:不合并,取两条细节

同事(lzc12138)在 `/mnt/model/corlorlight_models/lanzichang/Huawei/` 下有一套独立方案:
polar 仓 `luoss-lzc` @ `f2c1d3e7`(1728 行新增,35 文件,**只在本地未推远端**),
外加一整套未进版本控制的 vime `agentic-rl-*` 脚本族和 `utils/node_registry.py`(320 行)`[对比]`。

**不合并的三个理由:**

1. **驱动不同**。他那套的根本约束是"计算节点上没有 Docker 权限",所以才有
   Apptainer/SIF 转换、`runtime/local.py` 本地运行时、以及 `multi_machine/todo_8_3.md`
   里整页的 Pod securityContext 门禁(`privileged` / `CAP_SYS_ADMIN` / `/dev/fuse` /
   loop 设备 / `starter-suid`)`[对比]`。本拓扑的 polar 宿主能跑 Docker,这条约束不存在。
2. **动态性不同**。他的计算节点是固定主机名固定 IP(`atlas-4` / `atlas-27`),
   `node_registry.py` 解决的是"这两台已知机器里谁是 actor、网卡选哪个";
   本拓扑的计算节点由调度平台临时分配。那 320 行里的网卡枚举、管理网段筛选、
   主机名校验、connectivity 表,在这边没有对应问题 `[对比]`。
3. **他重新发明了上游已有的东西**。t2a 早就有 `tools/load_polar_profile.py` +
   profile/topology 分离 + 159 行 `restart_polar_host.sh`;他是在 `f0e8343a`
   那个不带 ascend_operator 的基线上从零搭的平行实现 `[对比]`。合过来只会打架。

**要取的两条:**

1. **`no_proxy` / `NO_PROXY` 注入对端 IP**。环境里存在 `http_proxy` 时,
   等 `/health` 的 curl 会走代理,失败得毫无线索 `[推断]`。这是本方案的真实缺口。
2. **交会文件原子写**。当前是 `echo "$IP" > file`,共享盘上对端可能读到空文件或半截内容
   `[推断]`。换 tempfile + `mv`。

**不取:** 他的时间戳容差带(`node_registry.py:197` 的 `age < -300`)。
那是给"写方不受自己控制"准备的;本方案两侧写方都是自己的脚本,判存在性更硬。

**当参考不当代码:** `guide/guide.md`(10 章)和 `multi_machine/todo_8_3.md`
记录了这批机器上 NPU 门禁、`/dev/shm`(需 1024Gi 内存型 `emptyDir`)、bind 挂载的
真实踩坑,碰到权限问题时值得翻。

## 六、落地顺序

在 `ascendc-szai` @ `3f294644` 上叠三个提交:

1. profile 四项站点值 token 化 + launcher 补默认值(含 A3 `ascend910_9391`、卡池、路径、端口)
2. launcher 注入 `no_proxy` / `NO_PROXY`
3. vime `start_vime_in_platform.sh`:交会文件改原子写,`POLAR_ROLLOUT_URL` 端口与 polar 侧对齐

做完后 `/tmp/polar-t2a-worktree.patch`(基线 `4f0e1a4e` 已死)即可丢弃 —— 它的内容
以 launcher 默认值的形式重新落地。

## 七、当前状态

| 项 | 状态 |
|---|---|
| vime 5 个提交 | 已落 `feature/swe-tasks`,**未 push** |
| polar `3f294644` | 已落 `ascendc-szai`,**未 push** |
| 上面三个提交 | 未做 |
| 真实 run | 从未跑通 |

推送注意:`ascendc-szai` 跟踪的是 `origin/feat/ascendc-rl-t2a`,裸 `git push` 会推错分支,
要 `git push -u origin ascendc-szai`。

遗留待定:`task_assets_dir` 指向的 `op_assets_cudallm_filtered189` 在共享盘上不存在 `[实测]`;
现有的只有 `/mnt/model/corlorlight_models/ljk/sft_workspace/benchmark\cudaLLM\level1\*`
(字面反斜杠 = 压平的 Windows 路径,66 个文件,且在 `ljk` 名下不在 `mingchengzou` 名下)。
