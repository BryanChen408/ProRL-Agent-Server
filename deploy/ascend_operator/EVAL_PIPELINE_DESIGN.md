# 固定入口的设计依据

`operator_runtime_t2a/tools/` 下的脚本会被只读挂进 agent workdir,agent 实测会 `cat` 它们。
所以那些"这条防的是什么攻击"的论证一律不留在脚本里 —— 在 RL 里那不是文档,是提示词。
本文件在 `deploy/` 下,不进 canonical 树,agent 读不到。

改脚本前先读这里。约束靠**写死的字面量 + preflight 断言**保证,不靠注释劝阻。

## 一、为什么要有"固定入口"这一层

上游让 agent 分别调 `evaluate_ascendc.sh` / `validate_*.py` / `msprof_*`。RL 需要在同一条
路径上塞进四件上游没有的事,只有收成一条命令才做得到:

1. **抢卡**(`npu_lease_exec.py`)—— 多容器共享卡池,agent 与 judge 都只在
   `POLAR_NPU_LEASE_POOL` 里 flock 抢一张,用完释放。锁文件命名与 polar
   `DockerRuntime.acquire_card()` 一致,锁目录 bind mount 进容器,三方共用同一套锁。
2. **预算**(generation / optimization 两阶段计数 + LIMIT_EXHAUSTED 收尾)。
3. **判分基准复位** —— 见第三节。
4. **统一错误分类** —— A / D / INFRA 三档,直接从 `error_type` 映射。

上游的做法是 `skill_script_hook` 用正则扒 stdout 猜标签,并**代为执行**被拦的脚本 ——
那会绕开上面全部四件事,所以我们不接 hook。CLAUDE.md 里原有的「Hook 机制说明」章已删。

## 二、agent 侧 / judge 侧同一条命令

判据:workdir 顶层有没有 `{op}/`。agent 容器有(它在那儿写代码),judge 容器没有
(只收到 tarball + `input/` 基准)。agent 侧额外做:自动打包 → 预算计数 → 内容哈希短路 →
退出时更新 `.best`;judge 侧一律跳过(一个 session 只判一次)。

## 三、判分基准无条件以数据集为准

golden(`model.py`)和用例(`{op}.json`)从 `input/` 覆盖 tarball 里的同名文件,防三件事:

- agent 改 golden 让参考实现迁就自己的错 kernel;
- agent 只交精简后的用例(被 abort 时交上来的就是砍过的),判分仍跑全量;
- agent 根本不需要把这两个文件打进 tarball。

31/31 个数据集算子的 `get_input_groups()` 都硬编码同名 json(无 `__file__` 推导),
所以改名成 `model.py` 后仍能读到同目录的 `{op}.json`。

**注入做两次**:第一次解包后,第二次在编译/安装之后、对拍之前。因为 Step2 会执行 agent
提供的代码(`setup.py`、`pip install` 触发的 import、编译期脚本),它完全可以在那时把基准
改回自己那份。第二次注入后立即 `chmod 444`。

## 四、解包后的两道清理

**隔离**:`$WORK` 完全由 agent 控制,而 `SK="$WORK/.claude/skills"` 正是 judge 跑
verification/msprof 的地方。`cp -r SRC DEST` 在 DEST 已存在时是**嵌套**而非覆盖,
所以 tarball 里带一份 `.claude/skills/.../verification_ascendc.py`(内容
`print("Result: pass")`)就能整个接管判分链。非恶意也可达:`tar czf x.tar.gz .` 会顺手
把项目级 `.claude` 打进去。故解包后先把 `{op}/` 之外的一切铲掉,`SK` 先铲后铺。

**purge**:删掉 tarball 里夹带的 `.so/.a/.o/.whl/build/dist`。实测 18_Index 的提交包里
就有。二进制与源码没有任何保证的对应关系 —— 极端情况源码是漂亮的 AscendC、`.so` 里是
torch 算的,而 AST 检测只看 `model_new_ascendc.py`,查不出来。RL 不需要恶意动机:
某次夹带碰巧拿高分,梯度就会强化它。

## 五、Step1 反退化为什么必须是 AST 而不是 grep

原来是 `grep -q "torch.ops.npu"` —— 文件任意位置(注释、docstring、未调用的 helper、
`if False` 分支)出现该字符串就算过。于是"forward 里写纯 torch + 注释里提一句"就能
编过 → 对拍必过(与 golden 逐位相同)→ speedup≈1.0 → reward 0.75。这是全流程投入产出比
最高的策略,GRPO 几十步内就会收敛到它,整批训练信号作废。

`validate_ascendc_impl.py` 做的是 AST 检测(Type1 无扩展导入 / Type2 forward 未调 kernel /
Type3 forward 里仍有 torch 计算 / Type4 逐元素 for 循环)。skill 本来就带,judge 已经把它
cp 进 `$SK` 却没用 —— judge 比 agent 自检还松是不能接受的。

**退化必须排在编译之前**:AST 检查器的输出常带"编译/compile"字眼(修复建议),排在后面
会被 compile 规则抢先命中而误分类。

## 六、Step2 编译必须 bdist_wheel + install

逐字对应上游 `evaluate_ascendc.sh:110-126`:`rm -rf build` → `cmake` → `make -j` →
`setup.py bdist_wheel` → `pip install dist/*.whl --force-reinstall`。**改这段要同步那份。**

打包+安装不是可选优化:AscendC 算子靠 `.so` 被 import 时执行 `TORCH_LIBRARY_IMPL` 才注册出
`torch.ops.npu.<op>`,只 make 不装 = 对拍时算子不存在(曾因此把写对的 18_Index 误判成
`correctness_failed`)。

与上游两处有意 delta:① 连 `dist/` 一起清,保证装的是本次编出来的 whl 而不是 tarball 里的
旧 whl;② 用 `$PY_BIN -m pip`(镜像里 python/pip 不一定在 PATH 上)。

编译**不套 `run_npu_phase`** —— AscendC 编译要数分钟,占着卡编译会堵死整个卡池。

## 七、测速契约:钉死 `--repeats 1`

由 `preflight.sh` 机械保证。

`msprof_perf_summary.py` quick 模式测的是 **device 侧核函数时间** —— 从 `task_time_*.csv` 里把
`kernel_type ∈ {AI_VECTOR_CORE, AI_CORE, MIX_AIV, MIX}` 的行累加。它的调用形态:

```python
wrapper = _generate_wrapper_script(..., repeats - 1, ...)   # 同进程内先跑 repeats-1 次预热
duration = duration / repeats                                # 采到的总 kernel 时间 ÷ repeats
```

**风险来自那个除法,不是来自 warmup。**按输入做 memo 的实现(`{input_key: result}`)输出随输入
变化,`detect_stateful_impl.py` 会正常放行;而测速在同一输入上连调 N 次,只有第一次发 kernel,
总时间却被除以 N ⇒ speedup 虚高 N 倍。`repeats=1` 时内部预热=0、除数=1,该放大不存在。
这是缓存探测覆盖不到、只有本约束挡得住的一类(闸门 D3 实测:cache/ok = 1.088)。

**纯缓存实现(输出不随输入变)不会虚高,而是测不出来** —— 它根本不发 kernel,
`compute_rows` 为空,解析返回 `"no compute rows found"`,判 benchmark 失败。

**`--compare` 不禁**:它是 standard 模式(8 轮采集 7 个 aic-metrics + sample),不做 `÷repeats`,
与缓存无关。我们不用它只是因为那些指标用不上、且采集慢 8 倍 —— 没有正确性理由去禁。
(上游 quickstart 的 Phase 5 用的就是 `--compare`;这是场景差异,不是分叉。)

其它对齐点:`--warmup 3` 对齐 `msprof_profile_run.sh:84`(不是 env.sh 的 WARMUP=5);
quick 模式的 warmup 跑在 msprof **之外**的独立进程里,预热的是 NPU/驱动,与进程内状态无关;
不传 `--device`,让它从 `run_npu_phase` 注入的 `ASCEND_RT_VISIBLE_DEVICES` 读,否则与租约打架;
取 `geomean_speedup` 而非 mean;新工具出的是**微秒**,`metrics.json` 是**毫秒**,要换算。

`msprof` 必须显式解析路径:`msprof_perf_summary.py:948` 是裸调 `"msprof"`,靠 env.sh 把
`BISHENGIR_BIN` 加进 PATH 是巧合,不能依赖。

## 八、Step2c 缓存/常量输出探测

判据:喂两组不同输入,输出必须跟着变;先用 golden 确认这两组本该产生不同输出(否则跳过)。
射程覆盖 `self._cache` / 全局缓存 / `lru_cache` / 返回常量 / 忽略输入。

**射程之外**:"shape 跟着变、数值瞎编"这类由对拍负责,不是这道闸门的事。数据集多数算子的
用例组 shape 不同(如 3_Add 的 `[128]` / `[256]`),所以形状敏感的假实现它抓不到。

退出码:0=通过 1=判定不真算 2=无法判定(固定入口此时只打印提示并放行,是已知软点)。

## 九、错误日志必须整份写文件

polar 会把 `judge_out/metrics_error.log` **整份下载**,交给
`operator_reward.classify_infra_error_text` 做 infra 二次分类。infra 签名
(`aclInit` / `InvalidDeviceId` / `getDeviceCntFailed`)出现在栈的**开头**,早先这里用
`tail -60` 传参 —— 头部被砍掉,polar 判不出 infra,本该 retry 不计分的环境故障被当成算子
失败给 0.2/0.3,直接毒化 GRPO。

超上限才截断,且**保头 + 保尾**,并如实置 `error_truncated=true`;
`error_bytes` / `error_sha256` 恒描述原始完整错误。

## 十、tarball 归一化

tarball 若打成 `tar -C {op} .`(内容直接在包根),解包后 `model_new_ascendc.py` 落在
`$WORK` 顶层 → `TASK_DIR=$WORK` → basename 得到 "work" 这种工作目录名而非算子名。
后果:① `verification_ascendc.py` 收到 op="work" 直接 FileNotFoundError;
② 隔离闸的 `_TOP_REL` 等于 `.`,整段被跳过。以 `--op_name` 为准,把内容挪进
`$WORK/$OP_NAME/` 再继续。
