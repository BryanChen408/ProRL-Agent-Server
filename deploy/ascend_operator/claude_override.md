# 本环境覆盖(优先级高于以上全部内容)

以上内容与本节冲突时,一律以本节为准。

## 固定入口

任何让你运行 `evaluate_ascendc.sh`、`evaluate_tilelang.sh`、`validate_ascendc_impl.py`、
`msprof_profile_run.sh`、`msprof_perf_summary.py`、`verification_ascendc.py` 的地方,
一律改跑这一条 —— 包括本文档以上各 Phase、以及你运行期调用任何 Skill 后读到的指示:

```bash
bash tools/ascendc_eval_pipeline.sh --op_name {op_name} \
     --impl output/submission/{op_name}_impl.tar.gz --out_dir judge_out
```

它一次完成:退化检测 → 编译 → 对拍 → 测速 → 写 `judge_out/metrics.json`,
并自动把 `{op_name}/` 打包成提交物、保留历史最优版本。

- 每轮修改后都跑一次。被中途截断时按历史最优版本判分,所以早跑、多跑不吃亏。
- 迭代时可加 `--incremental` 复用上次解包目录,走增量编译。
- 它评的是 **AscendC 提交物**。Phase 3 的 TileLang 阶段还没有 AscendC kernel,
  那时跑它只会得到 `submission_missing` 并白白消耗一次评测配额。
- 不要另跑 `cmake` / `make` / `python setup.py` / 自写测试脚本,也不要直接调 skill 里的
  AscendC 评测/对拍/测速脚本 —— 绕过它就没有基准复位、缓存检测和抢卡,结果不作数。

## Phase 3 的 TileLang 两个脚本(不走固定入口)

- `validate_tilelang_impl.py`(AST 退化检测,不占卡)—— **直接跑**,按上游 Phase 3 原样:
  ```bash
  python3 .claude/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py \
      {output_dir}/model_new_tilelang.py
  ```
- `verification_tilelang.py`(TileLang 功能验证,**占卡**)—— 必须经抢卡包装器,
  否则会抢走别的 session 正在用的卡:
  ```bash
  python3 tools/npu_lease_exec.py --pool "$POLAR_NPU_LEASE_POOL" \
      --lock-dir "$POLAR_NPU_LOCK_DIR" -- \
      python3 .claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py \
      {output_dir}
  ```
  按上游原文,TileLang 验证不是 correctness gate;失败但设计意图正确时可跳过并继续 Phase 4。

## 本环境无 Hook

上游的 `skill_script_hook.py` 拦截机制在本环境**未启用**:`.claude/settings.json` 只写了
`skillOverrides`,没有 hooks。你的 Bash 调用一律**直通执行**,不会被托管代跑。
凡上游文中说"由 Hook 拦截/代为执行"的地方,实际都是你自己在跑 —— 而按上方「固定入口」,
那些脚本一律改跑固定入口。

## 错误分类

固定入口输出末尾的 `错误分类:` 行是分类的权威来源:

| 分类 | 处理 |
|---|---|
| `通过` | 进入下一 Phase |
| `A类-代码/编译错误` | 走 4.5A 迭代 |
| `D类-精度不匹配` | 走 4.5D 迭代 |
| `INFRA-环境故障` | **不要迭代修复**,直接停止并说明 |

[A1] 照做:asc-devkit 就在 `$ASC_DEVKIT_DIR`。完整错误在 `judge_out/metrics_error.log`,先读它。

## Phase 2 / Phase 6

判分**恒用数据集原版全量用例**,`{op_name}/{op_name}.json` 在判分时会被覆盖。
所以 Phase 2(用例精简)只对你自己的迭代提速有意义,Phase 6(全量恢复)对判分无影响。
做或不做都可以,不要为它们花额外轮次。

## 提交物

- 唯一提交物:`output/submission/{op_name}_impl.tar.gz`,由固定入口自动打包。
- 工程目录必须是工作目录顶层的 `{op_name}/`,不要另建别名或带时间戳的目录。
- `{op_name}/model.py` 与 `{op_name}/{op_name}.json` 判分时会被数据集原版覆盖,改它们无效。
- 不要删除或移动 `output/submission/` 下的任何文件。

## 环境事实

- `asc-devkit` 挂在 `$ASC_DEVKIT_DIR`(= `/opt/asc-devkit`),与本机 CANN 同版本。
  API 文档在 `$ASC_DEVKIT_DIR/docs/zh/api/`,示例在 `$ASC_DEVKIT_DIR/examples/`,
  实现代码在 `$ASC_DEVKIT_DIR/impl/`。写 kernel 前用 `ascendc-docs-search` 查 API 签名与
  命名空间,不要凭记忆写。
- 文档里凡写 `asc-devkit/...` 的地方,一律读作 `$ASC_DEVKIT_DIR/...`。
- skill 文档里 `$ASC_DEVKIT_DIR/examples/00_introduction/...` 这类路径是旧版布局,
  本版实际是 `examples/{01_simd_cpp_api,02_simd_c_api,03_simt_api}/`,用 find 定位。
- 编译报 `no template named 'TQue'` / `did you mean 'AscendC::...'` 这类,是命名空间或签名
  记错,查 `$ASC_DEVKIT_DIR/docs/zh/api/` 核实,不要靠猜改。
- 检索文档一律从 `$ASC_DEVKIT_DIR` 根目录搜(Grep/Glob 的 path 填根目录),不要凭记忆
  猜子路径——docs/ 下有两棵树(`docs/zh/api/` 与 `docs/api/`),只搜子树会漏;
  搜不到时换关键词(去后缀、换同义词),不要直接下「API 不存在」的结论。

## 不可用

- **`[D2-2]` 那一步跳过**:`precision_forensics.py` 上游未随包发布(`tilelang2ascend-precision-tuning`
  没有 `scripts/` 目录)。D-2 的其余步骤照做 —— `[D2-1]` 调 Skill、`[D2-3]` 改代码、`[D2-4]` 跑固定入口。

## 禁止

- 自行设置 `ASCEND_RT_VISIBLE_DEVICES`。
- 运行 `npu-smi` 或任何探测 NPU 的命令。`SOC_VERSION` 已在环境变量里。
- 读取、修改或删除 `tools/` 与 `.claude/skills/*/scripts/` 下的脚本。
- 修改评测参数(SOC_VERSION / warmup / repeats / 精度阈值)。
- 向用户提问。这是非交互运行,没有人会回答。
