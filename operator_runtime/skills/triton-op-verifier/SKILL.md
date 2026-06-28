---
name: triton-op-verifier
description: >
  验证生成的 Triton 算子——通过固定入口 tools/triton_eval_pipeline.sh 跑 AST 退化检查 + NPU 数值
  正确性 + 性能采集，并以本次固定入口输出判定。失败未耗尽时读取 judge_out/metrics_error.log 的完整错误修复；
  出现 LIMIT_EXHAUSTED 时立即结束。禁止自写测试、禁止改评测脚本/参数。当需要验证 output/submission/{op_name}_impl.py
  的正确性或采集其性能时使用。
argument-hint: >
  输入：op_name、impl=output/submission/{op_name}_impl.py、task=src/{op_name}.py。
  输出：固定入口本次 Bash 输出中的 success、ast_check_ok、correctness_ok、error_type、attempt/LIMIT_EXHAUSTED
  与 judge_out/metrics_error.log 修复依据。固定入口：bash tools/triton_eval_pipeline.sh --op_name {op_name}
  --impl output/submission/{op_name}_impl.py --task src/{op_name}.py --out_dir judge_out。
---

# Triton 算子验证

你负责验证 `output/submission/{op_name}_impl.py` 是否编译运行正确、与 `src/{op_name}.py` 参考实现输出一致，并采集性能数据。

## 硬规则

你被调用来验证时，第一个动作必须是 `Bash` 运行固定入口：

```bash
bash tools/triton_eval_pipeline.sh --op_name {op_name} \
    --impl output/submission/{op_name}_impl.py \
    --task src/{op_name}.py \
    --out_dir judge_out
```

你必须：

- 只运行固定入口 `tools/triton_eval_pipeline.sh`。
- 只根据本次固定入口输出判断成功、失败、阶段次数和是否耗尽。
- 失败且未出现 `LIMIT_EXHAUSTED` 时，完整读取 `judge_out/metrics_error.log` 后再修复。
- 只修改 `output/submission/{op_name}_impl.py`。

你禁止：

- 自写测试、直接跑 `verify.py` / `benchmark.py`、或构造额外验证脚本。
- 读取、修改 `tools/`、`scripts/`、`.agents/skills/` 下的评测实现。
- 使用 `head` / `cat` / `sed -n` 等命令查看 `tools/triton_eval_pipeline.sh` 或 verifier scripts 来代替运行固定入口。
- 修改 warmup、repeats、精度阈值、framework latency、skip 参数。
- 出现 `LIMIT_EXHAUSTED` 后继续读文件、继续运行命令或继续修复。

⚠️ **禁止自写测试代码、禁止直接调用或修改 `scripts/` 下脚本、禁止改任何评测参数**——所有验证与性能采集**统一通过固定入口** `tools/triton_eval_pipeline.sh` 执行。它内部按固定顺序跑 AST 退化检查 → `verify.py`（NPU 数值）→ `benchmark.py`（性能），warmup/repeats/精度阈值、以及 framework 基线全部写死；**不存在也不要寻找** `--skip_framework` / `--framework_latency_ms` / `--verify_not_required` 之类开关。

## 用法

```bash
bash tools/triton_eval_pipeline.sh --op_name {op_name} \
    --impl output/submission/{op_name}_impl.py \
    --task src/{op_name}.py \
    --out_dir judge_out
```

固定入口只在本次 Bash 输出中回显成功/失败判定、`error_type`、阶段次数和下一步动作。失败且未出现 `LIMIT_EXHAUSTED` 时，完整错误写入 `judge_out/metrics_error.log`；先完整读取该文件，再只修改 `output/submission/` 后重跑本入口。若输出 `LIMIT_EXHAUSTED`，不要读取 `judge_out/metrics_error.log` 或 `judge_out/metrics.json`，不要再调用任何工具，直接结束任务。无需读 `tools/` 或 `scripts/`，也禁止修改它们。

## 固定入口输出字段

| 字段 | 含义 |
|------|------|
| `success` | 全流程是否成功（ast + correctness + perf 均通过） |
| `ast_check_ok` | AST 退化预检查是否通过 |
| `correctness_ok` | NPU 数值正确性是否通过 |
| `perf_data.speedup_vs_torch` | 相比 PyTorch 的加速比（**全 shape 几何平均** `(∏ sᵢ)^(1/n)`，仅取 status==pass 且为有限正数的 shape；NaN/Inf/0/负值不计入；全异常时为 `null`） |
| `error_type` / `error_file` | 失败分类与完整错误文件位置；完整 traceback 在 `judge_out/metrics_error.log` |

## 常见失败模式与修复方向（按 `error_type` 索引）

| error_type / 特征 | 修复方向 |
|---------|---------|
| AST 退化 Type 1/2/3 | 见 `CLAUDE.md` 退化子类型：把 `forward()` 的 torch 计算移进 `@triton.jit` kernel（Type1 完全无 kernel / Type2 有 kernel 但未被 forward 调用 / Type3 部分计算仍用 torch） |
| `correctness_failed` / shape mismatch | 检查 grid/block 配置、输出 buffer 形状、broadcasting、规约轴 |
| `correctness_failed` / 数值超差 | 检查 dtype 转换、累加顺序（必要时 fp32）、边界处理 |
| `triton_compile_failed` / `triton_lowering_failed` | 检查 `tl.constexpr` 参数、内存对齐、指针表达式（别传 `.data_ptr()`） |
| `triton_ub_overflow` / OOM | 减小 `BLOCK_SIZE_{M,N,K}`，优化内存复用（多为 B 类，可能需终止） |
| `benchmark_failed` | 简化 kernel、减少访存、调 block size |
| 超时 | 检查无限循环或过大 grid |

## 精度判定（评测器内置，**了解即可、不可改**）

`verify.py` 用基于 dtype 的 **MERE/MARE 双门限相对误差**（NPU Benchmark 标准），非 `torch.allclose`。判定须**同时满足**：

```
MERE < threshold      且      MARE < 10 × threshold
MERE = mean(|actual-golden| / max(|golden|, threshold))   # 平均相对误差
MARE = max (|actual-golden| / max(|golden|, threshold))   # 最大相对误差
```

比较前两侧统一升 float32；分母 `clamp(min=threshold)`（参考值小到 dtype 极限时退化为按绝对误差归一，避免零值附近误报）。

| 数据类型 | threshold | MERE 上限 | MARE 上限 |
|---------|-----------|-----------|-----------|
| `float16` | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| `bfloat16` | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| `float32` | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| `hifloat32` | 2⁻¹¹ ≈ 4.88e-4 | 4.88e-4 | 4.88e-3 |
| `float8_e4m3` / `float8_e5m2` | 0.125 / 0.25 | — | — |
| 其他（fallback） | 2⁻¹³ | 1.22e-4 | 1.22e-3 |

比对前置检查（按序，任一失败即 fail）：① 形状一致 → ② NaN 位置完全一致 → ③ Inf 位置与符号一致 → ④ `bool` 要求 `torch.equal` → ⑤ 仅在有限值 mask 上算 MERE/MARE。这些由固定入口写死，**agent 不可调整**。
