# Ascend Interpolate 算子系统性优化

## 概述

本文档覆盖 Ascend NPU 上 Triton Interpolate 算子（nearest / bilinear / bicubic / area）
的系统性性能优化方法。作为 `latency-optimizer` 的补充，应在完成通用优化点之后按 Phase 顺序执行。

## 适用算子

- `interpolate`
- `upsample_nearest2d`
- `upsample_bilinear2d`
- `upsample_bicubic2d`
- `adaptive_avg_pool2d`（对应 area 模式）

## Phase 1：坐标与权重 Host 预计算

### 触发条件

kernel 内在运行时计算 `scale`、`src_y`、`src_x`、floor 下标、双线性/双三次权重。

### 问题

- 同一行/列的坐标、权重对所有 channel/batch 重复计算。
- int32 floor、cast、分支判断容易退化为标量循环。
- CPU float、kernel float32、NPU vdiv 舍入路径不同，导致 bit 级精度差异。

### 优化策略

在 `ModelNew.forward()` 中用 `numpy.float32` / `torch.float32` 预计算：

| 模式 | 预计算内容 |
|------|-----------|
| nearest | `src_y`、`src_x` 或 `inv_scale` |
| bilinear | `y0/y1/x0/x1`、`ly/lx` |
| bicubic | 4×4 邻域下标、16 个 Keys' 权重 |
| area | `h_start/h_end`、`w_start/w_end`、count |

```python
import numpy as np

def _precompute_bilinear(H_in, W_in, H_out, W_out, align_corners, device):
    if align_corners:
        scale_y = np.float32(H_in - 1) / np.float32(max(H_out - 1, 1))
        scale_x = np.float32(W_in - 1) / np.float32(max(W_out - 1, 1))
        y_arr = np.arange(H_out, dtype=np.float32) * scale_y
        x_arr = np.arange(W_out, dtype=np.float32) * scale_x
    else:
        scale_y = np.float32(H_in) / np.float32(H_out)
        scale_x = np.float32(W_in) / np.float32(W_out)
        y_arr = (np.arange(H_out, dtype=np.float32) + 0.5) * scale_y - 0.5
        x_arr = (np.arange(W_out, dtype=np.float32) + 0.5) * scale_x - 0.5
        y_arr = np.clip(y_arr, 0, H_in - 1)
        x_arr = np.clip(x_arr, 0, W_in - 1)

    y0 = np.floor(y_arr).astype(np.int32)
    y1 = np.minimum(y0 + 1, H_in - 1)
    ly = (y_arr - y0.astype(np.float32)).astype(np.float32)

    x0 = np.floor(x_arr).astype(np.int32)
    x1 = np.minimum(x0 + 1, W_in - 1)
    lx = (x_arr - x0.astype(np.float32)).astype(np.float32)

    return (
        torch.tensor(y0, device=device), torch.tensor(y1, device=device),
        torch.tensor(x0, device=device), torch.tensor(x1, device=device),
        torch.tensor(ly, device=device), torch.tensor(lx, device=device),
    )
```

kernel 内仅加载预计算张量并执行融合：

```python
@triton.jit
def bilinear_kernel(input_ptr, output_ptr,
                    y0_ptr, y1_ptr, x0_ptr, x1_ptr,
                    ly_ptr, lx_ptr,
                    N, C, H_in, W_in, H_out, W_out,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_outputs = N * C * H_out * W_out
    for idx in range(pid * BLOCK_SIZE, num_outputs, tl.num_programs(0) * BLOCK_SIZE):
        tmp = idx
        w_out = tmp % W_out; tmp //= W_out
        h_out = tmp % H_out; tmp //= H_out
        c = tmp % C; n = tmp // C

        y0 = tl.load(y0_ptr + h_out)
        y1 = tl.load(y1_ptr + h_out)
        x0 = tl.load(x0_ptr + w_out)
        x1 = tl.load(x1_ptr + w_out)
        ly = tl.load(ly_ptr + h_out)
        lx = tl.load(lx_ptr + w_out)

        base = ((n * C + c) * H_in + y0) * W_in
        p00 = tl.load(input_ptr + base + x0)
        p01 = tl.load(input_ptr + base + x1)
        base = ((n * C + c) * H_in + y1) * W_in
        p10 = tl.load(input_ptr + base + x0)
        p11 = tl.load(input_ptr + base + x1)

        val = (1 - ly) * ((1 - lx) * p00 + lx * p01) + \
              ly      * ((1 - lx) * p10 + lx * p11)
        tl.store(output_ptr + ((n * C + c) * H_out + h_out) * W_out + w_out, val)
```

### 精度要点

- 预计算必须使用 `np.float32` / `torch.float32`，禁用 Python `float`（64 位）。
- 下标 clamp 在 host 完成，kernel 内不再判断边界。
- fp16/bf16 `align_corners=True` 的 scale 建议通过 NPU `vdiv` 计算（见 Phase 4）。

## Phase 2：UB 行预加载 + `tl.gather` 离散访存

### 触发条件

bilinear/bicubic 需要按运行时列下标从同一行取出多个像素，存在全局内存离散访问。

### 优化策略

1. 将连续整行加载到 UB。
2. 用 `tl.gather` 从 UB 行按列下标向量取出像素。

```python
# 为当前输出像素构造单元素向量下标
x0_vec = tl.full((1,), x0_scalar, tl.int32)
pixel = tl.gather(row, x0_vec, 0)
```

3. 在 program 内维护行缓存，避免同一输入行重复加载。

```python
last_y = -1
row_buf = tl.zeros((MAX_IN_W,), dtype=tl.float32)

for i in range(BLOCK_W):
    if y0[i] != last_y:
        row_buf = tl.load(input_ptr + base + y0[i] * W_in + tl.arange(0, W_in))
        last_y = y0[i]
    p00 = tl.gather(row_buf, x0_vec[i], 0)
```

### 适用条件

- bilinear / bicubic。
- `W_in` 不超过 UB 预算（通常 `W_in * sizeof(dtype)` 在几十 KB 内）。

## Phase 3：2D Vertical Tiling

### 触发条件

upsampling 场景下，相邻输出像素复用同一输入行，但每个 program 只处理一行。

### 优化策略

让单个 program 处理一个 2D tile：

```python
TILE_H = 4
TILE_W = 128

tiles_h = triton.cdiv(H_out, TILE_H)
tiles_w = triton.cdiv(W_out, TILE_W)
num_tiles = tiles_h * tiles_w * N * C
grid = (min(num_tiles, num_cores),)
```

在 tile 内按行加载输入，同一 `y0/y1` 行只加载一次，多列输出复用。

### TILE 选择

| 场景 | TILE_H | TILE_W |
|------|--------|--------|
| upsampling | 4~8 | 128~512 |
| downsampling | 1~2 | 256~1024 |
| 小图 | 1 | `W_out` |

约束：UB 占用 `< 192KB`。

## Phase 4：CANN 舍入对齐（`_npu_scale`）

### 触发条件

`align_corners=True` 或 fp16/bf16 下出现坐标缩放导致的精度边缘失败。

### 问题

CPU float32 除法舍入与 NPU `vdiv` 可能不同，导致 kernel 坐标表与 PyTorch/CANN 不完全一致。

### 优化策略

通过 NPU `vdiv` 计算 scale，再回 CPU 生成坐标表：

```python
def _npu_scale(num, den):
    a = torch.tensor([num], dtype=torch.float32, device='npu')
    b = torch.tensor([den], dtype=torch.float32, device='npu')
    return (a / b).cpu().item()

scale_h = _npu_scale(H_in - 1, max(H_out - 1, 1))
```

### 适用

- bilinear/bicubic `align_corners=True`
- fp16/bf16 精度敏感路径

## Phase 5：Kernel 路由与特化

### 触发条件

单一通用 kernel 在不同 mode / align_corners / dtype / shape 下效率差异大，
部分 shape 加速比明显偏低。

### 优化策略

在 `ModelNew.forward()` 中根据特征选择专用 kernel：

```python
def _route(self, x, size, scale_factor, mode, align_corners, ...):
    if mode == 'bilinear' and not align_corners and H_out > H_in:
        return self._bilinear_upsample_ac_false(x, ...)
    elif mode == 'bicubic' and align_corners:
        return self._bicubic_ac_true(x, ...)
    ...
```

### 设计原则

- 接口不变，仅内部路由。
- 无匹配时回退到通用 kernel。
- 路由开销 `< 0.1ms`。

## Phase 6：Bicubic `align_corners=True` 精度特化

### 触发条件

bicubic `align_corners=True` 出现随机 1 像素 fp32 边缘失败。

### 问题

在 kernel 内向量计算 Keys' 权重时，浮点舍入与 PyTorch C++ 参考存在差异。

### 优化策略

在 host 端预计算 4×4 邻域下标和 16 个权重，kernel 内只做 `tl.gather` 与确定性顺序累加：

```python
A = -0.75
def _cubic(t):
    t = abs(float(t))
    if t <= 1.0:
        return float(((A + 2.0) * t - (A + 3.0)) * t * t + 1.0)
    else:
        return float(((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A)

# 预计算 y_m1/y_0/y_p1/y_p2, x_m1/x_0/x_p1/x_p2 及 16 个 w_y*w_x
```

kernel 内：

```python
val = ZERO
for jj in range(4):
    y_idx = tl.load(y_idx_ptr[jj] + h_out)
    wy = tl.load(y_w_ptr[jj] + h_out)
    row = tl.load(input_ptr + base + y_idx * W_in + tl.arange(0, W_in))
    for ii in range(4):
        x_idx = tl.load(x_idx_ptr[ii] + w_out)
        wx = tl.load(x_w_ptr[ii] + w_out)
        pixel = tl.gather(row, x_idx_vec, 0)
        val = val + wy * wx * pixel
```

### 要点

- 权重和顺序必须与 PyTorch C++ 实现完全一致。
- 累加顺序固定，禁止向量化重排。

## Phase 7：编译选项 `multibuffer` / `unit_flag`

### 触发条件

插值 kernel 为内存密集型，大量 global load/store。

### 优化策略

kernel 调用时开启：

```python
kernel[grid](..., multibuffer=True)
```

对 tile 较小、计算密集度低的 kernel 可尝试 `unit_flag=True/False` 对比。

## 验证规则

每个 Phase 独立执行：

1. 修改后检查 `references/checklist.md`。
2. 执行 `verify.py`，要求 `passed_cases == total_cases`。
3. 执行 `benchmark.py`，性能不劣化则保留。

## 参考资料

- `latency-optimizer/references/checklist.md`
- `latency-optimizer/references/vector_core_partition.md`
- `latency-optimizer/references/scalar_to_vector.md`
- `latency-optimizer/references/discrete_memory_access.md`
- `latency-optimizer/references/loop-invariant-hoisting.md`
