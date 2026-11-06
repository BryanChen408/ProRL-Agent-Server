---
type: Runbook
title: 宽 NZ 子块错读：分离基址与平面跨度
description: 用固定A的Cube投影分别核对宽L1缓存的B子块基址和读取跨度；仅改C0基址仍可能错读，结论限定已实测的ND2NZ与helper参数。
tags: [task.debugging, topic.data_movement, topic.precision]
status: stable
sources: []
created_at: "2026-10-18T16:54:01Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# 宽 NZ 子块错读：分离基址与平面跨度

## 现象

在排查 AscendC Cube 矩阵乘的宽 L1 预取子块时，不能只用“单片复制算对了”判断宽缓存中的第二片也正确。这里用 `fresh_default/kernel/b_slice.h` 新建了一个 B-only 投影探针：A 每次都走同一条完整小块路径，B 先存入宽 NZ 缓存，再将其中一片送入 L0B，最后执行同一 `Mmad` 和 `Fixpipe`。执行 `run_panel.py --out npu_panel_001 --device 0 --gate root-authorized-after-c033` 后，直接片控制全部逐元素相等，但宽缓存中的若干模式稳定地产生数值错位，没有运行异常、NaN 或输出 guard 变化。

最容易误判的是 FP16 的宽 B：只把第二片基址从 `i*16*N` 改成 `i*16*C0` 后，仍仅有 `256/512` 个输出相等；把 half helper 的跨 N16 平面源步距也从 `16*16` 改成 `32*16`，才得到 `512/512`。这是同一受控诊断中的观测，不能据此宣布原完整矩阵乘已修复。

## 适用场景

- 工程形态：AscendC kernel 直调，纯 AIC、单 block 的 B-only Cube 数值投影诊断。
- API 路径：`DataCopy` 的 ND→NZ / 原样 GM→B1 路径，`LoadDataWithTranspose` 或 `LoadData3DParamsV2` 读取 B1→B2，再经 `Mmad` / `Fixpipe` 比对。
- 验证过的平台与版本：a2，实际 `Ascend910B4`；CANN `9.1.0`，Bisheng clang `15.0.5`，Python `3.11.14`，PyTorch `2.9.0+cpu` + `torch_npu 2.9.0`。实际编译为 `dav-c220-cube`，使用默认 auto-sync。
- 适用粒度：`exact_task`，仅下列完整诊断实现、已执行 dtype/宽度和 ND2NZ 字段；不得提升为跨布局通用 C0 公式。
- 路线优先级：`recommended_fastpath`，指优先采用这个单变量诊断顺序；没有证明生产 kernel 性能更快。
- 未验证场景：原完整 MatmulTransA 的 A 打包、持久调度、累加和尾块，bf16、其他尺寸/输入、其他 SoC/CANN；代码中的兼容重载并不表示这些场景已测。

## 根因

在本次明确列出的 ND2NZ 参数和读取链中，宽 NZ 缓存的子块仍继承宽缓存的平面跨度，将它当作独立紧凑 NZ 张量会同时误判子块的起点和跨平面的推进方式。

本次原始 ND2NZ 参数是 `ndNum=1`、`nValue=32`、`dValue=N`、`srcDValue=N`、`dstNzC0Stride=32`、`dstNzNStride=1`，其他矩阵跨度为零。与这些参数作设备对照的 raw 编码模型是：

`offset(r,c) = (c//C0)*32*C0 + r*C0 + c%C0`，其中 `C0=32/sizeof(T)`，偏移单位为 T 元素。

选取 `r=i*16+r_local` 后，子块起点是 `i*16*C0`；跨 C0 平面仍推进 `32*C0`，不能跟着局部块深度变成 `16*C0`。对 FP16，`sizeof(T)=2`、`C0=16`：第二片起点 `1*16*16=256` 个元素，跨 N16 平面推进 `32*16=512` 个元素。旧 half 重载忽略 `colC0Stride`，仍按 `j*k*16`、即 `j*16*16` 推进，所以改对起点后第二个平面仍错。对所测 FP32，`C0=8`，第二片起点为 `1*16*8=128`，而 `i*16*N` 给出 `256`；使用正确的 generic `col=32` 时只改起点即可消除本次错位。

上述算式只解释这次可重建的布局与观测；这里没有直接读回 L1/L0B，更没有证明任意 NZ、padding 或批矩阵布局都服从同一公式。

## 判别方法

1. 固定 A 的复制与读取路径、队列深度、缓冲大小、事件生命周期、`Mmad`、`Fixpipe` 和输入字节。分别输出 B 的两片，避免把 A 的宽预取或多片累加混进来。A 用稠密可逆的 Hadamard 投影，不用 one-hot 只抽少数行。
2. 每个 dtype/宽度先运行 mode 0：只把选中的 B 片做 ND→NZ，基址为零。若直接片、guard、finite 或重复字节检查失败，立即停止归因。
3. mode 1→2 只换 B1 元素基址；不要同时换 helper。FP16 且 `N=C0=16` 时两个公式相同，只有一个 N16 平面，不能拿这个点证明哪种公式正确。
4. FP16 mode 2→3 只换 half 重载循环中的跨平面源指针步距。保持 ND2NZ、基址、A 和所有队列不变。mode 3 与 mode 0 应在所测两片上完全一致。
5. 用 mode 3→4（FP16）或 2→4（FP32）对比实际 ND→NZ 与 raw 模型：仅去掉最终 ND2NZ 覆盖，读取链保持一致。B1 的完整物理分配先经原样复制初始化，故故意错误但有界的读取不会接触未初始化内容。
6. FP16 第二片 mode 3→5 只换成显式 generic3D、仍保留 `col=32`；再用 5→6 只换 `col=16`。FP32 同样比较 mode 2→6。helper 类型名称不能代替实际空间跨度。

模式定义如下；它们是诊断变体，不是七条待推荐的生产实现。

| mode | 最终 GM→B1 | 子块起点 | B helper |
|---:|---|---|---|
| 0 | 仅选中的片做 ND→NZ | 0 | 原 dtype helper，col=16 |
| 1 | 全宽 ND→NZ | i*16*N | 原 dtype helper，col=32 |
| 2 | 全宽 ND→NZ | i*16*C0 | 原 dtype helper，col=32 |
| 3 | 全宽 ND→NZ | i*16*C0 | half 跨平面步距改用 col*16 |
| 4 | 保留初始 raw 编码 | i*16*C0 | 与 mode 3 相同 |
| 5 | 全宽 ND→NZ | i*16*C0 | 显式 generic3D，col=32 |
| 6 | 全宽 ND→NZ | i*16*C0 | 显式 generic3D，col=16 |

## 设计决策

采用固定 A、逐片输出和单变量 B 对照，因为完整矩阵乘同时变化 A/B 的预取会让数值错误失去唯一切入点。A 的 `±1/4` 与 B 的小二进制分数可精确表示；CPU float64 参考再转 float32，与本次精确相等检查配套，避免把长累加或低精度舍入当成布局错位。两次重放使用相同输入，作用仅为检查不稳定性，不计为两个独立复现。

没有采用“先改基址再认定修好”、仅看 FP16 窄 B 控制、直接抄旧 helper 注释或构造未确认的 L1 读回接口。raw 编码只是有已初始化尾部的独立对照，设备结果对不上就不能当成布局事实。错误配置均保留数值失败，未改容差、删失败点或替换参考 oracle。当前路线仅优先用于诊断，没有性能测量或生产 fallback 推荐。

同一宽缓存问题可先用本卡的 B-only 对照分离地址与读取跨度；若需要确认完整内核是否真正激活非首子 tile，参见 [L1 预取分片控制诊断](l1_prefetch_subtile_controlled_failure.md)。两者执行的内核与证据范围不同，本卡结果不替代完整内核回归。API 参数语义分别查阅 [ND→NZ DataCopy](../../apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md)、[LoadDataWithTranspose](../../apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/load_data_with_transpose.md) 和 [LoadData3D](../../apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/load_data_3_d.md)。

## 系统信息

探针的完整核心文件是 `fresh_default/kernel/b_slice.h`、`matmul_tile.h`、两个 `probe_fp16.cpp` / `probe_fp32.cpp` 入口、`pybind11.cpp` 和 `run_panel.py`。构建脚本使用同一冻结的 CMake 打包入口；一次全新构建记录了实际 AIC 命令、源码、动态模块和编译器身份。运行器拒绝源码与构建回执不一致、已存在输出目录，以及导入模块 SHA 与构建产物不一致的情况。

编译前已检查当前安装包的原样 GM→B1 复制、ND2NZ、2D transpose / generic3D 的 dtype、对齐、几何、缓冲容量和事件预算。运行前检查目标设备空闲；逻辑设备由显式可见设备映射绑定。初始计划文件中的“未执行”描述保留为预注册快照，不能覆盖后续实际构建与上板结果。

## 完整代码

以下六个文件按实际冻结并执行的源码逐字内联。`matmul_tile.h` 原文件注释和 bf16 兼容重载原样保留，注释中的泛化说法不是额外验证结论；本卡只承认上文列出的 fp16/fp32 点。构建/模块身份记录属于支持证据，核心诊断的加载、执行、参考、全量输出保存与停止协议均在运行器中。

### `fresh_default/kernel/b_slice.h`

```cpp
#ifndef C021_B_SLICE_H
#define C021_B_SLICE_H
#include "matmul_tile.h"
#include <type_traits>

// A diagnostic projection, not a MatmulTransA candidate. The old dtype helpers
// are included byte-for-byte. A is always copied as one complete baseK tile.
// B1 has an initialized tail so deliberately wrong legal offsets stay inside
// allocated storage. No L1/L0B readback API is assumed or introduced.
template <typename T>
class BSliceProbe {
    static constexpr uint32_t M = 16, BK = 16, KL = 32;
    static constexpr uint32_t C0 = 32 / sizeof(T), GUARD = 256;
    AscendC::TQue<AscendC::TPosition::A1, 1> qa1;
    AscendC::TQue<AscendC::TPosition::A2, 1> qa2;
    AscendC::TQue<AscendC::TPosition::B1, 1> qb1;
    AscendC::TQue<AscendC::TPosition::B2, 1> qb2;
    AscendC::TQue<AscendC::TPosition::CO1, 2> qc;
public:
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR packed,
                              GM_ADDR c, uint32_t n, uint32_t slice,
                              uint32_t mode, AscendC::TPipe &pipe) {
        // Identical allocations and queue lifecycle for every mode at one N.
        // Largest L1 request (float,N32): 2*(256+1536)*4 = 14336 B.
        // Largest L0A/L0B/CO1 requests: 2048/4096/4096 B respectively.
        pipe.InitBuffer(qa1, 2, M * BK * sizeof(T));
        pipe.InitBuffer(qa2, 2, M * BK * sizeof(T));
        pipe.InitBuffer(qb1, 2, (KL + BK) * n * sizeof(T));
        pipe.InitBuffer(qb2, 2, BK * n * sizeof(T));
        pipe.InitBuffer(qc, 2, M * n * sizeof(float));
        AscendC::GlobalTensor<T> ga, gb, gp;
        AscendC::GlobalTensor<float> gc;
        ga.SetGlobalBuffer((__gm__ T *)a, M * BK);
        gb.SetGlobalBuffer((__gm__ T *)b, KL * n);
        gp.SetGlobalBuffer((__gm__ T *)packed, (KL + BK) * n);
        gc.SetGlobalBuffer((__gm__ float *)c, 2 * GUARD + M * n);

        auto a1 = qa1.template AllocTensor<T>();
        LoadNdGmToNzL1(a1, ga, M, BK, BK);
        qa1.EnQue(a1);
        a1 = qa1.template DeQue<T>();
        auto a2 = qa2.template AllocTensor<T>();
        LoadNzL1ToZzL0A(a2, a1, M, BK, M);
        qa2.EnQue(a2);

        auto b1 = qb1.template AllocTensor<T>();
        // Contract gate: the target's installed headers must confirm raw
        // GM->B1 DataCopy support before build. All length/alignment is 32B.
        AscendC::DataCopy(b1, gp, (KL + BK) * n);
        AscendC::PipeBarrier<PIPE_MTE2>();
        if (mode == 0) {
            LoadNdGmToNzL1(b1, gb[slice * BK * n], BK, n, n);
        } else if (mode != 4) {
            LoadNdGmToNzL1(b1, gb, KL, n, n);
        }
        qb1.EnQue(b1);
        b1 = qb1.template DeQue<T>();
        auto b2 = qb2.template AllocTensor<T>();
        uint32_t offset = mode == 0 ? 0 :
                          (mode == 1 ? slice * BK * n : slice * BK * C0);
        uint32_t col = mode == 0 || mode == 6 ? BK : KL;
        if (mode == 3 || mode == 4) {
            LoadStrideAware(b2, b1[offset], n, col);
        } else if (mode == 5 || mode == 6) {
            // Explicit template selects the original generic 3D helper,
            // bypassing its half overload. It is a diagnostic comparator.
            LoadNzL1ToZnL0B<T>(b2, b1[offset], BK, n, col);
        } else {
            LoadNzL1ToZnL0B(b2, b1[offset], BK, n, col);
        }
        qb2.EnQue(b2);
        auto c1 = qc.template AllocTensor<float>();
        a2 = qa2.template DeQue<T>();
        b2 = qb2.template DeQue<T>();
        AscendC::MmadParams mm;
        mm.m = M; mm.n = n; mm.k = BK; mm.cmatrixInitVal = true;
        AscendC::Mmad(c1, a2, b2, mm);
        qa2.FreeTensor(a2); qb2.FreeTensor(b2);
        qa1.FreeTensor(a1); qb1.FreeTensor(b1);
        qc.EnQue(c1);
        c1 = qc.template DeQue<float>();
        FixpipeNzL0cToNdGm(gc[GUARD], c1, M, n, n, QuantMode_t::NoQuant);
        AscendC::PipeBarrier<PIPE_FIX>();
        qc.FreeTensor(c1);
    }
private:
    __aicore__ inline void LoadStrideAware(const AscendC::LocalTensor<T> &dst,
                                          const AscendC::LocalTensor<T> &src,
                                          uint32_t n, uint32_t col) {
        if constexpr (std::is_same<T, half>::value) {
            AscendC::LoadData2dTransposeParams p;
            p.startIndex = 0; p.repeatTimes = BK / 16;
            p.srcStride = 1; p.dstGap = 1; p.dstFracGap = 0;
            for (uint32_t j = 0; j < n / 16; ++j) {
                // Only this per-N16 source stride differs from old helper.
                AscendC::LoadDataWithTranspose(dst[j * 16 * 16],
                                               src[j * col * 16], p);
            }
        } else {
            LoadNzL1ToZnL0B<T>(dst, src, BK, n, col);
        }
    }
};
#endif
```

### `fresh_default/kernel/matmul_tile.h`

```cpp
/**
 * @file matmul_tile.h
 *
 * Tile-level data movement helpers for the MatmulTransA cube kernel.
 *
 * The L1->L0A/L0B path is split by dtype:
 *   - float uses LoadData3DParamsV2 (single-fractal 4B layout).
 *   - half / bfloat16 use the canonical LoadData2D / LoadDataWithTranspose
 *     paths recommended for 2B fractal matmul, avoiding layout mismatch.
 * Fixpipe always writes CO1 (float) back to GM with optional quantPre cast.
 */
#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)  : load an (m, n) ND tile into an (m, n) Nz L1 buffer.
// Pre-condition: m % 16 == 0 && n % C0 == 0.
template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

// L1(Nz) -> L0A(Zz)
// Generic across element widths because LoadData3DParamsV2 expresses
// extents in *elements*, not bytes.
template<typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t m, uint32_t k, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) — generic version that works for half / bfloat16 / float.
template<typename T>
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = n;
    params.kExtension = n;
    params.mExtension = k;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0A(Zz) for half / bfloat16: canonical LoadData2D path.
// float 仍使用上方 LoadData3DParamsV2 路径。
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<half> &dst,
                                       const AscendC::LocalTensor<half> &src,
                                       uint32_t m, uint32_t k, uint32_t /*colC0Stride*/)
{
    AscendC::LoadData2DParams params;
    params.repeatTimes = m * k * sizeof(half) / 512;
    params.srcStride = 1;
    params.ifTranspose = false;
    AscendC::LoadData(dst, src, params);
}

__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<bfloat16_t> &dst,
                                       const AscendC::LocalTensor<bfloat16_t> &src,
                                       uint32_t m, uint32_t k, uint32_t /*colC0Stride*/)
{
    AscendC::LoadData2DParams params;
    params.repeatTimes = m * k * sizeof(bfloat16_t) / 512;
    params.srcStride = 1;
    params.ifTranspose = false;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) — half / bfloat16 专用 LoadDataWithTranspose 路径。
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<half> &dst,
                                       const AscendC::LocalTensor<half> &src,
                                       uint32_t k, uint32_t n, uint32_t /*colC0Stride*/)
{
    constexpr uint32_t BLOCK_SIZE = 16;
    AscendC::LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / BLOCK_SIZE;
    params.srcStride = 1;
    params.dstGap = 1;
    params.dstFracGap = 0;

    uint32_t nBlocks = n / BLOCK_SIZE;
    for (uint32_t i = 0; i < nBlocks; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * BLOCK_SIZE * BLOCK_SIZE],
                                       src[i * k * BLOCK_SIZE], params);
    }
}

__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<bfloat16_t> &dst,
                                       const AscendC::LocalTensor<bfloat16_t> &src,
                                       uint32_t k, uint32_t n, uint32_t /*colC0Stride*/)
{
    constexpr uint32_t BLOCK_SIZE = 16;
    AscendC::LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / BLOCK_SIZE;
    params.srcStride = 1;
    params.dstGap = 1;
    params.dstFracGap = 0;

    uint32_t nBlocks = n / BLOCK_SIZE;
    for (uint32_t i = 0; i < nBlocks; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * BLOCK_SIZE * BLOCK_SIZE],
                                       src[i * k * BLOCK_SIZE], params);
    }
}

// L0C(Nz) -> GM(ND): write an (m, n) Nz fragment back to an ND row of a
// larger GM matrix whose row stride is `dstStride` (in elements).
// `quantPre` carries any optional CO1 -> dst dtype cast (NoQuant for
// float, F322F16 for half, F322BF16 for bfloat16_t).
//
// Note: `QuantMode_t` is a *global* enum declared by the Bisheng cce
// intrinsics header, not a member of namespace AscendC.
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                          const AscendC::LocalTensor<float> &src,
                                          uint32_t m, uint32_t n, uint32_t dstStride,
                                          QuantMode_t quantPre)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = dstStride;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    params.quantPre = quantPre;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
```

### `fresh_default/kernel/probe_fp16.cpp`

```cpp
#include "b_slice.h"
extern "C" __global__ __aicore__ void c021_bslice_fp16(
    GM_ADDR a, GM_ADDR b, GM_ADDR packed, GM_ADDR c,
    uint32_t n, uint32_t slice, uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    BSliceProbe<half> probe;
    probe.Run(a,b,packed,c,n,slice,mode,pipe);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c021_bslice_fp16_do(uint32_t bd,void *stream,uint8_t *a,
    uint8_t *b,uint8_t *packed,uint8_t *c,uint32_t n,uint32_t slice,uint32_t mode) {
    c021_bslice_fp16<<<bd,nullptr,stream>>>(a,b,packed,c,n,slice,mode);
}
#endif
```

### `fresh_default/kernel/probe_fp32.cpp`

```cpp
#include "b_slice.h"
extern "C" __global__ __aicore__ void c021_bslice_fp32(
    GM_ADDR a, GM_ADDR b, GM_ADDR packed, GM_ADDR c,
    uint32_t n, uint32_t slice, uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    BSliceProbe<float> probe;
    probe.Run(a,b,packed,c,n,slice,mode,pipe);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void c021_bslice_fp32_do(uint32_t bd,void *stream,uint8_t *a,
    uint8_t *b,uint8_t *packed,uint8_t *c,uint32_t n,uint32_t slice,uint32_t mode) {
    c021_bslice_fp32<<<bd,nullptr,stream>>>(a,b,packed,c,n,slice,mode);
}
#endif
```

### `fresh_default/kernel/pybind11.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
using Fn = void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t,uint32_t);
extern "C" void c021_bslice_fp16_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t,uint32_t);
extern "C" void c021_bslice_fp32_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t,uint32_t);
static uint8_t *ptr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
}
void run(const at::Tensor &a,const at::Tensor &b,const at::Tensor &packed,
         const at::Tensor &c,uint32_t slice,uint32_t mode) {
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 &&
        a.device()==b.device() && a.device()==packed.device() && a.device()==c.device(),"same NPU required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && packed.is_contiguous() && c.is_contiguous(),"contiguous required");
    TORCH_CHECK(a.scalar_type()==b.scalar_type() && a.scalar_type()==packed.scalar_type() &&
        (a.scalar_type()==at::kHalf || a.scalar_type()==at::kFloat) && c.scalar_type()==at::kFloat,"fp16/fp32 inputs, fp32 output required");
    TORCH_CHECK(b.dim()==2 && b.size(0)==32 && (b.size(1)==16 || b.size(1)==32),"B must be [32,16/32]");
    const uint32_t n=b.size(1);
    TORCH_CHECK(a.numel()==256 && packed.numel()==48*n && c.numel()==512+16*n && slice<=1 && mode<=6,"bounded shape/slice/mode required");
    Fn fn=a.scalar_type()==at::kHalf ? c021_bslice_fp16_do : c021_bslice_fp32_do;
    fn(1,c10_npu::getCurrentNPUStream().stream(false),ptr(a),ptr(b),ptr(packed),ptr(c),n,slice,mode);
}
PYBIND11_MODULE(_c021_bslice_probe,m) {m.def("run",&run);}
```

### `run_panel.py`

```python
"""Prepared C021 B-path experiment. Do not run before root's post-C033 gate."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--gate", choices=["root-authorized-after-c033"], required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "fresh_build_receipt.json").read_text())
    if receipt.get("status") != "built" or not receipt.get("aic_compile_database_records"):
        raise RuntimeError("Need actual successful fresh AIC build receipt")
    kernel = root / "fresh_default/kernel"
    if receipt["source_sha256"] != {x.name: sha(x) for x in sorted(kernel.iterdir()) if x.is_file()}:
        raise RuntimeError("Source differs from actual build receipt")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "outputs").mkdir()
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    sys.path.insert(0, str(kernel / "build"))
    import _c021_bslice_probe as ext
    module = Path(ext.__file__).resolve()
    built = [a for a in receipt["artifacts"] if a["path"].endswith(module.name)]
    if len(built) != 1 or built[0]["sha256"] != sha(module):
        raise RuntimeError("Loaded module is not bound to successful build")
    def raw(t):
        return t.contiguous().view(torch.uint8).numpy().tobytes()
    def canonical(t):
        data = raw(t)
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "encoding": "IEEE754 little-endian contiguous bytes", "data_hex": data.hex(),
                "tensor_sha256": hashlib.sha256(data).hexdigest()}
    # Six direct controls first. Each diagnostic has both slices unless explicitly
    # selected only for the i=1 colstride/helper contrast.
    points = [(d, n, i, 0) for d, n in [("float16", 16), ("float16", 32), ("float32", 16)] for i in [0, 1]]
    points += [("float16", 16, i, 1) for i in [0, 1]]
    points += [("float16", 32, i, mode) for mode in [1, 2, 3, 4] for i in [0, 1]]
    points += [("float16", 32, 1, mode) for mode in [5, 6]]
    points += [("float32", 16, i, mode) for mode in [1, 2, 4] for i in [0, 1]]
    points += [("float32", 16, 1, 6)]
    if len(points) != 25 or len(set(points)) != 25:
        raise RuntimeError("Frozen point list is not the declared 25 points")
    metadata = {
        "schema_version": "c021.b_path_predeclared_panel.v1",
        "phase": "new_diagnostic_not_repaired_candidate", "pid": os.getpid(),
        "started_unix": time.time(), "python": sys.version, "platform": platform.platform(),
        "device_name": torch.npu.get_device_name(args.device), "logical_device": args.device,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "module_path": str(module), "module_sha256": sha(module), "runner_sha256": sha(__file__),
        "build_receipt_sha256": sha(root / "fresh_build_receipt.json"),
        "source_sha256": receipt["source_sha256"], "points": points, "repeats": 2,
        "repeat_independence": "Repeated same bytes, not independent inputs.",
        "reference": "CPU float64 A @ each B[16*i:16*(i+1)] then float32; exact dyadic operands",
        "M": 16, "baseK": 16, "wide_kLen": 32, "block_dim": 1,
        "A_path": "identical direct single 16x16 tile for every variant and both B slices",
        "sentinel": 12345.5, "guard_prefix_suffix_float32_elements": [256, 256],
        "B1_tail_value": -0.75,
        "pack_model": "offset(r,c)=(c//C0)*32*C0+r*C0+c%C0, C0=32/sizeof(T); model pending device contrast",
        "performance_or_official_claims": False,
    }
    write(out / "metadata.json", metadata)
    # Sylvester H16/4 is dense and invertible; it exposes all 16 B rows without
    # a one-hot input. B has unique magnitudes and coordinate-dependent signs.
    hadamard = [[(-0.25 if (r & k).bit_count() % 2 else 0.25) for k in range(16)] for r in range(16)]
    inputs = {}
    for name, n in [("float16", 16), ("float16", 32), ("float32", 16)]:
        dtype = getattr(torch, name)
        a = torch.tensor(hadamard, dtype=dtype)
        b = torch.tensor([[(-1 if (r * 5 + c * 3) % 7 < 3 else 1) * (r * 32 + c + 1) / 4096
                           for c in range(n)] for r in range(32)], dtype=dtype)
        c0 = 32 // a.element_size()
        packed = torch.full((48 * n,), -0.75, dtype=dtype)
        for r in range(32):
            for c in range(n):
                packed[(c // c0) * 32 * c0 + r * c0 + c % c0] = b[r, c]
        refs = [(a.double() @ b[16 * i:16 * (i + 1)].double()).float() for i in [0, 1]]
        inputs[f"{name}_n{n}"] = {"a": a, "b": b, "packed": packed, "ref0": refs[0], "ref1": refs[1]}
    torch.save(inputs, out / "inputs_reference.pt")
    write(out / "inputs_canonical.json", {k: {n: canonical(t) for n, t in v.items()} for k, v in inputs.items()})
    metadata["input_archive_sha256"] = sha(out / "inputs_reference.pt")
    metadata["input_canonical_sha256"] = sha(out / "inputs_canonical.json")
    write(out / "metadata.json", metadata)
    device_inputs = {k: {n: v[n].to(f"npu:{args.device}") for n in ["a", "b", "packed"]} for k, v in inputs.items()}
    rows, first_hashes = [], {}
    def maps():
        path = Path(f"/proc/{os.getpid()}/maps")
        return path.read_text() if path.exists() else "unavailable"
    (out / "process_maps_before.txt").write_text(maps())
    for repeat in range(2):
        for name, n, i, mode in points:
            family = f"{name}_n{n}"
            tensors, dn = inputs[family], device_inputs[family]
            key = f"r{repeat}_{family}_i{i}_mode{mode}"
            pending = {"key": key, "repeat": repeat, "dtype": name, "baseN": n, "slice": i,
                       "mode": mode, "phase": "output_initialization",
                       "input_sha256": {k: canonical(t)["tensor_sha256"] for k, t in tensors.items()}}
            write(out / "pending.json", pending)
            try:
                c = torch.full((512 + 16 * n,), metadata["sentinel"], dtype=torch.float32, device=f"npu:{args.device}")
                torch.npu.synchronize()
                pending["phase"] = "kernel_launch_and_completion"
                write(out / "pending.json", pending)
                started = time.monotonic()
                ext.run(dn["a"], dn["b"], dn["packed"], c, i, mode)
                torch.npu.synchronize()
                watchdog = time.monotonic() - started
                y = c.cpu()
                active, ref = y[256:256 + 16 * n].reshape(16, n), tensors[f"ref{i}"]
                guard = torch.cat([y[:256], y[256 + 16 * n:]])
                mismatch = (active != ref).nonzero()[:8].tolist()
                output = canonical(y)
                write(out / "outputs" / f"{key}.json", output)
                torch.save({"output_with_guards": y}, out / "outputs" / f"{key}.pt")
                point = (name, n, i, mode)
                row = dict(pending, phase="complete", numeric_total=active.numel(),
                           equal_elements=int((active == ref).sum()), max_abs_diff=float((active.double()-ref.double()).abs().max()),
                           finite_count=int(torch.isfinite(active).sum()), nan_count=int(torch.isnan(active).sum()), inf_count=int(torch.isinf(active).sum()),
                           guard_total=guard.numel(), guard_equal=int((guard == metadata["sentinel"]).sum()),
                           output_sha256=output["tensor_sha256"], output_json=f"outputs/{key}.json", output_archive=f"outputs/{key}.pt",
                           output_archive_sha256=sha(out / "outputs" / f"{key}.pt"), watchdog_seconds_not_latency=watchdog,
                           first_mismatch_indices=mismatch,
                           first_mismatch_values=[{"actual": float(active[tuple(ix)]), "reference": float(ref[tuple(ix)])} for ix in mismatch])
                row["numeric_pass"] = row["equal_elements"] == row["numeric_total"]
                row["guard_pass"] = row["guard_equal"] == row["guard_total"]
                row["repeat_equal"] = repeat == 0 or first_hashes[point] == output["tensor_sha256"]
                if repeat == 0:
                    first_hashes[point] = output["tensor_sha256"]
                rows.append(row)
                with (out / "records.jsonl").open("a") as stream:
                    stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
                print(json.dumps(clean(row), allow_nan=False), flush=True)
                if not row["guard_pass"] or row["finite_count"] != row["numeric_total"] or not row["repeat_equal"] or (mode == 0 and not row["numeric_pass"]):
                    write(out / "stop.json", {"reason": "guard, nonfinite, nondeterminism, or direct-control failure", "row": row})
                    raise SystemExit("Stop per predeclared protocol; preserve partial rows")
            except Exception:
                write(out / "runtime_failure.json", dict(pending, traceback=traceback.format_exc()))
                raise
    (out / "process_maps_after.txt").write_text(maps())
    write(out / "summary.json", {"rows": len(rows), "expected_rows": 50, "completed_declared_batch": len(rows) == 50,
                                "numeric_failures": sum(not r["numeric_pass"] for r in rows), "guard_failures": sum(not r["guard_pass"] for r in rows),
                                "repeat_failures": sum(not r["repeat_equal"] for r in rows), "finished_unix": time.time(),
                                "interpretation": "Numeric diagnostic failures are retained; infer mechanism only from declared same-input contrasts."})


if __name__ == "__main__":
    main()
```

## 预防

- 先列清物理布局与每个参数的单位，再分别推导基址和跨平面推进，不能只把 `N` 换成 `C0` 就结束。
- 查清实际选中的重载；显式 template 与普通重载可能走不同 API，传入 `colC0Stride` 不表示 half 重载会使用它。
- 把宽缓存中的各片单独投影，与仅复制该片的控制组比较；同时保留初始化尾部、完整 guard 和输出字节。
- 精确输入与小规模诊断只能缩小机制，不能代替完整算子、尾块和随机输入回归；推广前重新核对目标 CANN 的 API/布局/资源约束。

## 验证信号

这些信号来自全新构建的 B-only Cube 投影及其完整原始输出，不引用历史成功叙述；每条范围都限定到这次源文件、ND2NZ 参数和实际设备。

- 实际 Ascend910B4 / CANN 9.1.0 的 AIC 编译数据库包含两个入口的 `dav-c220-cube` 与默认 auto-sync，加载模块同实际构建产物一致，并出现在该次进程的前后映射中。（来源平台：a2，Ascend910B4；证据等级：artifact_verified）
- 25 个预声明点各按相同输入执行两次，共 50 次；12 次直接片控制均精确相等，14 次诊断数值失败全部保留；所有 active 输出 finite、每次 512 个 guard 元素均相等、重复输出字节相同。一次实验不因重放而升级证据等级。（来源平台：a2，Ascend910B4；证据等级：artifact_verified）
- FP16 N32，mode 1→2 只改基址：第一片输出相同，第二片 512 个输出全部改变，但两者都只有 256/512 相等；mode 2→3 只改 half 跨平面步距后，两片都从 256/512 到 512/512，并与各自 mode 0 输出字节相同。（来源平台：a2，Ascend910B4；证据等级：artifact_verified）
- FP32 N16，第二片 mode 1→2 只改基址，27/256 到 256/256；第一片两种基址均为零，其输出不变。（来源平台：a2，Ascend910B4；证据等级：artifact_verified）
- 所测 ND2NZ 与 raw 对照在两片上输出完全相同；FP16 第二片 generic3D/col32 与修正后的 transpose2D 输出相同，而 generic col16 为 256/512；FP32 第二片 generic col16 为 128/256。（来源平台：a2，Ascend910B4；证据等级：artifact_verified）

下表完整列出一次输入重放中的 25 个点，另一次的输出字节逐点一致；每行数字均来自同一 a2 设备上的 artifact_verified 观测。

| dtype / B 宽度 | 片 i | mode | 相等元素 / 总数 | 最大绝对差 | 来源平台 / 证据等级 |
|---|---:|---:|---:|---:|---|
| float16 / 16 | 0 | 0 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 16 | 1 | 0 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 0 | 0 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 0 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 0 | 0 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 1 | 0 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 16 | 0 | 1 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 16 | 1 | 1 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 0 | 1 | 256/512 | 0.68701171875 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 1 | 256/512 | 0.68701171875 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 0 | 2 | 256/512 | 0.68701171875 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 2 | 256/512 | 0.695556640625 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 0 | 3 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 3 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 0 | 4 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 4 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 5 | 512/512 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float16 / 32 | 1 | 6 | 256/512 | 0.695556640625 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 0 | 1 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 1 | 1 | 27/256 | 3.1953125 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 0 | 2 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 1 | 2 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 0 | 4 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 1 | 4 | 256/256 | 0.0 | a2（Ascend910B4）/ artifact_verified |
| float32 / 16 | 1 | 6 | 128/256 | 0.678466796875 | a2（Ascend910B4）/ artifact_verified |

原始完成摘要逐字如下；它明确将数值失败保留下来，没有把批次完成等同于全部精度通过。

```json
{
  "rows": 50,
  "expected_rows": 50,
  "completed_declared_batch": true,
  "numeric_failures": 14,
  "guard_failures": 0,
  "repeat_failures": 0,
  "finished_unix": 1792341612.195423,
  "interpretation": "Numeric diagnostic failures are retained; infer mechanism only from declared same-input contrasts."
}
```

## 失效边界

本结论的最窄范围是这次 B-only 投影，不是原完整矩阵乘的修复配方。未验证任意 padding、ND 批矩阵跨度、转置布局、其他元素宽度、其他平台或 CANN 版本；没有直接读回 L1/L0B，因此 raw 模型的吻合只能按测试参数陈述。FP16 N16 不能区分两种基址公式，第一片基址为零也不能说明第二片正确。

未决项：先把本诊断定位到的 B 读取改动放回原完整 kernel，再分别验证 A 打包、循环/同步、累加、尾块和完整输入覆盖；在此之前不能声称历史成功已认证。独立输入、其他布局和平台均未尝试。性能未测，运行器的 watchdog 耗时不能作为时延、吞吐或加速比。

## 坏实践

- 把 `i*baseK*N` 或 `i*baseK*C0` 写成无布局前提的通用偏移规则；两种数式都必须与真实 NZ 平面跨度和下游 helper 对齐。
- 只改基址，保留忽略宽 `colC0Stride` 的 half helper，然后用“部分列相等”当修复成功。
- 看到 generic3D 或 2D transpose 的某个控制组通过，就跳过其参数空间和实际重载；本次 col16 失败是反例。
- 把两个相同输入重放算成两个独立复现，把数值诊断失败隐藏为环境问题，或把投影通过当成完整算子通过。

## 误诊

“问题一定全在 A 侧”“只换 C0 基址即可”“旧 helper 注释说通用所以无需检查”都只是候选解释。固定 A 的实测只能隔离并支持本次 B 地址函数中的差异，不能证明原完整 kernel 没有 A、调度或同步问题。未经执行的 bf16/其他布局与历史口头成功叙述不进入验证信号。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/c021/execution_binding_public.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/metadata_public.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/summary_public.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/records_public.jsonl`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/inputs_canonical.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/verify_result.json#/contrasts`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/provenance_public.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/MANIFEST.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c021/README.md`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
