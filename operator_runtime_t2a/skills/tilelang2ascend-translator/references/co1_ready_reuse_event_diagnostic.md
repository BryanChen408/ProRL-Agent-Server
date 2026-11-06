---
type: "Runbook"
title: "CO1 复用诊断：追踪就绪与回收双向事件"
description: "在纯 AIC 的 CO1/Fixpipe 双缓冲中，沿安装头文件核对 M_FIX 就绪和 FIX_M 复用事件，并把屏障面板通过限定为观测。"
tags: [task.debugging, topic.memory, topic.data_movement]
status: stable
sources: []
created_at: '2026-10-18T12:34:27Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# CO1 复用诊断：追踪就绪与回收双向事件


## 现象

在 Ascend910B4 上用 kernel 直调检查手写 MatmulKernel 的 CO1 双缓冲时，CopyOut 在 DeQue 后、Fixpipe 前以及 Fixpipe 后、FreeTensor 前各有一个可选 PipeBarrier<PIPE_ALL>。旧探针通过一个运行时 mode 选择屏障；为看清当前编译路径，新探针把两个条件改成模板参数和 if constexpr，生成四个独立 kernel symbol 并编入同一个扩展。默认构建后，对空生命周期、首次排空、两个缓冲尚未回绕、发生回绕和连续多 tile 分别运行，四种屏障形式全部给出数值与 guard 通过。该现象没有出现可归因的失败，问题是如何解释“全部通过”，以及究竟在哪一步保护 L0C 的生产、搬出和再分配。

## 适用场景

工程形态为纯 AIC kernel 直调；API 路径为 TQue<CO1, 2> 的 AllocTensor/EnQue/DeQue/FreeTensor、Mmad 和 FixpipeParamsV220。已观测的平台为 a2（Ascend910B4），安装路径为 CANN 9.1.0，实际 AIC 目标为 dav-c220-cube。本规则的适用粒度为 operator_family，仅限具有相同 CO1→Fixpipe→GM 队列生命周期、且能取得实际安装头文件和构建产物绑定的矩阵计算探针。路线优先级为 recommended_fastpath：推荐的动作是先追踪就绪与复用两条队列事件，再解释屏障对照；这里没有推荐移除或添加宽屏障。其他平台、其他版本、手工管理 L0C、不同队列 mask/depth、跨核通信均未验证。

下面内联保留诊断夹具的完整核心源码及队列机制函数；其中固定容量、基本块和输入生成常量只定义本探针的验证域，不是生产算子的通用 tiling 建议。

## 判别方法

单一判别结论：在这份安装快照中，CO1 复用的源码调用位置是 FreeBuffer 调用 SetFlag<HardEvent::FIX_M>，后续同一物理 buffer 的 AllocBuffer 在保存的事件 ID 有效时调用 WaitFlag<HardEvent::FIX_M>；只看 FreeTensor 的 FREE 状态或两个物理 buffer 的数量，会漏掉事件 ID 的生产与消费位置。

1. 从实际生成的 AIC compile_commands 和 verbose 构建输出确认编译目标及 flags。这里的四个编译单元都使用 -g、-O3、dav-c220-cube、--cce-aicore-only 和 --cce-auto-sync；host 的 flags 不能替代 AIC flags。--cce-auto-sync 在命令中出现只能说明构建配置，不能独立证明 M↔FIX 依赖受到保护，本探针没有 off sweep 或最终 ISA 观测。
2. 对照安装头文件：TQue<CO1> 继承 TQueBind<GetBufferLogicPos(CO1,true), GetBufferLogicPos(CO1,false)>，在该快照对应 CO1/CO2；GetPhyType 在实际架构分支把两端映射成 L0C/GM。GetQueEvt(L0C,GM,true) 是 M_FIX，反向是 FIX_M。
3. 顺着完整函数核对就绪边：EnQue(c1Local) → SetFlag<M_FIX>(enQueEvtID)，DeQue() → WaitFlag<M_FIX>(ptr->enQueEvtID) → ReleaseEventID。这样追踪的是安装源码为 Cube 生产和 Fixpipe 消费选择的 HardEvent 调用。
4. 顺着完整函数核对复用边：Fixpipe → FreeTensor → FreeBuffer → 分配并 SetFlag<FIX_M>(ptr->freeBufEvtID) → state=FREE。下一次 AllocBuffer 选中同一 TBufType 后，在 ret->freeBufEvtID 有效时 WaitFlag<FIX_M>、释放并清空该 ID，然后返回可用 buffer。FreeTensor 本身没有立即调用 WaitFlag；它也不是只把 state 改成 FREE。
5. 用四个编译期 symbol 与相同已保存输入做数值/guard 对照，显式分开空输出与非空输出。驱动在 ext.run 后执行 torch.npu.synchronize()，再保存输出和比较；最后一次 Flush 中的单独 FreeTensor 不能被当作全局完成证明。只把原始记录覆盖的模式作为观测，遇到全通过仍保留未决边界。

这里的源级解释与运行观察相互限定：完整队列函数与类型选择给出安装快照中的 HardEvent 调用和事件 ID 生命周期，运行记录显示该夹具在当前默认构建下完成。本规则仅判定安装源码中的 HardEvent 调用和事件 ID 保存、释放与消费生命周期；不认证下层通用 dispatcher 的 PIPE 展开、最终指令发射、硬件等待完成，也不以此解释四种模式全部通过的硬件原因。未覆盖调度下是否存在竞争仍未验证。

本文保留完整 TQue/TQueBind、逻辑与物理位置映射、GetQueEvt、EnQue/DeQue 和分配/回收函数。更下层 SetFlagInternal、WaitFlagInternal、SetFlagImpl、WaitFlagImpl 四个通用展开单元作为支持材料保留原始完整字节及锚点，未逐字内联；因此正文结论止于上述 HardEvent 调用边界，不把它们展开成 PIPE 或最终硬件等待证明。

## 设计决策

采用编译期 BarrierMode，是为了让四个实现有独立 kernel symbol，可从生成源码和真实 AIC 命令绑定到同一个新扩展；旧运行时 mode 的结果仅为背景。保留两个 CO1 buffer 与一 tile 延迟 CopyOut，再用 Flush 排空末尾，目的是覆盖生产、消费与物理 buffer 回绕的生命周期。区分空输出和非空输出，防止以没有元素参与比较的 numeric_pass 代替数值验证。

源码中 “Two CO1 buffers guarantee...” 和 “Ensure ...” 等原注释原样保留，属于实现作者的解释，不是独立证据。两个 buffer 本身不证明安全；源级解释来自完整队列函数中的 HardEvent 类型选择、调用和事件 ID 保存/消费生命周期。原件里关于 pure AIC auto-sync 的文字也只按记录保留，不将构建 flag 当作 Cube 自动插入同步的证据。路线为 recommended_fastpath 的诊断步骤；没有观测支持的更改屏障方案未被选为配方。

## 系统信息

实测使用 Ascend910B4、容器内 logical device 0，ASCEND_RT_VISIBLE_DEVICES=2。设备绑定与运行身份只用于这次 panel。compiler 是 CANN 9.1.0 安装下的 bisheng，实际 AIC 编译库与 host PyBind 扩展通过源码 hash、生成文件和加载模块 hash 绑定。以下内联完整 matmul.h、matmul_tile.h、四个 probe、host dispatch、run_panel、observer，以及必要安装队列函数和事件映射；原始逐次观测与构建/运行绑定材料保留在配套审阅证据。两段进程各自的 metadata PID、summary PID 与 observer PID 一致，实际映射的 libruntime.so、libascendcl.so 解析到 cann-9.1.0，映射文件 hash 与安装版本原文已保存；版本范围来自本次实际运行映射与安装记录。构建工具属于配套依赖，不把框架本身当作本卡机制。安装快照没有固定上游 revision/blob，这是一份本地实验快照的诊断规则，不声明所有版本的 API 规范。

## 完整代码

以下代码逐字来自这次实验的完整原件，未把实际实现改写为示意代码。重用时需在具有相同 CANN、torch/torch_npu、构建工具和设备依赖的独立目录准备文件，并按脚本现有参数与资源授权运行。run_panel 还读取 fresh_build_receipt.json 作绑定记录；配套材料保留原构建收据及公开转换说明，新的复查应由实际新构建产生新收据，不能拿保存收据冒充新构建。本次策展只读取已有记录。

### fresh_default/kernel/matmul.h — complete MatmulKernel class and every definition

```cpp
#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
class MatmulKernel {
    static_assert(std::is_same<aType, bType>::value, "aType and bType must be the same type");

    // Nz fractal c0 size: 32 bytes / element size
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k, uint32_t lda, uint32_t ldb,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                uint32_t l1Prefetch,
                                AscendC::TPipe &pipe);
    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<bType> &bBlock,
                                        const AscendC::GlobalTensor<cType> &cBlock,
                                        uint32_t cStride = 0);
    __aicore__ inline void Flush();

private:
    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<bType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<bType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal);
    __aicore__ inline void CopyOut();

private:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 2> outQueueCO1;

    uint32_t l1Prefetch_;
    uint32_t baseM_, baseN_, baseK_;
    uint32_t baseMK_, baseKN_, baseMN_;
    uint32_t k;
    uint32_t aDValue, bDValue;

    // Ping-pong state for L0C double buffering: one buffer is being written back
    // via Fixpipe while the next MN tile is being computed in the other buffer.
    AscendC::LocalTensor<cType> pendingC1_;
    bool hasPendingC1_;
    AscendC::GlobalTensor<cType> pendingCBlock_;
    uint32_t pendingCStride_;

    AscendC::GlobalTensor<cType> cBlock_;
    uint32_t cStride_;
};

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb,
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t l1Prefetch,
    AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);

    l1Prefetch_ = l1Prefetch;
    baseM_ = baseM;
    baseN_ = baseN;
    baseK_ = baseK;
    baseMK_ = baseM_ * baseK_;
    baseKN_ = baseK_ * baseN_;
    baseMN_ = baseM_ * baseN_;
    this->k = k;
    aDValue = lda;
    bDValue = ldb;

    pipe.InitBuffer(inQueueA1, 2, baseMK_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK_ * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN_ * l1Prefetch_ * sizeof(bType));
    pipe.InitBuffer(inQueueB2, 2, baseKN_ * sizeof(bType));
    pipe.InitBuffer(outQueueCO1, 2, baseMN_ * sizeof(cType));

    hasPendingC1_ = false;
    cStride_ = 0;
    pendingCStride_ = 0;
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                                                        const AscendC::GlobalTensor<bType> &bBlock,
                                                                        const AscendC::GlobalTensor<cType> &cBlock,
                                                                        uint32_t cStride)
{
    cBlock_ = cBlock;
    cStride_ = (cStride == 0) ? baseN_ : cStride;

    AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
    uint32_t kTiles = k / baseK_;

    for (uint32_t outer = 0; outer < kTiles; outer += l1Prefetch_) {
        uint32_t count = (kTiles - outer < l1Prefetch_) ? (kTiles - outer) : l1Prefetch_;
        uint32_t kLen = count * baseK_;

        CopyA(aBlock[outer * baseK_], kLen);
        CopyB(bBlock[outer * baseK_ * bDValue], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK_, baseM_);
            // 使用 baseKN_ 作为 B 块在 L1(Nz) 中的步长，保证 half(C0=16) 与 float(C0=8) 都正确。
            SplitB(b1Local, i * baseKN_, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);

    // Flush the previous tile's Fixpipe while this tile's result stays queued.
    // Two CO1 buffers guarantee that Mmad of the next tile cannot overwrite the
    // buffer still being read by Fixpipe.
    if (hasPendingC1_) {
        CopyOut();
    }
    pendingC1_ = c1Local;
    pendingCBlock_ = cBlock_;
    pendingCStride_ = cStride_;
    hasPendingC1_ = true;
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::Flush()
{
    if (hasPendingC1_) {
        CopyOut();
        hasPendingC1_ = false;
    }
}


template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM_, kLen, aDValue);
    inQueueA1.EnQue(a1Local);
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::CopyB(
    const AscendC::GlobalTensor<bType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN_, bDValue);
    inQueueB1.EnQue(b1Local);
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM_, baseK_, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::SplitB(
    const AscendC::LocalTensor<bType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK_, baseN_, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::Compute(
    const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM_;
    mmadParams.n = baseN_;
    mmadParams.k = baseK_;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

template <typename aType, typename bType, typename cType, uint32_t BarrierMode>
__aicore__ inline void MatmulKernel<aType, bType, cType, BarrierMode>::CopyOut()
{
    auto c1Local = outQueueCO1.DeQue<cType>();
    // Ensure the Mmad that produced this CO1 buffer is fully done before
    // Fixpipe starts reading from L0C; otherwise L0C read/write conflicts
    // between cube and fixpipe can occur under pipelined execution.
    if constexpr (BarrierMode & 1) { AscendC::PipeBarrier<PIPE_ALL>(); }
    FixpipeNzL0cToNdGm(pendingCBlock_, c1Local, baseM_, baseN_, pendingCStride_);
    // Ensure Fixpipe finishes before the CO1 buffer is returned to the queue
    // and potentially reused by the next ComputeBlock.
    if constexpr (BarrierMode & 2) { AscendC::PipeBarrier<PIPE_ALL>(); }
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_CUSTOM_H
```

### fresh_default/kernel/matmul_tile.h — complete MatmulTile data movement dependencies

```cpp
#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)
// m % 16 == 0 && n % c0 == 0
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
// m % 16 == 0 && k % c0 == 0
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


// L1(Nz) -> L0B(Zn)
// 参数只有k,n, 只适用于整个基本块,
template <typename T>
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

// L1(Nz) -> L0B(Zn)
// 参数只有k,n, 只适用于整个基本块
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<int8_t> &dst,
                                       const AscendC::LocalTensor<int8_t> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    static constexpr uint32_t FRAC_M  = 16;
    static constexpr uint32_t FRAC_N  = 16;
    static constexpr uint32_t C0_SIZE = 32;
    static constexpr uint32_t FRAC_K  = C0_SIZE;
    uint32_t dstOffset = FRAC_K * n;
    uint32_t srcOffset = FRAC_K * C0_SIZE;
    // Nz -> Zn
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.repeatTimes = n / C0_SIZE;
    loadDataParams.srcStride = colC0Stride / FRAC_K;
    loadDataParams.dstGap = 1;
    loadDataParams.dstFracGap = 0;
    for (int i = 0; i < k / FRAC_K; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * dstOffset], src[i * srcOffset], loadDataParams);
    }
}



// L0C(Nz) -> GM(ND)
// 参数只有m, n，只适用于整个基本块搬出，如果不够基本块，需要补齐stride
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                      const AscendC::LocalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t dstStride = 0)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = dstStride == 0 ? n : dstStride;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
```

### fresh_default/kernel/probe_m0.cpp — compile-time mode 0: kernel and host launcher

```cpp
#include "matmul.h"
extern "C" __global__ __aicore__ void l0c_c036_m0(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> ga, gb;
    AscendC::GlobalTensor<float> gc;
    ga.SetGlobalBuffer((__gm__ half*)a, 32*16*32);
    gb.SetGlobalBuffer((__gm__ half*)b, 32*32*16);
    gc.SetGlobalBuffer((__gm__ float*)c, 32*16*16);
    MatmulKernel<half,half,float,0> mm;
    mm.Init(32,32,16,16,16,16,1,pipe);
    for (uint32_t i=0; i<tiles; ++i) {
        mm.ComputeBlock(ga[i*16*32], gb[i*32*16], gc[i*16*16], 16);
    }
    mm.Flush();
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void l0c_c036_m0_do(uint32_t bd, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint32_t tiles) {
    l0c_c036_m0<<<bd,nullptr,stream>>>(a,b,c,tiles);
}
#endif
```

### fresh_default/kernel/probe_m1.cpp — compile-time mode 1: kernel and host launcher

```cpp
#include "matmul.h"
extern "C" __global__ __aicore__ void l0c_c036_m1(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> ga, gb;
    AscendC::GlobalTensor<float> gc;
    ga.SetGlobalBuffer((__gm__ half*)a, 32*16*32);
    gb.SetGlobalBuffer((__gm__ half*)b, 32*32*16);
    gc.SetGlobalBuffer((__gm__ float*)c, 32*16*16);
    MatmulKernel<half,half,float,1> mm;
    mm.Init(32,32,16,16,16,16,1,pipe);
    for (uint32_t i=0; i<tiles; ++i) {
        mm.ComputeBlock(ga[i*16*32], gb[i*32*16], gc[i*16*16], 16);
    }
    mm.Flush();
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void l0c_c036_m1_do(uint32_t bd, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint32_t tiles) {
    l0c_c036_m1<<<bd,nullptr,stream>>>(a,b,c,tiles);
}
#endif
```

### fresh_default/kernel/probe_m2.cpp — compile-time mode 2: kernel and host launcher

```cpp
#include "matmul.h"
extern "C" __global__ __aicore__ void l0c_c036_m2(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> ga, gb;
    AscendC::GlobalTensor<float> gc;
    ga.SetGlobalBuffer((__gm__ half*)a, 32*16*32);
    gb.SetGlobalBuffer((__gm__ half*)b, 32*32*16);
    gc.SetGlobalBuffer((__gm__ float*)c, 32*16*16);
    MatmulKernel<half,half,float,2> mm;
    mm.Init(32,32,16,16,16,16,1,pipe);
    for (uint32_t i=0; i<tiles; ++i) {
        mm.ComputeBlock(ga[i*16*32], gb[i*32*16], gc[i*16*16], 16);
    }
    mm.Flush();
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void l0c_c036_m2_do(uint32_t bd, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint32_t tiles) {
    l0c_c036_m2<<<bd,nullptr,stream>>>(a,b,c,tiles);
}
#endif
```

### fresh_default/kernel/probe_m3.cpp — compile-time mode 3: kernel and host launcher

```cpp
#include "matmul.h"
extern "C" __global__ __aicore__ void l0c_c036_m3(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> ga, gb;
    AscendC::GlobalTensor<float> gc;
    ga.SetGlobalBuffer((__gm__ half*)a, 32*16*32);
    gb.SetGlobalBuffer((__gm__ half*)b, 32*32*16);
    gc.SetGlobalBuffer((__gm__ float*)c, 32*16*16);
    MatmulKernel<half,half,float,3> mm;
    mm.Init(32,32,16,16,16,16,1,pipe);
    for (uint32_t i=0; i<tiles; ++i) {
        mm.ComputeBlock(ga[i*16*32], gb[i*32*16], gc[i*16*16], 16);
    }
    mm.Flush();
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void l0c_c036_m3_do(uint32_t bd, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint32_t tiles) {
    l0c_c036_m3<<<bd,nullptr,stream>>>(a,b,c,tiles);
}
#endif
```

### fresh_default/kernel/pybind11.cpp — complete host dispatch

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
using Fn = void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void l0c_c036_m0_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void l0c_c036_m1_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void l0c_c036_m2_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void l0c_c036_m3_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
static uint8_t* ptr(const at::Tensor& t) {return static_cast<uint8_t*>(const_cast<void*>(t.data_ptr()));}
void run(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,uint32_t mode,uint32_t tiles) {
    TORCH_CHECK(a.scalar_type()==at::kHalf && b.scalar_type()==at::kHalf && c.scalar_type()==at::kFloat,"fp16/fp16/fp32 required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),"contiguous required");
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && a.device()==b.device() && a.device()==c.device(),"same NPU device required");
    TORCH_CHECK(a.numel()==32*16*32 && b.numel()==32*32*16 && c.numel()==32*16*16 && mode<=3 && tiles<=32,"bounded shapes/mode/tiles required");
    Fn fn[] = {l0c_c036_m0_do,l0c_c036_m1_do,l0c_c036_m2_do,l0c_c036_m3_do};
    auto stream=c10_npu::getCurrentNPUStream().stream(false);
    fn[mode](1,stream,ptr(a),ptr(b),ptr(c),tiles);
}
PYBIND11_MODULE(_c036_default_probe,m) {m.def("run",&run);}
```

### run_panel.py — complete driver and numeric/guard oracle

```python
"""Bounded C036 panel; execution requires the root's post-C014 NPU gate."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, choices=[1, 2], default=1)
    parser.add_argument("--repeat-start", type=int, choices=[0, 1], default=0)
    parser.add_argument("--gate", choices=["root-authorized-after-c014"], required=True)
    args = parser.parse_args()
    if args.repeat_start + args.repeats > 3:
        parser.error("at most three total repeats")
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "outputs").mkdir()
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    sys.path.insert(0, str(root / "fresh_default/kernel/build"))
    import _c036_default_probe as ext
    kernel = root / "fresh_default/kernel"
    metadata = {
        "schema_version": "c036.compiletime_default_panel.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "module_path": ext.__file__,
        "module_sha256": sha(ext.__file__),
        "script_sha256": sha(__file__),
        "source_sha256": {p.name: sha(p) for p in sorted(kernel.iterdir()) if p.is_file()},
        "build_receipt_sha256": sha(root / "fresh_build_receipt.json"),
        "process_pid": os.getpid(), "process_ppid": os.getppid(), "process_start_unix": time.time(),
        "logical_device": args.device,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "barrier_modes": [3, 1, 2, 0],
        "tile_counts": [0, 1, 2, 3, 32],
        "input_families": ["old_arithmetic", "tile_distinct_signed"],
        "repeats": args.repeats,
        "repeat_start": args.repeat_start,
        "repeat_independence": "Each repeat reuses identical saved input bytes; repeats are not independent inputs.",
        "reference": "CPU float64 bmm then float32; exact small dyadic operands",
        "sentinel": 12345.5,
        "fixed_resources": {"block_dim": 1, "capacity_tiles": 32, "M": 16, "N": 16, "K": 32,
                            "baseK": 16, "l1Prefetch": 1, "CO1_physical_buffers": 2},
        "auto_sync": "one actual default compiler configuration; effective AIC flags in build receipt",
        "no_performance_or_official_evaluation": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    a0 = ((torch.arange(32 * 16 * 32).reshape(32, 16, 32) * 7 + 3) % 17 - 8).half() / 16
    b0 = ((torch.arange(32 * 32 * 16).reshape(32, 32, 16) * 11 + 1) % 19 - 9).half() / 16
    t = torch.arange(32).reshape(32, 1, 1)
    m = torch.arange(16).reshape(1, 16, 1)
    k = torch.arange(32).reshape(1, 1, 32)
    a1 = ((t * 5 + m * 3 + k * 7) % 23 - 11).half() / 16
    k2 = torch.arange(32).reshape(1, 32, 1)
    n = torch.arange(16).reshape(1, 1, 16)
    b1 = ((t * 11 + k2 * 5 + n * 3) % 29 - 14).half() / 16
    inputs = {}
    for name, a, b in [("old_arithmetic", a0, b0), ("tile_distinct_signed", a1, b1)]:
        inputs[name] = {"a": a, "b": b, "ref": torch.bmm(a.double(), b.double()).float()}
    torch.save(inputs, out / "inputs_reference.pt")
    metadata["inputs_archive_sha256"] = sha(out / "inputs_reference.pt")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
        for name, tensors in inputs.items():
            an, bn = tensors["a"].to(f"npu:{args.device}"), tensors["b"].to(f"npu:{args.device}")
            tensor_hashes = {
                key: hashlib.sha256(tensors[key].contiguous().numpy().tobytes()).hexdigest()
                for key in ["a", "b", "ref"]
            }
            for tiles in [0, 1, 2, 3, 32]:
                for mode in [3, 1, 2, 0]:
                    key = f"r{repeat}_{name}_t{tiles}_m{mode}"
                    pending = {"key": key, "repeat": repeat, "input_family": name,
                               "tile_count": tiles, "mode": mode, "kernel_symbol": f"l0c_c036_m{mode}",
                               "module_sha256": metadata["module_sha256"], "tensor_sha256": tensor_hashes,
                               "phase": "output_initialization"}
                    (out / "pending.json").write_text(json.dumps(pending, indent=2) + "\n")
                    try:
                        c = torch.full((32, 16, 16), metadata["sentinel"], dtype=torch.float32,
                                       device=f"npu:{args.device}")
                        torch.npu.synchronize()
                        pending["phase"] = "kernel_launch_and_completion"
                        (out / "pending.json").write_text(json.dumps(pending, indent=2) + "\n")
                        start = time.monotonic()
                        ext.run(an, bn, c, mode, tiles)
                        torch.npu.synchronize()
                        watchdog_seconds = time.monotonic() - start
                        y = c.cpu()
                        torch.save({"output": y}, out / "outputs" / f"{key}.pt")
                        active, ref = y[:tiles], tensors["ref"][:tiles]
                        total = active.numel()
                        diff = (active.double() - ref.double()).abs()
                        mismatch = (active != ref).nonzero()
                        first = mismatch[:8].tolist()
                        row = dict(pending)
                        row.update({
                            "phase": "complete", "numeric_total": total,
                            "equal_elements": int((active == ref).sum()),
                            "max_abs_diff": float(diff.max()) if total else None,
                            "finite_count": int(torch.isfinite(active).sum()),
                            "nan_count": int(torch.isnan(active).sum()),
                            "inf_count": int(torch.isinf(active).sum()),
                            "first_mismatch_indices": first,
                            "first_mismatch_values": [{"actual": float(active[tuple(i)]),
                                                       "reference": float(ref[tuple(i)])} for i in first],
                            "guard_total": y[tiles:].numel(),
                            "guard_equal": int((y[tiles:] == metadata["sentinel"]).sum()),
                            "watchdog_seconds_not_latency": watchdog_seconds,
                            "output_file": f"outputs/{key}.pt",
                            "output_archive_sha256": sha(out / "outputs" / f"{key}.pt"),
                        })
                        row["numeric_pass"] = row["equal_elements"] == total
                        row["guard_pass"] = row["guard_equal"] == row["guard_total"]
                        rows.append(row)
                        with (out / "records.jsonl").open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(json.dumps(row), flush=True)
                        if not (row["numeric_pass"] and row["guard_pass"]):
                            row["failure_kind"] = "numeric_or_guard_validation_failed"
                            (out / "validation_failure.json").write_text(json.dumps(row, indent=2) + "\n")
                            raise SystemExit("Validation failed; stop without additional launches")
                    except Exception:
                        failure = dict(pending)
                        failure["traceback"] = traceback.format_exc()
                        (out / "runtime_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                        raise
    summary = {"rows": len(rows), "numeric_failures": sum(not r["numeric_pass"] for r in rows),
               "guard_failures": sum(not r["guard_pass"] for r in rows),
               "numeric_nonempty_rows": sum(r["numeric_total"] > 0 for r in rows),
               "empty_lifecycle_rows": sum(r["numeric_total"] == 0 for r in rows),
               "all_pass": all(r["numeric_pass"] and r["guard_pass"] for r in rows),
               "metadata_sha256": sha(out / "metadata.json"), "process_pid": os.getpid(), "finished_unix": time.time()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
```

### observe_panel_runtime.py — complete same-PID observer

```python
"""Run the unchanged panel and capture loaded libraries from that same PID."""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(__file__).resolve().parent
out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
receipt = {"schema_version": "c036.same_pid_runtime_maps.v1", "pid": os.getpid(),
           "ppid": os.getppid(), "started_unix": time.time(),
           "observer_sha256": sha(__file__), "runner_sha256": sha(root / "run_panel.py"),
           "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
           "argv": sys.argv[1:], "status": "running"}
try:
    sys.argv[0] = str(root / "run_panel.py")
    runpy.run_path(str(root / "run_panel.py"), run_name="__main__")
    receipt["status"] = "panel_completed"
except BaseException as exc:
    receipt["status"] = "panel_exception"
    receipt["exception"] = repr(exc)
    raise
finally:
    receipt["finished_panel_unix"] = time.time()
    selected = []
    try:
        map_lines = Path("/proc/self/maps").read_text().splitlines()
        receipt["maps_status"] = "captured"
    except Exception as exc:
        map_lines = []
        receipt["maps_status"] = "unavailable"
        receipt["maps_error"] = repr(exc)
    for line in map_lines:
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and any(s in parts[5] for s in ["/Ascend/", "cann-", "torch_npu"]):
            selected.append(line)
    receipt["mapped_lines"] = selected
    libraries = []
    for name in sorted({line.split(maxsplit=5)[5] for line in selected}):
        path = Path(name.removesuffix(" (deleted)"))
        row = {"mapped_path": name, "resolved_path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            try:
                row.update({"bytes": path.stat().st_size, "sha256": sha(path)})
            except Exception as exc:
                row["sha256"] = None
                row["hash_error"] = repr(exc)
        libraries.append(row)
    receipt["mapped_files"] = libraries
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    versions = []
    for folder in [cann, cann / "aarch64-linux", cann / "tools", cann / "aarch64-linux/asc"]:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and ("version" in path.name.lower() or path.name.endswith("install.info")) and path.stat().st_size < 256 * 1024:
                versions.append({"path": str(path), "resolved_path": str(path.resolve()),
                                 "sha256": sha(path), "content": path.read_text(errors="replace")})
    receipt["version_files"] = versions
    receipt["finished_capture_unix"] = time.time()
    if out.is_dir():
        (out / "same_pid_runtime_maps.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"observer_status": receipt["status"], "pid": receipt["pid"],
                      "mapped_files": len(libraries) if receipt["maps_status"] == "captured" else None, "version_files": len(versions)}), flush=True)
```

### 安装队列与事件映射的完整机制单元

### installed_toolchain/captured/aarch64-linux/asc/include/basic_api/kernel_tpipe.h — complete TQueBind declaration and TQue derivation

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask = 0> class TQueBind {
public:
    __aicore__ inline TQueBind();
    __aicore__ inline void FreeBuffer(TBufHandle buf);
    __aicore__ inline TBuffAddr GetBufferAddr(TBufHandle buf);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template <typename T> __aicore__ inline __sync_noalias__ LocalTensor<T> AllocTensor();
    template <typename T> __aicore__ inline __sync_noalias__ void AllocTensor(LocalTensor<T>& tensor);
#else
    template <typename T> __aicore__ inline __sync_alias__ LocalTensor<T> AllocTensor();
    template <typename T> __aicore__ inline __sync_alias__ void AllocTensor(LocalTensor<T>& tensor);
#endif
    template <typename T> __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    __aicore__ inline bool EnQue(TBufHandle buf);
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline void DeQue(LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline TBufHandle DeQue();
    template <TPosition srcUserPos, TPosition dstUserPos, typename T> __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline bool VacantInQue();
    __aicore__ inline bool HasTensorInQue();
    __aicore__ inline int32_t GetTensorCountInQue();
    __aicore__ inline bool HasIdleBuffer();
    __aicore__ inline void FreeAllEvent();
    template <typename T> __aicore__ inline TBufState GetState(const LocalTensor<T>& tensor) const;
    __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len);
    template <typename T>
    __aicore__ inline void InitBufHandle(T* bufPool, uint32_t index, TBufHandle bufhandle,
        uint32_t curPoolAddr, uint32_t len);
protected:
    static constexpr TQueConfig config = GetTQueConfig(mask);
    static constexpr bool nd2nz = config.nd2nz;
    static constexpr bool nz2nd = config.nz2nd;
    static constexpr bool scmBlockGroup = config.scmBlockGroup;
    static constexpr bool enableLoopQueue = config.enableLoopQueue;
    static constexpr TPosition srcPosition = src;
    static constexpr TPosition dstPosition = dst;
    static constexpr Hardware srcHardType = GetPhyType(src);
    static constexpr Hardware dstHardType = GetPhyType(dst);
    static constexpr HardEvent enQueEvt = GetQueEvt(srcHardType, dstHardType, true, nd2nz, nz2nd);
    static constexpr HardEvent freeBufEvt = GetQueEvt(srcHardType, dstHardType, false, nd2nz, nz2nd);
    static constexpr int32_t queDepth = depth;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
    static constexpr bool enableGlobalManageQue = EnableGlobalManageQue<GetBufferPos(src, dst)>(config);
    static constexpr pipe_t srcPipe = GetPipe(srcHardType, dstHardType, true);
    static constexpr pipe_t dstPipe = GetPipe(srcHardType, dstHardType, false);
    static constexpr int32_t maxBlockNum = 32;
    static constexpr uint8_t shiftBits = CalculatesShiftedBit(config.bufferLen);
    static constexpr uint8_t maxBufferBlock = GlobalManageQueConfig<GetBufferPos(src, dst)>::maxBufferBlock;
    static constexpr uint8_t bufIdOffset = GlobalManageQueConfig<GetBufferPos(src, dst)>::bufIdOffset;
#endif
    union {
        uint64_t value;
        struct {
            uint8_t bufNum = 0;
            uint8_t usedCount;
            uint16_t head;
            uint16_t tail;
            uint8_t bufUsedCount;
            uint8_t bufCursor;
        };

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        struct {
            uint32_t freeMask;
            uint8_t staticHead;
            uint8_t staticEnqueHead;
            uint8_t staticUsedCount;
            uint8_t staticBufUsedCount;
        };
#endif
    };
    typename TBufHandleAux<depth>::T que_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    typename BufInfoAux<enableGlobalManageQue, config>::type bufStart;
#else
    struct TBufType* bufStart;
#endif
    DEBUG_CODE(uint32_t bufLen);
    friend class TPipe;
    template <TPosition pos, int32_t d, auto m> friend class TQue;
    template<TPosition pos, uint32_t bufIDSize> friend class TBufPool;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPoolExtImpl;
#endif
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    uint64_t bufPoolHandle{0U};
#endif
private:
    __aicore__ inline void SetTBufPoolHandle(uint64_t bufPoolHandle);
    template <typename T> __aicore__ inline LocalTensor<T> Buf2Tensor(TBufHandle buf);
    __aicore__ inline TBufState GetState(const TBufHandle& handle) const;
    static constexpr bool isTQue = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline uint64_t GetNext(const int32_t len = 1);
#endif
    __aicore__ inline TBufHandle AllocBuffer();
    template <TPosition srcUserPos, TPosition dstUserPos> __aicore__ inline bool EnQue(TBufHandle buf);
    template <TPosition srcUserPos, TPosition dstUserPos> __aicore__ inline TBufHandle DeQue();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template <typename IMPL, typename A, typename B, typename L0cT, class C, const auto &MM_CFG, typename>
    friend class Impl::Detail::CubeOutBuffer;
    template <typename T> __aicore__ inline __sync_noalias__ LocalTensor<T> AllocTensor(int32_t num);
    template <typename T>
    friend __aicore__ inline uint64_t GetTQueHeadAddr(const T& que);
#endif
    template <typename T>
    friend __aicore__ constexpr bool UseBufIdSync();
};

// Template Args:
// pos - position for queue, such as VECIN/VECOUT/A1...
// mask - the 0th bit is nd2nz, 1 means data trans from nd format to nz format
//        the 1st bit is nz2nd, 1 means data trans from nz format to nd format
template <TPosition pos, int32_t depth, auto mask = 0>
class TQue : public TQueBind<GetBufferLogicPos(pos, true), GetBufferLogicPos(pos, false), depth, mask> {
public:
    __aicore__ inline TQue() = default;
private:
    friend class TPipe;
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPool;
    static constexpr bool isTQue = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPoolExtImpl;
    template <typename T>
    friend __aicore__ inline uint64_t GetTQueHeadAddr(const T& que);
#endif
};
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_event.h — GetPhyType

```cpp
__aicore__ constexpr Hardware GetPhyType(TPosition pos)
{
    ASSERT(pos != TPosition::MAX);
    Hardware hard = Hardware::UB;
    if (pos == TPosition::GM) {
        hard = Hardware::GM;
    } else if (pos == TPosition::A1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::A2) {
hard = Hardware::L0A;
    } else if (pos == TPosition::B1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::B2) {
        hard = Hardware::L0B;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002))
    } else if (pos == TPosition::C1) {
        hard = Hardware::UB;
    } else if (pos == TPosition::C2) {
        hard = Hardware::L0C;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::UB;
#elif (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3003)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#elif (__NPU_ARCH__ == 3002)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#elif (__NPU_ARCH__ == 3102)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
#elif (__NPU_ARCH__ == 5102)
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 3113))
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
#endif
    } else if (pos == TPosition::CO1) {
hard = Hardware::L0C;
    } else if (pos == TPosition::SHM) {
        hard = Hardware::L1;
    } else if (pos == TPosition::TSCM) {
        hard = Hardware::L1;
    }
    return hard;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_event.h — GetBufferLogicPos

```cpp
__aicore__ constexpr TPosition GetBufferLogicPos(TPosition pos, bool isSrc)
{
    ASSERT(pos != TPosition::GM);
    ASSERT(pos != TPosition::VECCALC);
    ASSERT(pos != TPosition::MAX);
    if (pos == TPosition::A1) {
        return isSrc ? TPosition::GM : TPosition::A1;
    } else if (pos == TPosition::B1) {
        return isSrc ? TPosition::GM : TPosition::B1;
    } else if (pos == TPosition::C1) {
        return isSrc ? TPosition::GM : TPosition::C1;
    } else if (pos == TPosition::A2) {
        return isSrc ? TPosition::A1 : TPosition::A2;
    } else if (pos == TPosition::B2) {
        return isSrc ? TPosition::B1 : TPosition::B2;
    } else if (pos == TPosition::C2) {
        return isSrc ? TPosition::C1 : TPosition::C2;
    } else if (pos == TPosition::CO1) {
        return isSrc ? TPosition::CO1 : TPosition::CO2;
    } else if (pos == TPosition::CO2) {
        return isSrc ? TPosition::CO2 : TPosition::GM;
    } else if (pos == TPosition::VECIN) {
        return isSrc ? TPosition::GM : TPosition::VECIN;
    } else if (pos == TPosition::VECOUT) {
        return isSrc ? TPosition::VECOUT : TPosition::GM;
    } else if (pos == TPosition::SPM) {
        return isSrc ? TPosition::VECOUT : TPosition::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        return isSrc ? TPosition::B1 : TPosition::C2PIPE2GM;
    }
    return TPosition::MAX;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_event.h — GetQueEvt(Hardware,Hardware,bool,...)

```cpp
__aicore__ constexpr HardEvent GetQueEvt(Hardware src, Hardware dst, bool fwdDirect, bool nd2nz = false,
                                         bool nz2nd = false)
{
    (void)(nz2nd);
    ASSERT((src == Hardware::GM) || (src == Hardware::UB) || (src == Hardware::L1) || (src == Hardware::L0A) ||
           (src == Hardware::L0B) || (src == Hardware::L0C));
    ASSERT(src != Hardware::MAX);
    ASSERT(dst != Hardware::MAX);
    if (src == Hardware::GM) {  // MTE2
        ASSERT(dst != Hardware::GM);
        ASSERT(dst != Hardware::L0C);
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
        if (dst == Hardware::UB) {  // MTE3
            return fwdDirect ? HardEvent::MTE2_V : HardEvent::V_MTE2;
        } else if (dst == Hardware::L1) {  // MTE1
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 1001) && (__NPU_ARCH__ != 2002)
            (void)(nd2nz);
#else
        // in v100/v200, nd2nz was simulated with vector intrins, so event changed event to mte3
            if (nd2nz) {
                return fwdDirect ? HardEvent::MTE3_MTE1 : HardEvent::MTE1_MTE3;
            }
#endif
            return fwdDirect ? HardEvent::MTE2_MTE1 : HardEvent::MTE1_MTE2;
        } else if (dst == Hardware::L0A) {
            return fwdDirect ? HardEvent::MTE2_M : HardEvent::M_MTE2;
        } else if (dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE2_M : HardEvent::M_MTE2;
        }
    } else if (src == Hardware::UB) {  // MTE3
        ASSERT(dst != Hardware::L0A);
        ASSERT(dst != Hardware::L0B);
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
        if (dst == Hardware::GM) {
            return fwdDirect ? HardEvent::V_MTE3 : HardEvent::MTE3_V;
        } else if (dst == Hardware::L1) {  // MTE1
            return fwdDirect ? HardEvent::MTE3_MTE1 : HardEvent::MTE1_MTE3;
        } else if (dst == Hardware::L0C) {
            return fwdDirect ? HardEvent::V_V : HardEvent::MAX;  // HardEvent::M_V
        } else if (dst == Hardware::UB) {
            return fwdDirect ? HardEvent::MTE2_MTE3 : HardEvent::MTE3_MTE2;
        }
    } else if (src == Hardware::L1) {  // MTE1
        ASSERT(dst != Hardware::GM);
        ASSERT(dst != Hardware::L1);
        ASSERT(dst != Hardware::L0C);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002))
        ASSERT(dst != Hardware::BIAS);
        ASSERT(dst != Hardware::FIXBUF);
#endif
        if (dst == Hardware::UB) {
            return fwdDirect ? HardEvent::MTE1_V : HardEvent::V_MTE1;
        } else if (dst == Hardware::L0A) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        } else if (dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        } else if (dst == Hardware::FIXBUF) {
            return fwdDirect ? HardEvent::MTE1_FIX : HardEvent::FIX_MTE1;
        } else if (dst == Hardware::BIAS) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        }
    } else if (src == Hardware::L0A) {
        ASSERT(dst == Hardware::L0C);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
    } else if (src == Hardware::L0B) {
        ASSERT(dst == Hardware::L0C);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002))
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_V : HardEvent::V_M;
    }
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3003))
    } else if (src == Hardware::L0C) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
        ASSERT(dst == Hardware::GM || dst == Hardware::UB || dst == Hardware::L1);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
#else
        ASSERT(dst == Hardware::GM || dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
#endif
    }
#elif (__NPU_ARCH__ == 3102)
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM || dst == Hardware::UB);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#elif defined(__NPU_ARCH__) && ( \
      ((__NPU_ARCH__ == 3113)))
    } else if (src == Hardware::L0C) {
        ASSERT(dst == Hardware::GM || dst == Hardware::UB || dst == Hardware::L1);
        return fwdDirect ? HardEvent::M_FIX : HardEvent::FIX_M;
    }
#else
    }
#endif
    return HardEvent::MAX;
}
```





### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — AllocTensor()

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::AllocTensor()
{
    static_assert((depth != 0), "must use AllocTensor<LocalTensor&> api while tque's depth is zero");
    auto buf = AllocBuffer();
    return Buf2Tensor<T>(buf);
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — FreeTensor(LocalTensor&)

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline void TQueBind<src, dst, depth, mask>::FreeTensor(LocalTensor<T>& input)
{
    FreeBuffer(input.GetBufferHandle());
    return;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — EnQue(LocalTensor)

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(const LocalTensor<T>& input)
{
    auto buf = input.GetBufferHandle();
    return EnQue(reinterpret_cast<TBufHandle>(buf));
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — EnQue(TBufHandle)

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(TBufHandle buf)
{
    auto ptr = reinterpret_cast<TBufType*>(buf);
    if constexpr (depth != 0) {
        ASCENDC_DEBUG_ASSERT((this->usedCount < depth),
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "usedCount is %d, which exceed depth limits %d", static_cast<int32_t>(usedCount),
                depth));

        if constexpr (depth == 1) {
            this->que_ = buf;
        } else {
            this->que_[this->tail] = buf;
        }
        this->usedCount++;
    }
    ASCENDC_DEBUG_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum));
    ASCENDC_DEBUG_ASSERT((ptr->state == TBufState::OCCUPIED) || (ptr->state == TBufState::DEQUE),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "ptr state is %d, which should be OCCUPIED / DEQUE", static_cast<int32_t>(ptr->state)));
    DEBUG_CODE(ptr->state = TBufState::ENQUE);
    if constexpr (depth == 0) {
        // If the AIC is not entered, the AIV does not process any event ID.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
            GetBuffImpl<srcPipe, true>(ptr->bufId);
            ReleaseBuffImpl<srcPipe, true>(ptr->bufId);
        } else
#endif
        {
        if constexpr ((GetPosition(src, dst) != TPosition::TSCM)) {
            SetFlag<enQueEvt>(ptr->enQueEvtID);
        }
        }
    } else {
        /* Add for TSCM
        * for 220, aiv just send message, no need add this set/wait
        */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
            GetBuffImpl<srcPipe, true>(ptr->bufId);
            ReleaseBuffImpl<srcPipe, true>(ptr->bufId);
        } else
#endif
        {
#if __NPU_ARCH__ == 2201
        // If the AIC is not entered, the AIV does not process any event ID.
        if (g_coreType != AIV || (GetPosition(src, dst) != TPosition::TSCM)) {
            auto enQueEvtID = GetTPipePtr()->AllocEventID<enQueEvt>();
            SetFlag<enQueEvt>(enQueEvtID);
            ptr->enQueEvtID = enQueEvtID;
        }
#else
        auto enQueEvtID = GetTPipePtr()->AllocEventID<enQueEvt>();
        SetFlag<enQueEvt>(enQueEvtID);
        ptr->enQueEvtID = enQueEvtID;
#endif
        }
        if constexpr (depth != 1) {
            if (++this->tail >= depth) {
                this->tail = 0;
            }
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufEnque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst),
        static_cast<uint8_t>(GetPosition(src, dst)), reinterpret_cast<uint64_t>(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return true;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — DeQue<T>()

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::DeQue()
{
    static_assert((depth != 0), "must use DeQue<LocalTensor&> api while tque's depth is zero");
    auto buf = DeQue();
    auto ret = Buf2Tensor<T>(buf);
    return ret;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — DeQue() handle

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline __sync_alias__ TBufHandle TQueBind<src, dst, depth, mask>::DeQue()
{
    TBufHandle buf;
    if constexpr (depth == 1) {
        buf = this->que_;
    } else {
        buf = this->que_[this->head];
    }
    ASCENDC_DEBUG_ASSERT((buf != nullptr), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "buf can not be nullptr"));
    auto ptr = reinterpret_cast<TBufType*>(buf);

#if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)
    ASCENDC_DEBUG_ASSERT((ptr->state == TBufState::ENQUE),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "ptr state is %d, which can only be ENQUE", static_cast<int32_t>(ptr->state)));
#endif
    ASCENDC_DEBUG_ASSERT((this->usedCount > 0),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "usedCount is %d, which can only larger than 0",
            static_cast<int32_t>(this->usedCount)));
    this->usedCount--;
    /* Add for TSCM
     * for 220, aiv just send message, no need add this set/wait
     */
    DEBUG_CODE(ptr->state = TBufState::DEQUE);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
        GetBuffImpl<dstPipe, false>(ptr->bufId);
        ReleaseBuffImpl<dstPipe, false>(ptr->bufId);
    } else
#endif
    {
#if __NPU_ARCH__ == 2201
    if (g_coreType != AIV || (GetPosition(src, dst) != TPosition::TSCM)) {
        if (ptr->enQueEvtID != INVALID_TEVENTID) {
            WaitFlag<enQueEvt>(ptr->enQueEvtID);
            GetTPipePtr()->ReleaseEventID<enQueEvt>(ptr->enQueEvtID);
            ptr->enQueEvtID = INVALID_TEVENTID;
        }
    }
#else
    if (ptr->enQueEvtID != INVALID_TEVENTID) {
        WaitFlag<enQueEvt>(ptr->enQueEvtID);
        GetTPipePtr()->ReleaseEventID<enQueEvt>(ptr->enQueEvtID);
        ptr->enQueEvtID = INVALID_TEVENTID;
    }
#endif
    }
    if constexpr (depth != 1) {
        if (++this->head >= depth) {
            this->head = 0;
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufDeque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst), static_cast<uint8_t>(GetPosition(src, dst)),
        (uint64_t)(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return reinterpret_cast<TBufHandle>(buf);
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — FreeBuffer(TBufHandle)

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline void TQueBind<src, dst, depth, mask>::FreeBuffer(TBufHandle buf)
{
    auto ptr = reinterpret_cast<TBufType*>(buf);
    ASCENDC_DEBUG_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum));
    ASCENDC_DEBUG_ASSERT((ptr->state != TBufState::FREE),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "ptr state is %d, which can not be FREE", static_cast<int32_t>(ptr->state)));
    if constexpr (depth == 0) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
            GetBuffImpl<dstPipe, true>(ptr->bufId);
            ReleaseBuffImpl<dstPipe, true>(ptr->bufId);
        } else
#endif
        {
        if constexpr (!IsAivTscm(src, dst)) {
            // in 220 version, event changed from v to M_MTE1 on condition C1 -> C2
            SetFlag<freeBufEvt>(ptr->freeBufEvtID);
        }
        }
    } else {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
            GetBuffImpl<dstPipe, true>(ptr->bufId);
            ReleaseBuffImpl<dstPipe, true>(ptr->bufId);
        } else
#endif
        {
            if constexpr (!IsAivTscm(src, dst)) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 1001) && (__NPU_ARCH__ != 2002)
                // in 220 version, event changed from v to M_MTE1 on condition C1 -> C2
                ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
                SetFlag<freeBufEvt>(ptr->freeBufEvtID);
                if constexpr (enableLoopQueue) {
                    ptr->freeBufEvt = freeBufEvt;
                }
#else
                if constexpr (src == TPosition::C1 || (src == TPosition::CO2 && dst == TPosition::VECIN)) {
                    SetFlag<freeBufEvt>(0); // insert pipe_v without eventID
                    ASCENDC_DEBUG_ASSERT((ptr->freeBufEvtID == INVALID_TEVENTID),
                                KERNEL_LOG_INTERNAL(KERNEL_ERROR, "freebuf event id can not be -1"));
                } else {
                    ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
                    SetFlag<freeBufEvt>(ptr->freeBufEvtID);
                }
#endif
            } else if constexpr (srcHardType == Hardware::GM) {
                if ASCEND_IS_AIC {
                    ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
                    SetFlag<freeBufEvt>(ptr->freeBufEvtID);
                    if constexpr (enableLoopQueue) {
                        ptr->freeBufEvt = freeBufEvt;
                    }
                }
            }
        }
    }
    ptr->state = TBufState::FREE;
    if constexpr (depth != 0) {
        this->bufUsedCount--;
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufFree(static_cast<uint8_t>(bufferType), static_cast<uint8_t>(GetPosition(src, dst)),
        (uint64_t)(absAddr + ptr->address), static_cast<uint64_t>(ptr->dataLen));
#endif // ASCENDC_CPU_DEBUG
    return;
}
```

### installed_toolchain/captured/aarch64-linux/asc/impl/basic_api/kernel_tquebind_impl.h — AllocBuffer()

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline TBufHandle TQueBind<src, dst, depth, mask>::AllocBuffer()
{
    DEBUG_CODE(int32_t size = 0);
    ASCENDC_DEBUG_ASSERT((bufNum > 0),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "bufNum is %d, which must be larger than 0", static_cast<int32_t>(bufNum)));
    TBufType* ret;
    do {
        ret = this->bufStart + this->bufCursor;
        if constexpr (config.bufferNumber != 1) {
            this->bufCursor += 1;
            if (this->bufCursor == this->bufNum) {
                this->bufCursor = 0;
            }
        }
        if (ret->state == TBufState::FREE) {
            ret->state = TBufState::OCCUPIED;
            if constexpr (IsAivTscm(src, dst)) {
                if constexpr (srcHardType == Hardware::UB) {
                    break;
                } else if constexpr (srcHardType == Hardware::GM) {
                    if ASCEND_IS_AIV {
                        break;
                    }
                }
            }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
            if constexpr(UseBufIdSync<TQueBind<src, dst, depth, mask>>()) {
                GetBuffImpl<srcPipe, false>(ret->bufId);
                ReleaseBuffImpl<srcPipe, false>(ret->bufId);
            } else
#endif
            {
                if (ret->freeBufEvtID != INVALID_TEVENTID) {
                    if constexpr (enableLoopQueue) {
                        if (freeBufEvt == ret->freeBufEvt) {
                            WaitFlag<freeBufEvt>(ret->freeBufEvtID);
                            GetTPipePtr()->ReleaseEventID<freeBufEvt>(ret->freeBufEvtID);
                            ret->freeBufEvtID = INVALID_TEVENTID;
                        } else if (freeBufEvt == HardEvent::V_MTE2 && ret->freeBufEvt == HardEvent::MTE3_V) {
                            WaitFlag<HardEvent::MTE3_V>(ret->freeBufEvtID);
                            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(ret->freeBufEvtID);
                            ret->freeBufEvtID = INVALID_TEVENTID;
                            TEventID evtId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
                            SetFlag<HardEvent::MTE3_MTE2>(evtId);
                            WaitFlag<HardEvent::MTE3_MTE2>(evtId);
                            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtId);
                        } else if (freeBufEvt == HardEvent::MTE3_V && ret->freeBufEvt == HardEvent::V_MTE2) {
                            WaitFlag<HardEvent::V_MTE2>(ret->freeBufEvtID);
                            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(ret->freeBufEvtID);
                            ret->freeBufEvtID = INVALID_TEVENTID;
                        } else {
                            ASCENDC_DEBUG_ASSERT(false,
                                KERNEL_LOG_INTERNAL(KERNEL_ERROR, "there is something wrong with free buf event"));
                        }
                    } else {
                        WaitFlag<freeBufEvt>(ret->freeBufEvtID);
                        GetTPipePtr()->ReleaseEventID<freeBufEvt>(ret->freeBufEvtID);
                        ret->freeBufEvtID = INVALID_TEVENTID;
                    }
                }
            }
            break;
        }
#if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)
        ASCENDC_DEBUG_ASSERT((++size <= this->bufNum),
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "size is %d, which exceed limits %d", size, static_cast<int32_t>(this->bufNum)));
#endif
    } while (true);
    this->bufUsedCount++;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufAlloc(static_cast<uint8_t>(bufferType), static_cast<uint8_t>(GetPosition(src, dst)),
        reinterpret_cast<uint64_t>(absAddr + ret->address), static_cast<uint64_t>(ret->dataLen));
    if (this->bufPoolHandle != 0U) {
        AscendCUpdateTbufPoolStatus(this->bufPoolHandle, false);
        AscendCTBufPoolResetCheck(static_cast<uint8_t>(GetPosition(srcPosition, dstPosition)),
            reinterpret_cast<uint64_t>(absAddr + ret->address),
            static_cast<uint64_t>(ret->dataLen),
            this->bufPoolHandle);
    }
#endif // ASCENDC_CPU_DEBUG
    return reinterpret_cast<TBufHandle>(ret);
}
```

## 面板观测与绑定核对

下列两份 summary 是本次驱动实际输出原件，不把 120 行重复记录逐条转录到知识正文。完整原始记录、两段同 PID 映射与全部保存输出的 canonical bits 均保留在配套审阅材料；公开化中的路径脱敏和 SHA 命名空间分别记录，不算新 NPU 复现。（来源平台：a2；证据等级：artifact_verified）

### 首段原始 summary

```json
{
  "rows": 40,
  "numeric_failures": 0,
  "guard_failures": 0,
  "numeric_nonempty_rows": 32,
  "empty_lifecycle_rows": 8,
  "all_pass": true,
  "metadata_sha256": "c0eb8c4737eb734c4722516fb1c85514c7363d0dc141e49f8c7e547c7d6bcdf3",
  "process_pid": 372447,
  "finished_unix": 1792325914.8400192
}
```

### 同输入重复段原始 summary

```json
{
  "rows": 80,
  "numeric_failures": 0,
  "guard_failures": 0,
  "numeric_nonempty_rows": 64,
  "empty_lifecycle_rows": 16,
  "all_pass": true,
  "metadata_sha256": "fd0e22cbec15bb8e301fe7e666b2b4c24c364eb73238223cb886564532043d0b",
  "process_pid": 372529,
  "finished_unix": 1792325928.7872164
}
```

两段记录合起来覆盖四种编译期屏障组合、两个保存输入族，以及空、首次排空、尚未回绕、首次回绕和连续多 tile 生命周期；每个组合重复三次。逐条复核结果如下，分母含义不互换。（来源平台：a2；证据等级：artifact_verified）

| 编译期模式 | launch | 非空 / 空生命周期 | 非空 exact / 数值分母 | guard 相等 / guard 分母 |
| --- | --- | --- | --- | --- |
| 不加额外宽屏障 | 30 | 24 / 6 | 58368 / 58368 | 187392 / 187392 |
| 只在 Fixpipe 前加 | 30 | 24 / 6 | 58368 / 58368 | 187392 / 187392 |
| 只在 Fixpipe 后加 | 30 | 24 / 6 | 58368 / 58368 | 187392 / 187392 |
| 两处都加 | 30 | 24 / 6 | 58368 / 58368 | 187392 / 187392 |
| 全部 | 120 | 96 / 24 | 233472 / 233472 | 749568 / 749568 |

原始记录中的空路径有 numeric_total=0、equal_elements=0、max_abs_diff=null，而 guard_total=guard_equal=8192；首个非空路径有 numeric_total=equal_elements=256、max_abs_diff=0、guard_total=guard_equal=7936。它们说明 numeric_pass 对空分母会自然为真，因此空路径只作生命周期与 guard 检查。上述为原始字段的摘录与汇总；没有补造失败输出。（来源平台：a2；证据等级：artifact_verified）

队列事件说明只取自本文保留的完整因果单元：逻辑/物理位置映射、GetQueEvt 的 HardEvent 类型选择、EnQue/DeQue、FreeTensor/FreeBuffer 和 AllocTensor/AllocBuffer。结论止于 SetFlag/WaitFlag 调用及事件 ID 生命周期，不证明下层 PIPE 展开或硬件等待完成。四模式全通过不能替代这些源码证据，也不能从该面板反推出宽屏障的一般必要性。

## 预防

先核对实际 AIC 编译链，再核对 TQue 的逻辑位置、硬件位置、就绪事件和回收事件；检查必须追到具体 buffer 保存的 event ID。对没有数值元素的空路径只验生命周期与 guard。对重复运行检查保存输入 hash，避免把同一输入的重复次数当作独立输入覆盖或跨轨迹复现。

## 验证信号

来源为一次普通 Matmul CO1/Fixpipe 的默认构建与两段连续 panel，完整核心实现与两份原始 summary 已内联，全部逐次记录、保存数值 bits 和绑定材料保留在配套审阅证据；第二段重用相同保存输入，不构成另一条独立来源。

- 新构建的四个编译期 symbol 均由实际 AIC 命令构建，源码/生成文件/加载扩展绑定到同一个新模块；本次只有一种默认 compiler 配置。两段进程各自的 metadata、summary、observer PID 一致，实际映射的两项主要 runtime 库路径与 hash 一致，并保存安装版本原文。（来源平台：a2；证据等级：artifact_verified）
- 120 次 launch 中 96 次非空，233472/233472 个数值元素精确相等，非空 max_abs_diff=0；24 次空输出的 max_abs_diff=null，只计生命周期与 guard。所有 launch 的 guard 合计 749568/749568 相等。（来源平台：a2；证据等级：artifact_verified）
- 两个输入族各自保存的输入字节在三次 repeat 中相同；repeat 不增加输入独立性，也不增加轨迹来源数量。四个屏障模式在该面板内均未出现数值或 guard 失败。（来源平台：a2；证据等级：artifact_verified）

## 失效边界

本卡不判断宽屏障在一般场景中冗余或必要，不声称修复了旧竞争，也没有吞吐收益结论。原始记录中的 watchdog_seconds_not_latency 是完成等待的 watchdog 记录，不能当作 kernel latency 或性能比较。没有 auto-sync-off sweep、最终 ISA、事件发射追踪或 overlap 测量；编译 flags 不能弥补这些缺口。

未决项优先项：若要回答旧 race 是否存在或宽屏障是否必要，须在能触发真实失败的独立实验中继续追根因到最终指令、buffer 复用与执行重叠机制，再建立可归因对照。本面板没有复现失败，因此没有 failure-before/repair-after 结论。其他 shape、dtype、资源规模、pipeline 深度和输入分布未验证；其他平台未验证，推断可迁移时仍需重新核对安装代码和上板验证。两段 panel、四个 symbol、两个输入族和三次 repeat 均属于同一次来源，内部只按 single_run 处理。

## 坏实践

把 FreeTensor 简化成“只标 FREE”，或写成“调用时立即等待 Fixpipe 完成”，都会误读回收事件的生产/消费位置。只看两个 buffer 或源码保证性注释就判安全，会跳过 M_FIX/FIX_M 依赖链。以全通过推导宽屏障普遍无用、把 --cce-auto-sync 的出现归因为 Cube 同步插入、把空输出算成精度样本、把重复输入算成独立复现，均超出这次证据。

## 误诊

旧运行时 mode 面板与旧模块/源码恢复只说明旧 binding 背景；本卡不倒推旧 device 的历史身份。没有本次失败修复路线，也没有失败路线客观回退结论。安装源码中的保证性注释属于作者解释；本卡的诊断依赖完整队列函数、HardEvent 类型选择与事件 ID 生命周期，未把注释或 trace 外 CPU 验收写成新的上板实测。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/c036/runtime_cann_evidence.json#/runs`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/fresh_build_receipt.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/actual_aic_commands.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/raw_receipts/npu_first40_records.jsonl`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/raw_receipts/npu_repeat80_records.jsonl`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/canonical_saved_tensors.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/canonical_cpu_consistency.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/installed_headers/manifest.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/provenance_map.json`；`cannbot_non_level4_split_review_20261019/evidence_public/c036/README.md`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
