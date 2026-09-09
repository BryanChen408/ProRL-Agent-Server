// {op_name} device kernel — elementwise 骨架(dtype 分发 + 尾块 + 32B 对齐 + 双 buffer)
// 数学在 Compute() 里,默认是 DataCopy 恒等拷贝(能编过、能注册、能跑通打包链路)。
// 必须按本题改写 tiling、buffer、尾块有效长度和全部输出,不能只替换 Compute。
// dtypeSize==2 仅是 half 占位,不代表支持 BF16;整数/混合 dtype 须显式接线,不能只删 host 检查。
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Kernel{OpName} {
public:
    __aicore__ inline Kernel{OpName}() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t formerNum, int64_t formerLength,
                                int64_t tailLength, int64_t tileLength)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();

        if (blockIdx < formerNum) {
            this->blockLength = formerLength;
            int64_t offset = formerLength * blockIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);
        } else {
            this->blockLength = tailLength;
            int64_t tailIdx = blockIdx - formerNum;
            int64_t offset = formerLength * formerNum + tailLength * tailIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);
        }
        this->tileLength = tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;

        int64_t alignNum = 32 / static_cast<int64_t>(sizeof(T));
        int64_t alignedTailLen = ((tailTileLength + alignNum - 1) / alignNum) * alignNum;

        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (tileNum > 0) {
            CopyIn(tileNum - 1, alignedTailLen);
            Compute(tileNum - 1, alignedTailLen);
            CopyOut(tileNum - 1, alignedTailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], curTileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // TODO: 换成你的算子数学。默认是恒等拷贝 y = x(只为打通打包/注册链路)。
        // 例: AscendC::Abs(yLocal, xLocal, curTileLength);       // |x|
        //     AscendC::Adds(yLocal, xLocal, (T)1, curTileLength); // x + 1
        AscendC::DataCopy(yLocal, xLocal, curTileLength);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, curTileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    int64_t blockLength;
    int64_t tileLength;
};

extern "C" __global__ __aicore__ void {op_name}_kernel(
    GM_ADDR x, GM_ADDR y,
    int64_t formerNum, int64_t formerLength,
    int64_t tailLength, int64_t tileLength, int64_t dtypeSize)
{
    if (dtypeSize == 2) {
        Kernel{OpName}<half> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    } else {
        Kernel{OpName}<float> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
}
