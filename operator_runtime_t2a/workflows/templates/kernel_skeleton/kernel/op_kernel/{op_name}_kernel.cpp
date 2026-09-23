// {op_name} device kernel — elementwise 骨架(1 输入;dtype 分发 + 尾块 + 32B 对齐 + 双 buffer)
// 数学在 Compute() 里,默认把第 1 个输入恒等拷贝到输出(能编过、能注册、跑通打包链路,
// 不代表实现了 reference)。全部普通 Tensor 输入按相同类型搬运，host 检查类型一致。
// 必须按本题改写 tiling、buffer、尾块有效长度和全部输出,不能只替换 Compute。
// dtypeFlag 按实际 scalar_type 分派，区分 FP16/BF16/整数；bool 占位仅按 uint8_t 搬运。
// 类型/API/转换参考 .claude/skills/tilelang2ascend-translator/SKILL.md 与 archive_tasks/rms_norm/。
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Kernel{OpName} {
public:
    __aicore__ inline Kernel{OpName}() {}

    __aicore__ inline void Init(GM_ADDR in0, GM_ADDR y, int64_t formerNum, int64_t formerLength,
                                int64_t tailLength, int64_t tileLength)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();

        if (blockIdx < formerNum) {
            this->blockLength = formerLength;
            int64_t offset = formerLength * blockIdx;
            in0Gm.SetGlobalBuffer((__gm__ T *)in0 + offset, formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);
        } else {
            this->blockLength = tailLength;
            int64_t tailIdx = blockIdx - formerNum;
            int64_t offset = formerLength * formerNum + tailLength * tailIdx;
            in0Gm.SetGlobalBuffer((__gm__ T *)in0 + offset, tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);
        }
        this->tileLength = tileLength;

        pipe.InitBuffer(inQueue0, BUFFER_NUM, tileLength * sizeof(T));
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
        AscendC::LocalTensor<T> in0Local = inQueue0.AllocTensor<T>();
        AscendC::DataCopy(in0Local, in0Gm[progress * this->tileLength], curTileLength);
        inQueue0.EnQue(in0Local);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> in0Local = inQueue0.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // TODO: 换成你的算子数学。默认恒等拷贝 y = in0(只为打通打包/注册链路)。
        // 例: AscendC::Abs(yLocal, in0Local, curTileLength);       // |x|
        // 混合类型输入需按 reference 分别实现类型与 buffer，并调整 host 检查。
        AscendC::DataCopy(yLocal, in0Local, curTileLength);

        outQueueY.EnQue<T>(yLocal);
        inQueue0.FreeTensor(in0Local);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, curTileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue0;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> in0Gm;
    AscendC::GlobalTensor<T> yGm;
    int64_t blockLength;
    int64_t tileLength;
};

extern "C" __global__ __aicore__ void {op_name}_kernel(
    GM_ADDR in0, GM_ADDR y,
    int64_t formerNum, int64_t formerLength, int64_t tailLength, int64_t tileLength, int64_t dtypeFlag)
{
    if (dtypeFlag == 0) {
        Kernel{OpName}<float> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 1) {
        Kernel{OpName}<half> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 2) {
        Kernel{OpName}<bfloat16_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 3) {
        Kernel{OpName}<int32_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 4) {
        Kernel{OpName}<int64_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 5) {
        Kernel{OpName}<int8_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 6) {
        Kernel{OpName}<uint8_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 7) {
        Kernel{OpName}<uint8_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 8) {
        Kernel{OpName}<int16_t> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
    else if (dtypeFlag == 9) {
        Kernel{OpName}<double> op;
        op.Init(in0, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
}
