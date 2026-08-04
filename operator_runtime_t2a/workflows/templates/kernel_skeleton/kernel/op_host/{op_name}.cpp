// {op_name} op_host — 校验 + tiling + EXEC_KERNEL_CMD 启动
// 结构取自已验证的 elementwise 模板:正确的核间划分 + UB 感知 tileLength + 32B 对齐,
// 没有 blockLength=0 / 尾块未处理这类坑。
#include <algorithm>
#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

// 由 build 从 op_kernel/{op_name}_kernel.cpp 的 __global__ 入口自动生成。
#include "aclrtlaunch_{op_name}_kernel.h"

namespace ascend_kernel {

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

at::Tensor {op_name}(const at::Tensor &self)
{
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat,
                "{op_name}: only float16 and float32 are supported, got ", self.scalar_type());
    TORCH_CHECK(self.is_contiguous(), "{op_name}: input must be contiguous");

    at::Tensor output = at::empty_like(self);

    int64_t totalLength = self.numel();
    if (totalLength == 0) {
        return output;
    }
    int64_t dtypeSize = self.element_size();

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    if (coreNum <= 0) { coreNum = 1; }
    uint64_t ubSize = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int64_t totalLengthCore = (totalLength + coreNum - 1) / coreNum;
    int64_t totalLengthCoreAlign = (totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) /
                                   CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t bufferCoefficient = dtypeSize * 4;  // 按 UB 分配表调整(in/out 双 buffer)
    if (bufferCoefficient <= 0) { bufferCoefficient = 1; }
    int64_t maxTileElements = static_cast<int64_t>(ubSize) / bufferCoefficient;
    int64_t alignElements = 32 / (dtypeSize > 0 ? dtypeSize : 1);
    if (alignElements <= 0) { alignElements = 1; }
    int64_t tileLength = (maxTileElements / alignElements) * alignElements;
    if (tileLength <= 0) { tileLength = alignElements; }

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD({op_name}_kernel, blockDim,
                    self, output,
                    formerNum, formerLength, tailLength, tileLength, dtypeSize);

    return output;
}

}  // namespace ascend_kernel
