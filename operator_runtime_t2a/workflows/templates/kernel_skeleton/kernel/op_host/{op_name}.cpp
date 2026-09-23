// {op_name} op_host — 校验 + tiling + EXEC_KERNEL_CMD 启动
// 签名由 prepare 按 model.py 生成(与 register.cpp/ops.h/model_new 自洽);
// elementwise 占位不是本题语义契约:按原始 reference 核对完整计算、输出与分支。
// empty_like、连续性检查、单输出接线与 tiling 均需按本题改写,不只改 Compute。
// 要改签名请四处同步;构建/loader 机制不重建,设计复用 CLAUDE.md 指向的 cannbot 资料。
#include <algorithm>
#include <cstdint>
#include <tuple>

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
    TORCH_CHECK(self.is_contiguous(), "{op_name}: self must be contiguous");
    // CANNBot rms_norm 的显式 dtypeFlag 分派；element_size 只用于字节数计算。
    int64_t _dtypeFlag = -1;
    switch (self.scalar_type()) {
        case at::kFloat: _dtypeFlag = 0; break;
        case at::kHalf: _dtypeFlag = 1; break;
        case at::kBFloat16: _dtypeFlag = 2; break;
        case at::kInt: _dtypeFlag = 3; break;
        case at::kLong: _dtypeFlag = 4; break;
        case at::kChar: _dtypeFlag = 5; break;
        case at::kBool: _dtypeFlag = 6; break;
        case at::kByte: _dtypeFlag = 7; break;
        case at::kShort: _dtypeFlag = 8; break;
        case at::kDouble: _dtypeFlag = 9; break;
        default: TORCH_CHECK(false, "{op_name}: extend dtype dispatch for ", self.scalar_type());
    }

    at::Tensor _output = at::empty_like(self);

    int64_t _totalLength = self.numel();
    if (_totalLength == 0) {
        return _output;
    }
    int64_t _dtypeSize = self.element_size();

    auto _ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t _coreNum = static_cast<int64_t>(_ascendc_platform->GetCoreNumAiv());
    if (_coreNum <= 0) { _coreNum = 1; }
    uint64_t _ubSize = 0;
    _ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, _ubSize);

    int64_t _totalLengthCore = (_totalLength + _coreNum - 1) / _coreNum;
    int64_t _totalLengthCoreAlign = (_totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) /
                                   CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

    int64_t _usedCoreNum = (_totalLength + _totalLengthCoreAlign - 1) / _totalLengthCoreAlign;
    int64_t _formerNum = _usedCoreNum - 1;
    int64_t _formerLength = _totalLengthCoreAlign;
    int64_t _tailLength = _totalLength - _formerNum * _formerLength;

    int64_t _bufferCoefficient = _dtypeSize * 4;  // 全部输入 queue + 输出 queue，各双 buffer
    if (_bufferCoefficient <= 0) { _bufferCoefficient = 1; }
    int64_t _maxTileElements = static_cast<int64_t>(_ubSize) / _bufferCoefficient;
    int64_t _alignElements = 32 / (_dtypeSize > 0 ? _dtypeSize : 1);
    if (_alignElements <= 0) { _alignElements = 1; }
    int64_t _tileLength = (_maxTileElements / _alignElements) * _alignElements;
    if (_tileLength <= 0) { _tileLength = _alignElements; }

    uint32_t _blockDim = static_cast<uint32_t>(_usedCoreNum);

    EXEC_KERNEL_CMD({op_name}_kernel, _blockDim,
                    self, _output, _formerNum, _formerLength, _tailLength, _tileLength, _dtypeFlag);

    return _output;
}

}  // namespace ascend_kernel
