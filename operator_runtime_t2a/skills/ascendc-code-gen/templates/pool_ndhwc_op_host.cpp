/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



// ============================================================
// Pooling 算子 op_host 模板
// 适用: Pooling 等算子
// 使用: 复制到 csrc/ops/<op_name>/op_host/<op_name>.cpp，
//       替换 <op_name>/<OpName>/<dtype> 等占位符
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<op_name>.h"


static inline int64_t AvgPool3DOutputShape(
    const int64_t inputSize, const int64_t kernelSize, const int64_t padL, const int64_t stride, const bool ceilMode) {
    int64_t outputSize = (stride == 0) ? -1 :
                         (inputSize + padL * 2 - kernelSize + (ceilMode ? stride - 1 : 0)) /stride + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= inputSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

static inline int64_t divRtn(const int64_t x, const int64_t y) {
  if (y == 0) { return 0; }
  int64_t q = x / y;
  int64_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  };
  return q;
}

static inline int64_t MaxPool3DOutputShape(const int64_t inputSize, const int64_t kernelSize, const int64_t padL,
                                            const int64_t padR, const int64_t stride, const int64_t dilation,
                                            const bool ceilMode) {
  int64_t outputSize =
      divRtn(inputSize + padL + padR - dilation * (kernelSize - 1) - 1 + (ceilMode ? stride - 1 : 0), stride) + 1;

  if (ceilMode) {
    if ((outputSize - 1) * stride >= inputSize + padL) {
      --outputSize;
    }
  }
  return outputSize;
}

struct TilingSplitParams {
    int64_t formerNum, tailNum, formerLength, tailLength;
    uint64_t usedCoreNum;
};

static inline TilingSplitParams ComputeTilingSplit(int64_t outputNum, int64_t coreNum) {
    TilingSplitParams p;
    TORCH_CHECK(coreNum != 0, "ComputeTilingSplit: coreNum should not be 0");
    if (outputNum < coreNum) {
        p.formerNum = outputNum;
        p.tailNum = 0UL;
        p.formerLength = 1UL;
        p.tailLength = 0UL;
        p.usedCoreNum = static_cast<uint64_t>(outputNum);
    } else if (outputNum % coreNum == 0UL) {
        p.formerNum = coreNum;
        p.tailNum = 0UL;
        p.formerLength = outputNum / coreNum;
        p.tailLength = 0UL;
        p.usedCoreNum = static_cast<uint64_t>(coreNum);
    } else {
        p.formerNum = outputNum % coreNum;
        p.tailNum = coreNum - p.formerNum;
        p.formerLength = outputNum / coreNum + 1UL;
        p.tailLength = outputNum / coreNum;
        p.usedCoreNum = static_cast<uint64_t>(coreNum);
    }
    return p;
}

at::Tensor pooling_op(const at::Tensor& self,
                      at::IntArrayRef kernel_size,
                      at::IntArrayRef stride,
                      at::IntArrayRef padding) {    //ceil_mode, count_include_pad, divisor_override...用例信息参数传递

    // ---- 获取硬件参数 ----核数量，ub空间大小参数
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    // 参数解析
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inputD = input.size(2);
    int64_t inputH = input.size(3);
    int64_t inputW = input.size(4);
    int64_t kernelD = kernel_size[0];
    int64_t kernelH = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[1];
    int64_t kernelW = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[2];
    // ... stride, padding, ceil_mode, count_include_pad, divisor_override 解析

    // 参数校验
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat || self.scalar_type() == at::kBFloat16,
                "<op_name>: only float16, float32 and bfloat16 are supported, got ", self.scalar_type());
    //数据类型、长度、取值范围校验...

    // 计算输出形状并创建,ceilmode=1需要特殊处理计算输出形状
    int64_t outputD = AvgPool3DOutputShape(inputD, kernelD, padD, strideD, ceilMode);   // maxpool根据MaxPool3DOutputShape适配
    int64_t outputH = AvgPool3DOutputShape(inputH, kernelH, padH, strideH, ceilMode);   // maxpool根据MaxPool3DOutputShape适配
    int64_t outputW = AvgPool3DOutputShape(inputW, kernelW, padW, strideW, ceilMode);   // maxpool根据MaxPool3DOutputShape适配

    // 准备输入数据，NCDHWC → NDHWC格式
    at::Tensor xNDHWC = self.permute({0, 2, 3, 4, 1}).contiguous();
    at::Tensor outputNDHWC = at::empty({N, outputD, outputH, outputW, C},
                                 self.options().dtype(self.scalar_type()));

    // 计算tiling切分参数-核间处理，每个核处理数据量
    auto tiling = ComputeTilingSplit(N * outputD * outputH * outputW, coreNum);
    int64_t formerNum = tiling.formerNum;
    int64_t tailNum = tiling.tailNum;
    int64_t formerLength = tiling.formerLength;
    int64_t tailLength = tiling.tailLength;
    uint64_t usedCoreNum = tiling.usedCoreNum;

    // 计算tiling切分参数-核内处理，UB空间单次处理的数据量， UB空间使用最好预留1k
    int64_t alignNum = 32 / sizeof(T);  
    int64_t alignC = ((channels + alignNum - 1) / alignNum) * alignNum;
    //...其他参数

    //windowWNum计算公式
    int64_t indiceSize = 0;            // avgpool: 0, maxpool: 4 (sizeof(int32_t))
    int64_t rowExtra = (kW - 1) * alignC * (elemSize + 4);
    int64_t perWindow = alignC * (sW * (elemSize + 4) + 4 + indiceSize);
    int64_t windowWNum = (ubSizeLimit - rowExtra) / perWindow;
    if (windowWNum < 1) windowWNum = 1;
    if (windowWNum > oW) windowWNum = oW;

    // 启动kernel
    EXEC_KERNEL_CMD(pooling_op, xNDHWC, output, N, C, inputD, inputH, inputW, outputD, outputH, outputW, ...); //用例信息、tiling切分参数传递

    // NDHWC → NCDHW格式
    output = outputNDHWC.permute({0, 4, 1, 2, 3}).contiguous();
    return output;
}