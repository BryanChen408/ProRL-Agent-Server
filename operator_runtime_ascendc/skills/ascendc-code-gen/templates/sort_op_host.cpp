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
// sort类算子 op_host 参考代码
// 适用: sort、topk等需要行内排序的算子
// 使用: 本文件演示了硬件参数获取、分核思路、chunkSize等，务必参考
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<op_name>.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> sort_op(const at::Tensor &self) {
    // ---- 获取硬件参数 ----
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t CORE_NUM = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t UB_SIZE_LIMIT = static_cast<int64_t>(ubSize);

    auto sizes = self.sizes();
    int64_t sortLength = sizes.back();

    // 计算行数
    int64_t totalRows = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()) - 1; i++) {
        totalRows *= sizes[i];
    }

    // chunkSize: 最大32倍数使得 chunkSize * 12 * 4 <= 192KB
    int64_t chunkSize = UB_SIZE_LIMIT / sizeof(float) / 12;
    chunkSize = (chunkSize / 32) * 32;

    int64_t numChunks = (sortLength + chunkSize - 1) / chunkSize;

    // sortLength 较小时缩小 chunkSize
    if (sortLength <= chunkSize) {
        chunkSize = (sortLength + 31) / 32 * 32;
        if (chunkSize == 0) chunkSize = 32;
        numChunks = 1;
    }

    // 核间切分参数
    int64_t formerNum = 0, formerRows = 0, tailNum = 0, tailRows = 0;
    int64_t usedCoreNum = 0;

    int64_t avgRows = totalRows / CORE_NUM;
    if (avgRows == 0) {
        formerNum  = totalRows; formerRows = 1;
    } else if (totalRows % CORE_NUM == 0) {
        formerNum  = CORE_NUM; formerRows = avgRows;
    } else {
        formerNum  = totalRows % CORE_NUM; formerRows = avgRows + 1;
        tailNum    = CORE_NUM - formerNum; tailRows = avgRows;
    }
    usedCoreNum = formerNum + tailNum;
    if (usedCoreNum == 0) usedCoreNum = 1;

    // Workspace: 每核 sortLength*4 floats (WS_A + WS_B)
    int64_t perCoreWsFloats = sortLength * 4;
    int64_t totalWsFloats = perCoreWsFloats * usedCoreNum;
    int64_t totalWorkspace = totalWsFloats * sizeof(float) + 16 * 1024 * 1024;
    auto workspace = at::empty({totalWorkspace},
        at::TensorOptions().dtype(at::kByte).device(self.device()));

    auto values  = at::empty_like(self);
    auto indices = at::empty(sizes, self.options().dtype(at::kInt));
    
    /*
     * 填充 tiling 结构体并 launch kernel
     * 例如：
     * EXEC_KERNEL_CMD(<op_name>, blockDim,
     * self, values, indices, workspace, 
     * formerNum, ......其他tiling参数);
     */
}
}  // namespace ascend_kernel
