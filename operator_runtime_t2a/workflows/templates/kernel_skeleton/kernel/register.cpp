#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

// 用 TORCH_LIBRARY 注册(不加 PYBIND11_MODULE)。注册在 .so 被 dlopen 时触发,
// 由 model_new_ascendc.py 的 torch.ops.load_library(...) 完成加载。
// m.def 的签名要和 ops.h、op_host/{op_name}.cpp 的实现、model_new 的调用一致。
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("{op_name}(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("{op_name}", TORCH_FN(ascend_kernel::{op_name}));
}

}  // namespace
