// {op_name}: implement the reference contract before evaluation.
#include "ops.h"
#include "torch_kernel_helper.h"
#include "aclrtlaunch_{op_name}_kernel.h"

namespace ascend_kernel {

at::Tensor {op_name}(const at::Tensor &x)
{
    // Design outputs, dtype dispatch, tiling and kernel launch for this operator.
    TORCH_CHECK(false, "{op_name}: host/kernel implementation is missing; implement the reference contract");
}

} // namespace ascend_kernel
