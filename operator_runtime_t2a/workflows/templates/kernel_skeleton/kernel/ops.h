#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// host 函数声明。签名按你的算子改(输入个数/标量参数/返回类型),
// register.cpp 的 m.def 和 model_new_ascendc.py 的调用要保持一致。
at::Tensor {op_name}(const at::Tensor &x);

} // namespace ascend_kernel

#endif // OPS_H
