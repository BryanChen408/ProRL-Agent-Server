// {op_name}: device entry placeholder, not an implementation.
// Choose arguments, dtypes, buffers and computation from the reference.
// Update the host launch arguments together with this entry.
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void {op_name}_kernel()
{
    // TODO: implement the operator. The generated host rejects execution until implemented.
}
