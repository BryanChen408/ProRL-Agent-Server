import torch
import torch_npu
import os
device = int(os.getenv('LOCAL_RANK'))
print(device)
torch.npu.set_device(device)
# Call the hccl init process
torch.distributed.init_process_group(backend='hccl', init_method='env://')
a = torch.tensor(1).npu()
torch.distributed.all_reduce(a)
print('Hccl: ', a, device)
