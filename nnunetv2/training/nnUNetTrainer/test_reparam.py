import torch
import torch.nn as nn
from UniRepLKTrainer import UniRepLKUNet
import time
strides = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
device = torch.device(type='cuda', index=1)
m1 = torch.Tensor(size=(1, 1, 112, 112, 128)).to(device)
model = UniRepLKUNet(1, 105, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                     LK_kernels=[[3], [3],
                                 [13],
                                 [13], [3]], pool_op_kernel_sizes=strides, conv_kernel_sizes=[[3, 3, 3]] * 6,
                     enable_deep_supervision=True).to(device)

weights = torch.load('/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/UniRepLKTrainer_base_hybridv2_ft__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth')['network_weights']

model.load_state_dict(weights)

m1 = torch.Tensor(size = (1, 1, 112, 112, 128)).to(device)

# t1 = time.time()
# s1 = model(m1)
# t2 = time.time()
# print('before reparam: ', t2-t1)

for m in model.modules():
    if hasattr(m, 'reparameterize'):
        m.reparameterize()

t1 = time.time()
s1 = model(m1)
t2 = time.time()
print('after reparam: ', t2-t1)


