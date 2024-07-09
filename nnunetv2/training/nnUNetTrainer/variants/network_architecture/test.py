import torch
import torch.nn as nn
from STUNet_head import STUNet

# Assuming STUNet is defined properly in your environment
# and pool_op_kernel_sizes, conv_kernel_sizes are set appropriately
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

stunet = nn.Sequential(
    STUNet(1, 1, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
           pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
           enable_deep_supervision=True),
    nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global Average Pooling
    nn.Flatten())  # Flatten the output

# Create a dummy input tensor
d = torch.randn(2, 1, 128, 128, 128)

# Pass the tensor through the model
output = stunet(d)

# Print the shape of the output
print(output.shape)
