from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.dims = dims

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0],
                          use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                              stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                  for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 16

    def get_feature_map_channels(self):
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.dims[:5]

    def forward(self, x, hierarchical = False):
        skips = []

        for d in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
        if hierarchical:
            return skips
        else:
            return x

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        # self.norm1 = nn.BatchNorm3d(output_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        # self.norm2 = nn.BatchNorm3d(output_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)



