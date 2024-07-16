import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn3d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose3d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn3d(cin), nn.ReLU6(inplace=True),
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn3d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        # Check for NaNs
        # assert not torch.isnan(x).any(), 'NaNs detected after up_sample'

        for idx, layer in enumerate(self.conv):
            x = layer(x)
            # Check for NaNs
            # assert not torch.isnan(x).any(), f'NaNs detected after layer {idx} in conv'
        return x



class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True, use_IN = False, out_channel = 1):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule

        if sbn:
            bn3d = nn.SyncBatchNorm
        elif use_IN:
            bn3d = nn.InstanceNorm3d
        else:
            bn3d = nn.BatchNorm3d

        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn3d) for (cin, cout) in zip(channels[:-1], channels[1:])])

        self.proj = nn.Conv3d(channels[-1], out_channel, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]

            # x = self.dec[i](x)
            x = checkpoint(self.dec[i], x)
            # print('check dec x, ', i, torch.isnan(x).any(), torch.isinf(x).any())

        x = self.proj(x)
        # print('check projeced x, ',  torch.isnan(x).any(), torch.isinf(x).any())

        return x

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class DSDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn3d = nn.SyncBatchNorm if sbn else nn.BatchNorm3d
        # bn3d = nn.SyncBatchNorm if sbn else nn.InstanceNorm3d

        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn3d) for (cin, cout) in zip(channels[:-1], channels[1:])])

        self.proj = nn.ModuleList([nn.Conv3d(cout, 1, kernel_size=1) for cout in channels[1:]])
        # self.proj = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        ds = []
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
            ds.append(self.proj[i](x))
        return ds

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class SMiMDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn3d = nn.SyncBatchNorm if sbn else nn.BatchNorm3d
        # bn3d = nn.SyncBatchNorm if sbn else nn.InstanceNorm3d

        self.dec = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=channels[0],
                out_channels= channels[-1], kernel_size=32, stride = 16, padding=8),
            bn3d(channels[-1]), nn.ReLU6(inplace=True),
        )

        self.proj = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec):
        x = to_dec
        x = self.dec(x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class SMiMTwoDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn3d = nn.SyncBatchNorm if sbn else nn.BatchNorm3d
        # bn3d = nn.SyncBatchNorm if sbn else nn.InstanceNorm3d

        self.dec = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=channels[0],
                out_channels= channels[2], kernel_size=8, stride =4, padding=2),
            bn3d(channels[2]), nn.ReLU6(inplace=True),
            nn.ConvTranspose3d(
                in_channels=channels[2],
                out_channels=channels[-1], kernel_size=8, stride=4, padding=2),
            bn3d(channels[-1]), nn.ReLU6(inplace=True),
        )

        self.proj = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec):
        x = to_dec[0]
        x = self.dec(x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)



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
            x = checkpoint(self.conv_blocks_context[d], x)
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



