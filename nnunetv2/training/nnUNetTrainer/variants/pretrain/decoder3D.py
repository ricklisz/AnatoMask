import math
from typing import List
import time
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_3tuple
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

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

        for idx, layer in enumerate(self.conv):
            x = layer(x)
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
            x = self.dec[i](x)
        x = self.proj(x)

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

if __name__ == '__main__':
    device = torch.device(type='cuda', index=3)

    a1 = torch.cuda.memory_allocated(device=device)
    t1 = time.time()
    decoder = LightDecoder(16,sbn=False, width = 512, out_channel = 1).to(device)
    input = [
        torch.Tensor(size =(2,512,7,7,8)).to(device),
        torch.Tensor(size =(2, 256, 14, 14, 16)).to(device),
        torch.Tensor(size =(2, 128, 28, 28, 32)).to(device),
        torch.Tensor(size =(2, 64, 56, 56, 64)).to(device)
    ]

    output = decoder(input)
    t2 = time.time()
    a2 = torch.cuda.memory_allocated(device=device)
    print(output.shape)
    print(a2-a1)
    print(t2-t1)

