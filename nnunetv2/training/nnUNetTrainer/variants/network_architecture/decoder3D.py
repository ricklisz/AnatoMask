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

def get_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_3tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    else:
        padding = to_3tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] == kernel_size[2] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        # return nn.BatchNorm3d(dim)
        return nn.InstanceNorm3d(dim, affine=True)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.down = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1, 1)


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv3d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        self.need_contiguous = (not deploy) or kernel_size >= 7
        self.kernel_size = kernel_size
        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def _f(x):
            return x + self.compute_residual(x)
            # return self.compute_residual(x)
        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out

class UniBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False, use_sync_bn=False, drop_path=0.0,  layer_scale_init_value = 1e-6):
        super().__init__()
        self.module1 = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.module2 = UniRepLKNetBlock(dim=output_channels, kernel_size=kernel_size,drop_path=drop_path,  layer_scale_init_value =  layer_scale_init_value)
        self.act2 = nn.LeakyReLU(inplace=True)
        if use_1x1conv:
            self.conv3 = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=True)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.module1(x)
        y = self.module2(y)
        if self.conv3:
            x = self.conv3(x)
            y += x
        return self.act2(y)

class UniDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True, use_IN = False, LK_kernels = [3,3,3,3], out_channel = 1):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
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

        self.dec = nn.ModuleList([UniBlock(cin, cout, k) for (cin, cout, k) in zip(channels[:-1], channels[1:], LK_kernels)])

        self.proj = nn.Conv3d(channels[-1], out_channel, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
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
class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=2, hidden_dim=2048, act='gelu'):
        super(ProjHead, self).__init__()
        self.nlayers = nlayers
        self.layers = nn.ModuleList()

        for i in range(nlayers):
            if i == 0:
                layer_in_dim = in_dim
            else:
                layer_in_dim = hidden_dim

            if i == nlayers - 1:
                layer_out_dim = out_dim
            else:
                layer_out_dim = hidden_dim

            self.layers.append(nn.Linear(layer_in_dim, layer_out_dim))

            if i < nlayers - 1:  # Add activation and normalization to all but last layer
                if act.lower() == 'gelu':
                    self.layers.append(nn.GELU())
                else:
                    self.layers.append(nn.ReLU())  # Default to ReLU if not GELU

                self.layers.append(nn.LayerNorm(layer_out_dim))  # Simplified normalization choice

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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

class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU6(), # why relu? Who knows
            nn.InstanceNorm3d(channels, affine=True)
        )
        self.predict = nn.Conv3d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class SegDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True, use_IN = False, out_channel = 1):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule

        self.dec = SegFormerDecoder(channels[-1]//2, channels[:-1], [8,4,2,1])

        self.proj = SegFormerSegmentationHead(
            channels[-1]//2, out_channel, num_features=n
        )

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        features = self.dec(to_dec)
        x = self.proj(features)
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

class DistillDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True, use_IN = False, out_channel = 1, embed_dim = 1024):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
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
        self.proj2 = ProjHead(in_dim=up_sample_ratio**3, out_dim=embed_dim)
        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
            # print('check dec x, ', i, torch.isnan(x).any(), torch.isinf(x).any())

        x = self.proj(x)
        x = self.proj2(x)
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

if __name__ == '__main__':
    device = torch.device(type='cuda', index=3)

    a1 = torch.cuda.memory_allocated(device=device)
    t1 = time.time()
    decoder = SegDecoder(16,sbn=False, width = 512, out_channel = 1).to(device)
    # decoder = LightDecoder(16,sbn=False, width = 512, out_channel = 1).to(device)
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

