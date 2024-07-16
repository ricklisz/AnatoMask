import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
_cur_active: torch.Tensor = None            # B1ff
# todo: try to use `gather` for speed?
def _get_active_ex_or_ii(H, W, D, returning_active_ex=True):
    h_repeat, w_repeat, d_repeat = H // _cur_active.shape[-3], W // _cur_active.shape[-2], D// _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3).repeat_interleave(d_repeat, dim=4)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi, di

def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x = x * _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D = x.shape[4], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x

def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
    bhwdc = x.permute(0,2,3,4,1)
    nc = bhwdc[ii]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)  # use BN1d to normalize this flatten feature `nc`
    bchwd = torch.zeros_like(bhwdc)
    bchwd[ii] = nc
    bhwdc = bchwd.permute(0,4,1,2,3)
    return bhwdc

class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool3d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool3d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm3d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm3d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseGroupNorm(nn.GroupNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, num_groups, num_channels, eps=1e-6, sparse=True):
        super().__init__( num_groups, num_channels, eps)
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 5:  # BHWDC or BCHWD
            if self.sparse:
                ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
                bhwdc = x.permute(0, 2, 3, 4, 1)
                nc = bhwdc[ii]
                nc = super(SparseGroupNorm, self).forward(nc)
                x = torch.zeros_like(bhwdc)
                x[ii] = nc
                return x.permute(0,4,1,2,3)
            else:
                return super(SparseGroupNorm, self).forward(x)
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseGroupNorm, self).forward(x)

    def __repr__(self):
        return super(SparseGroupNorm, self).__repr__()[
               :-1] + f', sp={self.sparse})'

class GRNwithNHWDC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, D, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class SparseGRN(nn.Module):
    def __init__(self, dim, use_bias=True, sparse=True):
        super().__init__()
        self.use_bias = use_bias
        self.sparse = sparse
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        if x.ndim != 5:  # Assuming we are dealing with 5D tensors
            raise NotImplementedError("SparseGRN supports only 5D tensors")

        if self.sparse:
            # Get active indices
            ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)

            # Apply GRN on active indices
            nc = x[ii]
            Gx = torch.norm(nc, p=2, dim=1, keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            normalized_nc = (self.gamma * (nc * Nx) + self.beta) if self.use_bias else (self.gamma * (nc * Nx))

            # Create a zero tensor and fill in the normalized values
            x_new = torch.zeros_like(x)
            x_new[ii] = normalized_nc
            return x_new
        else:
            # Apply GRN to the entire tensor
            Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            return (self.gamma * Nx + 1) * x + self.beta if self.use_bias else (self.gamma * Nx + 1) * x

    def __repr__(self):
        return f"SparseGRN(dim={self.gamma.size(1)}, use_bias={self.use_bias}, sparse={self.sparse})"


class SparseInstanceNorm(nn.InstanceNorm1d):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, num_features, eps=1e-6, sparse=True):
        super().__init__(num_features, eps, affine=True)
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 5:  # BHWDC or BCHWD
            if self.sparse:
                ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
                bchwd = x
                cn = x[ii[0], :, ii[1],  ii[2],  ii[3]].transpose(0, 1)
                cn = super(SparseInstanceNorm, self).forward(cn)
                x = torch.zeros_like(bchwd)
                x[ii[0], :, ii[1], ii[2], ii[3]] = cn.t()
                return x
            else:
                return super(SparseInstanceNorm, self).forward(x)
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseInstanceNorm, self).forward(x)

    def __repr__(self):
        return super(SparseInstanceNorm, self).__repr__()[
               :-1] + f', sp={self.sparse})'

class SparseAdaptiveAvgPooling(nn.AdaptiveAvgPool3d):
    def __init__(self, output_size, sparse=True):
        super().__init__(output_size)
        self.output_size = output_size
        self.sparse = sparse
    def forward(self, x):   # shape: BCHWD
        unmasked_positions = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=True)
        mean = (x * unmasked_positions).sum(dim=(2,3,4), keepdims=True) / (unmasked_positions.sum(dim=(2,3,4), keepdims=True)  + 1e-6)
        return mean         # shape: BC111

class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 5:  # BHWDC or BCHWD
            if self.data_format == "channels_last":  # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], D=x.shape[3], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:  # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
                    bhwdc = x.permute(0, 2, 3, 4, 1)
                    nc = bhwdc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                    x = torch.zeros_like(bhwdc)
                    x[ii] = nc
                    return x.permute(0, 4, 1, 2, 3)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[
               :-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'


class SparseConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=True, ks=7):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim)  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,4,1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)  # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0,4,1,2,3)  # (N, H, W, C) -> (N, C, H, W)

        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3],D=x.shape[4], returning_active_ex=True)

        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'

class SparseEncoder(nn.Module):
    """
       SparseEncoder Class for converting dense models to sparse and encoding inputs.

       Parameters:
       -----------
       cnn : nn.Module
           CNN model to be converted to a sparse model.
       input_size : tuple
           The size of the input data.
       sbn : bool, default=False
           Flag to indicate whether to use synchronized batch normalization.
       verbose : bool, default=False
           Flag to indicate whether to print detailed conversion information.
       """
    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_ratio, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv3d):
            if not getattr(m, 'skip_sparse_conversion', False):
                m: nn.Conv3d
                bias = m.bias is not None
                oup = SparseConv3d(
                    m.in_channels, m.out_channels,
                    kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                    dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
                )
                oup.weight.data.copy_(m.weight.data)
                if bias:
                    oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool3d):
            m: nn.MaxPool3d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation,
                                   return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool3d):
            m: nn.AvgPool3d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode,
                                   count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, nn.GroupNorm):
            m: nn.GroupNorm
            oup = SparseGroupNorm(m.num_groups, m.num_channels, eps=m.eps)

        elif isinstance(m, nn.InstanceNorm3d):
            m: nn.InstanceNorm3d
            oup = SparseInstanceNorm(m.num_features, m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.AdaptiveAvgPool3d):
            m: nn.AdaptiveAvgPool3d
            oup = SparseAdaptiveAvgPooling(output_size = (1,1,1))

        elif isinstance(m, GRNwithNHWDC):  # Replace DenseGRN with the actual name of your dense GRN class
            m: GRNwithNHWDC
            oup = SparseGRN(dim=m.num_features, use_bias=True, sparse=True)
            oup.gamma.data.copy_(m.gamma.data)
            if m.use_bias:
                oup.beta.data.copy_(m.beta.data)

        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            m: nn.BatchNorm3d
            oup = (SparseSyncBatchNorm3d if sbn else SparseBatchNorm3d)(m.weight.shape[0], eps=m.eps,
                                                                        momentum=m.momentum, affine=m.affine,
                                                                        track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError

        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)