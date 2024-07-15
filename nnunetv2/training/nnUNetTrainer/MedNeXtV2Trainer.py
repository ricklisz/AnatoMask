from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
# from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

enc_norm_type='group'
dec_norm_type='group'
checkpoint_style = None
deep_supervision = True

class MedNeXtV2Trainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 0
        self.initial_lr = 1e-4
        self.device = torch.device(type='cuda', index=0)

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        return MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_classes,
            exp_r=2,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=enable_deep_supervision,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2] * 9,
            checkpoint_style = checkpoint_style,
            enc_norm_type=enc_norm_type,
            dec_norm_type=dec_norm_type
        )

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                           self.initial_lr,
                                           weight_decay=self.weight_decay,
                                           eps=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.num_epochs)
        return optimizer, lr_scheduler


class MedNeXtV2Trainer_S(MedNeXtV2Trainer):
    pass

class MedNeXtV2Trainer_S_ft(MedNeXtV2Trainer_S):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class MedNeXtV2Trainer_B(MedNeXtV2Trainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        return MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_classes,
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=enable_deep_supervision,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2] * 9,
            checkpoint_style = checkpoint_style,
            enc_norm_type=enc_norm_type,
            dec_norm_type=dec_norm_type
        )

class MedNeXtV2Trainer_B_ft(MedNeXtV2Trainer_B):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class MedNeXtV2Trainer_M(MedNeXtV2Trainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        return MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_classes,
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=False,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = checkpoint_style,
            enc_norm_type=enc_norm_type,
            dec_norm_type=dec_norm_type
        )

class MedNeXtV2Trainer_M_ft(MedNeXtV2Trainer_M):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

class MedNeXtV2Trainer_L(MedNeXtV2Trainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        return MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_classes,
            exp_r= [3,4,8,8,8,8,8,4,3],     # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=enable_deep_supervision,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = checkpoint_style,
            enc_norm_type=enc_norm_type,
            dec_norm_type=dec_norm_type
        )

class MedNeXtV2Trainer_L_ft(MedNeXtV2Trainer_L):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class MedNeXt(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 enc_norm_type='layer', dec_norm_type='layer',dim='3d'
                 ):

        super().__init__()

        self.decoder = Decoder()
        self.decoder.deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.n_channels = n_channels
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True

        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=enc_norm_type,
                dim=dim
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=enc_norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=enc_norm_type,
                dim=dim
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=enc_norm_type,
            dim=dim
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=enc_norm_type,
                dim=dim
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=enc_norm_type,
            dim=dim
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=enc_norm_type,
                dim=dim
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=enc_norm_type,
            dim=dim
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=enc_norm_type,
                dim=dim
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=dec_norm_type,
            dim=dim
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=dec_norm_type,
                dim=dim
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=dec_norm_type,
            dim=dim
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=dec_norm_type,
                dim=dim
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=dec_norm_type
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=dec_norm_type,
                dim=dim
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=dec_norm_type,
            dim=dim
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=dec_norm_type,
                dim=dim
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes,dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if self.do_ds:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes,dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes,dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes,dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes,dim=dim)

        self.block_counts = block_counts

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
        return [self.n_channels, self.n_channels*2, self.n_channels*4, self.n_channels*8, self.n_channels*16]

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x, hierarchical=False):

        ls = []

        x = self.stem(x)

        x_res = [None] * 4
        x_ds = [None] * 5  # x_ds_0 to x_ds_4

        if self.outside_block_checkpointing:
            for idx in range(4):
                x_res[idx] = self.iterative_checkpoint(getattr(self, f'enc_block_{idx}'), x)
                if hierarchical:
                    ls.append(x_res[idx])
                x = checkpoint.checkpoint(getattr(self, f'down_{idx}'), x_res[idx], self.dummy_tensor)
            x = self.iterative_checkpoint(self.bottleneck, x)
            if hierarchical:
                ls.append(x)

            if self.do_ds and hasattr(self, f'out_4'):
                x_ds[4] = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)
            for idx in reversed(range(4)):
                x_up = checkpoint.checkpoint(getattr(self, f'up_{idx}'), x, self.dummy_tensor)
                dec_x = x_res[idx] + x_up
                x = self.iterative_checkpoint(getattr(self, f'dec_block_{idx}'), dec_x)
                if self.do_ds and hasattr(self, f'out_{idx}'):
                    x_ds[idx] = checkpoint.checkpoint(getattr(self, f'out_{idx}'), x, self.dummy_tensor)
                del x_res[idx], x_up
            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)
        else:
            for idx in range(4):
                x_res[idx] = getattr(self, f'enc_block_{idx}')(x)
                if hierarchical:
                    ls.append(x_res[idx])
                x = getattr(self, f'down_{idx}')(x_res[idx])

            x = self.bottleneck(x)
            if hierarchical:
                ls.append(x)

            if hasattr(self, 'out_4'):
                x_ds[4] = self.out_4(x)
            for idx in reversed(range(4)):
                x_up = getattr(self, f'up_{idx}')(x)
                dec_x = x_res[idx] + x_up
                x = getattr(self, f'dec_block_{idx}')(dec_x)
                if self.do_ds and hasattr(self, f'out_{idx}'):
                    x_ds[idx] = getattr(self, f'out_{idx}')(x)
                del x_res[idx], x_up
            x = self.out_0(x)

        if hierarchical:
            return ls
        elif self.do_ds:
            return [x, *x_ds[1:]]
        else:
            return x

class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='3d'
                 ):

        super().__init__()

        self.do_res = do_res

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels)
            self.conv2 = conv(
                in_channels = in_channels,
                out_channels = exp_r*in_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
            self.conv3 = conv(
                in_channels = exp_r*in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        elif norm_type == 'layer':
            self.norm = nn.Sequential(
                to_channels_last(),
                LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_last'
            ))
            self.conv2 = nn.Linear(in_channels, exp_r * in_channels)
            # Third convolution (Compression) layer with Conv3D 1x1x1
            self.conv3 = nn.Sequential(
                nn.Linear(exp_r * in_channels, out_channels),
                to_channels_first()
            )
        # GeLU activations
        self.act = nn.GELU()
        # self.grn = GRN(exp_r * in_channels)


    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        # x1 = self.grn(x1)
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d'):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d'):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim)

        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                output_padding=1
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
            output_padding=1
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 4, 1)