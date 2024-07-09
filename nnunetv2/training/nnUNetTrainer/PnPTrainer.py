from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import math
import copy
import ml_collections
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from torch import distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from sklearn.model_selection import KFold, train_test_split
from torch.utils.checkpoint import checkpoint


class PnPTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-4
        self.momentum = 0.9599
        self.device = torch.device(type='cuda', index=5)
        # self.device = torch.device('cuda')
        self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        normalization =  configuration_manager.normalization_schemes
        print('Using normalization:  ', normalization)
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])

        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict(
            {'size': (16, 16, 16)})  # config.patches = ml_collections.ConfigDict({'size': (16, 16)})
        config.hidden_size = 768

        config.dim = 192
        config.dim_expand = 768
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 3072
        config.transformer.num_heads = 6
        config.transformer.num_layers = 4
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.1

        config.classifier = 'seg'
        config.representation_size = None
        config.resnet_pretrained_path = None
        # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
        config.patch_size = 16

        config.decoder_channels = (256, 128, 64, 16)
        config.n_classes = 26
        config.activation = 'softmax'
        config.batch_size = 2

        config.patches.grid = (16, 16, 16)  # config.patches.grid = (16, 16)
        config.resnet = ml_collections.ConfigDict()
        config.resnet.num_layers = (3, 4, 9)
        config.resnet.width_factor = 1

        config.decoder_channels = (256, 128, 64, 16)
        config.skip_channels = [512, 256, 64, 16]
        config.n_classes = num_classes  # 9
        config.n_skip = 3
        config.activation = 'softmax'

        img_size = (112,112,128)

        config.patches.grid = (
        int(img_size[0] / config.patch_size), int(img_size[1] / config.patch_size),
        int(img_size[2] / config.patch_size))
        ###
        config_n_patches = int(img_size[0] / config.patch_size) * int(
            img_size[1] / config.patch_size) * int(img_size[2] / config.patch_size)
        config.h = int(img_size[0] / config.patch_size)
        config.w = int(img_size[1] / config.patch_size)
        config.l = int(img_size[2] / config.patch_size)

        return network(in_channel=num_input_channels, out_channel=num_classes, training=True, config=config)

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
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr,weight_decay=self.weight_decay, eps=1e-4)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=True)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.num_epochs)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            # self.batch_size = self.configuration_manager.batch_size
            self.batch_size = 2
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            # global_batch_size = self.configuration_manager.batch_size
            global_batch_size = 2
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            print("worker", my_rank, "oversample", oversample_percents[my_rank])
            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]



def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class network(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True, config=None):
        super(network, self).__init__()
        self.dim = 512
        self.hybrid_model = ResNetV2(block_units=(2, 3, 5), width_factor=1)
        self.decoder_channels = (256, 128, 64, 16)
        self.skip_channels = [256, 128, 64, 16]
        channels = [16, 32, 64, 128, 256, 512]

        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=1, padding=0),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )

        self.decoder1 = nn.Sequential(
            Conv3dReLU(self.dim + self.skip_channels[0], self.decoder_channels[0], kernel_size=1, padding=0),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0] + self.skip_channels[1], self.decoder_channels[1], kernel_size=1,
                       padding=0),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1] + self.skip_channels[2], self.decoder_channels[2], kernel_size=1,
                       padding=0),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[2] + channels[0], channels[0], kernel_size=1, padding=0),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )

        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.cluster1 = CCM(config, self.skip_channels[0])
        self.cluster2 = CCM(config, self.skip_channels[1])
        self.cluster3 = CCM(config, self.skip_channels[2])

        self.cluster_center = nn.Parameter(torch.randn(1, config.n_classes, config.dim))

        self.segmentation_head = nn.Conv3d(channels[0], out_channel, kernel_size=3, padding=1)
        self.sdn1 = SDM(channels[4], channels[5])
        self.sdn2 = SDM(channels[3], channels[4])
        self.sdn3 = SDM(channels[2], channels[3])

        self.conv1 = DecoderResBlock(3 * config.n_classes, 3 * config.n_classes)
        self.conv2 = nn.Conv3d(3 * config.n_classes, config.n_classes, kernel_size=3, padding=1)

        self.intra_loss0 = intra_loss(self.skip_channels[2], config.dim, config.n_classes, 1 / 2)

    def forward(self, x, label):

        t1 = self.encoder1(x)
        print('t1',t1.shape)

        x, features = self.hybrid_model(t1)

        print('resnet output, ', x.shape)
        for i in features:
            print(i.shape)

        x = self.up(x)
        print('first upsampled', x.shape)
        class_token0, refined_center = self.cluster1(features[0], self.cluster_center)
        class_token0 = F.interpolate(class_token0, scale_factor=8, mode="trilinear")
        print('token0', class_token0.shape)

        features[0] = self.sdn1(features[0], x)
        print('features0',  features[0].shape, x.shape)
        x = torch.cat((x, features[0]), 1)
        print('before decoder1', x.shape)
        x = self.decoder1(x)
        print('after decoder1', x.shape)

        x = self.up(x)
        print('upsampled',x.shape)
        class_token1, refined_center = self.cluster2(features[1], refined_center)
        class_token1 = F.interpolate(class_token1, scale_factor=4, mode="trilinear")
        features[1] = self.sdn2(features[1], x)
        x = torch.cat((x, features[1]), 1)
        print('before decoder2', x.shape)
        x = self.decoder2(x)
        print('after decoder2', x.shape)

        x = self.up(x)
        class_token2, refined_center = self.cluster3(features[2], refined_center)
        loss0 = self.intra_loss0(refined_center, features[2], label)
        class_token2 = F.interpolate(class_token2, scale_factor=2, mode="trilinear")
        features[2] = self.sdn3(features[2], x)
        x = torch.cat((x, features[2]), 1)
        x = self.decoder3(x)

        x = self.up(x)
        x = torch.cat((x, t1), 1)
        x = self.decoder4(x)
        print('after decoder 4', x.shape)

        x = self.segmentation_head(x)

        # fuse class token features
        class_token = torch.cat((torch.cat((class_token0, class_token1), 1), class_token2), 1)
        print('class token before encod', class_token.shape)
        class_token = self.conv2(self.conv1(class_token))
        print('after encoding', class_token.shape)

        x = torch.sigmoid(x) * class_token + class_token

        return x, loss0


class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv3dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv3d(16, width, kernel_size=7, stride=2, bias=False, padding=3)),       # 原来是3
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*2, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*2, cout=width*2, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*2, cout=width*4, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _, _ = x.size()

        x = self.root(x)
        features.append(x)
        x = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = np.int32(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size()[2], right_size)
                feat = torch.zeros((b, x.size()[1], x.size()[2], x.size()[3], x.size()[4]), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]] = x[:]
            else:
                feat = x
            features.append(feat)

        x = self.body[-1](x)

        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.dim / self.num_attention_heads)  # 768 / 6 = 128
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.dim, self.all_head_size)
        self.key = Linear(config.dim, self.all_head_size)
        self.value = Linear(config.dim, self.all_head_size)

        self.out = Linear(config.dim, config.dim)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_attention_heads, config.n_classes, config.n_classes))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores + self.position_embeddings  # RPE

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.dim, config.hidden_size)
        self.fc2 = Linear(config.hidden_size, config.dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

        self.attention_norm = LayerNorm(config.dim, eps=1e-6)
        self.ffn_norm = LayerNorm(config.dim, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class self_attention(nn.Module):
    def __init__(self, config):
        super(self_attention, self).__init__()
        num_layers = 1
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.dim, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, class_center):
        for layer_block in self.layer:
            class_center = layer_block(class_center)

        encoded = self.encoder_norm(class_center)

        return encoded


# center: B * N * C
# feature_embeddings: B * C * H * W * L
# mask_embeddings: B * N * H * W * L
def center_update(value, soft_embeddings, center):
    center_residual = torch.matmul(soft_embeddings.flatten(2), value.flatten(2).transpose(-1, -2))
    center = center + center_residual
    return center


class CCM(nn.Module):
    def __init__(self, config, in_channel):
        super(CCM, self).__init__()
        self.SA = self_attention(config)
        self.key_projection = DecoderResBlock(in_channel, config.dim)
        self.value_projection = DecoderResBlock(in_channel, config.dim)
        self.resblock3 = DecoderResBlock(config.n_classes, config.n_classes)
        self.resblock4 = DecoderResBlock(config.n_classes, config.n_classes)

        self.classes = config.n_classes
        self.softmax = Softmax(dim=-2)  # softmax on the dimension of class numbers

    def forward(self, feature, class_center):
        b, c, h, w, l = feature.size()
        class_center = self.SA(class_center)
        key = self.key_projection(feature)
        value = self.value_projection(feature)

        mask_embeddings = torch.matmul(class_center, key.flatten(2))
        mask_embeddings = mask_embeddings / math.sqrt(self.classes)
        soft_embeddings = self.softmax(mask_embeddings)

        soft_embeddings = soft_embeddings.contiguous().view(b, self.classes, h, w, l)
        mask_embeddings = soft_embeddings + self.resblock4(self.resblock3(soft_embeddings))
        refined_center = center_update(value, soft_embeddings, class_center)
        return mask_embeddings, refined_center


class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv3dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)

        return x


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1

        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1

        kernel[:, :, 0, 2, 2] = -1
        kernel[:, :, 2, 0, 2] = -1
        kernel[:, :, 2, 2, 0] = -1

        kernel[:, :, 2, 2, 2] = 1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        print('diff', x_diff.shape, guidance_diff.shape)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)

class Conv3dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGNReLU, self).__init__(conv, gn, gelu)


class Conv3dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGN, self).__init__(conv, gn)


class intra_loss(nn.Module):
    def __init__(self, in_channels, dim, num_classes, scale, use_batchnorm=True):
        super(intra_loss, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.scale = scale
        self.dim = dim
        self.LN1 = LayerNorm(dim, eps=1e-6)
        self.LN2 = LayerNorm(dim, eps=1e-6)
        self.conv = Conv3dReLU(
            in_channels,
            dim,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, center, feature, label):
        # skipped feature: BN + RELU
        label = one_hot(label, self.num_classes).squeeze(2)
        # label = torch.nn.functional.one_hot(label, self.num_classes)
        label = F.interpolate(label.float(), scale_factor=self.scale, mode="trilinear")
        pred_center = self.LN1(torch.matmul(label.flatten(2), self.conv(feature).flatten(2).permute(0, 2, 1)))
        difference = (pred_center - self.LN2(center))

        difference_threshold = torch.clamp(difference, min=-1.0, max=1.0)
        loss = torch.mean(torch.square(difference_threshold))
        return loss

def one_hot(label, num_classes):
    tensor_list = []
    for i in range(num_classes):
        temp_prob = label == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


if __name__ == '__main__':
    strides = [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,1]]
    device = torch.device(type='cuda', index=5)
    m1 = torch.Tensor(size = (2, 1, 112, 112, 128)).to(device)

    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict(
        {'size': (16, 16, 16)})  # config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768

    config.dim = 192
    config.dim_expand = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 6
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 26
    config.activation = 'softmax'
    config.batch_size = 2

    config.patches.grid = (16, 16, 16)  # config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 105  # 9
    config.n_skip = 3
    config.activation = 'softmax'

    img_size = (112, 112, 128)

    config.patches.grid = (
        int(img_size[0] / config.patch_size), int(img_size[1] / config.patch_size),
        int(img_size[2] / config.patch_size))
    ###
    config_n_patches = int(img_size[0] / config.patch_size) * int(
        img_size[1] / config.patch_size) * int(img_size[2] / config.patch_size)
    config.h = int(img_size[0] / config.patch_size)
    config.w = int(img_size[1] / config.patch_size)
    config.l = int(img_size[2] / config.patch_size)

    model = network(in_channel=1, out_channel=105, training=True, config=config).to(device)

    print(model)

    num_elements = 2 * 1 * 112 * 112 * 128
    # Create a 1D tensor with values from 0 to 104, repeated enough times to match the size of l1
    index_tensor = torch.arange(105).repeat(num_elements // 105 + 1)[:num_elements]
    l1 = index_tensor.reshape(2, 1, 112, 112, 128).to(device)

    s1 = model(m1, l1)
    for j in s1:
        print(j.shape)
