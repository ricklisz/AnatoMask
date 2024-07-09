# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import torch
# from torch import nn
# import numpy as np
# from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
# from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.nn.functional as F
# from torch import distributed as dist
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import trunc_normal_, DropPath, to_3tuple
#
# try:
#     from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ops_3D_DCN import modules as opsm
# except:
#     opsm = None
#
# class UniRepLKTrainer(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.num_epochs = 1000
#         self.initial_lr = 1e-4
#         self.momentum = 0.9599
#         self.device = torch.device(type='cuda', index=7)
#         self.weight_decay = 1e-5
#
#     @staticmethod
#     def build_network_architecture(plans_manager,
#                                    dataset_json,
#                                    configuration_manager,
#                                    num_input_channels,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         num_classes = label_manager.num_segmentation_heads
#         kernel_sizes = [[3, 3, 3]] * 6
#         normalization =  configuration_manager.normalization_schemes
#         print('Using normalization:  ', normalization)
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#         return UniRepLKUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[24 * x for x in [1, 2, 4, 8, 16, 16]],
#                         groups = [3 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
#                                                            self.configuration_manager,
#                                                            self.num_input_channels,
#                                                            enable_deep_supervision=True).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Compiling network...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers()
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             self.was_initialized = True
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr,weight_decay=self.weight_decay, eps=1e-4)
#         # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=True)
#
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.num_epochs)
#         # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#
#         return optimizer, lr_scheduler
#
#     def _set_batch_size_and_oversample(self):
#         if not self.is_ddp:
#             # set batch size to what the plan says, leave oversample untouched
#             self.batch_size = self.configuration_manager.batch_size
#         else:
#             # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
#             batch_sizes = []
#             oversample_percents = []
#
#             world_size = dist.get_world_size()
#             my_rank = dist.get_rank()
#
#             global_batch_size = self.configuration_manager.batch_size
#             assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
#                                                     'GPUs... Duh.'
#
#             batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)
#
#             for rank in range(world_size):
#                 if (rank + 1) * batch_size_per_GPU > global_batch_size:
#                     batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
#                 else:
#                     batch_size = batch_size_per_GPU
#
#                 batch_sizes.append(batch_size)
#
#                 sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
#                 sample_id_high = np.sum(batch_sizes)
#
#                 if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
#                     oversample_percents.append(0.0)
#                 elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
#                     oversample_percents.append(1.0)
#                 else:
#                     percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
#                     oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
#                                                     sample_id_low / global_batch_size) / percent_covered_by_this_rank)
#                     oversample_percents.append(oversample_percent_here)
#
#             print("worker", my_rank, "oversample", oversample_percents[my_rank])
#             print("worker", my_rank, "batch_size", batch_sizes[my_rank])
#             # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
#             # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])
#
#             self.batch_size = batch_sizes[my_rank]
#             self.oversample_foreground_percent = oversample_percents[my_rank]
#
# class UniRepLKTrainer_small(UniRepLKTrainer):
#     @staticmethod
#     def build_network_architecture(plans_manager,
#                                    dataset_json,
#                                    configuration_manager,
#                                    num_input_channels,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         num_classes = label_manager.num_segmentation_heads
#         kernel_sizes = [[3, 3, 3]] * 6
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#
#         return UniRepLKUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[24 * x for x in [1, 2, 4, 8, 16, 16]],
#                         groups = [3 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
#
# class UniRepLKTrainer_small_ft(UniRepLKTrainer_small):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#
#
# class UniRepLKTrainer_base(UniRepLKTrainer):
#     @staticmethod
#     def build_network_architecture(plans_manager,
#                                    dataset_json,
#                                    configuration_manager,
#                                    num_input_channels,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         num_classes = label_manager.num_segmentation_heads
#         normalization = configuration_manager.normalization_schemes
#         print('Using normalization:  ', normalization)
#
#         kernel_sizes = [[3, 3, 3]] * 6
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#
#         return UniRepLKUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
#                             LK_kernels=[[3],[3],
#                                       [13],
#                                       [13],[3]],
#                             pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
# class UniRepLKTrainer_base_hybridv2(UniRepLKTrainer_base):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#
#
# class UniRepLKTrainer_large(UniRepLKTrainer):
#     @staticmethod
#     def build_network_architecture(plans_manager,
#                                    dataset_json,
#                                    configuration_manager,
#                                    num_input_channels,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         num_classes = label_manager.num_segmentation_heads
#         kernel_sizes = [[3, 3, 3]] * 6
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#
#         return UniRepLKUNet(num_input_channels, num_classes, depth=[2] * 6, dims=[72 * x for x in [1, 2, 4, 8, 16, 16]],
#                         groups = [3 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
# class UniRepLKTrainer_large_ft(UniRepLKTrainer_large):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#
#
# class UniRepLKTrainer_huge(UniRepLKTrainer):
#     @staticmethod
#     def build_network_architecture(plans_manager,
#                                    dataset_json,
#                                    configuration_manager,
#                                    num_input_channels,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         num_classes = label_manager.num_segmentation_heads
#         kernel_sizes = [[3, 3, 3]] * 6
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#         return UniRepLKUNet(num_input_channels, num_classes, depth=[3] * 6, dims=[96 * x for x in [1, 2, 4, 8, 16, 16]],
#                         groups = [3 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
#
# class UniRepLKTrainer_huge_ft(UniRepLKTrainer_huge):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.deep_supervision = True
#
# class UniRepLKUNet(nn.Module):
#     def __init__(self, input_channels, num_classes,LK_kernels,depth=[2,2,2,2,2,2], dims=[1, 2, 4, 8, 16,16]*48,
#                  pool_op_kernel_sizes=None, conv_kernel_sizes=None, drop_path_rate=0.,enable_deep_supervision=True):
#         super().__init__()
#         self.conv_op = nn.Conv3d
#         self.input_channels = input_channels
#         self.num_classes = num_classes
#         self.LK_kernels = LK_kernels
#
#         self.final_nonlin = lambda x: x
#         self.decoder = Decoder()
#         self.decoder.deep_supervision = enable_deep_supervision
#         self.upscale_logits = False
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
#         print('=========== drop path rates: ', dp_rates)
#
#         self.pool_op_kernel_sizes = pool_op_kernel_sizes
#         self.conv_kernel_sizes = conv_kernel_sizes
#         self.conv_pad_sizes = []
#         for krnl in self.conv_kernel_sizes:
#             self.conv_pad_sizes.append([i // 2 for i in krnl])
#
#         num_pool = len(pool_op_kernel_sizes)
#
#         assert num_pool == len(dims) - 1
#
#         # encoder
#         self.conv_blocks_context = nn.ModuleList()
#         stage = nn.Sequential(
#             BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
#             *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
#               range(depth[0] - 1)])
#
#         self.conv_blocks_context.append(stage)
#
#         cur = 0
#         for d in range(1, num_pool + 1):
#             # if self.LK_kernels[d-1][0] >= 1:
#             if d < num_pool:
#                 stage = nn.Sequential(
#                     ModResBlock2(dims[d - 1], dims[d], kernel_size=self.LK_kernels[d - 1][0], use_1x1conv=True),
#                     *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
#                                     for _ in range(depth[d] - 1)])
#                 cur += depth[d]
#             else:
#                 stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
#                                                 stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
#                                   *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
#                                     for _ in range(depth[d] - 1)])
#                 cur += depth[d]
#             self.conv_blocks_context.append(stage)
#
#         # upsample_layers
#         self.upsample_layers = nn.ModuleList()
#         for u in range(num_pool):
#             upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
#             self.upsample_layers.append(upsample_layer)
#
#         # decoder
#         self.conv_blocks_localization = nn.ModuleList()
#         for u in range(num_pool):
#             stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
#                                                 self.conv_pad_sizes[-2 - u], use_1x1conv=True),
#                                   *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
#                                                   self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
#             self.conv_blocks_localization.append(stage)
#
#         # outputs
#         self.seg_outputs = nn.ModuleList()
#         for ds in range(len(self.conv_blocks_localization)):
#             self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))
#
#         self.upscale_logits_ops = []
#         for usl in range(num_pool - 1):
#             self.upscale_logits_ops.append(lambda x: x)
#
#     def forward(self, x):
#         skips = []
#         seg_outputs = []
#
#         for d in range(len(self.conv_blocks_context) - 1):
#             x = self.conv_blocks_context[d](x)
#             skips.append(x)
#         x = self.conv_blocks_context[-1](x)
#
#         for u in range(len(self.conv_blocks_localization)):
#             x = self.upsample_layers[u](x)
#             x = torch.cat((x, skips[-(u + 1)]), dim=1)
#             x = self.conv_blocks_localization[u](x)
#             seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
#
#         if self.decoder.deep_supervision:
#             return tuple([seg_outputs[-1]] + [i(j) for i, j in
#                                               zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
#         else:
#             return seg_outputs[-1]
#
# class BasicResBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
#         super().__init__()
#         self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
#         self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
#         self.act1 = nn.LeakyReLU(inplace=True)
#         self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
#         self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
#         self.act2 = nn.LeakyReLU(inplace=True)
#         # self.se = SEBlock(output_channels, output_channels//4)
#
#         if use_1x1conv:
#             self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
#         else:
#             self.conv3 = None
#
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.act1(self.norm1(y))
#         y = self.norm2(self.conv2(y))
#
#         # y = self.se(y)
#
#         if self.conv3:
#             x = self.conv3(x)
#         y += x
#         return self.act2(y)
#
# class Upsample_Layer_nearest(nn.Module):
#     def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
#         super().__init__()
#         self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
#         self.pool_op_kernel_size = pool_op_kernel_size
#         self.mode = mode
#
#     def forward(self, x):
#         x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
#         x = self.conv(x)
#         return x
#
# class DownsampleBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_size=1, padding=1, stride=1):
#         super().__init__()
#         self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
#         # self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
#         # self.norm = LayerNorm(output_channels, eps=1e-6, data_format="channels_first")
#         self.norm = nn.InstanceNorm3d(output_channels, affine=True)
#         self.act = nn.LeakyReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x
#
# class GRNwithNHWDC(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
#     This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
#     We assume the inputs to this layer are (N, H, W, C)
#     """
#     def __init__(self, dim, use_bias=True):
#         super().__init__()
#         self.use_bias = use_bias
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
#         if self.use_bias:
#             self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
#
#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         if self.use_bias:
#             return (self.gamma * Nx + 1) * x + self.beta
#         else:
#             return (self.gamma * Nx + 1) * x
#
# class to_channels_first(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x.permute(0, 4, 1, 2, 3)
#
# class to_channels_last(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x.permute(0, 2, 3, 4, 1)
#
# def build_norm_layer(dim,
#                      norm_layer,
#                      in_format='channels_last',
#                      out_format='channels_last',
#                      eps=1e-6):
#     layers = []
#     if norm_layer == 'BN':
#         if in_format == 'channels_last':
#             layers.append(to_channels_first())
#         layers.append(nn.BatchNorm3d(dim))
#         if out_format == 'channels_last':
#             layers.append(to_channels_last())
#     elif norm_layer == 'LN':
#         if in_format == 'channels_first':
#             layers.append(to_channels_last())
#         layers.append(nn.LayerNorm(dim, eps=eps))
#         if out_format == 'channels_first':
#             layers.append(to_channels_first())
#     else:
#         raise NotImplementedError(
#             f'build_norm_layer does not support {norm_layer}')
#     return nn.Sequential(*layers)
#
# def build_act_layer(act_layer):
#     if act_layer == 'ReLU':
#         return nn.ReLU(inplace=True)
#     elif act_layer == 'SiLU':
#         return nn.SiLU(inplace=True)
#     elif act_layer == 'GELU':
#         return nn.GELU()
#     raise NotImplementedError(f'build_act_layer does not support {act_layer}')
#
# def get_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
#                attempt_use_lk_impl=True):
#     kernel_size = to_3tuple(kernel_size)
#     if padding is None:
#         padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
#     else:
#         padding = to_3tuple(padding)
#     need_large_impl = kernel_size[0] == kernel_size[1] == kernel_size[2] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
#
#     if attempt_use_lk_impl and need_large_impl:
#         print('---------------- trying to import iGEMM implementation for large-kernel conv')
#         try:
#             from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
#             print('---------------- found iGEMM implementation ')
#         except:
#             DepthWiseConv2dImplicitGEMM = None
#             print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
#         if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
#                 and out_channels == groups and stride == 1 and dilation == 1:
#             print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
#             return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
#     return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                      padding=padding, dilation=dilation, groups=groups, bias=bias)
#
# def get_bn(dim, use_sync_bn=False):
#     if use_sync_bn:
#         return nn.SyncBatchNorm(dim)
#     else:
#         # return nn.BatchNorm3d(dim)
#         return nn.InstanceNorm3d(dim, affine=True)
#
# class SEBlock(nn.Module):
#     """
#     Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
#     We assume the inputs to this layer are (N, C, H, W)
#     """
#     def __init__(self, input_channels, internal_neurons):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool3d((1,1,1))
#         self.down = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons,
#                               kernel_size=1, stride=1, bias=True)
#         self.up = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels,
#                             kernel_size=1, stride=1, bias=True)
#         self.input_channels = input_channels
#         # self.nonlinear = nn.ReLU(inplace=True)
#         self.nonlinear = nn.LeakyReLU(inplace=True)
#
#     def forward(self, inputs):
#         x = self.pool(inputs)
#         x = self.down(x)
#         x = self.nonlinear(x)
#         x = self.up(x)
#         x = F.sigmoid(x)
#         return inputs * x.view(-1, self.input_channels, 1, 1, 1)
#
# def fuse_bn(conv, bn):
#     conv_bias = 0 if conv.bias is None else conv.bias
#     std = (bn.running_var + bn.eps).sqrt()
#     return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
#
# def fuse_in(conv, inst_norm):
#     conv_bias = 0 if conv.bias is None else conv.bias
#     # For instance norm, the standard deviation calculation is not needed.
#     # Use the affine parameters directly.
#     return conv.weight * inst_norm.weight.reshape(-1, 1, 1, 1, 1), inst_norm.bias + conv_bias
#
#
# def convert_dilated_to_nondilated(kernel, dilate_rate):
#     identity_kernel = torch.ones((1, 1, 1, 1, 1)).to(kernel.device)
#     if kernel.size(1) == 1:
#         #   This is a DW kernel
#         dilated = F.conv_transpose3d(kernel, identity_kernel, stride=dilate_rate)
#         return dilated
#     else:
#         #   This is a dense or group-wise (but not DW) kernel
#         slices = []
#         for i in range(kernel.size(1)):
#             dilated = F.conv_transpose3d(kernel[:,i:i+1,:,:,:], identity_kernel, stride=dilate_rate)
#             slices.append(dilated)
#         return torch.cat(slices, dim=1)
#
# def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
#     large_k = large_kernel.size(2)
#     dilated_k = dilated_kernel.size(2)
#     equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
#     equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
#     rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
#     merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 6)
#     return merged_kernel
#
#
# class BaselineConvBlock(nn.Module):
#     """
#     Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
#     We assume the inputs to this block are (N, C, H, W)
#     """
#     def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=13, stride=1, padding=6, bias=True)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class DilatedReparamBlock(nn.Module):
#     """
#     Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
#     We assume the inputs to this block are (N, C, H, W)
#     """
#     def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
#         super().__init__()
#         self.lk_origin = get_conv3d(channels, channels, kernel_size, stride=1,
#                                     padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
#                                     attempt_use_lk_impl=attempt_use_lk_impl)
#         self.attempt_use_lk_impl = attempt_use_lk_impl
#
#         #   Default settings. We did not tune them carefully. Different settings may work better.
#         if kernel_size == 17:
#             self.kernel_sizes = [5, 9, 3, 3, 3]
#             self.dilates = [1, 2, 4, 5, 7]
#         elif kernel_size == 15:
#             self.kernel_sizes = [5, 7, 3, 3, 3]
#             self.dilates = [1, 2, 3, 5, 7]
#         elif kernel_size == 13:
#             self.kernel_sizes = [5, 7, 3, 3, 3]
#             self.dilates = [1, 2, 3, 4, 5]
#         elif kernel_size == 11:
#             self.kernel_sizes = [5, 5, 3, 3, 3]
#             self.dilates = [1, 2, 3, 4, 5]
#         elif kernel_size == 9:
#             self.kernel_sizes = [5, 5, 3, 3]
#             self.dilates = [1, 2, 3, 4]
#         elif kernel_size == 7:
#             self.kernel_sizes = [5, 3, 3]
#             self.dilates = [1, 2, 3]
#         elif kernel_size == 5:
#             self.kernel_sizes = [3, 3]
#             self.dilates = [1, 2]
#         else:
#             raise ValueError('Dilated Reparam Block requires kernel_size >= 5')
#
#         if not deploy:
#             self.origin_bn = get_bn(channels, use_sync_bn)
#             for k, r in zip(self.kernel_sizes, self.dilates):
#                 self.__setattr__('dil_conv_k{}_{}'.format(k, r),
#                                  nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
#                                            padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
#                                            bias=False))
#                 self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))
#
#     def forward(self, x):
#         if not hasattr(self, 'origin_bn'):      # deploy mode
#             return self.lk_origin(x)
#         out = self.origin_bn(self.lk_origin(x))
#         for k, r in zip(self.kernel_sizes, self.dilates):
#             conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
#             bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
#             out = out + bn(conv(x))
#         return out
#
#     def merge_dilated_branches(self):
#         if hasattr(self, 'origin_bn'):
#             # origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
#             origin_k, origin_b = fuse_in(self.lk_origin, self.origin_bn)
#             for k, r in zip(self.kernel_sizes, self.dilates):
#                 conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
#                 bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
#                 # branch_k, branch_b = fuse_bn(conv, bn)
#                 branch_k, branch_b = fuse_in(conv, bn)
#                 origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
#                 origin_b += branch_b
#             merged_conv = get_conv3d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
#                                     padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
#                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
#             merged_conv.weight.data = origin_k
#             merged_conv.bias.data = origin_b
#             self.lk_origin = merged_conv
#             self.__delattr__('origin_bn')
#             for k, r in zip(self.kernel_sizes, self.dilates):
#                 self.__delattr__('dil_conv_k{}_{}'.format(k, r))
#                 self.__delattr__('dil_bn_k{}_{}'.format(k, r))
#
# class UniRepLKNetBlock(nn.Module):
#
#     def __init__(self,
#                  dim,
#                  kernel_size,
#                  drop_path=0.,
#                  layer_scale_init_value=1e-6,
#                  deploy=False,
#                  attempt_use_lk_impl=True,
#                  with_cp=False,
#                  use_sync_bn=False,
#                  ffn_factor=4):
#         super().__init__()
#         self.with_cp = with_cp
#         if deploy:
#             print('------------------------------- Note: deploy mode')
#         if self.with_cp:
#             print('****** note with_cp = True, reduce memory consumption but may slow down training ******')
#
#         self.need_contiguous = (not deploy) or kernel_size >= 7
#         self.kernel_size = kernel_size
#         if kernel_size == 0:
#             self.dwconv = nn.Identity()
#             self.norm = nn.Identity()
#         elif deploy:
#             self.dwconv = get_conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
#                                      dilation=1, groups=dim, bias=True,
#                                      attempt_use_lk_impl=attempt_use_lk_impl)
#             self.norm = nn.Identity()
#         elif kernel_size >= 7:
#             self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
#                                               use_sync_bn=use_sync_bn,
#                                               attempt_use_lk_impl=attempt_use_lk_impl)
#
#             # self.dwconv = BaselineConvBlock(dim, kernel_size, deploy=deploy,
#             #                                   use_sync_bn=use_sync_bn,
#             #                                   attempt_use_lk_impl=attempt_use_lk_impl)
#
#             self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
#         elif kernel_size == 1:
#             self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
#                                     dilation=1, groups=1, bias=deploy)
#             self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
#         else:
#             assert kernel_size in [3, 5]
#             self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
#                                     dilation=1, groups=dim, bias=deploy)
#             self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
#
#         # self.se = SEBlock(dim, dim // 4)
#
#         ffn_dim = int(ffn_factor * dim)
#         # self.pwconv1 = nn.Sequential(
#         #     to_channels_last(),
#         #     nn.Linear(dim, ffn_dim))
#         # self.act = nn.Sequential(
#         #     nn.GELU(),
#         #     GRNwithNHWDC(ffn_dim, use_bias=not deploy))
#         # if deploy:
#         #     self.pwconv2 = nn.Sequential(
#         #         nn.Linear(ffn_dim, dim),
#         #         to_channels_first())
#         # else:
#         #     self.pwconv2 = nn.Sequential(
#         #         nn.Linear(ffn_dim, dim, bias=False),
#         #         to_channels_first(),
#         #         get_bn(dim, use_sync_bn=use_sync_bn))
#
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
#                                   requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
#                                                          and layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def compute_residual(self, x):
#         # y = self.se(self.norm(self.dwconv(x)))
#         y = self.norm(self.dwconv(x))
#         # y = self.pwconv2(self.act(self.pwconv1(y)))
#         if self.gamma is not None:
#             y = self.gamma.view(1, -1, 1, 1, 1) * y
#         return self.drop_path(y)
#
#     def forward(self, inputs):
#         def _f(x):
#             return x + self.compute_residual(x)
#             # return self.compute_residual(x)
#         if self.with_cp and inputs.requires_grad:
#             out = checkpoint.checkpoint(_f, inputs)
#         else:
#             out = _f(inputs)
#         return out
#
#     # def reparameterize(self):
#     #     if hasattr(self.dwconv, 'merge_dilated_branches'):
#     #         self.dwconv.merge_dilated_branches()
#     #     if hasattr(self.norm, 'running_var'):
#     #         std = (self.norm.running_var + self.norm.eps).sqrt()
#     #         if hasattr(self.dwconv, 'lk_origin'):
#     #             self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1, 1)
#     #             self.dwconv.lk_origin.bias.data = self.norm.bias + (
#     #                         self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
#     #         else:
#     #             conv = nn.Conv3d(self.dwconv.in_channels, self.dwconv.out_channels, self.kernel_size,
#     #                              padding=self.dwconv.padding, groups=self.dwconv.groups, bias=True)
#     #             conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1, 1)
#     #             conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
#     #             self.dwconv = conv
#     #         self.norm = nn.Identity()
#     #     if self.gamma is not None:
#     #         final_scale = self.gamma.data
#     #         self.gamma = None
#     #     else:
#     #         final_scale = 1
#     #     if self.act[1].use_bias and len(self.pwconv2) == 3:
#     #         grn_bias = self.act[1].beta.data
#     #         self.act[1].__delattr__('beta')
#     #         self.act[1].use_bias = False
#     #         linear = self.pwconv2[0]
#     #         grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
#     #         bn = self.pwconv2[2]
#     #         std = (bn.running_var + bn.eps).sqrt()
#     #         new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
#     #         new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
#     #         linear_bias = 0 if linear.bias is None else linear.bias.data
#     #         linear_bias += grn_bias_projected_bias
#     #         new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
#     #         self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])
#
#     def reparameterize(self):
#         if hasattr(self.dwconv, 'merge_dilated_branches'):
#             self.dwconv.merge_dilated_branches()
#
#         # Adjusting for InstanceNorm
#         if isinstance(self.norm, nn.InstanceNorm3d):  # Assuming 3D, change as needed
#             if hasattr(self.dwconv, 'lk_origin'):
#
#                 self.dwconv.lk_origin.weight.data *= self.norm.weight.view(-1, 1, 1, 1, 1)
#                 self.dwconv.lk_origin.bias.data = self.norm.bias + self.dwconv.lk_origin.bias
#             else:
#                 conv = nn.Conv3d(self.dwconv.in_channels, self.dwconv.out_channels, self.kernel_size,
#                                  padding=self.dwconv.padding, groups=self.dwconv.groups, bias=True)
#                 conv.weight.data = self.dwconv.weight * self.norm.weight.view(-1, 1, 1, 1, 1)
#                 conv.bias.data = self.norm.bias + self.dwconv.bias
#                 self.dwconv = conv
#
#         if self.gamma is not None:
#             final_scale = self.gamma.data
#             self.gamma = None
#         else:
#             final_scale = 1
#
#         if self.act[1].use_bias and len(self.pwconv2) == 3:
#             grn_bias = self.act[1].beta.data
#             self.act[1].__delattr__('beta')
#             self.act[1].use_bias = False
#             linear = self.pwconv2[0]
#             grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
#
#             # Assuming 'bn' is replaced by instance norm
#             inst_norm = self.pwconv2[2]
#             new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
#             new_linear.weight.data = linear.weight * inst_norm.weight.view(-1, 1)
#             linear_bias = 0 if linear.bias is None else linear.bias.data
#             linear_bias += grn_bias_projected_bias
#             new_linear.bias.data = inst_norm.bias + linear_bias * inst_norm.weight
#             self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])
#
#
# class ModResBlock1(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False, use_sync_bn=False):
#         super().__init__()
#         self.module1 = DownsampleBlock(input_channels, output_channels, 3, 1, 2)
#         self.module2 = nn.Sequential(
#             *[
#         nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding),
#         nn.InstanceNorm3d(output_channels, affine=True),
#         SEBlock(output_channels, output_channels//4),
#             ]
#         )
#         self.act2 = nn.LeakyReLU(inplace=True)
#         if use_1x1conv:
#             self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=2)
#         else:
#             self.conv3 = None
#
#     def forward(self, x):
#         y = self.module1(x)
#         y = self.module2(y)
#         if self.conv3:
#             x = self.conv3(x)
#         y += x
#         return self.act2(y)
#
#
# class ModResBlock2(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False, use_sync_bn=False):
#         super().__init__()
#         self.module1 = DownsampleBlock(input_channels, output_channels, 3, 1, 2)
#         self.module2 = UniRepLKNetBlock(dim=output_channels, kernel_size=kernel_size)
#         self.act2 = nn.LeakyReLU(inplace=True)
#         if use_1x1conv:
#             self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=2)
#         else:
#             self.conv3 = None
#
#     def forward(self, x):
#         y = self.module1(x)
#         y = self.module2(y)
#         if self.conv3:
#             x = self.conv3(x)
#         y += x
#         return self.act2(y)
#
# class LayerNorm(nn.Module):
#     r""" LayerNorm implementation used in ConvNeXt
#     LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#         self.reshape_last_to_first = reshape_last_to_first
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
#             return x
#
# if __name__ == '__main__':
#     strides = [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,1]]
#     device = torch.device(type='cuda', index=7)
#     m1 = torch.Tensor(size = (1, 1, 112, 112, 128)).to(device)
#     model = UniRepLKUNet(1,105,depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
#                         LK_kernels=[[3],[3],
#                                       [13],
#                                       [13],[3]], pool_op_kernel_sizes=strides, conv_kernel_sizes= [[3, 3, 3]] * 6,
#                       enable_deep_supervision=True).to(device)
#
#     print(model)
#     import time
#     t1 = time.time()
#     s1 = model(m1)
#     t2 = time.time()
#     print('inference: ', t2-t1)
#     for j in s1:
#         print(j.shape)
#     from fvcore.nn import FlopCountAnalysis, flop_count_table
#     flops = FlopCountAnalysis(model, m1)
#     print('Total flops: ', flops.total())
#     print(flop_count_table(flops))
#
#
