# from nnunetv2.training.nnUNetTrainer.SomeTrainer import SomeTrainer
# from variants.network_architecture.STUNet_head import STUNet
# from variants.network_architecture.encoder3D import SparseEncoder
# from variants.network_architecture.decoder3D import LightDecoder
# from variants.network_architecture.spark3D import SparK
#
# import torch
# from torch import nn
# from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
# from torch.nn.parallel import DistributedDataParallel as DDP
#
#
# class STUNetTrainer(SomeTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.num_epochs = 1000
#         self.initial_lr = 5e-5
#         self.device = torch.device(type='cuda', index=4)
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
#         strides = configuration_manager.pool_op_kernel_sizes[1:]
#         if len(strides) > 5:
#             strides = strides[:5]
#         while len(strides) < 5:
#             strides.append([1, 1, 1])
#         return STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
#                       pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
#                       enable_deep_supervision=enable_deep_supervision)
#
#
#
#
