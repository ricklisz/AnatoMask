# region Description of the region
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
import time
from time import sleep
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from timm.utils import ModelEma
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from STUNet_head import STUNet
from MedNeXt_head import MedNeXt
from encoder3D import SparseEncoder
from decoder3D import LightDecoder, DistillDecoder
from spark3D_HPM import SparK
from torch.cuda.amp import GradScaler, autocast
import sys
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from typing import Union, Tuple, List

import math
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from utils.lr_control import lr_wd_annealing, get_param_groups
from utils import dist
import monai

# Training transforms, data augmentation pipelines
def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                            rotation_for_DA: dict,
                            deep_supervision_scales: Union[List, Tuple],
                            mirror_axes: Tuple[int, ...],
                            do_dummy_2d_data_aug: bool,
                            order_resampling_data: int = 3,
                            order_resampling_seg: int = 1,
                            border_val_seg: int = -1,
                            use_mask_for_norm: List[bool] = None,
                            is_cascaded: bool = False,
                            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                            ignore_label: int = None) -> AbstractTransform:
    tr_transforms = []
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    # First augmentation transform, dont change
    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    if do_dummy_2d_data_aug:
        tr_transforms.append(Convert2DTo3DTransform())


    # CHANGE HERE!!!!!!

    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
    #                                            p_per_channel=0.5))
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
    #                                                     p_per_channel=0.5,
    #                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
    #                                                     ignore_axes=ignore_axes))
    # tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    # tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                           mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if is_cascaded:
        assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
        tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
        tr_transforms.append(ApplyRandomBinaryOperatorTransform(
            channel_idx=list(range(-len(foreground_labels), 0)),
            p_per_sample=0.4,
            key="data",
            strel_size=(1, 8),
            p_per_label=1))
        tr_transforms.append(
            RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                key="data",
                p_per_sample=0.2,
                fill_with_other_class_p=0,
                dont_do_if_covers_more_than_x_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        # the ignore label must also be converted
        tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                   if ignore_label is not None else regions,
                                                                   'target', 'target'))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                          output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                              is_cascaded: bool = False,
                              foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                              regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                              ignore_label: int = None) -> AbstractTransform:
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))

    if is_cascaded:
        val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        # the ignore label must also be converted
        val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                    if ignore_label is not None else regions,
                                                                    'target', 'target'))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                           output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms

def calculate_3d_ssim_per_patch(img, rec, patch_size=(16, 16, 16)):
    B, C, H, W, D = img.shape
    ssim_values = torch.zeros((B, 1, H // patch_size[0], W // patch_size[1], D // patch_size[2]), device = img.device)
    # mse_values = torch.zeros_like(ssim_values)

    # ssim3D = monai.metrics.regression.SSIMMetric(spatial_dims=3)

    # Calculate the starting index for each patch
    patch_indices = [(h, w, d) for h in range(0, H, patch_size[0])
                                for w in range(0, W, patch_size[1])
                                for d in range(0, D, patch_size[2])]

    for b in range(B):  # Iterate over each sample in the batch
        for (h, w, d) in patch_indices:
            # Extract the current patch for img and rec
            patch_img = img[b, :, h:h+patch_size[0], w:w+patch_size[1], d:d+patch_size[2]]
            patch_rec = rec[b, :, h:h+patch_size[0], w:w+patch_size[1], d:d+patch_size[2]]

            # Calculate 3D SSIM for the current patch. Assuming ssim3D returns a scalar value per patch
            # ssim_patch = ssim3D(patch_img.unsqueeze(0), patch_rec.unsqueeze(0)).item()  # Add batch dimension back, then remove it
            ssim_patch = calculate_ncc_3d_torch(patch_img.unsqueeze(0), patch_rec.unsqueeze(0)).item()
            # mse_patch = ((patch_img - patch_rec) ** 2).mean().item()  # Calculate MSE and convert to Python scalar

            # Store the SSIM and MSE values
            ssim_values[b, :, h // patch_size[0], w // patch_size[1], d // patch_size[2]] = ssim_patch
            # mse_values[b, :, h // patch_size[0], w // patch_size[1], d // patch_size[2]] = mse_patch

    return torch.reshape(ssim_values, (B, -1))

def calculate_ncc_3d_torch(patch_a, patch_b):
    """
    Calculate the normalized cross-correlation for batches of 3D patches using PyTorch.
    Assumes patch_a and patch_b are tensors of shape (N, C, H, W, D) where:
    N is the batch size,
    C is the number of channels (set to 1 for grayscale patches),
    H, W, D are the dimensions of the patches.
    """
    # Ensure patches are of float type
    patch_a = patch_a.float()
    patch_b = patch_b.float()

    # Calculate means across the spatial dimensions (H, W, D) for each patch in the batch
    mean_a = torch.mean(patch_a, dim=(2, 3, 4), keepdim=True)
    mean_b = torch.mean(patch_b, dim=(2, 3, 4), keepdim=True)

    # Calculate standard deviations across the spatial dimensions (H, W, D) for each patch in the batch
    std_a = torch.std(patch_a, dim=(2, 3, 4), keepdim=True)
    std_b = torch.std(patch_b, dim=(2, 3, 4), keepdim=True)

    # Avoid division by zero
    std_a[std_a == 0] = 1
    std_b[std_b == 0] = 1

    # Normalize patches
    norm_a = (patch_a - mean_a) / std_a
    norm_b = (patch_b - mean_b) / std_b

    # Calculate NCC
    ncc = torch.mean(norm_a * norm_b, dim=(2, 3, 4))

    return ncc

def get_score(ssim, mse, mask):
    ssim = ssim.to(mask.device, non_blocking=True)
    mse = mse.to(mask.device, non_blocking=True)
    norm_ssim = (ssim - ssim.min()) / (ssim.max() - ssim.min())
    norm_ssim = norm_ssim * mask
    norm_mse = (mse - mse.min()) / (mse.max() - mse.min())
    norm_mse = norm_mse * mask
    weighting1 = norm_mse - norm_ssim+1
    return weighting1


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
                    self.layers.append(nn.LeakyReLU())  # Default to ReLU if not GELU

                self.layers.append(nn.LayerNorm(layer_out_dim))  # Simplified normalization choice

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FinalModel(nn.Module):
    def __init__(self, base_model, head):
        super(FinalModel, self).__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, x):
        # Pass input through the base model
        x = self.base_model(x)
        # Now pass the output of the base model to the head
        x = self.head(x)
        return x

# endregion

device = torch.device("cuda:5")

pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# STUNet_B
head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
               enable_deep_supervision=True).to(device)
# STUNet_L
# from GC import STUNet
# head = STUNet(1,1,depth=[2] * 6, dims=[64 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
#               enable_deep_supervision=True).to(device)
# STUNet_H
# head = STUNet(1,1,depth=[3] * 6, dims=[96 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
#             enable_deep_supervision=True).to(device)

model_name = 'STUNet_B'

class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

use_hog = False

enc = SparseEncoder(head, input_size=(112, 112, 128), sbn=False).to(device)
if use_hog:
    dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 3).to(device)
    loss_dec = None
else:
    dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 1).to(device)
    loss_dec = None

# model_without_ddp = SparK(
#     sparse_encoder=enc, dense_decoder=dec, loss_decoder= loss_dec, mask_ratio=0.6,
#     densify_norm='in', use_hog=use_hog
# ).to(device)

model_without_ddp = SparK(
    sparse_encoder=enc, dense_decoder=dec, loss_decoder= None, mask_ratio=0.6,
    densify_norm='in', use_hog=use_hog
).to(device)

model_without_ddp = FinalModel(model_without_ddp, ProjHead(4096, 1024)).to(device)
# model_without_ddp = FinalModel(model_without_ddp, nn.Identity()).to(device)

# model_without_ddp = torch.compile(model_without_ddp)

model_ema = ModelEma(model_without_ddp, decay=0.999, device=device, resume='')

# checkpoint = torch.load('/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/Pretraining/STUNet_B_HPM_no_dec_add_loss/STUNet_B_head_latest.pt')
# from collections import OrderedDict
#
# pretrained_state = checkpoint['network_weights']
# new_weights = OrderedDict((key.replace('module.', ''), value) for key, value in pretrained_state.items())
# for k in new_weights.keys():
#     print(k)
#
# for key, values in new_weights.items():
#
#     if values.dtype in [torch.float16, torch.float32, torch.float64]:
#         print(f"{key}: Mean = {values.mean().item()}")
#     else:
#         print(f"{key}: Mean cannot be computed for dtype {values.dtype}")
#
#     # Check for NaNs and Infs in the tensor
#     if torch.isnan(values).any():
#         print(f"{key} contains NaN values.")
#     elif torch.isinf(values).any():
#         print(f"{key} contains Inf values.")
#     else:
#         print(f"{key} looks fine - no NaN or Inf values detected.")
#
# missing, unexpected = model_without_ddp.load_state_dict(new_weights, strict=False)
# model_ema.ema.load_state_dict(new_weights, strict=False)
#
# assert len(missing) == 0, f'load_state_dict missing keys: {missing}'
# # assert len(unexpected) == 0, f'load_state_dict unexpected keys: {unexpected}'
#
# del pretrained_state, new_weights

model = LocalDDP(model_without_ddp)

# # Chnage this every time...
fold = 0
epoch = 1000
batch_size = 4
opt = 'adamw'
ada = 0.999
lr = 1e-4
weight_decay = 1e-5
clip = 12
wd = 0.04
wde = 0.2
wp_ep = 8
warmup = 20
AMP = False
guide = True
alpha = 0.9
student_temp, teacher_temp, temp2 = 0.1, 0.04, 0.04

output_folder = '/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/Pretraining/' + model_name +'_HPM_no_dec_distill'
timestamp = datetime.now()
maybe_mkdir_p(output_folder)

log_file = join(output_folder, 'training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt'%
                     (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                      timestamp.second))
#
def print_to_log_file(*args, also_print_to_console=True, add_timestamp=True):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)
#

preprocessed_dataset_folder = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans_3d_fullres'
splits_file = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/splits_final.json'
splits = load_json(splits_file)

all_keys = splits[fold]['train']
tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)
# tr_keys, _ = train_test_split(tr_keys, test_size=0.6, random_state=42)

dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                           folder_with_segs_from_previous_stage=None,
                           num_images_properties_loading_threshold=0)
dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                            folder_with_segs_from_previous_stage=None,
                            num_images_properties_loading_threshold=0)

dataset_json =load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/dataset.json')
plans = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans.json')
plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration('3d_fullres')
label_manager = plans_manager.get_label_manager(dataset_json)

patch_size = configuration_manager.patch_size
dim = len(patch_size)
rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
initial_patch_size = get_patch_size(patch_size[-dim:],
                                    *rotation_for_DA.values(),
                                    (0.85, 1.25))

dl_tr = nnUNetDataLoader3D(dataset_tr, batch_size,
                           initial_patch_size,
                           configuration_manager.patch_size,
                           label_manager,
                           oversample_foreground_percent=0.33,
                           sampling_probabilities=None, pad_sides=None)

dl_val = nnUNetDataLoader3D(dataset_val, batch_size,
                           configuration_manager.patch_size,
                           configuration_manager.patch_size,
                           label_manager,
                           oversample_foreground_percent=0.33,
                           sampling_probabilities=None, pad_sides=None)

iters_train = len(dataset_tr) // batch_size
iters_val = len(dataset_val) // batch_size

deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
mirror_axes = (0, 1, 2)

tr_transforms = get_training_transforms(
    patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, False,
    order_resampling_data=3, order_resampling_seg=1,
    use_mask_for_norm=configuration_manager.use_mask_for_norm,
    is_cascaded=False, foreground_labels=label_manager.foreground_labels,
    regions=label_manager.foreground_regions if label_manager.has_regions else None,
    ignore_label=label_manager.ignore_label)

val_transforms = get_validation_transforms(
    deep_supervision_scales,
    is_cascaded=False,
    foreground_labels=label_manager.foreground_labels,
    regions=label_manager.foreground_regions if
    label_manager.has_regions else None,
    ignore_label=label_manager.ignore_label)

allowed_num_processes = get_allowed_n_proc_DA()

mt_gen_train = LimitedLenWrapper(iters_train, data_loader=dl_tr, transform=tr_transforms,
                                 num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                 pin_memory= True, wait_time=0.02)

mt_gen_val = LimitedLenWrapper(iters_val, data_loader=dl_val,
                               transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                               num_cached=3, seeds=None, pin_memory=True,
                               wait_time=0.02)

# # build optimizer and lr_scheduler
param_groups = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
opt_clz = {
    'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
    'adamw': partial(torch.optim.AdamW, betas=(0.9, ada)),
    #     'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, ada), max_grad_norm=5.0),
}[opt]

optimizer = opt_clz(params=param_groups, lr=lr, weight_decay=weight_decay)
# print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup, epoch, 1e-6)
# optimizer.load_state_dict(checkpoint['optimizer_state'])
#
# for i in range(202):
#     scheduler.step()

it = 0
epoch_loss = []
epoch_ema_loss = []
val_loss = []
optimizer.zero_grad()

scaler = GradScaler()
best_val_loss = 1e4
val_every = 1
logger = nnUNetLogger()

ema_loss_vector = None
ema_loss_vector_val = None

for i in range(epoch):
    model.train()
    per_loss = 0.0
    per_p_loss, per_t_loss = 0.0, 0.0
    print_to_log_file('')
    print_to_log_file(f'Epoch {i}')
    print_to_log_file()


    print_to_log_file(
        f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
    logger.log('epoch_start_timestamps', time.time(), i)
    # add this
    if i < epoch//4:
        model_ema.decay = 0.999 + i / (epoch//4) * (0.9999 - 0.999)
    else:
        model_ema.decay = 0.9999

    for idx in range(iters_train):

        inp = next(mt_gen_train)
        inp = inp['data']
        inp = inp.to(device, non_blocking=True)

        if AMP:
            # with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast():
                if model_ema is not None:
                    with torch.no_grad():
                        _, _, loss_pred = model_ema.ema(inp)
                mask = model_ema.ema.generate_mask(loss_pred, guide = guide, epoch=i, total_epoch=epoch-1)

                mask = mask.to(device, non_blocking=True)
                inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis=False)
                loss_p, loss_target = model.module.forward_loss(inpp, recc, mask)

                loss_learn = model.module.forward_learning_loss(
                    loss_pred,
                    loss_target.detach()
                )
                loss = loss_p + loss_learn

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale the gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

        else:
            # if model_ema is not None:
            #     with torch.no_grad():
            #         _, _, loss_pred = model_ema.ema(inp)
            # mask = model_ema.ema.generate_mask(loss_pred, guide = guide, epoch=i, total_epoch=epoch-1)
            # mask = mask.to(device, non_blocking=True)
            # inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis=False)
            # loss_p, loss_target = model.module.forward_loss(inpp, recc, mask)
            # loss_learn = model.module.forward_learning_loss(
            #     loss_pred,
            #     loss_target.detach()
            # )
            # loss = loss_p + loss_learn

            ### Ablation on if need to predict loss

            # mask1 = (torch.rand((batch_size, 1, 7, 7, 8), device=device) > 0.6).bool()
            mask1 = model.module.base_model.mask(batch_size, device)

            if model_ema is not None:
                with torch.no_grad():
                    inp1, rec = model_ema.ema.base_model(inp, active_b1ff=mask1)
                    # l2_loss = torch.nn.functional.smooth_l1_loss(rec, inp1).mean(dim=2, keepdim=False)
                    l2_loss = ((rec - inp1) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
                    non_active = mask1.logical_not().int().view(mask1.shape[0], -1)  # (B, 1, f, f) => (B, L)
                    recon_loss = l2_loss * non_active
            #     mask = model_ema.ema.generate_mask(ema_loss_vector, guide=guide, epoch=i, total_epoch=epoch - 1)
            # else:
            mask, easy_mask = model_ema.ema.base_model.generate_mask(recon_loss, guide=guide, epoch=i, total_epoch=epoch - 1)
            # mask, easy_mask = model_ema.ema.generate_mask(my_score, guide=guide, epoch=i, total_epoch=epoch - 1)
            mask = mask.to(device, non_blocking=True)
            # no_mask = (torch.rand((batch_size, 1, 7, 7, 8), device=device) > 0.0 ).bool()
            # with torch.no_grad():
            #     _, teacher_rec = model_ema.ema.base_model(inp, active_b1ff=no_mask)
            teacher_rec = rec

            inpp, student_rec = model.module.base_model(inp, active_b1ff=mask, vis=False)

            student_rec = student_rec / student_temp
            teacher_rec = teacher_rec / teacher_temp

            teacher_rec_c = F.softmax((teacher_rec) / temp2, dim=-1)
            teacher_rec_c = teacher_rec_c.detach()
            teacher_rec_c = (1 - ((i + 1) / epoch)) * teacher_rec_c + (i + 1) / epoch * inp1
            recon_loss = torch.sum(-teacher_rec_c * F.log_softmax(student_rec, dim=-1), dim=-1)

            non_active = mask.logical_not().int().view(mask.shape[0], -1)  # (B, 1, f, f) => (B, L)

            loss_p = recon_loss.mul_(non_active).sum()/ (non_active.sum() + 1e-8)
            loss = loss_p

            optimizer.zero_grad()
            loss.backward()
            # optimize
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            optimizer.step()

        model_ema.update(model)
        loss = loss.item()

        if not math.isfinite(loss):
            print(loss)
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', flush=True)
            sys.exit(-1)
        per_loss += loss
        per_p_loss += loss_p.item()
        # per_t_loss += loss_t.item()
        torch.cuda.synchronize()
        it += 1
    scheduler.step()
    logger.log('epoch_end_timestamps', time.time(), i)
    epoch_loss.append(per_loss / iters_train)

    if i == 0:
        ema_loss = alpha * (per_loss / iters_train) + (1 - alpha) * (per_loss / iters_train)
    else:
        ema_loss = alpha * ema_loss + (1 - alpha) * (per_loss / iters_train)

    epoch_ema_loss.append(ema_loss)
    print('Epoch ', i, ' Train AVG Loss: ', per_loss / iters_train)
    print('Epoch ', i, ' Train EMA Loss: ', ema_loss)

    print('Train Pixel Loss: ', per_p_loss / iters_train)
    # print('Train Learn Loss: ', per_t_loss / iters_train)

    logger.log('train_losses', per_loss / iters_train, i)
    # logger.log('train_ema_losses', ema_loss, i)
    print_to_log_file('train_loss', np.round(logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    print_to_log_file(
        f"Epoch time: {np.round(logger.my_fantastic_logging['epoch_end_timestamps'][-1] - logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    if i % val_every == 0:
        model.eval()

        with torch.no_grad():
            val_per_loss = 0
            per_p_loss, per_t_loss = 0.0, 0.0
            for idx in range(iters_val):
                inp = next(mt_gen_val)
                inp = inp['data']
                inp = inp.to(device, non_blocking=True)
                if AMP:
                    with autocast():
                        if model_ema is not None:
                            with torch.no_grad():
                                inp, rec, _ = model_ema.ema(inp, active_b1ff=mask1)
                        mask = model_ema.ema.generate_mask(loss_pred, guide = guide, epoch=i, total_epoch=epoch-1)
                        mask = mask.to(device, non_blocking=True)
                        inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis=False)
                        loss_p, loss_target = model.module.forward_loss(inpp, recc, mask)
                        loss_learn = model.module.forward_learning_loss(
                            loss_pred,
                            loss_target.detach()
                        )
                        loss = loss_p + loss_learn

                else:
                    # if model_ema is not None:
                    #     with torch.no_grad():
                    #         _, _, loss_pred = model_ema.ema(inp)
                    # mask = model_ema.ema.generate_mask(loss_pred, guide = guide, epoch=i, total_epoch=epoch-1)
                    # mask = mask.to(device, non_blocking=True)
                    # inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis=False)
                    # loss_p, loss_target = model.module.forward_loss(inpp, recc, mask)
                    # loss_learn = model.module.forward_learning_loss(
                    #     loss_pred,
                    #     loss_target.detach()
                    # )
                    # loss = loss_p + loss_learn

                    ### Ablation
                    mask1 = model.module.base_model.mask(batch_size, device)

                    if model_ema is not None:
                        with torch.no_grad():
                            inp1, rec = model_ema.ema.base_model(inp, active_b1ff=mask1)
                            # l2_loss = torch.nn.functional.smooth_l1_loss(rec, inp1).mean(dim=2, keepdim=False)
                            l2_loss = ((rec - inp1) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
                            non_active = mask1.logical_not().int().view(mask1.shape[0], -1)  # (B, 1, f, f) => (B, L)
                            recon_loss = l2_loss * non_active
                    #     mask = model_ema.ema.generate_mask(ema_loss_vector, guide=guide, epoch=i, total_epoch=epoch - 1)
                    # else:
                    mask, easy_mask = model_ema.ema.base_model.generate_mask(recon_loss, guide=guide, epoch=i,
                                                                             total_epoch=epoch - 1)
                    # mask, easy_mask = model_ema.ema.generate_mask(my_score, guide=guide, epoch=i, total_epoch=epoch - 1)
                    mask = mask.to(device, non_blocking=True)
                    no_mask = (torch.rand((batch_size, 1, 7, 7, 8), device=device) > 0.0).bool()
                    with torch.no_grad():
                        _, teacher_rec = model_ema.ema.base_model(inp, active_b1ff=no_mask)
                    inpp, student_rec = model.module.base_model(inp, active_b1ff=mask, vis=False)

                    student_rec = student_rec / student_temp
                    teacher_rec = teacher_rec / teacher_temp

                    teacher_rec_c = F.softmax(teacher_rec/ temp2, dim=-1)
                    teacher_rec_c = teacher_rec_c.detach()
                    teacher_rec_c = (1 - ((i + 1) / epoch)) * teacher_rec_c + (i + 1) / epoch * inp1

                    recon_loss = torch.sum(-teacher_rec_c * F.log_softmax(student_rec, dim=-1), dim=-1)

                    non_active = mask.logical_not().int().view(mask.shape[0], -1)  # (B, 1, f, f) => (B, L)

                    loss_p = recon_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)

                    # l2_loss = ((recc - inpp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
                    # non_active = mask.logical_not().int().view(mask.shape[0], -1)  # (B, 1, f, f) => (B, L)
                    # recon_loss = l2_loss * non_active
                    # full_rec = model.module.unpatchify(recc)
                    # ssim = calculate_3d_ssim_per_patch(inp, full_rec)
                    # my_score = get_score(ssim, l2_loss, non_active)
                    # loss_p = my_score.mul_(non_active).sum() / (
                    #         non_active.sum() + 1e-8)


                    # easy_mask = easy_mask.to(device, non_blocking=True)
                    # easy_mask = ~((~easy_mask) * (~mask))
                    # loss_t, _ = model.module.forward_loss(rec.detach(), recc, easy_mask)
                    # if i >= 250:
                    #     loss = loss_p + loss_t /100
                    # else:
                    #     loss = loss_p
                    loss = loss_p

                loss = loss.item()
                val_per_loss += loss
                per_p_loss += loss_p.item()
                # per_t_loss += loss_t.item()
        val_loss.append(val_per_loss/iters_val)
        print('Val AVG Loss: ', val_per_loss / iters_val)
        print('Val pixel Loss: ', per_p_loss / iters_val)
        # print('Val learn Loss: ', per_t_loss / iters_val)
        logger.log('val_losses', val_per_loss / iters_val, i)
        print_to_log_file('val_loss', np.round(logger.my_fantastic_logging['val_losses'][-1], decimals=4))

        if (val_per_loss/iters_val) < best_val_loss:
            best_val_loss = val_per_loss / iters_val
            print('New best loss!')
            print_to_log_file(f"Yayy! New best val loss: {np.round(best_val_loss, decimals=4)}")
            checkpoint = {
                'network_weights': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'grad_scaler_state': None,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'current_epoch': i
            }
            torch.save(checkpoint, join(output_folder, model_name + '_head_best.pt'))

        plt.figure()
        plt.plot()
        epochs = list(range(1, len(epoch_loss) + 1))
        plt.plot(epochs, epoch_loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.plot(epochs, epoch_ema_loss, label='Training EMA loss')

        # Adding title and labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(join(output_folder, 'progress.png'))
        plt.close()

    checkpoint = {
        'network_weights': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'grad_scaler_state': None,
        'train_loss': epoch_loss,
        'val_loss': val_loss,
        'current_epoch': i
    }
    torch.save(checkpoint, join(output_folder, model_name + '_head_latest.pt'))

