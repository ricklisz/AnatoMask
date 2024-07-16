# region Description of the region
import torch
import time
from time import sleep
from datetime import datetime
import numpy as np
from torch import autocast
import sys
from torch import distributed as dist
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
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
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
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

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

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

def build_dataloader(preprocessed_dataset_folder, splits_file, fold, dataset_json, plans, batch_size):
    splits = load_json(splits_file)
    tr_keys = splits[fold]['train']
    val_keys = splits[fold]['val']

    dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                               folder_with_segs_from_previous_stage=None,
                               num_images_properties_loading_threshold=0)
    dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                                folder_with_segs_from_previous_stage=None,
                                num_images_properties_loading_threshold=0)

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
                               oversample_foreground_percent=0.8,
                               sampling_probabilities=None, pad_sides=None)
    patch = (160,160,160)
    # patch = (112,112,128)
    dl_val = nnUNetDataLoader3D(dataset_val, 1,
                                patch,
                                patch,
                                label_manager,
                                oversample_foreground_percent=0.7,
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
                                     pin_memory=True, wait_time=0.02)

    mt_gen_val = LimitedLenWrapper(iters_val, data_loader=dl_val,
                                   transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                   num_cached=3, seeds=None, pin_memory=True,
                                   wait_time=0.02)

    return mt_gen_train, mt_gen_val, iters_train, iters_val

class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def load_pretrained_weights(network, fname, verbose=False):

    saved_model = torch.load(fname)

    if fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    new_file3 = OrderedDict()
    for old_key, value in pretrained_dict.items():
        if 'encoder' in old_key:
            new_key = old_key.split('sp_cnn.')[-1]  # This extracts the part after the last '.'
            new_file3[new_key] = value

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    mod_dict = mod.state_dict()

    for key, _ in mod_dict.items():
        if ('conv_blocks' in key):
            if (key in new_file3) and (mod_dict[key].shape == new_file3[key].shape):
                print('This layer worked: ', key)
            else:
                print('This layer not worked: ', key)

    mod.load_state_dict(new_file3, strict=False)

def build_loss(label_manager, configuration_manager, is_ddp = False):
    if label_manager.has_regions:
        loss = DC_and_BCE_loss({},
                               {'batch_dice': configuration_manager.batch_dice,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': is_ddp},
                               use_ignore_label=label_manager.ignore_label is not None,
                               dice_class=MemoryEfficientSoftDiceLoss)
    else:
        loss = DC_and_CE_loss({'batch_dice': configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    weights[-1] = 0

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    weights = weights / weights.sum()
    # now wrap the loss
    loss = DeepSupervisionWrapper(loss, weights)
    return loss

def validation_step(batch, device, network, loss, label_manager):
    data = batch['data']
    target = batch['target']
    data = data.to(device, non_blocking=True)
    if isinstance(target, list):
        target = [i.to(device, non_blocking=True) for i in target]
    else:
        target = target.to(device, non_blocking=True)

    with autocast(device.type, enabled=True):
        output = network(data)
        del data
        l = loss(output, target)

    # we only need the output with the highest output resolution
    output = output[0]
    target = target[0]

    # the following is needed for online evaluation. Fake dice (green line)
    axes = [0] + list(range(2, len(output.shape)))

    if label_manager.has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

    if label_manager.has_ignore_label:
        if not label_manager.has_regions:
            mask = (target != label_manager.ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            target[target == label_manager.ignore_label] = 0
        else:
            mask = 1 - target[:, -1:]
            # CAREFUL that you don't rely on target after this line!
            target = target[:, :-1]
    else:
        mask = None

    tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

    tp_hard = tp.detach().cpu().numpy()
    fp_hard = fp.detach().cpu().numpy()
    fn_hard = fn.detach().cpu().numpy()
    if not label_manager.has_regions:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

    return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

def on_validation_epoch_end(val_outputs, current_epoch, logger, is_ddp = False):
    outputs_collated = collate_outputs(val_outputs)
    tp = np.sum(outputs_collated['tp_hard'], 0)
    fp = np.sum(outputs_collated['fp_hard'], 0)
    fn = np.sum(outputs_collated['fn_hard'], 0)

    if  is_ddp:
        world_size = dist.get_world_size()

        tps = [None for _ in range(world_size)]
        dist.all_gather_object(tps, tp)
        tp = np.vstack([i[None] for i in tps]).sum(0)

        fps = [None for _ in range(world_size)]
        dist.all_gather_object(fps, fp)
        fp = np.vstack([i[None] for i in fps]).sum(0)

        fns = [None for _ in range(world_size)]
        dist.all_gather_object(fns, fn)
        fn = np.vstack([i[None] for i in fns]).sum(0)

        losses_val = [None for _ in range(world_size)]
        dist.all_gather_object(losses_val, outputs_collated['loss'])
        loss_here = np.vstack(losses_val).mean()
    else:
        loss_here = np.mean(outputs_collated['loss'])

    global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                       zip(tp, fp, fn)]]
    mean_fg_dice = np.nanmean(global_dc_per_class)
    logger.log('mean_fg_dice', mean_fg_dice, current_epoch)
    logger.log('dice_per_class_or_region', global_dc_per_class,current_epoch)
    logger.log('val_losses', loss_here,current_epoch)


# endregion


