# region Description of the region
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
import time
from time import sleep
from datetime import datetime
import numpy as np
from timm.utils import ModelEma
from STUNet_head import STUNet
from encoder3D import SparseEncoder
from decoder3D import LightDecoder
from AnatoMask import SparK

import sys
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
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
from torch.cuda.amp import GradScaler, autocast

import math
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from utils.lr_control import lr_wd_annealing, get_param_groups
from utils import dist

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

# endregion
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import device_count
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int)
    return parser.parse_args()


def main_pt():
    # set your device here
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,7,6'
    os.environ["OMP_NUM_THREADS"] = "8"

    # args = get_args()
    # local_rank = args.local_rank

    dist.init_process_group(backend='nccl')

    is_ddp = dist.is_available() and dist.is_initialized()
    local_rank = dist.get_rank()

    # print what device we are using
    if is_ddp:  # implicitly it's clear that we use cuda in this case
        device = torch.device(type='cuda', index=local_rank)
        print(f"I am local rank {local_rank}. {device_count()} GPUs are available. The world size is "
              f"{dist.get_world_size()}."
              f"Setting device to {device}")

    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
    conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

    # STUNet_B
    # head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
    #                enable_deep_supervision=True).to(device)
    # STUNet_L
    # head = STUNet(1,1,depth=[2, 2, 2, 2, 2, 2], dims=[64 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
    #               enable_deep_supervision=True).to(device)
    # STUNet_H
    from GC import STUNet
    head = STUNet(1,1,depth=[3,3,3,3,3,3], dims=[96 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
                enable_deep_supervision=True).to(device)

    model_name = 'STUNet_H'
    input_size = (112,112,128)
    enc = SparseEncoder(head, input_size=input_size, sbn=True).to(device)
    dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 1536, out_channel = 1).to(device)
    loss_dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 1536, use_IN= True, out_channel = 1).to(device)

    model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, loss_decoder=loss_dec, mask_ratio=0.6,
        densify_norm='in'
    ).to(device)

    # model_without_ddp = torch.compile(model_without_ddp)

    model = DDP(model_without_ddp, device_ids=[dist.get_rank()],
                                                             find_unused_parameters=True, broadcast_buffers=False)

    model_ema = ModelEma(model_without_ddp, decay=0.999, device=device, resume='')

    # Chnage this every time...
    fold = 0
    epoch = 1000
    # Your batch size here
    global_batch_size = 4
    global_oversample_percents = 0.33
    opt = 'adamw'
    ada = 0.999
    warmup = 10
    lr = 2e-5
    weight_decay = 3e-5
    clip = 12
    val_every = 2
    AMP = False
    guide = True

    batch_sizes = []
    oversample_percents = []

    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
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

        if sample_id_high / global_batch_size < (1 - global_oversample_percents):
            oversample_percents.append(0.0)
        elif sample_id_low / global_batch_size > (1 - global_oversample_percents):
            oversample_percents.append(1.0)
        else:
            percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
            oversample_percent_here = 1 - (((1 - global_oversample_percents) -
                                            sample_id_low / global_batch_size) / percent_covered_by_this_rank)
            oversample_percents.append(oversample_percent_here)

    print("worker", my_rank, "oversample", oversample_percents[my_rank])
    print("worker", my_rank, "batch_size", batch_sizes[my_rank])

    batch_size = batch_sizes[my_rank]
    oversample_foreground_percent = oversample_percents[my_rank]



    output_folder = '/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset666_Big/Pretraining/' + model_name +'_HPM'

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

    preprocessed_dataset_folder = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans_3d_fullres'
    splits_file = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/splits_final.json'
    splits = load_json(splits_file)

    all_keys = splits[fold]['train']
    tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

    dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                               folder_with_segs_from_previous_stage=None,
                               num_images_properties_loading_threshold=0)

    dataset_json = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/dataset.json')
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

    iters_train = len(dataset_tr) // global_batch_size

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

    allowed_num_processes = get_allowed_n_proc_DA()

    mt_gen_train = LimitedLenWrapper(iters_train, data_loader=dl_tr, transform=tr_transforms,
                                     num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                     pin_memory= True, wait_time=0.02)

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
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup,epoch,1e-6)

    it = 0
    best_val_loss = 1e4
    epoch_loss = []
    val_loss = []

    if is_ddp:
        dist.barrier()

    optimizer.zero_grad()
    early_clipping = clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]

    scaler = GradScaler()

    logger = nnUNetLogger()
    for i in range(epoch):
        model.train()
        per_loss = 0.0
        per_p_loss = 0.0

        if dist.get_rank() == 0:
            print_to_log_file('')
            print_to_log_file(f'Epoch {i}')
            print_to_log_file()

            print_to_log_file(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
            logger.log('epoch_start_timestamps', time.time(), i)
        if i < epoch // 4:
            model_ema.decay = 0.999 + i / (epoch // 4) * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

        for idx in range(iters_train):

            inp = next(mt_gen_train)
            inp = inp['data']
            inp = inp.to(device, non_blocking=True)

            if AMP:
                with torch.cuda.amp.autocast():
                    mask1 = model.module.mask(batch_size, device)
                    if model_ema is not None:
                        with torch.no_grad():
                            inp1, rec1 = model_ema.ema(inp, active_b1ff=mask1)
                            l2_loss = ((rec1 - inp1) ** 2).mean(dim=2, keepdim=False)
                            non_active = mask1.logical_not().int().view(mask1.shape[0], -1)  # (B, 1, f, f) => (B, L)
                            recon_loss = l2_loss * non_active

                    mask, easy_mask = model_ema.ema.generate_mask(recon_loss, guide=guide, epoch=i,
                                                                  total_epoch=epoch - 1)
                    mask = mask.to(device, non_blocking=True)
                    inpp, recc = model(inp, active_b1ff=mask, vis=False)
                    loss_p, _ = model.module.forward_loss(inpp, recc, mask)

                    loss = loss_p
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Unscale the gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                mask1 = model.module.mask(batch_size, device)
                if model_ema is not None:
                    with torch.no_grad():
                        inp1, rec1 = model_ema.ema(inp, active_b1ff=mask1)
                        l2_loss = ((rec1 - inp1) ** 2).mean(dim=2, keepdim=False)
                        non_active = mask1.logical_not().int().view(mask1.shape[0], -1)  # (B, 1, f, f) => (B, L)
                        recon_loss = l2_loss * non_active

                mask, easy_mask = model_ema.ema.generate_mask(recon_loss, guide=guide, epoch=i,
                                                              total_epoch=epoch - 1)
                mask = mask.to(device, non_blocking=True)
                inpp, recc = model(inp, active_b1ff=mask, vis=False)
                loss_p, _ = model.module.forward_loss(inpp, recc, mask)

                loss = loss_p
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
                optimizer.step()
            model_ema.update(model)
            loss = loss.item()
            if not math.isfinite(loss):
                print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
                sys.exit(-1)
            per_loss += loss
            per_p_loss += loss_p.item()
            # torch.cuda.synchronize()
            it += 1
        scheduler.step()
        epoch_loss.append(per_loss / iters_train)

        if dist.get_rank() == 0:
            logger.log('epoch_end_timestamps', time.time(), i)
            print('Epoch ', i, ' AVG Loss: ', per_loss / iters_train)
            logger.log('train_losses', per_loss / iters_train, i)
            print_to_log_file('train_loss', np.round(logger.my_fantastic_logging['train_losses'][-1], decimals=4))
            print_to_log_file(
                f"Epoch time: {np.round(logger.my_fantastic_logging['epoch_end_timestamps'][-1] - logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")
            checkpoint = {
                'network_weights': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'grad_scaler_state': None,
                'train_loss': epoch_loss,
                'current_epoch': i
            }
            torch.save(checkpoint, join(output_folder, model_name + '_head_latest.pt'))


if __name__ == '__main__':
    main_pt()
