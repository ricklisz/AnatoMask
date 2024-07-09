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
from STUNet_head import STUNet
from MedNeXt_head import MedNeXt
from UniRep_head import UniRepLKUNet
from encoder3D import SparseEncoder
from decoder3D import DSDecoder
from spark3D_DS import SparK
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
    #
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


device = torch.device("cuda:4")
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
#               enable_deep_supervision=True).to(device)

head = STUNet(1,1,depth=[2,2,2,2,2,2], dims=[64, 128, 256, 512, 1024,1024], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
              enable_deep_supervision=True).to(device)

model_name = 'STUNet_L_DS'
mask_ratio_para = 0.9

class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

enc = SparseEncoder(head, input_size=(112,112,128), sbn=False).to(device)
dec = DSDecoder(enc.downsample_ratio,sbn=False, width = 1024).to(device)
model_without_ddp = SparK(
    sparse_encoder=enc, dense_decoder=dec,mask_ratio=mask_ratio_para,
    densify_norm='in'
).to(device)
model = LocalDDP(model_without_ddp)

fold = 0
epoch = 1000
batch_size = 4
opt = 'adamw'
ada = 0.999
lr = 8e-5
weight_decay = 1e-5
clip = 12
wd = 0.04
wde = 0.2
wp_ep = 20
output_folder = '/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/Pretraining/' + model_name +'_mask_ratio_'+str(mask_ratio_para)
timestamp = datetime.now()
maybe_mkdir_p(output_folder)
weights = [1, 1, 1, 1]

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

normalization = configuration_manager.normalization_schemes
print('Using normalization:  ', normalization)

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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)
it = 0
epoch_loss = []
val_loss = []

model.train()
optimizer.zero_grad()
early_clipping = clip > 0 and not hasattr(optimizer, 'global_grad_norm')
late_clipping = hasattr(optimizer, 'global_grad_norm')
if early_clipping:
    params_req_grad = [p for p in model.parameters() if p.requires_grad]

best_val_loss = 1e9
val_every = 1
logger = nnUNetLogger()
for i in range(epoch):
    per_loss = 0.0
    print_to_log_file('')
    print_to_log_file(f'Epoch {i}')
    print_to_log_file()

    print_to_log_file(
        f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
    logger.log('epoch_start_timestamps', time.time(), i)

    for idx in range(iters_train):
        inp = next(mt_gen_train)
        inp = inp['data']
       #
        inp = inp.to(device, non_blocking=True)
        loss = model(inp, active_b1ff=None, vis=False)
        total_loss = sum(w * l for w, l in zip(weights, loss))
        optimizer.zero_grad()
        total_loss.backward()
        total_loss = total_loss.item()
        if not math.isfinite(total_loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {total_loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        per_loss += total_loss
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        # torch.cuda.synchronize()
        it +=1
    scheduler.step()
    logger.log('epoch_end_timestamps', time.time(), i)
    epoch_loss.append(per_loss / iters_train)

    print('Epoch ', i, ' AVG Loss: ', per_loss / iters_train)
    logger.log('train_losses', per_loss / iters_train, i)
    print_to_log_file('train_loss', np.round(logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    print_to_log_file(
        f"Epoch time: {np.round(logger.my_fantastic_logging['epoch_end_timestamps'][-1] - logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    if i % val_every == 0:
        model.eval()
        val_per_loss = 0
        for idx in range(iters_val):
            inp = next(mt_gen_val)
            inp = inp['data']
            inp = inp.to(device, non_blocking=True)
            loss = model(inp, active_b1ff=None, vis=False)
            total_loss = sum(w * l for w, l in zip(weights, loss))
            total_loss = total_loss.item()
            val_per_loss += total_loss

        val_loss.append(val_per_loss/iters_val)
        print('Val AVG Loss: ', val_per_loss / iters_val)
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
        'current_epoch': epoch
    }
    torch.save(checkpoint, join(output_folder, model_name + '_head_latest.pt'))







