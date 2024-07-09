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
from timm.utils import ModelEma
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from STUNet_head import STUNet
from MedNeXt_head import MedNeXt
from encoder3D import SparseEncoder
from decoder3D import LightDecoder, SMiMTwoDecoder, SMiMDecoder
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

device = torch.device("cuda:0")

pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# STUNet_B
head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
               enable_deep_supervision=True).to(device)

d = torch.Tensor(size = (1,1,112,112,128)).to(device)


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
    loss_dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 1).to(device)
else:
    dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 1).to(device)
    loss_dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 1).to(device)

model_without_ddp = SparK(
    sparse_encoder=enc, dense_decoder=dec, loss_decoder= loss_dec, mask_ratio=0.6,
    densify_norm='in', use_hog=use_hog
).to(device)

model_ema = ModelEma(model_without_ddp, decay=0.999, device=device, resume='')

model = LocalDDP(model_without_ddp)

fold = 0
epoch = 1000
batch_size = 4
opt = 'adamw'
ada = 0.999
lr = 2e-4
weight_decay = 1e-5
clip = 12
wd = 0.04
wde = 0.2
wp_ep = 20
warmup = 20
AMP = False



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

it = 0
epoch_loss = []
val_loss = []

model.train()
optimizer.zero_grad()
early_clipping = clip > 0 and not hasattr(optimizer, 'global_grad_norm')
late_clipping = hasattr(optimizer, 'global_grad_norm')
if early_clipping:
    params_req_grad = [p for p in model.parameters() if p.requires_grad]

scaler = GradScaler()
best_val_loss = 1e9
val_every = 1

iters_train = 1
iters_val = 1

for i in range(epoch):
    per_loss = 0.0

    # add this
    if i < 100:
        model_ema.decay = 0.999 + i / 100 * (0.9999 - 0.999)
    else:
        model_ema.decay = 0.9999

    for idx in range(iters_train):
        # inp = next(mt_gen_train)
        # inp = inp['data']
        inp = d

        inp = inp.to(device, non_blocking=True)

        if AMP:
            with torch.cuda.amp.autocast():
                if model_ema is not None:
                    with torch.no_grad():
                        _, _, loss_pred = model_ema.ema(inp)
                mask = model_ema.ema.generate_mask(loss_pred, epoch=i, total_epoch=epoch)
                inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis =False)
                loss_p, loss_target = model.module.forward_loss(inpp, recc, loss_pred)

                loss_learn = model.module.forward_learning_loss(
                    loss_pred,
                    loss_p.detach()
                )
                print(loss_p.item(), loss_learn.item())
                loss = loss_p + loss_learn

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale the gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            break
        else:
            if model_ema is not None:
                with torch.no_grad():
                    _, _, loss_pred = model_ema.ema(inp)
            mask = model_ema.ema.generate_mask(loss_pred, epoch=i, total_epoch=epoch)
            inpp, recc, loss_pred = model(inp, active_b1ff=mask, vis=False)
            loss_p, loss_target = model.module.forward_loss(inpp, recc, loss_pred)
            loss_learn = model.module.forward_learning_loss(
                loss_pred,
                loss_target.detach()
            )
            loss = loss_p + loss_learn
            # Sum losses across all GPUs
            optimizer.zero_grad()
            loss.backward()
            # optimize
            grad_norm = None
            if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, clip).item()
            optimizer.step()

        model_ema.update(model)
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        per_loss += loss
        # torch.cuda.synchronize()
        it += 1
    scheduler.step()
    epoch_loss.append(per_loss / iters_train)

    print('Epoch ', i, ' AVG Loss: ', per_loss / iters_train)

    if i % val_every == 0:
        model.eval()
        val_per_loss = 0
        for idx in range(iters_val):
            # inp = next(mt_gen_val)
            # inp = inp['data']
            #
            inp = d
            inp = inp.to(device, non_blocking=True)
            if AMP:
                with autocast():
                    loss = model(inp, active_b1ff=None, vis=False)
            else:
                loss = model(inp, active_b1ff=None, vis=False)

            loss = loss.item()
            val_per_loss += loss

        val_loss.append(val_per_loss/iters_val)

        if (val_per_loss/iters_val) < best_val_loss:
            best_val_loss = val_per_loss / iters_val
            print('New best loss!')

