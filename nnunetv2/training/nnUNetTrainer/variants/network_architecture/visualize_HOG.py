import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import json
from utility import get_validation_transforms, build_dataloader, LocalDDP
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from STUNet_head import STUNet
from MedNeXt_head import MedNeXt
from encoder3D import SparseEncoder
from decoder3D import LightDecoder, DSDecoder
from spark3D import SparK, HOGLayer, HOGLayerC
# from spark3D_DS import SparK
from collections import OrderedDict


preprocessed_dataset_folder = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans_3d_fullres'
splits_file = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/splits_final.json'
fold = 0
batch_size = 2
mask_ratio = 0.6
dataset_json = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/dataset.json')
plans = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans.json')
mt_gen_train, mt_gen_val, iters_train, iters_val = build_dataloader(preprocessed_dataset_folder, splits_file, fold, dataset_json, plans, batch_size)
device = torch.device("cuda:0")

img1 = next(mt_gen_val)['data'].to(device, non_blocking=True)

layer = HOGLayer(nbins=3, pool=1).to(device)

hogged = [layer(img1[:, :, i, :, :]).unsqueeze(2) for i in range(img1.shape[2])]
hogged = torch.cat(hogged, 2)
slice_num = 64
inp_bchw = img1[:,:,:,:,slice_num]
hog_chan = 0
hogged = hogged[:,hog_chan,:,:,slice_num].unsqueeze(1)

plt.figure()
for col, (title, bchw) in enumerate(zip(['input', 'hog'], [inp_bchw, hogged])):
    plt.subplot2grid((1, 2), (0, col))
    plt.imshow(bchw[0,0].cpu().numpy(), plt.cm.gray)
    plt.title(title)
    plt.axis('off')
plt.show()
plt.savefig('masked_hog.png')
plt.close()