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
from spark3D import SparK
# from spark3D_DS import SparK
from collections import OrderedDict

checkpoint = torch.load('/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset666_Big/Pretraining/STUNet_B_mask60_total_plans/STUNet_B_head_best.pt')

preprocessed_dataset_folder = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset666_Big/nnUNetPlans_3d_fullres'
splits_file = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset666_Big/splits_final.json'
fold = 0
batch_size = 2
mask_ratio = 0.6
dataset_json = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset666_Big/dataset.json')
plans = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans.json')
mt_gen_train, mt_gen_val, iters_train, iters_val = build_dataloader(preprocessed_dataset_folder, splits_file, fold, dataset_json, plans, batch_size)
device = torch.device("cuda:0")
print('Restoring from: ', checkpoint['current_epoch'])
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
              enable_deep_supervision=True).to(device)

enc = SparseEncoder(head, input_size=(112,112,128), sbn=True).to(device)
dec = LightDecoder(enc.downsample_ratio,sbn=True, width = 512).to(device)
# dec = DSDecoder(enc.downsample_ratio,sbn=False, width = 512).to(device)

model = SparK(
    sparse_encoder=enc, dense_decoder=dec,mask_ratio=mask_ratio,
    densify_norm='in'
).to(device)

model.eval()
[p.requires_grad_(False) for p in model.parameters()]

pretrained_state = checkpoint['network_weights']
new_weights = OrderedDict((key.replace('module.', ''), value) for key, value in pretrained_state.items())
for k in new_weights.keys():
    print(k)

missing, unexpected = model.load_state_dict(new_weights, strict=False)
assert len(missing) == 0, f'load_state_dict missing keys: {missing}'
# assert len(unexpected) == 0, f'load_state_dict unexpected keys: {unexpected}'
del pretrained_state, new_weights

active_b1ff = (torch.rand((2,1,7,7,8), device=device) > mask_ratio).bool()
img1 = next(mt_gen_val)['data'].to(device, non_blocking=True)

import pickle

#
# with open('sample1.pkl', 'wb') as file:
#     pickle.dump(img1.detach().cpu(), file)

with open('sample1.pkl', 'rb') as file:
    img1 = pickle.load(file).to(device, non_blocking=True)

inp, masked, rec_or_inp = model(img1, active_b1ff=active_b1ff, vis=True)
masked_title = 'rand masked' if active_b1ff is None else 'specified masked'

slice_num = 56
# inp_bchw = inp[:,:,:,:,slice_num]
# masked_bchw, rec_or_inp = masked[:,:,:,:,slice_num], rec_or_inp[:,:,:,:,slice_num]

inp_bchw = inp[:,:,:,slice_num, :]
masked_bchw, rec_or_inp = masked[:,:,:,slice_num, :], rec_or_inp[:,:,:,slice_num, :]


plt.figure()
for col, (title, bchw) in enumerate(zip(['input', masked_title, 'reconstructed'], [inp_bchw, masked_bchw, rec_or_inp])):
    plt.subplot2grid((1, 3), (0, col))
    plt.imshow(bchw[0,0].cpu().numpy(), plt.cm.gray)
    plt.title(title)
    plt.axis('off')
plt.show()
plt.savefig('masked.png')
plt.close()

train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
epochs = list(range(1, len(train_loss) + 1))

plt.figure()
plt.plot()
plt.plot(epochs, train_loss, label='Training loss')
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
plt.savefig('output.png')
plt.close()


