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
from spark3D_HPM import SparK
# from spark3D_DS import SparK
from collections import OrderedDict
from timm.utils import ModelEma
import pickle

checkpoint = torch.load('/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/Pretraining/STUNet_B_HPM_no_dec/STUNet_B_head_latest.pt')
# checkpoint = torch.load('/home/yl_li/STUNet/nnUNet-1.7.1/nnUNet_results/Dataset501_Total/Pretraining/STUNet_B_DS_mask_ratio_0.6/STUNet_B_DS_head_best.pt')

preprocessed_dataset_folder = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans_3d_fullres'
splits_file = '/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/splits_final.json'
fold = 0
batch_size = 1
mask_ratio = 0.6
roi = (160,160,160)
dataset_json = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/dataset.json')
plans = load_json('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset501_Total/nnUNetPlans.json')
mt_gen_train, mt_gen_val, iters_train, iters_val = build_dataloader(preprocessed_dataset_folder, splits_file, fold, dataset_json, plans, batch_size)
device = torch.device("cuda:6")
print('Restoring from: ', checkpoint['current_epoch'])
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2,2,2]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
              enable_deep_supervision=True).to(device)

enc = SparseEncoder(head, input_size=roi, sbn=False).to(device)
dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel=1).to(device)
loss_dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, use_IN=True, out_channel=1).to(device)
loss_dec = None
model = SparK(
    sparse_encoder=enc, dense_decoder=dec, loss_decoder= loss_dec, mask_ratio=0.6,
    densify_norm='in', use_hog=False
).to(device)

model_ema = ModelEma(model, decay=0.999, device=device, resume='')

model.eval()

[p.requires_grad_(False) for p in model.parameters()]

pretrained_state = checkpoint['network_weights']
new_weights = OrderedDict((key.replace('module._orig_mod.', ''), value) for key, value in pretrained_state.items())
for k in new_weights.keys():
    print(k)

missing, unexpected = model.load_state_dict(new_weights, strict=False)
model_ema.ema.load_state_dict(new_weights, strict=False)

# assert len(missing) == 0, f'load_state_dict missing keys: {missing}'
# assert len(unexpected) == 0, f'load_state_dict unexpected keys: {unexpected}'
del pretrained_state, new_weights

active_b1ff = (torch.rand((batch_size,1,roi[0]//16,roi[1]//16,roi[2]//16), device=device) > mask_ratio).bool()

stuff = next(mt_gen_val)
img1 = stuff['data'].to(device, non_blocking=True)
lab1 = stuff['target'][0].to(device, non_blocking=True)

with open('mask1.pkl', 'rb') as file:
    active_b1ff = pickle.load(file).to(device, non_blocking=True)
with open('sample1.pkl', 'rb') as file:
    img1 = pickle.load(file).to(device, non_blocking=True)
with open('label1.pkl', 'rb') as file:
    lab1 = pickle.load(file).to(device, non_blocking=True)

# with open('sample1.pkl', 'wb') as file:
#     pickle.dump(img1.detach().cpu(), file)
# with open('mask1.pkl', 'wb') as file:
#     pickle.dump(active_b1ff.detach().cpu(), file)
# with open('label1.pkl', 'wb') as file:
#     pickle.dump(lab1.detach().cpu(), file)

avg_pred = torch.zeros(batch_size,1000).to(device)
avg_bchwd = torch.zeros(batch_size,roi[0]//16*roi[1]//16*roi[2]//16).to(device)

iter = 20

for i in range(iter):
    active_b1ff = (torch.rand((batch_size, 1, roi[0]//16, roi[1]//16, roi[2]//16),
                              device=device) > mask_ratio).bool()
    with torch.no_grad():
        inp1, rec1 = model_ema.ema(img1, active_b1ff=active_b1ff,vis = False)
        # inp1,masked1,rec1 = model_ema.ema(img1, active_b1ff=active_b1ff,vis = True)
        # l2_loss = ((rec1 - inp1) ** 2)
        l2_loss = ((rec1 - inp1) ** 2).mean(dim=2, keepdim=False)
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1) # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss * non_active
        # temp = ((rec1 - inp1) ** 2) * active_b1ff.logical_not().int()
        # _, _, loss_bchwd, _ = model_ema.ema(img1, active_b1ff=active_b1ff,vis = True)
        # avg_pred += loss_pred
        # recon_loss = (recon_loss - recon_loss.min()) / (recon_loss.max() - recon_loss.min())
        avg_bchwd += recon_loss

# avg_pred /= 20
# avg_bchwd /=  iter

# avg_bchwd_flat = model.patchify(avg_bchwd).mean(dim=2, keepdim=False)
# print(avg_bchwd_flat.shape)
mask, _ = model_ema.ema.generate_mask(avg_bchwd, guide=True, epoch=998, total_epoch=1000)

mask = mask.to(device, non_blocking=True)
first_masked = active_b1ff.repeat_interleave(16, 2).repeat_interleave(16,3).repeat_interleave(16,4)
masked_bchwd = mask.repeat_interleave(16, 2).repeat_interleave(16,3).repeat_interleave(16,4)
inp_bchwd, _, rec_or_inp = model(img1, active_b1ff=mask, vis=True)

print(inp_bchwd.shape)
slice_num = 80
inp_bchw = img1[:,:,slice_num,:, :]
first_masked = first_masked[:,:,slice_num,:, :]
masked_bchw, rec_or_inp = masked_bchwd[:,:,slice_num,:, :], rec_or_inp[:,:,slice_num,:, :]
lb_bchw = lab1[:,:,slice_num,:, :][0,0].cpu().numpy().astype('int')
# loss_bchw = loss_bchwd[:,:, slice_num,:,:]
loss_bchw = avg_bchwd.view(batch_size, 1, 10, 10, 10).repeat_interleave(16, 2).repeat_interleave(16,3).repeat_interleave(16,4)[:,:,slice_num,:, :]

plt.figure()
for col, (title, bchw) in enumerate(zip(['input', 'first mask', 'second mask','reconstructed'], [inp_bchw, first_masked, masked_bchw, rec_or_inp])):
    plt.subplot2grid((1, 4), (0, col))
    plt.imshow(bchw[0,0].cpu().numpy(), plt.cm.gray)
    plt.title(title)
    plt.axis('off')
plt.show()
plt.savefig('masked_hpm.png')
plt.close()

plt.figure(figsize=(10, 5))  # Optional: You can specify the figure size
# Pltting the first image on the left
plt.subplot2grid((1, 2), (0, 0))
plt.imshow(active_b1ff[0, 0][3].cpu().numpy(), cmap=plt.cm.gray)
plt.title("First Image")  # Optional: You can add a title to the subplot
# Plotting the second image on the right
plt.subplot2grid((1, 2), (0, 1))
plt.imshow(avg_bchwd.view(1, 1, 10, 10, 10)[0, 0][3].cpu().numpy())
plt.title("Second Image")  # Optional: You can add a title to the subplot
# Saving the figure to a file
plt.savefig('debug.png')
# Closing the figure to free up memory
plt.close()

from segmentation_mask_overlay import overlay_masks

def recode(np_arr):
    # Find the unique elements and their order of appearance
    unique, inverse = np.unique(np_arr.flatten(), return_inverse=True)
    # Re-encode the array based on the order of unique elements
    reencoded = inverse.reshape(np_arr.shape)
    return reencoded

lb_bchw = np.eye(np.max(lb_bchw) + 1)[lb_bchw][:,:,1:]
# [Optional] prepare labels
mask_labels = [f"Mask_{i}" for i in range(lb_bchw.shape[-1])]
plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
# [Optional] prepare colors
cmap = plt.cm.tab20(np.arange(np.max(lb_bchw.shape[-1])))[..., :-1]
alpha = 1.3
beta = 0.3
fig = array = overlay_masks(inp_bchw[0,0].cpu().numpy(), lb_bchw, mpl_figsize = (10,10), mpl_dpi = 120, alpha = alpha, beta = beta)

# Extract the base image, first mask, and second mask
base_image = inp_bchw[0, 0].cpu().numpy()
first_mask = first_masked[0, 0].cpu().numpy()
second_mask = masked_bchw[0, 0].cpu().numpy()

# Create a function to generate a white mask
def create_white_mask(mask):
    white_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    masked_areas = mask == 1  # Adjust this condition based on how your mask is defined
    white_mask[masked_areas] = [200, 200, 200]
    return white_mask

# Generate white masks for first and second masks
white_mask_first = create_white_mask(first_mask)
white_mask_second = create_white_mask(second_mask)

# Create figure for plotting
plt.figure(figsize=(25, 5))

plt.subplot2grid((1, 5), (0, 0))
plt.imshow(base_image*first_mask, cmap=plt.cm.gray)
# plt.imshow(first_mask, cmap=plt.cm.gray, alpha=1)
plt.title('First mask overlaid')
plt.axis('off')


# Plot the overlay from 'overlay_masks' function
plt.subplot2grid((1, 5), (0, 1))
plt.imshow(fig)
plt.title('Overlay from overlay_masks')
plt.axis('off')

# Plot the original image with 'pred_loss' overlaid
plt.subplot2grid((1, 5), (0, 2))
plt.imshow(base_image, cmap=plt.cm.gray)
plt.imshow(loss_bchw[0, 0].cpu().numpy(), cmap='jet', alpha=0.6)
plt.title('pred_loss overlaid')
plt.axis('off')

# Plot the base image and overlay the white mask for the first mask
plt.subplot2grid((1, 5), (0, 3))
plt.imshow(base_image, cmap=plt.cm.gray)
plt.imshow(white_mask_first, alpha=0.6)  # Overlay the white mask with some transparency
plt.title('first mask with white masked regions')
plt.axis('off')

# Plot the base image and overlay the white mask for the second mask
plt.subplot2grid((1, 5), (0, 4))
plt.imshow(base_image, cmap=plt.cm.gray)
plt.imshow(white_mask_second, alpha=0.6)
plt.title('second mask with white masked regions')
plt.axis('off')

plt.show()
plt.savefig('loss_mask_with_white.png')
plt.close()

