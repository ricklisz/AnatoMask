import os
import nibabel as nib
import json
import numpy as np
import torch

chkp = torch.load('/home/yl_li/STUNet/SelfMedMAE/SelfMedMAE_output/ssl-framework/mae3d_vit_base_Dataset501/ckpts/checkpoint_latest.pth.tar')

print(chkp['epoch'])

for k,v in enumerate(chkp):
    print(k, v)

# raw_folder = '/scr/yl_li/segmentation_data/nnUNet_raw_data_base/nnUNet_raw_data/Dataset221_AutoPETII_2023/labelsTr'
#
# with open('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset221_AutoPETII_2023/tumor_splits_final.json', 'r') as file:
#     orig_json = json.load(file)
# print(len(orig_json))

# for k,v in enumerate(orig_json):
#     print(k)
#     print(v['val'][0])
#
# images_with_ones = []
# for filename in os.listdir(raw_folder):
#     if filename.endswith(".nii.gz") or filename.endswith(".gz"):  # Adjust the extension according to your image files
#         file_path = os.path.join(raw_folder, filename)
#         img = nib.load(file_path)
#         img_data = img.get_fdata()
#
#         if np.any(img_data == 1):  # Check if there's any 1 in the image data
#             images_with_ones.append(filename.split('.nii.gz')[0])
#             print(filename)
#
# print(len(images_with_ones))
#
#
# for outer_key in range(5):
#     for inner_key in ['train', 'val']:
#         orig_json[outer_key][inner_key] = [item for item in orig_json[outer_key][inner_key] if item in images_with_ones]
#
# print(orig_json)
#
# with open('/scr/yl_li/segmentation_data/nnUNet_preprocessed/Dataset221_AutoPETII_2023/tumor_splits_final.json','w') as file:
#     json.dump(orig_json, file, indent=4)  # 'indent=4' for pretty printing
#
#
