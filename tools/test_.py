###################################### mask
import torch
mask_ratio = 0.86
x = torch.rand([20,196,3])
N, L, D = x.shape  # batch, length, dim
out_x, out_y = 16, 16
block_size_x = int(out_x * mask_ratio)
block_size_y = int(out_y * mask_ratio)

start_x = torch.randint(0, out_x - block_size_x-1, (N,))
start_y = torch.randint(0, out_y - block_size_y-1, (N,))

# Generate the binary mask: 0 is keep, 1 is remove
mask = torch.zeros([N, out_x, out_y], device=x.device)

# ids_keep = start_x + start_y * out_x
for i in range(N):
    mask[i,start_x[i]:(start_x[i]+block_size_x+1),start_y[i]:(start_y[i]+block_size_y+1)]=1

mask = mask.flatten(1)

# Calculate ids_keep
ids_keep = torch.nonzero(mask==0, as_tuple=True)[1].view(N, -1)

# Calculate ids_restore
ids_restore = torch.arange(0, L).unsqueeze(0).repeat(N, 1)

# Calculate x_masked
expanded_ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
# x_masked = torch.gather(x, dim=1, index=expanded_ids_keep)



# noise = torch.rand(5, 6, device='cuda')  # noise in [0, 1]

# # # sort noise for each sample
# ids_shuffle = torch.argsort(
#     noise, dim=1)  # ascend: small is keep, large is remove
# ids_restore = torch.argsort(ids_shuffle, dim=1)
print(111)


###################################### mask vis
# import sys
# import os
# import requests

# import torch
# import numpy as np

# import matplotlib.pyplot as plt
# from PIL import Image

# import mmpretrain
# from mmpretrain.apis import init_model
# def prepare_model(config, chkpt_dir):
#     # build model
#     model = init_model(config, chkpt_dir)
#     return model
# chkpt_dir = '/scratch/yw6594/hpml/mmpretrain/omit/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth'
# config = '/scratch/yw6594/hpml/mmpretrain/configs_mask/mae/maevit_vis.py'
# model_mae = prepare_model(config, chkpt_dir)


# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
# img_url = '/scratch/yw6594/hpml/mmpretrain/data/imagenet100/train/n01558993/n01558993_3430.JPEG' # bird
# img = Image.open(img_url)
# img = img.resize((224, 224))
# img = np.array(img) / 255.

# assert img.shape == (224, 224, 3)

# # normalize by ImageNet mean and std
# img = img - imagenet_mean
# img = img / imagenet_std


# x = torch.tensor(img)

# # make it a batch-like
# x = x.unsqueeze(dim=0)
# x = torch.einsum('nhwc->nchw', x)

# run MAE
# y, mask = model_mae.visualize(x.float())
# y = model_mae.head.unpatchify(y)