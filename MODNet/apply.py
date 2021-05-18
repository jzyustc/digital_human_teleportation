import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pymatting import *

from models.models.modnet import MODNet

# ------- 0. get settings --------
fg_path = "results/inputs/fg.jpg"  # change it
bg_path = "results/inputs/bg.JPEG"  # change it
output_path = "results/outputs/"  # change it
ckpt_path = "results/models/modnet_photographic_portrait_matting.ckpt"  # change it

# define hyper-parameters
ref_size = 512  # change it

print('Process image: ', fg_path)

# ------- 1. load model --------

# load models
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load(ckpt_path))
modnet.eval()

# ------- 2. applying process --------

# define image to tensor transform
im_transform = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]
)

# load image
im = np.asarray(Image.open(fg_path))

# reformat to rgb-channels
if len(im.shape) == 2:
	im = im[:, :, None]
if im.shape[2] == 1:
	im = np.repeat(im, 3, axis=2)
elif im.shape[2] == 4:
	im = im[:, :, :3]

# convert image to PyTorch tensor
im = im_transform(Image.fromarray(im))

# add mini-batch dim
im = im[None, :, :, :]

# resize image for input
im_b, im_c, im_h, im_w = im.shape
if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
	if im_w >= im_h:
		im_rh = ref_size
		im_rw = int(im_w / im_h * ref_size)
	elif im_w < im_h:
		im_rw = ref_size
		im_rh = int(im_h / im_w * ref_size)
else:
	im_rh = im_h
	im_rw = im_w

im_rw = im_rw - im_rw % 32
im_rh = im_rh - im_rh % 32
im_raw = im.clone()
im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

# inference
_, _, mask = modnet(im.cuda(), True)

# ------- 3. output --------

# save raw image
im = im_raw[0].permute(1, 2, 0).data.cpu().numpy() * 0.5 + 0.5
Image.fromarray(((im * 255).astype('uint8'))).save(os.path.join(output_path, "raw_image.png"))
print("saved : ", os.path.join(output_path, "raw_image.png"))

# resize and save mask
mask = F.interpolate(mask, size=(im_h, im_w), mode='area')
mask = mask[0, 0].data.cpu().numpy()
Image.fromarray(((mask * 255).astype('uint8')), mode='L').save(os.path.join(output_path, "mask.png"))
print("saved : ", os.path.join(output_path, "mask.png"))

# save the foreground after multi-level foreground estimation
foreground = estimate_foreground_ml(im, mask)
Image.fromarray((foreground * 255).astype(np.uint8)).save(os.path.join(output_path, "fg.png"))
print("saved : ", os.path.join(output_path, "fg.png"))

# green background
background = np.zeros_like(im)
background[:, :] = [0, 177 / 255, 64 / 255]
new_image = blend(foreground, background, mask)
Image.fromarray((new_image * 255).astype(np.uint8)).save(os.path.join(output_path, "green_bg.png"))
print("saved : ", os.path.join(output_path, "green_bg.png"))

# real background
b_image = Image.open(bg_path).convert('RGB')
b_image = b_image.resize((im.shape[1], im.shape[0]), Image.ANTIALIAS)
b_image = np.array(b_image, dtype=np.float32) / 255.
b_new_image = blend(foreground, b_image, mask)
Image.fromarray((b_new_image * 255).astype(np.uint8)).save(os.path.join(output_path, "real_bg.png"))
print("saved : ", os.path.join(output_path, "real_bg.png"))
