import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pymatting import *

from MODNet.models.models.modnet import MODNet

'''
STEP 1 : Matting from a rgb image
'''


def step1_matting(fg_image, ckpt_path="MODNet/results/models/modnet_photographic_portrait_matting.ckpt",
				  ref_size=512, fg_name="fg.png", mask_name="mask.png", results_path="results/"):
	# ------- 0. get settings --------

	print('STEP 1 : Matting')
	# ------- 1. load model --------

	use_gpu = False
	device = torch.device("cuda") if use_gpu else torch.device("cpu")

	# load models
	modnet = MODNet(backbone_pretrained=False).to(device)
	modnet = nn.DataParallel(modnet)
	if use_gpu:
		modnet.load_state_dict(torch.load(ckpt_path))
	else:
		modnet.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
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
	im = np.asarray(fg_image)

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
		else:
			im_rw = ref_size
			im_rh = int(im_h / im_w * ref_size)
	else:
		im_rh = im_h
		im_rw = im_w

	im_rw = im_rw - im_rw % 32
	im_rh = im_rh - im_rh % 32
	im_raw = im.clone()
	im = nn.Upsample(size=(im_rh, im_rw), mode='area')(im)

	# inference
	_, _, mask = modnet(im.to(device), True)

	# ------- 3. output --------

	# resize and save mask
	mask = nn.Upsample(size=(im_h, im_w), mode='area')(mask)
	mask = mask[0, 0].data.cpu().numpy()
	Image.fromarray(((mask * 255).astype('uint8')), mode='L').save(os.path.join(results_path, mask_name))
	print("Matting : mask saved : ", os.path.join(results_path, mask_name))

	# save the foreground after multi-level foreground estimation
	im = im_raw[0].permute(1, 2, 0).data.cpu().numpy() * 0.5 + 0.5
	foreground = estimate_foreground_ml(im, mask)
	Image.fromarray((foreground * 255).astype(np.uint8)).save(os.path.join(results_path, fg_name))
	print("Matting : foreground saved : ", os.path.join(results_path, fg_name))


if __name__ == "__main__":
	# ---- settings ----

	# define path
	results_path = "results/"  # change it
	fg_image_name = "human.jpg"  # change it
	ckpt_path = "MODNet/results/models/modnet_photographic_portrait_matting.ckpt"  # change it

	# define hyper-parameters
	ref_size = 512  # change it

	# image
	fg_image = Image.open(os.path.join(results_path, fg_image_name))

	step1_matting(fg_image, ckpt_path=ckpt_path, ref_size=ref_size, results_path=results_path)
