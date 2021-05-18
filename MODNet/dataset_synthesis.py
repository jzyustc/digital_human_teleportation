import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.models.modnet import MODNet

ckpt_path = "results/models/modnet_photographic_portrait_matting.ckpt"

for dataset_type in ["man_in_white/", "man_with_pose/", "mixed/", "woman_with_food/"]:

	raw_dataset_path = "YourPathForDatesets/paixin/" + dataset_type
	comp_dataset_path = "YourPathForDatesets/paixin_comp_mod/" + dataset_type

	raw_path = os.path.join(comp_dataset_path, "raw_image")
	mask_path = os.path.join(comp_dataset_path, "mask_image")

	if not os.path.exists(comp_dataset_path): os.mkdir(comp_dataset_path)
	if not os.path.exists(raw_path): os.mkdir(raw_path)
	if not os.path.exists(mask_path): os.mkdir(mask_path)

	# define hyper-parameters
	ref_size = 512

	# define image to tensor transform
	im_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]
	)

	modnet = MODNet(backbone_pretrained=False)
	modnet = nn.DataParallel(modnet).cuda()
	modnet.load_state_dict(torch.load(ckpt_path))
	modnet.eval()

	# inference results
	im_names = os.listdir(raw_dataset_path)
	for im_name in im_names:
		print('Process image: {0}'.format(im_name))

		# read image
		im = Image.open(os.path.join(raw_dataset_path, im_name))

		# unify image channels to 3
		im = np.asarray(im)
		if len(im.shape) == 2:
			im = im[:, :, None]
		if im.shape[2] == 1:
			im = np.repeat(im, 3, axis=2)
		elif im.shape[2] == 4:
			im = im[:, :, 0:3]

		# convert image to PyTorch tensor
		im = Image.fromarray(im)
		im = im_transform(im)

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
		_, _, matte = modnet(im.cuda(), True)

		# saved file name
		image_name = im_name.split('.')[0] + '.png'

		# save raw image
		im = im_raw[0].permute(1, 2, 0).data.cpu().numpy() * 0.5 + 0.5
		Image.fromarray(((im * 255).astype('uint8'))).save(os.path.join(raw_path, image_name))

		# resize and save matte
		matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
		matte = matte[0, 0].data.cpu().numpy()
		Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(mask_path, image_name))
